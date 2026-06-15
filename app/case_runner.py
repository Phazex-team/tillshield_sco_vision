"""End-to-end case analyzer.

Given a case id, resolve its CCTV window, run perception, run the
reasoning chain, persist the VLM run row, write the evidence package,
wrap the result with the decision policy, and update the case row.

This is the function called by the API's reprocess endpoint and by the
background analysis worker. It is intentionally synchronous and
mockable: the perception pipeline is invoked via dependency-injection
hooks so tests can replace real model calls with stubs.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from sqlalchemy.orm import Session

from db.models import Case, PosEvent, VideoWindow, VlmRun
from pos.correlation import plan_window


log = logging.getLogger(__name__)


def analyze_case(session: Session,
                 case_id: str,
                 *,
                 perception_runner: Optional[Callable] = None,
                 vlm_runner: Optional[Callable] = None,
                 prompt_version: str = "return_review_v1") -> dict:
    """Run the full analysis for ``case_id`` and persist results.

    ``perception_runner`` and ``vlm_runner`` are optional injection
    points used by tests. Production code leaves them ``None`` and the
    function builds the real chain provider.
    """
    from app import audit
    from evidence.package import write_package
    from reasoning.decision_policy import (
        OUTCOME_INVALID_VIDEO,
        OUTCOME_VERIFIED,
        decide,
        summary_from_vlm,
    )

    case = session.get(Case, case_id)
    if case is None:
        raise KeyError(f"case {case_id} not found")
    pos = session.get(PosEvent, case.pos_event_id) if case.pos_event_id else None
    if pos is None:
        raise ValueError("case has no POS event linked")

    # 1. Resolve window
    plan = plan_window(session, case.camera_id, pos.pos_event_at)
    window = VideoWindow(
        case_id=case.id,
        camera_id=case.camera_id,
        requested_start_at=plan.requested_start,
        requested_end_at=plan.requested_end,
        actual_start_at=plan.actual_start,
        actual_end_at=plan.actual_end,
        segment_ids=list(plan.matched_segment_ids),
        status="SUCCEEDED" if plan.is_valid else "FAILED",
        failure_reason=plan.invalid_reason,
    )
    session.add(window)
    session.flush()

    audit.record(session, action="case.window_resolved",
                 entity_type="case", entity_id=case.id,
                 actor_type="analyzer",
                 after={"window_id": window.id,
                        "coverage_ratio": plan.coverage_ratio,
                        "valid": plan.is_valid,
                        "invalid_reason": plan.invalid_reason})

    if not plan.is_valid:
        case.outcome = OUTCOME_INVALID_VIDEO
        case.status = "CLOSED"
        case.invalid_reason = plan.invalid_reason
        case.decision_policy_version = "v1"
        case.closed_at = datetime.now(timezone.utc)
        package = write_package(session, case.id)
        audit.record(session, action="case.decided",
                     entity_type="case", entity_id=case.id,
                     actor_type="analyzer",
                     after={"outcome": case.outcome,
                            "package_sha256": package["sha256"]})
        session.commit()
        return {"case_id": case.id, "outcome": case.outcome,
                "invalid_reason": case.invalid_reason,
                "package": package["uri"]}

    # 2. Perception (injected for tests; real pipeline lazy-loads here)
    perception_result = None
    if perception_runner is None:
        try:
            from perception.pipeline import run_perception
            perception_runner = run_perception
        except Exception as exc:
            log.warning("perception pipeline unavailable: %s", exc)
    if perception_runner is not None:
        try:
            perception_result = perception_runner(session, case, window)
        except Exception:
            log.exception("perception failed; treating as obstructed evidence")
            perception_result = None

    # 3. Reasoning
    vlm_result_dict: dict = {}
    if vlm_runner is None:
        try:
            from reasoning.providers import build_active_provider
            from reasoning.providers.base import EvidenceManifest
            from app.config import load_config
            chain = build_active_provider(load_config())
            manifest = EvidenceManifest(
                case_id=case.id,
                camera_id=case.camera_id,
                window_start_ts=plan.requested_start.isoformat(),
                window_end_ts=plan.requested_end.isoformat(),
                frames=[],  # filled by perception pipeline in production
                tracks=(perception_result or {}).get("tracks", []),
                ocr=(perception_result or {}).get("ocr", []),
                metadata={"prompt_version": prompt_version},
            )
            vlm_runner = lambda s, c, w: _adapt_vlm_result(  # noqa: E731
                chain.analyze_evidence(manifest), prompt_version)
        except Exception as exc:
            log.warning("VLM chain unavailable: %s", exc)

    if vlm_runner is not None:
        try:
            vlm_result_dict = vlm_runner(session, case, window) or {}
        except Exception as exc:
            log.exception("VLM run raised")
            vlm_result_dict = {"error": str(exc),
                               "provider": "chain", "model_name": "chain"}

    # 4. Persist VLM run row
    run = VlmRun(
        case_id=case.id,
        provider=vlm_result_dict.get("provider", "chain"),
        model_name=vlm_result_dict.get("model_name", "chain"),
        model_snapshot=vlm_result_dict.get("model_snapshot"),
        prompt_version=prompt_version,
        input_manifest={"window_id": window.id,
                        "perception": _summarise_perception(perception_result)},
        output_json=vlm_result_dict.get("parsed", {}),
        status="FAILED" if vlm_result_dict.get("error") else "SUCCEEDED",
        latency_ms=vlm_result_dict.get("latency_ms"),
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        error=vlm_result_dict.get("error"),
    )
    session.add(run)
    session.flush()

    # 5. Decision policy
    summary = summary_from_vlm(
        vlm_result_dict.get("parsed", {}),
        footage_valid=True,
        obstructed=_perception_obstructed(perception_result),
        camera_gap=False,
    )
    if vlm_result_dict.get("error"):
        summary.contradictions.append(f"vlm error: {vlm_result_dict['error']}")
    decision = decide(summary)
    case.outcome = decision.outcome
    case.risk_score = decision.risk_score
    case.risk_reasons = list(decision.reasons)
    case.decision_policy_version = decision.policy_version
    case.status = "CLOSED" if decision.outcome != OUTCOME_VERIFIED \
        else "CLOSED"  # ↑ even VERIFIED still routes through reviewer queue
    case.closed_at = datetime.now(timezone.utc)

    # 6. Evidence package
    package = write_package(session, case.id)

    audit.record(session, action="case.decided",
                 entity_type="case", entity_id=case.id,
                 actor_type="analyzer",
                 after={"outcome": case.outcome,
                        "risk_score": decision.risk_score,
                        "reasons": list(decision.reasons),
                        "package_sha256": package["sha256"]})
    session.commit()
    return {
        "case_id": case.id,
        "outcome": case.outcome,
        "risk_score": case.risk_score,
        "reasons": list(decision.reasons),
        "package": package["uri"],
        "vlm_run_id": run.id,
    }


def _adapt_vlm_result(vlm_result, prompt_version: str) -> dict:
    return {
        "provider": vlm_result.provider,
        "model_name": vlm_result.model_name,
        "parsed": dict(vlm_result.parsed or {}),
        "latency_ms": vlm_result.latency_ms,
        "error": vlm_result.error,
        "prompt_version": prompt_version,
    }


def _summarise_perception(perception_result: Optional[dict]) -> dict:
    if not perception_result:
        return {"tracks": 0, "keyframes": 0, "ocr": 0}
    return {
        "tracks": len(perception_result.get("tracks") or []),
        "keyframes": len(perception_result.get("keyframes") or []),
        "ocr": len(perception_result.get("ocr") or []),
        "limitations": perception_result.get("limitations") or [],
    }


def _perception_obstructed(perception_result: Optional[dict]) -> bool:
    if not perception_result:
        return False
    if perception_result.get("obstructed") is not None:
        return bool(perception_result["obstructed"])
    limitations = perception_result.get("limitations") or []
    return any("obstruct" in str(l).lower() for l in limitations)
