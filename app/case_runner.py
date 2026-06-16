"""End-to-end case analyzer.

Given a case id, this:

1. Resolves the CCTV window via ``pos.correlation.plan_window``.
2. Reconstructs the actual window MP4 from the matched immutable
   segments via ``video.window_builder.build_window``. If the build
   fails, the case is marked ``INVALID_VIDEO`` and analysis stops.
3. Runs the perception pipeline on the rebuilt window.
4. Extracts up to ``manifest_max_frames`` frames from the window MP4
   and encodes them as ``data:image/jpeg;base64`` entries in the
   ``EvidenceManifest``. The reasoning chain NEVER sees frames=[] for
   a valid window.
5. Calls the active provider chain (Qwen3-VL primary, Gemma fallback).
6. Persists the VLM run row + the perception evidence + writes a
   versioned evidence package.
7. Wraps the result with the deterministic decision policy.
"""
from __future__ import annotations

import base64
import io
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from sqlalchemy.orm import Session

from db.models import (
    Case,
    PosEvent,
    VideoSegment,
    VideoWindow,
    VlmRun,
)
from pos.correlation import plan_window


log = logging.getLogger(__name__)


def analyze_case(session: Session,
                 case_id: str,
                 *,
                 perception_runner: Optional[Callable] = None,
                 vlm_runner: Optional[Callable] = None,
                 prompt_version: str = "return_review_v1",
                 manifest_max_frames: int = 12) -> dict:
    """Run the full analysis for ``case_id`` and persist results.

    ``perception_runner`` and ``vlm_runner`` are dependency-injection
    hooks. Tests usually only override ``vlm_runner`` so they exercise
    real window-building + real frame extraction.
    """
    from app import audit
    from evidence.package import write_package
    from evidence.persistence import persist_perception
    from reasoning.decision_policy import (
        OUTCOME_INVALID_VIDEO,
        OUTCOME_VERIFIED,
        decide,
        summary_from_vlm,
    )
    from video.window_builder import build_window

    case = session.get(Case, case_id)
    if case is None:
        raise KeyError(f"case {case_id} not found")
    pos = session.get(PosEvent, case.pos_event_id) if case.pos_event_id else None
    if pos is None:
        raise ValueError("case has no POS event linked")

    # ----- 1. Resolve window from segment index ----------------------
    storage_root = _storage_root()
    plan = plan_window(session, case.camera_id, pos.pos_event_at)
    window = VideoWindow(
        case_id=case.id,
        camera_id=case.camera_id,
        requested_start_at=plan.requested_start,
        requested_end_at=plan.requested_end,
        actual_start_at=plan.actual_start,
        actual_end_at=plan.actual_end,
        segment_ids=list(plan.matched_segment_ids),
        status="PENDING",
    )
    session.add(window)
    session.flush()

    # ----- 1b. NVR on-demand retrieval (preferred-but-NONBLOCKING) ----
    # Try to pull the historical window from the NVR for this tx. This
    # only annotates metadata + (if export is enabled and succeeds)
    # provides the exact clip; otherwise the local recorder/segment path
    # below is used. Never raises.
    nvr_clip = _try_nvr_window(case, pos.pos_event_at, window, storage_root)

    audit.record(session, action="case.window_resolved",
                 entity_type="case", entity_id=case.id,
                 actor_type="analyzer",
                 after={"window_id": window.id,
                        "coverage_ratio": plan.coverage_ratio,
                        "valid": plan.is_valid,
                        "invalid_reason": plan.invalid_reason,
                        "acquisition_source": window.acquisition_source,
                        "nvr": window.nvr_metadata})

    out_path = (storage_root / "cases" / f"case_id={case.id}" / "window"
                / f"window_{window.id}.mp4")

    # ----- 2. Acquire the window MP4: NVR clip first, else local ------
    if nvr_clip:
        build = _adopt_clip(nvr_clip, plan)
        window.acquisition_source = "nvr_clip_retrieved"
    else:
        # Fall back to the existing local recorded-segment path.
        if not plan.is_valid:
            window.status = "FAILED"
            window.failure_reason = plan.invalid_reason
            # Leave the NVR state (e.g. nvr_recording_found_no_export) so
            # operators can see footage exists on the NVR even when local
            # segments are missing; only default it when NVR was off.
            if not window.acquisition_source:
                window.acquisition_source = "local_no_segments"
            return _close_invalid(session, case, plan.invalid_reason)
        segments = (session.query(VideoSegment)
                    .filter(VideoSegment.id.in_(plan.matched_segment_ids))
                    .order_by(VideoSegment.start_at.asc()).all())
        build = build_window(
            segments=segments,
            requested_start=plan.requested_start,
            requested_end=plan.requested_end,
            out_path=out_path,
        )
        if not build.ok:
            window.status = "FAILED"
            window.failure_reason = build.failure_reason
            audit.record(session, action="case.window_build_failed",
                         entity_type="case", entity_id=case.id,
                         actor_type="analyzer",
                         after={"window_id": window.id,
                                "failure_reason": build.failure_reason})
            return _close_invalid(session, case, build.failure_reason)
        window.acquisition_source = "local_segments_used"

    window.path = build.out_path
    window.sha256 = build.sha256
    window.actual_start_at = build.actual_start_at
    window.actual_end_at = build.actual_end_at
    window.status = "SUCCEEDED"

    # ----- 3. Perception ---------------------------------------------
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

    # Persist perception evidence so the evidence package + graph see
    # real rows, not in-memory dicts.
    if perception_result:
        try:
            persist_perception(session, case_id=case.id,
                               window_id=window.id,
                               perception=perception_result)
        except Exception:
            log.exception("perception persist failed (continuing)")

    # ----- 4. Build the evidence manifest with REAL frames -----------
    window_start_naive = build.actual_start_at or plan.requested_start
    sampled_frames = _extract_keyframe_data_urls(
        window_path=build.out_path,
        window_start_ts=window_start_naive,
        keyframes=(perception_result or {}).get("keyframes") or [],
        max_frames=manifest_max_frames,
    )

    # ----- 5. Reasoning ----------------------------------------------
    # Build the manifest once. Both the real chain and any injected
    # ``vlm_runner`` receive it as a fourth positional arg, so tests can
    # assert that ``manifest.frames`` is non-empty for a valid window.
    from reasoning.providers.base import EvidenceManifest
    manifest = EvidenceManifest(
        case_id=case.id,
        camera_id=case.camera_id,
        window_start_ts=(build.actual_start_at
                         or plan.requested_start).isoformat(),
        window_end_ts=(build.actual_end_at
                       or plan.requested_end).isoformat(),
        frames=sampled_frames,
        tracks=(perception_result or {}).get("tracks") or [],
        ocr=(perception_result or {}).get("ocr") or [],
        metadata={"prompt_version": prompt_version},
    )

    vlm_result_dict: dict = {}
    if vlm_runner is None:
        try:
            from app.config import load_config
            from reasoning.providers import build_active_provider
            chain = build_active_provider(load_config())
            vlm_runner = lambda s, c, w, m: _adapt_vlm_result(  # noqa: E731
                chain.analyze_evidence(m), prompt_version)
        except Exception as exc:
            log.warning("VLM chain unavailable: %s", exc)

    if vlm_runner is not None:
        try:
            # Production / new-style runners take (session, case, window,
            # manifest). Old test-style runners take (session, case,
            # window); fall back transparently.
            try:
                vlm_result_dict = vlm_runner(session, case, window,
                                             manifest) or {}
            except TypeError:
                vlm_result_dict = vlm_runner(session, case, window) or {}
        except Exception as exc:
            log.exception("VLM run raised")
            vlm_result_dict = {"error": str(exc),
                               "provider": "chain", "model_name": "chain"}

    # ----- 6. Persist VLM run row ------------------------------------
    run = VlmRun(
        case_id=case.id,
        provider=vlm_result_dict.get("provider", "chain"),
        model_name=vlm_result_dict.get("model_name", "chain"),
        model_snapshot=vlm_result_dict.get("model_snapshot"),
        prompt_version=prompt_version,
        input_manifest={"window_id": window.id,
                        "frame_count": len(sampled_frames),
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

    # ----- 7. Decision policy ----------------------------------------
    # Track-gated: pass the perception_result through so the policy can
    # derive physical_item_track from real persisted tracks. Without
    # this, the VLM could upgrade a case alone — the contract the
    # K-series fixed in summary_from_vlm.
    summary = summary_from_vlm(
        vlm_result_dict.get("parsed", {}),
        footage_valid=True,
        obstructed=_perception_obstructed(perception_result),
        camera_gap=False,
        perception_result=perception_result,
    )
    if vlm_result_dict.get("error"):
        summary.contradictions.append(f"vlm error: {vlm_result_dict['error']}")
    decision = decide(summary)
    case.outcome = decision.outcome
    case.risk_score = decision.risk_score
    case.risk_reasons = list(decision.reasons)
    case.decision_policy_version = decision.policy_version
    case.status = "CLOSED"
    case.closed_at = datetime.now(timezone.utc)

    # ----- 8. Evidence package ---------------------------------------
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
        "window_id": window.id,
        "window_path": window.path,
    }


def _close_invalid(session: Session, case: Case,
                   reason: Optional[str]) -> dict:
    from app import audit
    from evidence.package import write_package
    from reasoning.decision_policy import OUTCOME_INVALID_VIDEO

    case.outcome = OUTCOME_INVALID_VIDEO
    case.status = "CLOSED"
    case.invalid_reason = reason
    case.decision_policy_version = "v1"
    case.closed_at = datetime.now(timezone.utc)
    package = write_package(session, case.id)
    audit.record(session, action="case.decided",
                 entity_type="case", entity_id=case.id,
                 actor_type="analyzer",
                 after={"outcome": case.outcome,
                        "invalid_reason": reason,
                        "package_sha256": package["sha256"]})
    session.commit()
    return {"case_id": case.id, "outcome": case.outcome,
            "invalid_reason": reason, "package": package["uri"]}


def _storage_root() -> Path:
    from app.config import load_config
    return load_config().storage_root


def _camera_cfg(camera_id: str) -> dict:
    from app.config import load_config
    for c in load_config().cameras:
        if c.get("id") == camera_id:
            return c
    return {}


def _try_nvr_window(case, pos_event_at, window, storage_root) -> Optional[str]:
    """Run NVR on-demand acquisition for this case window.

    Annotates ``window.acquisition_source`` + ``window.nvr_metadata`` and
    returns a local clip path when the exact historical clip was actually
    exported (Stage B); otherwise returns ``None`` so the caller uses the
    local recorded segments. NEVER raises — NVR is non-blocking."""
    try:
        from video.nvr_dahua import (
            acquire_window, load_nvr_config, STATE_CLIP_RETRIEVED)
        nvr_cfg = load_nvr_config(_camera_cfg(case.camera_id))
        out = (storage_root / "cases" / f"case_id={case.id}" / "window"
               / f"nvr_{window.id}.mp4")
        acq = acquire_window(nvr_cfg, pos_event_at,
                             camera_id=case.camera_id, out_path=str(out))
        window.nvr_metadata = acq.metadata
        if acq.attempted:
            window.acquisition_source = acq.state
        if acq.state == STATE_CLIP_RETRIEVED and acq.clip_path:
            return acq.clip_path
    except Exception:
        log.exception("nvr acquisition step failed (non-fatal)")
    return None


@dataclass
class _ClipBuild:
    """build_window-compatible result for an NVR-exported clip."""
    out_path: str
    sha256: Optional[str]
    actual_start_at: Optional[datetime]
    actual_end_at: Optional[datetime]
    ok: bool = True
    failure_reason: Optional[str] = None


def _adopt_clip(clip_path: str, plan) -> _ClipBuild:
    import hashlib
    h = hashlib.sha256()
    with open(clip_path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return _ClipBuild(out_path=clip_path, sha256=h.hexdigest(),
                      actual_start_at=plan.requested_start,
                      actual_end_at=plan.requested_end)


def _extract_keyframe_data_urls(*, window_path: str,
                                window_start_ts: datetime,
                                keyframes: list,
                                max_frames: int) -> list[dict]:
    """Decode a small number of frames from the built window and encode
    them as data URLs the provider chain can ship.

    Prefers timestamps recommended by perception keyframes; otherwise
    samples evenly across the window. If decoding fails entirely the
    function returns ``[]`` and the caller logs a limitation.
    """
    try:
        import cv2  # type: ignore
    except Exception:
        log.warning("cv2 unavailable; sending zero frames to VLM")
        return []
    if not window_path or not os.path.exists(window_path):
        log.warning("window file %r missing on disk", window_path)
        return []

    cap = cv2.VideoCapture(window_path)
    if not cap.isOpened():
        log.warning("cv2 could not open window %r", window_path)
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        cap.release()
        log.warning("window %r reports zero frames", window_path)
        return []

    # Choose frame indices: pull from keyframe roles first, dedupe,
    # cap at max_frames, then top up with evenly spaced indices.
    indices: list[int] = []
    for kf in keyframes:
        idx = kf.get("frame_idx") if isinstance(kf, dict) else None
        if isinstance(idx, int) and 0 <= idx < frame_count:
            indices.append(idx)
    indices = sorted(set(indices))[:max_frames]
    if len(indices) < max_frames:
        step = max(1, frame_count // max_frames)
        for i in range(0, frame_count, step):
            if i in indices:
                continue
            indices.append(i)
            if len(indices) >= max_frames:
                break
    indices = sorted(indices)[:max_frames]

    # Anchor frame timestamps to the actual CCTV window start so each
    # frame carries a wall-clock value aligned with the POS event,
    # not a synthetic placeholder.
    base_ts = _ensure_naive_utc(window_start_ts)

    out: list[dict] = []
    from PIL import Image
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            continue
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        out.append({
            "frame_id": f"frame_{idx:06d}",
            "frame_idx": int(idx),
            "ts": _ts_for_index(idx, fps, base_ts).isoformat(),
            "image_url": f"data:image/jpeg;base64,{b64}",
        })
    cap.release()
    return out


def _ensure_naive_utc(dt: datetime) -> datetime:
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _ts_for_index(idx: int, fps: float, base_ts: datetime) -> datetime:
    """Frame timestamp = ``base_ts + idx/fps``. ``base_ts`` is the
    real CCTV window start."""
    from datetime import timedelta
    return base_ts + timedelta(seconds=idx / max(fps, 1.0))


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
        return {"tracks": 0, "keyframes": 0, "ocr": 0, "detections": 0}
    return {
        "tracks": len(perception_result.get("tracks") or []),
        "keyframes": len(perception_result.get("keyframes") or []),
        "ocr": len(perception_result.get("ocr") or []),
        "detections": len(perception_result.get("detections") or []),
        "limitations": perception_result.get("limitations") or [],
    }


def _perception_obstructed(perception_result: Optional[dict]) -> bool:
    if not perception_result:
        return False
    if perception_result.get("obstructed") is not None:
        return bool(perception_result["obstructed"])
    limitations = perception_result.get("limitations") or []
    return any("obstruct" in str(l).lower() for l in limitations)
