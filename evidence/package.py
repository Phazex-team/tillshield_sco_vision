"""Evidence package builder (PRODUCTION_SPEC §12).

Produces an immutable, append-only JSON package for each case + writes
its manifest to ``storage/cases/case_id=<uuid>/package/``. The package
contains POS payload, video window metadata, perception results, the
VLM run, decision policy output, reviewer actions, and the
sha256 of the package itself.

Tracked artifact rows live in the ``artifacts`` table; the package
references them by uri + sha so reviewers can prove what evidence was
considered.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import Artifact, Case, PosEvent, ReviewAction, VlmRun


log = logging.getLogger(__name__)


PACKAGE_DIR_NAME = "package"


def _storage_root() -> Path:
    from app.config import load_config
    return load_config().storage_root


def case_dir(case_id: str) -> Path:
    return _storage_root() / "cases" / f"case_id={case_id}"


def _jsonable(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def build_package(session: Session, case_id: str) -> dict:
    """Assemble the package JSON for ``case_id`` from DB state.

    The returned dict is NOT yet persisted; pass it to ``write_package``
    to materialise it on disk and write the manifest hash.
    """
    case = session.get(Case, case_id)
    if case is None:
        raise KeyError(f"case {case_id!r} not found")

    pos = session.get(PosEvent, case.pos_event_id) if case.pos_event_id else None
    artifacts = session.execute(
        select(Artifact).where(Artifact.case_id == case.id)
    ).scalars().all()
    vlm_runs = session.execute(
        select(VlmRun).where(VlmRun.case_id == case.id)
        .order_by(VlmRun.started_at.asc())
    ).scalars().all()
    reviews = session.execute(
        select(ReviewAction).where(ReviewAction.case_id == case.id)
        .order_by(ReviewAction.created_at.asc())
    ).scalars().all()

    return {
        "case_id": case.id,
        "pos_event": _serialise_pos(pos) if pos else None,
        "case": {
            "camera_id": case.camera_id,
            "status": case.status,
            "outcome": case.outcome,
            "risk_score": case.risk_score,
            "risk_reasons": case.risk_reasons,
            "decision_policy_version": case.decision_policy_version,
            "opened_at": _jsonable(case.opened_at),
            "closed_at": _jsonable(case.closed_at),
            "invalid_reason": case.invalid_reason,
        },
        "artifacts": [_serialise_artifact(a) for a in artifacts],
        "reasoning": [
            {
                "provider": r.provider,
                "model_name": r.model_name,
                "model_snapshot": r.model_snapshot,
                "prompt_version": r.prompt_version,
                "status": r.status,
                "latency_ms": r.latency_ms,
                "output": r.output_json,
                "error": r.error,
            }
            for r in vlm_runs
        ],
        "reviews": [
            {
                "id": r.id,
                "reviewer_id": r.reviewer_id,
                "action": r.action,
                "outcome": r.outcome,
                "labels": r.labels,
                "notes": r.notes,
                "created_at": _jsonable(r.created_at),
            }
            for r in reviews
        ],
        "audit": {
            "built_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def write_package(session: Session, case_id: str) -> dict:
    """Build + persist the package. Returns the saved dict including
    the package_sha256 and on-disk path. Appends a versioned filename
    so prior packages are preserved (PRODUCTION_SPEC: packages are
    append/versioned, never overwritten silently)."""
    payload = build_package(session, case_id)
    blob = json.dumps(payload, indent=2, sort_keys=True,
                      default=str).encode()
    sha = hashlib.sha256(blob).hexdigest()
    payload["audit"]["package_sha256"] = sha

    out_dir = case_dir(case_id) / PACKAGE_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    # Versioned filename: pkg_<timestamp>_<sha8>.json
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fname = f"pkg_{ts}_{sha[:8]}.json"
    out_path = out_dir / fname
    final_blob = json.dumps(payload, indent=2, sort_keys=True,
                            default=str).encode()
    out_path.write_bytes(final_blob)

    # Record the package as an artifact row so downstream callers find
    # it via the case-artifacts relationship.
    art = Artifact(
        case_id=case_id,
        artifact_type="PACKAGE",
        uri=str(out_path),
        sha256=hashlib.sha256(final_blob).hexdigest(),
        mime_type="application/json",
        artifact_metadata={"versioned_filename": fname},
    )
    session.add(art)
    session.flush()
    return {
        "case_id": case_id,
        "uri": str(out_path),
        "sha256": art.sha256,
        "payload": payload,
    }


def latest_package_for_case(session: Session,
                            case_id: str) -> Optional[dict]:
    """Return the most recent persisted package payload for a case, or
    ``None`` if none yet exists."""
    rows = session.execute(
        select(Artifact)
        .where(Artifact.case_id == case_id,
               Artifact.artifact_type == "PACKAGE")
        .order_by(Artifact.created_at.desc())
    ).scalars().all()
    for art in rows:
        try:
            with open(art.uri, "rb") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
    return None


def _serialise_pos(ev) -> dict:
    return {
        "id": ev.id,
        "store_id": ev.store_id,
        "terminal_id": ev.terminal_id,
        "transaction_id": ev.transaction_id,
        "line_id": ev.line_id,
        "event_type": ev.event_type,
        "pos_event_at": _jsonable(ev.pos_event_at),
        "staff_id": ev.staff_id,
        "sku": ev.sku,
        "item_description": ev.item_description,
        "amount": ev.amount,
        "currency": ev.currency,
    }


def _serialise_artifact(a: Artifact) -> dict:
    return {
        "id": a.id,
        "artifact_type": a.artifact_type,
        "uri": a.uri,
        "sha256": a.sha256,
        "mime_type": a.mime_type,
        "frame_ts": _jsonable(a.frame_ts),
        "frame_idx": a.frame_idx,
        "metadata": a.artifact_metadata,
    }
