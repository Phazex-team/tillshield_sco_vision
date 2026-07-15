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

from db.models import (
    Artifact,
    Case,
    Detection,
    Keyframe,
    OcrResult,
    PosEvent,
    ReviewAction,
    Track,
    TrackObservation,
    VlmRun,
)


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
    detections = session.execute(
        select(Detection).where(Detection.case_id == case.id)
        .order_by(Detection.frame_idx.asc())
    ).scalars().all()
    tracks = session.execute(
        select(Track).where(Track.case_id == case.id)
        .order_by(Track.first_seen_ts.asc())
    ).scalars().all()
    keyframes = session.execute(
        select(Keyframe).where(Keyframe.case_id == case.id)
        .order_by(Keyframe.frame_idx.asc())
    ).scalars().all()
    ocr_rows = session.execute(
        select(OcrResult).where(OcrResult.case_id == case.id)
    ).scalars().all()
    track_obs = session.execute(
        select(TrackObservation).where(
            TrackObservation.track_id.in_([t.id for t in tracks]) if tracks
            else False)
    ).scalars().all() if tracks else []

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
        "perception": {
            "detections": [_serialise_detection(d) for d in detections],
            "tracks": [_serialise_track(t) for t in tracks],
            "track_observations": [_serialise_observation(o)
                                    for o in track_obs],
            "keyframes": [_serialise_keyframe(k) for k in keyframes],
            "ocr": [_serialise_ocr(o) for o in ocr_rows],
            # Saved Falcon detection stills. Unlike raw artifacts (whose
            # ``uri`` is an on-disk path), each entry carries a browser-
            # fetchable ``url`` served by app.api.video.detection_snapshot.
            "detection_snapshots": [
                _serialise_detection_snapshot(a, case.id) for a in artifacts
                if a.artifact_type == "DETECTION_SNAPSHOT"
            ],
        },
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
                # Lightweight processing-time observability. Only the
                # timings sub-dict is exposed (not the full input
                # manifest) so base64 frame URLs / large provider
                # metadata never leak into the package payload. Missing
                # data renders as a missing key in JSON; the UI shows
                # ``—`` rather than synthesising a number.
                "processing_timings_ms":
                    (r.input_manifest or {}).get("processing_timings_ms"),
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


_PACKAGE_SHA_PLACEHOLDER = "0" * 64


def write_package(session: Session, case_id: str) -> dict:
    """Build + persist the package. Three distinct hashes are recorded:

      * ``audit.content_sha256``: sha256 of the payload BEFORE the
        package_sha256 field exists. Stable across reads of the same
        case state (modulo ``built_at``).
      * ``audit.package_sha256``: self-verifying hash. Computed as
        sha256 of the full file with the package_sha256 field value
        replaced by 64 zeros. ``evidence.verify_package_file`` can
        reproduce it offline to detect tampering.
      * ``audit.file_sha256``: the LITERAL sha256 of the bytes on
        disk. ``Artifact.sha256`` stores this same value, so a
        reviewer can verify integrity with a one-liner
        ``sha256sum pkg_*.json`` without knowing the zero-field scheme.

    Returns a dict with all three hashes so callers can use whichever
    is appropriate. Append-versioned filename so prior packages are
    preserved.
    """
    payload = build_package(session, case_id)

    # 1) content_sha256 — sha of the payload BEFORE any audit hash
    # fields are added. Stable across reads of the same case state
    # (modulo audit.built_at).
    content_blob = json.dumps(payload, indent=2, sort_keys=True,
                              default=str).encode()
    content_sha = hashlib.sha256(content_blob).hexdigest()
    payload["audit"]["content_sha256"] = content_sha

    # 2) self-verifying package_sha256 (zeroed-field scheme). This is
    # the ONLY hash embedded in the file body, and it is reproducible
    # offline via ``evidence.verify_package_file``. We deliberately do
    # NOT embed a "file_sha256" field — a file cannot contain its own
    # literal sha (any attempt to do so changes the bytes and breaks
    # the equality). Reviewers who want the literal hash use
    # ``sha256sum pkg_*.json`` or read ``Artifact.sha256``.
    payload["audit"]["package_sha256"] = _PACKAGE_SHA_PLACEHOLDER
    placeholder_blob = json.dumps(payload, indent=2, sort_keys=True,
                                  default=str).encode()
    package_self_sha = hashlib.sha256(placeholder_blob).hexdigest()
    final_blob = placeholder_blob.replace(
        b'"package_sha256": "' + _PACKAGE_SHA_PLACEHOLDER.encode() + b'"',
        b'"package_sha256": "' + package_self_sha.encode() + b'"',
    )

    # 3) Literal file hash = sha256 of what we are about to write.
    literal_file_sha = hashlib.sha256(final_blob).hexdigest()

    out_dir = case_dir(case_id) / PACKAGE_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fname = f"pkg_{ts}_{literal_file_sha[:8]}.json"
    out_path = out_dir / fname
    out_path.write_bytes(final_blob)

    # Artifact.sha256 == LITERAL bytes-on-disk hash (matches
    # ``sha256sum pkg_*.json``). Metadata carries the other two so
    # downstream tooling can tell them apart unambiguously.
    art = Artifact(
        case_id=case_id,
        artifact_type="PACKAGE",
        uri=str(out_path),
        sha256=literal_file_sha,
        mime_type="application/json",
        artifact_metadata={
            "versioned_filename": fname,
            "content_sha256": content_sha,
            "package_self_sha256": package_self_sha,
            "literal_file_sha256": literal_file_sha,
            "hash_scheme": "sha256_with_zeroed_field",
        },
    )
    session.add(art)
    session.flush()
    return {
        "case_id": case_id,
        "uri": str(out_path),
        # Literal sha of the bytes on disk (== Artifact.sha256 and
        # what ``sha256sum pkg_*.json`` reports). This value is NOT
        # embedded inside the package JSON; it can only be computed by
        # reading the final file.
        "sha256": literal_file_sha,
        "literal_file_sha256": literal_file_sha,
        # Self-verifying hash (matches ``audit.package_sha256`` field
        # inside the file). Reproducible offline by zeroing that field.
        "package_self_sha256": package_self_sha,
        "content_sha256": content_sha,
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


def _serialise_detection(d: Detection) -> dict:
    return {
        "id": d.id, "label": d.label, "score": d.score,
        "bbox_xyxy": d.bbox_xyxy, "frame_id": d.frame_id,
        "frame_idx": d.frame_idx, "frame_ts": _jsonable(d.frame_ts),
        "query": d.query,
    }


def _serialise_track(t: Track) -> dict:
    return {
        "id": t.id, "tracker_id": t.tracker_id, "label": t.label,
        "first_seen_ts": _jsonable(t.first_seen_ts),
        "last_seen_ts": _jsonable(t.last_seen_ts),
        "confidence": t.confidence,
        "zones": t.zones or [], "events": t.events or [],
        "physical_item_candidate": t.physical_item_candidate,
        "receipt_candidate": t.receipt_candidate,
    }


def _serialise_observation(o: TrackObservation) -> dict:
    return {
        "id": o.id, "track_id": o.track_id, "detection_id": o.detection_id,
        "frame_id": o.frame_id, "frame_idx": o.frame_idx,
        "frame_ts": _jsonable(o.frame_ts),
        "bbox_xyxy": o.bbox_xyxy,
    }


def _serialise_keyframe(k: Keyframe) -> dict:
    return {
        "id": k.id, "role": k.role, "frame_id": k.frame_id,
        "frame_idx": k.frame_idx, "frame_ts": _jsonable(k.frame_ts),
        "track_id_ref": k.track_id_ref, "uri": k.uri,
    }


def _serialise_ocr(o: OcrResult) -> dict:
    return {
        "id": o.id, "frame_id": o.frame_id, "bbox_xyxy": o.bbox_xyxy,
        "text": o.text, "confidence": o.confidence,
        "engine": o.engine, "crop_uri": o.crop_uri,
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


def _serialise_detection_snapshot(a: Artifact, case_id: str) -> dict:
    """Serialise a DETECTION_SNAPSHOT artifact for the reviewer UI.

    ``uri`` is an on-disk path the browser cannot fetch, so we resolve the
    filename (from metadata, falling back to the uri basename) into the
    ``/video/cases/{case_id}/detection-snapshots/{filename}`` route that
    serves the PNG.
    """
    meta = a.artifact_metadata or {}
    filename = meta.get("filename") or (
        a.uri.rsplit("/", 1)[-1] if a.uri else "")
    return {
        "id": a.id,
        "url": (f"/api/v1/video/cases/{case_id}"
                f"/detection-snapshots/{filename}") if filename else None,
        "filename": filename,
        "frame_idx": a.frame_idx,
        "frame_ts": _jsonable(a.frame_ts) or meta.get("frame_ts"),
        "box_count": meta.get("box_count"),
    }
