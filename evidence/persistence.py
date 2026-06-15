"""Persist perception evidence into structured DB tables + artifacts.

The phase-1 evidence graph projected POS_EVENT / CASE / VIDEO_WINDOW /
VIDEO_SEGMENT / ARTIFACT / VLM_CLAIM / REVIEW_ACTION out of relational
tables that already existed. This module adds the missing perception
node types — DETECTION / TRACK / TRACK_OBSERVATION / OCR_RESULT /
KEYFRAME — so the package + graph contain the actual visual evidence
the reasoning chain saw, not just counts.

The schema is deliberately portable: SQLite for dev, Postgres for
production. JSON columns hold structured payloads.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Iterable

from sqlalchemy.orm import Session


log = logging.getLogger(__name__)


def persist_perception(session: Session,
                       *,
                       case_id: str,
                       window_id: str,
                       perception: dict) -> dict:
    """Write detections / tracks / keyframes / OCR rows for this case.

    Returns counts of each table written. The caller is responsible for
    committing the session; perception writes always happen inside the
    same transaction as the rest of the analyze step.
    """
    from db.models import (
        Detection,
        Keyframe,
        OcrResult,
        Track,
        TrackObservation,
    )

    detections_in = perception.get("detections") or []
    tracks_in = perception.get("tracks") or []
    keyframes_in = perception.get("keyframes") or []
    ocr_in = perception.get("ocr") or []

    # ---- detections ----
    det_rows: list[Detection] = []
    for idx, d in enumerate(detections_in):
        row = Detection(
            case_id=case_id,
            video_window_id=window_id,
            label=str(d.get("label") or ""),
            score=float(d.get("score") or 0.0),
            bbox_xyxy=list(d.get("bbox_xyxy") or []),
            frame_id=str(d.get("frame_id") or f"frame_{idx:06d}"),
            frame_idx=int(d.get("frame_idx") or 0),
            frame_ts=_parse_dt(d.get("ts")),
            query=str(d.get("query") or "") or None,
        )
        session.add(row)
        det_rows.append(row)
    if det_rows:
        session.flush()

    # ---- tracks + observations ----
    track_rows: list[Track] = []
    for t in tracks_in:
        row = Track(
            case_id=case_id,
            video_window_id=window_id,
            label=str(t.get("label") or ""),
            tracker_id=str(t.get("track_id") or ""),
            first_seen_ts=_parse_dt(t.get("first_seen_ts")),
            last_seen_ts=_parse_dt(t.get("last_seen_ts")),
            confidence=float(t.get("confidence") or 0.0),
            zones=list(t.get("zones") or []),
            events=list(t.get("events") or []),
            physical_item_candidate=bool(t.get("physical_item_candidate")),
            receipt_candidate=bool(t.get("receipt_candidate")),
        )
        session.add(row)
        track_rows.append(row)
    if track_rows:
        session.flush()

    # Map detection list indices to actual Detection rows for FK use.
    det_idx_to_row = {i: d for i, d in enumerate(det_rows)}
    for t_in, t_row in zip(tracks_in, track_rows):
        for det_idx in (t_in.get("detections") or []):
            try:
                det_row = det_idx_to_row[int(det_idx)]
            except (KeyError, ValueError, TypeError):
                continue
            session.add(TrackObservation(
                track_id=t_row.id,
                detection_id=det_row.id,
                frame_id=det_row.frame_id,
                frame_idx=det_row.frame_idx,
                frame_ts=det_row.frame_ts,
                bbox_xyxy=list(det_row.bbox_xyxy or []),
            ))

    # ---- keyframes ----
    kf_rows: list[Keyframe] = []
    for kf in keyframes_in:
        row = Keyframe(
            case_id=case_id,
            video_window_id=window_id,
            role=str(kf.get("role") or ""),
            frame_id=str(kf.get("frame_id") or ""),
            frame_idx=int(kf.get("frame_idx") or 0),
            frame_ts=_parse_dt(kf.get("ts")),
            track_id_ref=str(kf.get("track_id") or "") or None,
            uri=str(kf.get("uri") or "") or None,
        )
        session.add(row)
        kf_rows.append(row)

    # ---- OCR ----
    ocr_rows: list[OcrResult] = []
    for o in ocr_in:
        row = OcrResult(
            case_id=case_id,
            video_window_id=window_id,
            frame_id=str(o.get("frame_id") or ""),
            bbox_xyxy=list(o.get("bbox_xyxy") or []),
            text=str(o.get("text") or ""),
            confidence=float(o.get("confidence") or 0.0),
            engine=str(o.get("engine") or "falcon"),
            crop_uri=str(o.get("crop_uri") or "") or None,
        )
        session.add(row)
        ocr_rows.append(row)

    if kf_rows or ocr_rows:
        session.flush()

    return {
        "detections": len(det_rows),
        "tracks": len(track_rows),
        "keyframes": len(kf_rows),
        "ocr": len(ocr_rows),
    }


def _parse_dt(value: Any):
    if value is None or isinstance(value, datetime):
        return value if not isinstance(value, datetime) \
            or value.tzinfo is None else value.replace(tzinfo=None)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    return None
