"""Per-case timeline view.

Aggregates POS event time, video window bounds, perception keyframes,
VLM runs, and review actions into one chronological list. Used by the
reviewer UI.
"""
from __future__ import annotations

from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import (
    Artifact,
    Case,
    PosEvent,
    ReviewAction,
    VideoWindow,
    VlmRun,
)


def timeline_for_case(session: Session, case_id: str) -> list[dict]:
    events: list[dict] = []
    case = session.get(Case, case_id)
    if case is None:
        return events

    if case.pos_event_id:
        pos = session.get(PosEvent, case.pos_event_id)
        if pos:
            events.append(_event(pos.pos_event_at, "POS_EVENT",
                                 f"{pos.event_type} txn={pos.transaction_id}"))

    if case.opened_at:
        events.append(_event(case.opened_at, "CASE_OPENED", "case opened"))

    windows = session.execute(
        select(VideoWindow).where(VideoWindow.case_id == case.id)
    ).scalars().all()
    for w in windows:
        if w.requested_start_at:
            events.append(_event(w.requested_start_at,
                                 "WINDOW_REQUESTED",
                                 f"window requested ({w.status})"))
        if w.actual_end_at:
            events.append(_event(w.actual_end_at, "WINDOW_AVAILABLE",
                                 "window ready"))

    arts = session.execute(
        select(Artifact).where(Artifact.case_id == case.id,
                               Artifact.artifact_type == "KEYFRAME")
        .order_by(Artifact.frame_ts.asc())
    ).scalars().all()
    for a in arts:
        if a.frame_ts:
            events.append(_event(a.frame_ts, "KEYFRAME",
                                 f"keyframe {a.frame_idx or '?'}"))

    runs = session.execute(
        select(VlmRun).where(VlmRun.case_id == case.id)
        .order_by(VlmRun.started_at.asc())
    ).scalars().all()
    for r in runs:
        if r.started_at:
            events.append(_event(r.started_at, "VLM_RUN",
                                 f"{r.provider} {r.status}"))

    revs = session.execute(
        select(ReviewAction).where(ReviewAction.case_id == case.id)
        .order_by(ReviewAction.created_at.asc())
    ).scalars().all()
    for r in revs:
        if r.created_at:
            events.append(_event(r.created_at, "REVIEW_ACTION",
                                 f"{r.action} -> {r.outcome or '-'}"))

    if case.closed_at:
        events.append(_event(case.closed_at, "CASE_CLOSED",
                             f"closed -> {case.outcome or '-'}"))

    events.sort(key=lambda e: e["ts"])
    return events


def _event(ts, kind: str, label: str) -> dict:
    return {"ts": ts.isoformat() if ts else None,
            "kind": kind, "label": label}
