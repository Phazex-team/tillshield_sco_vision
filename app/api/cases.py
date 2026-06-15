"""Case query + reprocess endpoints."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select

from app.case_runner import analyze_case  # module-level so tests can monkeypatch


router = APIRouter(prefix="/cases", tags=["cases"])


def _serialise_case(case, pos_event=None, latest_window=None) -> dict:
    return {
        "id": case.id,
        "pos_event_id": case.pos_event_id,
        "camera_id": case.camera_id,
        "status": case.status,
        "outcome": case.outcome,
        "risk_score": case.risk_score,
        "risk_reasons": case.risk_reasons,
        "decision_policy_version": case.decision_policy_version,
        "opened_at": case.opened_at.isoformat() if case.opened_at else None,
        "closed_at": case.closed_at.isoformat() if case.closed_at else None,
        "invalid_reason": case.invalid_reason,
        "pos_event": _serialise_pos(pos_event) if pos_event else None,
        "latest_window": _serialise_window(latest_window)
            if latest_window else None,
    }


def _serialise_window(w) -> dict:
    return {
        "id": w.id,
        "status": w.status,
        "path": w.path,
        "actual_start_at": w.actual_start_at.isoformat()
            if w.actual_start_at else None,
        "actual_end_at": w.actual_end_at.isoformat()
            if w.actual_end_at else None,
        "failure_reason": w.failure_reason,
    }


def _serialise_pos(ev) -> dict:
    return {
        "id": ev.id,
        "store_id": ev.store_id,
        "terminal_id": ev.terminal_id,
        "transaction_id": ev.transaction_id,
        "line_id": ev.line_id,
        "event_type": ev.event_type,
        "pos_event_at": ev.pos_event_at.isoformat()
            if ev.pos_event_at else None,
        "staff_id": ev.staff_id,
        "sku": ev.sku,
        "item_description": ev.item_description,
        "amount": ev.amount,
        "currency": ev.currency,
    }


@router.get("")
def list_cases(
    status: Optional[str] = None,
    outcome: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
) -> dict:
    from db.models import Case, PosEvent
    from db.session import get_sessionmaker

    SM = get_sessionmaker()
    with SM() as s:
        q = select(Case)
        if status:
            q = q.where(Case.status == status)
        if outcome:
            q = q.where(Case.outcome == outcome)
        q = q.order_by(Case.opened_at.desc()).limit(limit)
        cases = s.execute(q).scalars().all()
        pos_events = {}
        if cases:
            ids = [c.pos_event_id for c in cases if c.pos_event_id]
            if ids:
                pe = s.execute(
                    select(PosEvent).where(PosEvent.id.in_(ids))
                ).scalars().all()
                pos_events = {p.id: p for p in pe}
        return {
            "items": [
                _serialise_case(c, pos_events.get(c.pos_event_id))
                for c in cases
            ],
            "count": len(cases),
        }


@router.get("/{case_id}")
def get_case(case_id: str) -> dict:
    from db.models import Case, PosEvent, VideoWindow
    from db.session import get_sessionmaker

    SM = get_sessionmaker()
    with SM() as s:
        case = s.get(Case, case_id)
        if case is None:
            raise HTTPException(status_code=404, detail="case not found")
        pos = s.get(PosEvent, case.pos_event_id) if case.pos_event_id else None
        # The most recent SUCCEEDED window is the one a reviewer should
        # be able to scrub. If none succeeded, we still return the row
        # so the UI can show the failure_reason inline.
        latest = (s.query(VideoWindow)
                  .filter(VideoWindow.case_id == case.id)
                  .order_by(VideoWindow.id.desc()).first())
        return _serialise_case(case, pos, latest)


@router.post("/{case_id}/reprocess", status_code=202)
def reprocess(case_id: str) -> dict:
    """Reset + immediately re-run analysis for ``case_id``. Audited."""
    from app import audit
    from db.models import Case
    from db.session import get_sessionmaker

    SM = get_sessionmaker()
    with SM() as s:
        case = s.get(Case, case_id)
        if case is None:
            raise HTTPException(status_code=404, detail="case not found")
        before = {"status": case.status, "outcome": case.outcome}
        case.status = "REPROCESSING"
        case.outcome = None
        case.risk_score = None
        case.risk_reasons = None
        case.invalid_reason = None
        case.closed_at = None
        audit.record(
            s,
            action="case.reprocess_requested",
            entity_type="case", entity_id=case.id,
            actor_type="api",
            before=before,
            after={"status": case.status},
        )
        s.commit()

    # Run analysis in a fresh session so audit + decision land together.
    with SM() as s:
        try:
            result = analyze_case(s, case_id)
        except Exception as exc:
            raise HTTPException(status_code=500,
                                detail=f"analyze failed: {exc}")
        return {"case_id": case_id, **result}
