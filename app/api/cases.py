"""Case query + reprocess endpoints."""
from __future__ import annotations

import atexit
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select

from app.case_runner import analyze_case  # module-level so tests can monkeypatch


log = logging.getLogger(__name__)

router = APIRouter(prefix="/cases", tags=["cases"])

# Reprocess runs ``analyze_case`` (NVR clip export + ffmpeg + perception +
# VLM) which can take minutes, so it must NOT block the request. A single
# worker serialises analyses — concurrent reprocess requests queue rather
# than contending for the GPU / model servers.
_REPROCESS_POOL = ThreadPoolExecutor(max_workers=1,
                                     thread_name_prefix="reprocess")
atexit.register(lambda: _REPROCESS_POOL.shutdown(wait=False))


def _run_reprocess(case_id: str, prior: dict) -> None:
    """Background worker: run the full analysis in its own session. On an
    unexpected failure the case is restored to its pre-reprocess state (so
    it is never left stuck in REPROCESSING) and the failure is audited."""
    from app import audit
    from db.models import Case
    from db.session import get_sessionmaker

    SM = get_sessionmaker()
    try:
        with SM() as s:
            analyze_case(s, case_id)
    except Exception as exc:  # noqa: BLE001 — must not crash the worker
        log.exception("background reprocess failed for case %s", case_id)
        try:
            with SM() as s:
                case = s.get(Case, case_id)
                if case is not None and case.status == "REPROCESSING":
                    case.status = prior.get("status") or "CLOSED"
                    case.outcome = prior.get("outcome")
                    case.invalid_reason = f"reprocess_failed: {exc}"[:480]
                    audit.record(s, action="case.reprocess_failed",
                                 entity_type="case", entity_id=case_id,
                                 actor_type="api",
                                 before={"status": "REPROCESSING"},
                                 after={"status": case.status,
                                        "error": str(exc)})
                    s.commit()
        except Exception:
            log.exception("failed to record reprocess failure for %s",
                          case_id)


def _drain_reprocess_pool() -> None:
    """Block until all queued reprocess jobs have finished. Single worker
    + FIFO means a sentinel that completes guarantees prior jobs are done.
    Used by tests and any caller that needs the result synchronously."""
    _REPROCESS_POOL.submit(lambda: None).result()


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
        # Which path produced the window (local vs NVR on-demand) + the
        # NVR query observability so operators see what was attempted.
        "acquisition_source": getattr(w, "acquisition_source", None),
        "nvr": getattr(w, "nvr_metadata", None),
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


def _claim_for_reprocess(case_id: str) -> dict:
    """Set the case to REPROCESSING (audited) and return its prior state.
    Raises 404 if the case is missing. Does NOT submit the analysis."""
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
    return before


def _run_retime_then_reprocess(case_id: str, prior: dict) -> None:
    """Background worker: re-time the case's slow-motion segments to real
    time, then run the normal analysis. Retime failure is non-fatal — the
    reprocess still runs on whatever segments exist."""
    try:
        from video.retime import retime_segments_for_case
        summary = retime_segments_for_case(case_id)
        log.info("retime for case %s: considered=%s retimed=%s",
                 case_id, summary.get("segments_considered"),
                 summary.get("retimed"))
    except Exception:
        log.exception("retime failed for case %s (continuing to reprocess)",
                      case_id)
    _run_reprocess(case_id, prior)


@router.post("/{case_id}/reprocess", status_code=202)
def reprocess(case_id: str) -> dict:
    """Reset the case and queue a re-analysis as a BACKGROUND job.

    Returns 202 immediately with ``status="REPROCESSING"``. The analysis
    (NVR clip export, perception, VLM, decision) runs on a single-worker
    pool; poll ``GET /cases/{id}`` for the final status/outcome. Audited.
    """
    before = _claim_for_reprocess(case_id)
    _REPROCESS_POOL.submit(_run_reprocess, case_id, before)
    return {"case_id": case_id, "status": "REPROCESSING",
            "detail": "reprocess started; poll GET /cases/{id} "
                      "for the outcome"}


@router.post("/{case_id}/retime-reprocess", status_code=202)
def retime_and_reprocess(case_id: str) -> dict:
    """Re-time the case's slow-motion CCTV segments to real time, THEN
    reprocess. For cases recorded before the recorder fps fix, whose clip
    plays too slow and doesn't cover the transaction. Real-time / already
    correct segments are left untouched. Background job; poll GET
    /cases/{id}."""
    before = _claim_for_reprocess(case_id)
    _REPROCESS_POOL.submit(_run_retime_then_reprocess, case_id, before)
    return {"case_id": case_id, "status": "REPROCESSING",
            "detail": "retiming segments, then reprocessing; "
                      "poll GET /cases/{id} for the outcome"}
