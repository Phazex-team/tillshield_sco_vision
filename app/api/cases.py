"""Case query + reprocess endpoints."""
from __future__ import annotations

import atexit
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
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

# Outbound result export to the refund agent runs on its OWN single worker so
# the (slow) video render + HTTP upload never delays the analysis queue. The
# export is fully guarded and one-way — it cannot affect analysis/evidence.
_EXPORT_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="export")
atexit.register(lambda: _EXPORT_POOL.shutdown(wait=False))


# ---------------------------------------------------------------------------
# Reprocess job registry + hang watchdog.
#
# The pool is single-worker with no per-job cancellation, and analyze_case's
# perception (GPU) stage has no timeout — so a true hang (as opposed to an
# error, which drains cleanly) wedges the ONE worker forever and head-of-line
# blocks the whole queue. This registry lets a supervisor (auto_analyzer) tell
# apart three states for every REPROCESSING case: queued-in-pool, actively
# running, or a genuine orphan — and detect when the active job has hung.
# ---------------------------------------------------------------------------
_JOBS_LOCK = threading.Lock()
_QUEUED_IDS: set[str] = set()          # submitted to the pool, not yet started
_ACTIVE: Optional[dict] = None         # {"case_id": str, "started": monotonic}
_QUARANTINED: set[str] = set()         # finalized by the watchdog; worker must
                                       # not resurrect these to OPEN


def register_queued(case_id: str) -> None:
    with _JOBS_LOCK:
        _QUEUED_IDS.add(case_id)


def _register_started(case_id: str) -> None:
    global _ACTIVE
    with _JOBS_LOCK:
        _QUEUED_IDS.discard(case_id)
        _ACTIVE = {"case_id": case_id, "started": time.monotonic()}


def _register_done(case_id: str) -> None:
    global _ACTIVE
    with _JOBS_LOCK:
        if _ACTIVE and _ACTIVE.get("case_id") == case_id:
            _ACTIVE = None
        _QUARANTINED.discard(case_id)


def in_flight_ids() -> set[str]:
    """Case ids this process is currently responsible for — queued in the
    pool or actively running. A REPROCESSING case NOT in this set is a true
    orphan (its worker died with the process) and is safe to reap."""
    with _JOBS_LOCK:
        ids = set(_QUEUED_IDS)
        if _ACTIVE:
            ids.add(_ACTIVE["case_id"])
        return ids


def active_job() -> Optional[dict]:
    with _JOBS_LOCK:
        return dict(_ACTIVE) if _ACTIVE else None


def check_reprocess_hang(timeout_sec: float) -> Optional[dict]:
    """Return {case_id, elapsed_sec} when the active job has been running
    longer than ``timeout_sec``, else None."""
    job = active_job()
    if not job:
        return None
    elapsed = time.monotonic() - job["started"]
    if elapsed < timeout_sec:
        return None
    return {"case_id": job["case_id"], "elapsed_sec": elapsed}


def quarantine_case(case_id: str, elapsed_sec: float) -> None:
    """Finalize a wedged case to CLOSED/REVIEW so it leaves the queue and is
    never re-claimed into another hang. Marks it quarantined so the (still
    stuck) worker thread can't resurrect it if it ever returns."""
    from app import audit
    from db.models import Case
    from db.session import get_sessionmaker
    with _JOBS_LOCK:
        _QUARANTINED.add(case_id)
    try:
        SM = get_sessionmaker()
        with SM() as s:
            case = s.get(Case, case_id)
            if case is not None:
                before = {"status": case.status, "outcome": case.outcome}
                case.status = "CLOSED"
                case.outcome = case.outcome or "REVIEW"
                case.invalid_reason = (
                    f"reprocess_timeout: worker wedged {elapsed_sec:.0f}s")[:480]
                case.closed_at = datetime.now(timezone.utc)
                audit.record(s, action="case.reprocess_timeout",
                             entity_type="case", entity_id=case_id,
                             actor_type="watchdog", before=before,
                             after={"status": "CLOSED",
                                    "outcome": case.outcome,
                                    "elapsed_sec": round(elapsed_sec, 1)})
                s.commit()
    except Exception:
        log.exception("failed to quarantine wedged case %s", case_id)


def _run_reprocess(case_id: str, prior: dict,
                   pre_roll_sec=None, post_roll_sec=None) -> None:
    """Background worker: run the full analysis in its own session. On an
    unexpected failure the case is restored to its pre-reprocess state (so
    it is never left stuck in REPROCESSING) and the failure is audited.

    ``pre_roll_sec`` / ``post_roll_sec`` (when set) widen the analysis
    window beyond the config defaults — used by retime+reprocess."""
    from app import audit
    from db.models import Case
    from db.session import get_sessionmaker

    SM = get_sessionmaker()
    _register_started(case_id)
    try:
        with SM() as s:
            analyze_case(s, case_id,
                         pre_roll_sec=pre_roll_sec,
                         post_roll_sec=post_roll_sec)
    except Exception as exc:  # noqa: BLE001 — must not crash the worker
        log.exception("background reprocess failed for case %s", case_id)
        # If the watchdog already quarantined this case (declared it wedged),
        # do NOT restore it to OPEN — that would re-queue a poison case.
        with _JOBS_LOCK:
            quarantined = case_id in _QUARANTINED
        if quarantined:
            log.warning("reprocess of %s failed after watchdog quarantine; "
                        "leaving it CLOSED/REVIEW", case_id)
        else:
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
    else:
        # SCO Phase 7a: the refund-agent export is the legacy refund flow.
        # In SCO mode it stays disabled by default; an SCO-shaped exporter
        # is deferred to v1.1. The legacy module remains on disk so the
        # refund flow can be re-enabled by flipping the config flag.
        try:
            from app.config import load_config
            cfg = load_config()
            refund_export_enabled = bool(
                ((cfg.raw.get("integrations") or {})
                 .get("refund_agent") or {})
                .get("enabled", False)
            )
        except Exception:
            refund_export_enabled = False
        if refund_export_enabled:
            try:
                from pos.refund_agent_export import maybe_export_case
                _EXPORT_POOL.submit(maybe_export_case, case_id)
            except Exception:
                log.exception(
                    "failed to queue refund-agent export for %s", case_id)
        else:
            log.debug(
                "sco mode: refund-agent export disabled (case=%s); "
                "SCO exporter deferred to v1.1", case_id)
    finally:
        _register_done(case_id)


def _drain_reprocess_pool() -> None:
    """Block until all queued reprocess jobs have finished. Single worker
    + FIFO means a sentinel that completes guarantees prior jobs are done.
    Used by tests and any caller that needs the result synchronously."""
    _REPROCESS_POOL.submit(lambda: None).result()


# Asia/Dubai is a fixed UTC+4 offset (no DST) — use a fixed tz so we don't
# depend on the tzdata package being present in the image.
_DUBAI_TZ = timezone(timedelta(hours=4))


def _local_iso(dt):
    """Render a naive-UTC timestamp in Dubai local time for DISPLAY.

    The POS receipt and the camera's burned-in clock are both Dubai local,
    but we store times as naive UTC (correct for matching the UTC video
    index). Without this, the UI shows POS/window times 4h behind what the
    video actually displays. Storage is unchanged; this is display-only.
    """
    if dt is None:
        return None
    if getattr(dt, "tzinfo", None) is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(_DUBAI_TZ).isoformat()


def _vlm_mismatch(vlm_output: dict):
    """VLM's basket-match headline as a yes/no/uncertain mismatch flag.
    'yes' = the VLM flagged a count OR identity mismatch."""
    pcm = vlm_output.get("physical_count_match")
    sim = vlm_output.get("semantic_identity_match")
    if pcm == "no" or sim == "no":
        return "yes"
    if pcm == "yes" and sim in ("yes", None):
        return "no"
    if pcm is None and sim is None:
        return None
    return "uncertain"


def _fl_mismatch(vlm_manifest: dict, pos_event):
    """Independent Falcon count vs POS basket size -> yes/no mismatch.
    None when the FL count wasn't produced (e.g. no video)."""
    flc = (vlm_manifest or {}).get("fl_audit_zone_count") or {}
    fl_count = flc.get("count") if isinstance(flc, dict) else None
    if fl_count is None:
        return None
    pos_count = len(_pos_items(pos_event)) if pos_event else 0
    return "yes" if int(fl_count) != int(pos_count) else "no"


def _signals(vlm_output, vlm_manifest, run_status, case,
             falcon_enabled: bool) -> dict:
    """Per-case health of the two independent analysers, for a UI badge:
      'ok'          — ran and produced output
      'failed'      — ran but errored / produced no usable result
      'unavailable' — no video / couldn't run (e.g. INVALID_VIDEO)
      'disabled'    — turned off in config (FL only)
      'pending'     — not analysed yet
    """
    inv = getattr(case, "invalid_reason", None)
    vlm_output = vlm_output or {}
    # VLM
    if run_status is None:
        vlm = "unavailable" if inv else "pending"
    elif run_status == "FAILED":
        vlm = "failed"
    elif (vlm_output.get("physical_count_match") is not None
          or vlm_output.get("matched_items") is not None):
        vlm = "ok"
    else:
        vlm = "failed"
    # FL
    if not falcon_enabled:
        fl = "disabled"
    elif ((vlm_manifest or {}).get("fl_audit_zone_count") or {}).get(
            "count") is not None:
        fl = "ok"
    elif vlm_manifest is None:
        fl = "unavailable" if inv else "pending"
    else:
        fl = "unavailable"
    return {"vlm": vlm, "fl": fl}


def _falcon_enabled() -> bool:
    try:
        from app.config import load_config
        return bool((load_config().raw.get("models") or {})
                    .get("falcon", {}).get("enabled", True))
    except Exception:
        return True


def _serialise_case(case, pos_event=None, latest_window=None,
                    vlm_output=None, vlm_manifest=None,
                    vlm_run_status=None, falcon_enabled=True) -> dict:
    # Surface the headline VLM observation (from the latest run) so the case
    # grid can show it as a column without a per-row fetch.
    vlm_output = vlm_output or {}
    return {
        "id": case.id,
        "pos_event_id": case.pos_event_id,
        "camera_id": case.camera_id,
        "status": case.status,
        "outcome": case.outcome,
        "risk_score": case.risk_score,
        "risk_reasons": case.risk_reasons,
        "decision_policy_version": case.decision_policy_version,
        "opened_at": _local_iso(case.opened_at),
        "closed_at": _local_iso(case.closed_at),
        "invalid_reason": case.invalid_reason,
        "customer_present": vlm_output.get("customer_present"),
        # Two independent mismatch signals for the queue grid: the VLM's
        # basket-match headline, and Falcon's count vs the POS basket.
        "vlm_mismatch": _vlm_mismatch(vlm_output),
        "fl_mismatch": _fl_mismatch(vlm_manifest, pos_event),
        # Per-case health of the two analysers (ran / failed / unavailable /
        # disabled / pending) for the UI badges.
        "signals": _signals(vlm_output, vlm_manifest, vlm_run_status, case,
                            falcon_enabled),
        "pos_event": _serialise_pos(pos_event) if pos_event else None,
        "latest_window": _serialise_window(latest_window)
            if latest_window else None,
    }


def _serialise_window(w) -> dict:
    return {
        "id": w.id,
        "status": w.status,
        "path": w.path,
        "actual_start_at": _local_iso(w.actual_start_at),
        "actual_end_at": _local_iso(w.actual_end_at),
        "failure_reason": w.failure_reason,
        # Which path produced the window (local vs NVR on-demand) + the
        # NVR query observability so operators see what was attempted.
        "acquisition_source": getattr(w, "acquisition_source", None),
        "nvr": getattr(w, "nvr_metadata", None),
    }


def _pos_items(ev) -> list[dict]:
    """The POS basket line items for this transaction, normalised for the
    UI. They live in ``raw_payload['items']`` (the TillShield agent's
    per-line list) — surfaced here so the case detail can show the basket
    the video is being matched against."""
    rp = ev.raw_payload or {}
    if isinstance(rp, str):
        import json
        try:
            rp = json.loads(rp)
        except Exception:
            rp = {}
    items = rp.get("items")
    if not items and isinstance(rp.get("raw_payload"), dict):
        items = rp["raw_payload"].get("items")
    out: list[dict] = []
    for it in (items or []):
        if not isinstance(it, dict):
            continue
        out.append({
            "description": it.get("description") or it.get("name") or "",
            "sku": it.get("sku") or it.get("barcode") or it.get("plu"),
            "quantity": it.get("quantity") or it.get("qty"),
            "amount": (it.get("totalAmount") if it.get("totalAmount") is not None
                       else it.get("amount") or it.get("total")),
        })
    return out


def _serialise_pos(ev) -> dict:
    return {
        "id": ev.id,
        "store_id": ev.store_id,
        "terminal_id": ev.terminal_id,
        "transaction_id": ev.transaction_id,
        "line_id": ev.line_id,
        "event_type": ev.event_type,
        "pos_event_at": _local_iso(ev.pos_event_at),
        "pos_event_end_at": _local_iso(ev.pos_event_end_at),
        "staff_id": ev.staff_id,
        "sku": ev.sku,
        "item_description": ev.item_description,
        "amount": ev.amount,
        "currency": ev.currency,
        "items": _pos_items(ev),
    }


@router.get("")
def list_cases(
    status: Optional[str] = None,
    outcome: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
) -> dict:
    from db.models import Case, PosEvent, VlmRun
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
        vlm_outputs = {}
        if cases:
            ids = [c.pos_event_id for c in cases if c.pos_event_id]
            if ids:
                pe = s.execute(
                    select(PosEvent).where(PosEvent.id.in_(ids))
                ).scalars().all()
                pos_events = {p.id: p for p in pe}
            # Latest VlmRun per case (for the handover/item columns). One
            # query ordered newest-first; keep the first row seen per case.
            case_ids = [c.id for c in cases]
            runs = s.execute(
                select(VlmRun)
                .where(VlmRun.case_id.in_(case_ids))
                .order_by(VlmRun.started_at.desc())
            ).scalars().all()
            vlm_manifests = {}
            vlm_statuses = {}
            for run in runs:
                if run.case_id not in vlm_outputs:
                    vlm_outputs[run.case_id] = run.output_json or {}
                    vlm_manifests[run.case_id] = run.input_manifest or {}
                    vlm_statuses[run.case_id] = run.status
        fal_on = _falcon_enabled()
        return {
            "items": [
                _serialise_case(c, pos_events.get(c.pos_event_id),
                                vlm_output=vlm_outputs.get(c.id),
                                vlm_manifest=vlm_manifests.get(c.id),
                                vlm_run_status=vlm_statuses.get(c.id),
                                falcon_enabled=fal_on)
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
        # Show the window the MOST RECENT analysis actually used. Window
        # ids are UUIDs (not time-orderable), so a reprocessed case has
        # several windows and id.desc() returned a random/stale one. The
        # latest VlmRun records its window_id in input_manifest — that's
        # the clip behind the current verdict.
        from db.models import VlmRun
        latest = None
        run = (s.query(VlmRun)
               .filter(VlmRun.case_id == case.id)
               .order_by(VlmRun.started_at.desc()).first())
        if run and isinstance(run.input_manifest, dict):
            wid = run.input_manifest.get("window_id")
            if wid:
                latest = s.get(VideoWindow, wid)
        if latest is None:
            # No analysis yet (or no window_id) — any window for the case,
            # preferring a SUCCEEDED one, so the UI can still show status.
            latest = (s.query(VideoWindow)
                      .filter(VideoWindow.case_id == case.id,
                              VideoWindow.status == "SUCCEEDED").first()
                      or s.query(VideoWindow)
                      .filter(VideoWindow.case_id == case.id).first())
        return _serialise_case(
            case, pos, latest,
            vlm_output=(run.output_json if run else None),
            vlm_manifest=(run.input_manifest if run else None),
            vlm_run_status=(run.status if run else None),
            falcon_enabled=_falcon_enabled())


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


def _run_retime_then_reprocess(case_id: str, prior: dict,
                               pre_roll_sec=None, post_roll_sec=None) -> None:
    """Background worker: re-time the case's slow-motion segments to real
    time, then run the normal analysis. Retime failure is non-fatal — the
    reprocess still runs on whatever segments exist.

    ``pre_roll_sec`` / ``post_roll_sec`` widen the window for BOTH the
    retime (so extra segments get re-timed) and the reprocess."""
    try:
        from video.retime import retime_segments_for_case
        summary = retime_segments_for_case(case_id,
                                           pre_roll_sec=pre_roll_sec,
                                           post_roll_sec=post_roll_sec)
        log.info("retime for case %s: considered=%s retimed=%s",
                 case_id, summary.get("segments_considered"),
                 summary.get("retimed"))
    except Exception:
        log.exception("retime failed for case %s (continuing to reprocess)",
                      case_id)
    _run_reprocess(case_id, prior, pre_roll_sec=pre_roll_sec,
                   post_roll_sec=post_roll_sec)


@router.post("/{case_id}/reprocess", status_code=202)
def reprocess(case_id: str) -> dict:
    """Reset the case and queue a re-analysis as a BACKGROUND job.

    Returns 202 immediately with ``status="REPROCESSING"``. The analysis
    (NVR clip export, perception, VLM, decision) runs on a single-worker
    pool; poll ``GET /cases/{id}`` for the final status/outcome. Audited.
    """
    before = _claim_for_reprocess(case_id)
    register_queued(case_id)
    _REPROCESS_POOL.submit(_run_reprocess, case_id, before)
    return {"case_id": case_id, "status": "REPROCESSING",
            "detail": "reprocess started; poll GET /cases/{id} "
                      "for the outcome"}


# Hard ceiling on a single window (matches nvr.max_window_sec). Keeps an
# operator from requesting an absurd pre/post that would pull in minutes of
# footage and blow up perception time.
_MAX_WINDOW_SEC = 900


@router.post("/{case_id}/retime-reprocess", status_code=202)
def retime_and_reprocess(
    case_id: str,
    pre_roll_sec: Optional[float] = Query(
        None, ge=0, le=_MAX_WINDOW_SEC,
        description="Seconds of footage BEFORE the POS event "
                    "(default 90 when omitted)."),
    post_roll_sec: Optional[float] = Query(
        None, ge=0, le=_MAX_WINDOW_SEC,
        description="Seconds of footage AFTER the POS event "
                    "(default 60 when omitted)."),
) -> dict:
    """Re-time the case's slow-motion CCTV segments to real time, THEN
    reprocess. For cases recorded before the recorder fps fix, whose clip
    plays too slow and doesn't cover the transaction. Real-time / already
    correct segments are left untouched. Background job; poll GET
    /cases/{id}.

    ``pre_roll_sec`` / ``post_roll_sec`` let an operator WIDEN the window
    (e.g. 180s pre / 120s post) when the default 90/60 misses the action.
    Omit either to keep its config default. Their sum is capped at
    ``_MAX_WINDOW_SEC``."""
    if (pre_roll_sec is not None and post_roll_sec is not None
            and pre_roll_sec + post_roll_sec > _MAX_WINDOW_SEC):
        raise HTTPException(
            status_code=400,
            detail=(f"pre_roll_sec + post_roll_sec must be <= "
                    f"{_MAX_WINDOW_SEC}s (got "
                    f"{pre_roll_sec + post_roll_sec:.0f}s)"))
    before = _claim_for_reprocess(case_id)
    register_queued(case_id)
    _REPROCESS_POOL.submit(_run_retime_then_reprocess, case_id, before,
                           pre_roll_sec, post_roll_sec)
    win = []
    if pre_roll_sec is not None:
        win.append(f"{pre_roll_sec:.0f}s pre")
    if post_roll_sec is not None:
        win.append(f"{post_roll_sec:.0f}s post")
    win_txt = (" (window: " + ", ".join(win) + ")") if win else ""
    return {"case_id": case_id, "status": "REPROCESSING",
            "detail": "retiming segments, then reprocessing" + win_txt
                      + "; poll GET /cases/{id} for the outcome"}
