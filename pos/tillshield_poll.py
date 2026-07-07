"""TillShield POS-agent poller (additive to the push endpoints).

The local TillShield agent exposes a per-workstation transaction feed
(discovered via ``GET /pos/data/info``):

    GET /pos/data/transactions?workstationId=<id>&start=<ISO>&end=<ISO>

This module polls that feed every ``poll_every_seconds`` for each
allow-listed SCO workstation, applies the checkout policy (accepted
event type + optional amount gate + allow-listed workstation that maps
to a configured camera), and reuses ``pos.tillshield.ingest_tillshield_batch``
so the downstream case flow is identical to the push path.

Hard rules (PRODUCTION_SPEC):
  * Never use poll time as ``pos_event_at`` — always the row's
    ``transactionDate``.
  * Idempotent: app-side natural-key dedup means re-fetching an
    overlapping window never opens a second case.
  * A temporarily-unavailable agent must never crash the app.

Cursor: per workstation we persist ``(last_txn_at, last_txn_id)`` in
``IntegrationPollState`` so a restart resumes instead of replaying the
whole feed. The next window starts inclusively at ``last_txn_at`` and the
boundary row is dropped by idempotency.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import IntegrationPollState
from pos.ingest import resolve_camera_for_pos_event
from pos.tillshield import _accepted_event_types, _normalise_event_type
from pos.tillshield_schemas import TillShieldBatch, TillShieldTransaction


log = logging.getLogger(__name__)

SOURCE_SYSTEM = "tillshield_agent"

# Ignored-by-reason keys surfaced in every poll summary.
IGNORE_REASONS = (
    "ignored_non_return_events",
    "ignored_non_negative_events",
    "ignored_unconfigured_workstation_events",
    "ignored_unmapped_workstation_events",
)

# An http getter takes (url, params, timeout) and returns parsed JSON.
HttpGet = Callable[[str, dict, float], Any]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TillShieldPollConfig:
    enabled: bool = False
    base_url: str = ""
    info_path: str = "/pos/data/info"
    transactions_path: str = "/pos/data/transactions"
    poll_every_seconds: int = 300
    request_timeout_sec: float = 30.0
    poll_lookback_seconds: int = 900
    # Re-query this far behind the cursor each cycle so late-arriving rows
    # (the agent lands data in ~10-min batches) are not skipped; app-side
    # idempotency dedupes the overlap. Transient agent errors (e.g. 500
    # during a batch refresh) are retried instead of stalling the cursor.
    poll_overlap_seconds: int = 600
    poll_retry_attempts: int = 3
    poll_retry_backoff_sec: float = 2.0
    # Default flipped to False for SCO mode: a sale has a positive total
    # so the legacy refund-counter filter (only negative totals) would
    # drop every SCO transaction. Deployments that still want refund-counter
    # behavior set this to True in integrations.tillshield.
    require_negative_amount: bool = False
    allowed_workstation_ids: list[str] = field(default_factory=list)
    workstation_camera_map: dict[str, str] = field(default_factory=dict)
    source_system: str = SOURCE_SYSTEM


def _ts_section(cfg) -> dict:
    integrations = (cfg.raw.get("integrations") if cfg else None) or {}
    return integrations.get("tillshield") or {}


def load_poll_config(cfg) -> TillShieldPollConfig:
    ts = _ts_section(cfg)
    ws_map = {str(k): str(v) for k, v in
              (ts.get("workstation_camera_map") or {}).items() if v}
    allowed = [str(x) for x in (ts.get("allowed_workstation_ids") or [])]
    return TillShieldPollConfig(
        enabled=bool(ts.get("poll_enabled", False)),
        base_url=str(ts.get("base_url") or "").rstrip("/"),
        info_path=str(ts.get("info_path") or "/pos/data/info"),
        transactions_path=str(
            ts.get("transactions_path") or "/pos/data/transactions"),
        poll_every_seconds=int(ts.get("poll_every_seconds", 300) or 0),
        request_timeout_sec=float(ts.get("request_timeout_sec", 30) or 30),
        poll_lookback_seconds=int(ts.get("poll_lookback_seconds", 900) or 0),
        poll_overlap_seconds=int(ts.get("poll_overlap_seconds", 600) or 0),
        poll_retry_attempts=int(ts.get("poll_retry_attempts", 3) or 1),
        poll_retry_backoff_sec=float(ts.get("poll_retry_backoff_sec", 2) or 0),
        require_negative_amount=bool(ts.get("require_negative_amount", False)),
        allowed_workstation_ids=allowed,
        workstation_camera_map=ws_map,
    )


def validate_poll_config(cfg) -> list[str]:
    """Return a list of human-readable misconfiguration issues. Empty
    list means OK. Only validates when polling is enabled."""
    pc = load_poll_config(cfg)
    if not pc.enabled:
        return []
    issues: list[str] = []
    if not pc.base_url:
        issues.append("tillshield.poll_enabled but base_url is missing")
    if not pc.transactions_path:
        issues.append("tillshield.poll_enabled but transactions_path is missing")
    if pc.poll_every_seconds <= 0:
        issues.append(
            f"tillshield.poll_every_seconds must be > 0 (got "
            f"{pc.poll_every_seconds})")
    if pc.request_timeout_sec <= 0:
        issues.append("tillshield.request_timeout_sec must be > 0")
    if not pc.allowed_workstation_ids:
        issues.append(
            "tillshield.poll_enabled but allowed_workstation_ids is empty")

    seen: set[str] = set()
    for ws in pc.allowed_workstation_ids:
        if ws in seen:
            issues.append(
                f"duplicate workstation {ws!r} in allowed_workstation_ids")
        seen.add(ws)

    camera_ids = {c.get("id") for c in (cfg.cameras or []) if c.get("id")}
    for ws in pc.allowed_workstation_ids:
        cam = pc.workstation_camera_map.get(ws)
        if not cam:
            issues.append(
                f"workstation {ws!r} is allow-listed but missing from "
                f"workstation_camera_map")
            continue
        if cam not in camera_ids:
            issues.append(
                f"workstation {ws!r} maps to camera {cam!r} which is not "
                f"defined under cameras:")
    return issues


# ---------------------------------------------------------------------------
# Agent contract adapter + policy
# ---------------------------------------------------------------------------

def _str_or_none(v) -> Optional[str]:
    return str(v) if v is not None else None


def agent_row_to_tillshield(row: dict) -> TillShieldTransaction:
    """Map one POS-agent row ({_meta, items, summary, transaction}) onto
    the canonical ``TillShieldTransaction``. Raises if the row lacks the
    identifying transaction fields."""
    txn = row.get("transaction") or {}
    summary = row.get("summary") or {}
    return TillShieldTransaction(
        transaction_id=str(txn["transactionId"]),
        transaction_date=txn["transactionDate"],
        transaction_end_date=txn.get("transactionEndDate"),
        transaction_type=str(txn.get("transactionType") or ""),
        store_id=str(txn["storeId"]),
        workstation_id=str(txn["workstationId"]),
        operator_id=_str_or_none(txn.get("operatorId")),
        cashier_name=txn.get("cashierName"),
        currency=txn.get("currency"),
        total_items=summary.get("totalItems"),
        total_amount=summary.get("totalAmount"),
        items=row.get("items") or [],
        payload=row,  # full raw row preserved for audit
    )


@dataclass
class RowDecision:
    accept: bool
    reason: Optional[str]  # one of IGNORE_REASONS (without the "ignored_" use)
    txn: Optional[TillShieldTransaction]
    camera_id: Optional[str]


def classify_row(row: dict, pc: TillShieldPollConfig, cfg) -> RowDecision:
    """Decide whether a raw agent row should open an SCO checkout case.

    Order: accepted event type -> amount gate -> allow-listed workstation ->
    workstation maps to a configured camera. Never raises on a normal
    non-matching row."""
    try:
        txn = agent_row_to_tillshield(row)
    except Exception as exc:  # malformed row — count as non-return noise
        log.warning("tillshield poll: skipping malformed row: %s", exc)
        return RowDecision(False, "ignored_non_return_events", None, None)

    accepted = _accepted_event_types(cfg)
    if _normalise_event_type(txn.transaction_type) not in accepted:
        return RowDecision(False, "ignored_non_return_events", txn, None)

    amt = txn.total_amount
    if pc.require_negative_amount and not (amt is not None and amt < 0):
        return RowDecision(False, "ignored_non_negative_events", txn, None)

    if txn.workstation_id not in pc.allowed_workstation_ids:
        return RowDecision(
            False, "ignored_unconfigured_workstation_events", txn, None)

    camera_id = resolve_camera_for_pos_event(
        txn.store_id, txn.workstation_id, cfg)
    if not camera_id:
        return RowDecision(
            False, "ignored_unmapped_workstation_events", txn, camera_id)

    return RowDecision(True, None, txn, camera_id)


# ---------------------------------------------------------------------------
# HTTP fetch
# ---------------------------------------------------------------------------

def _default_http_get(url: str, params: dict, timeout: float) -> Any:
    import httpx
    with httpx.Client(timeout=timeout) as c:
        r = c.get(url, params=params)
        r.raise_for_status()
        return r.json()


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()


def fetch_workstation_rows(pc: TillShieldPollConfig,
                           workstation_id: str,
                           start: datetime,
                           end: datetime,
                           *,
                           http_get: HttpGet) -> list[dict]:
    """Fetch the raw agent rows for one workstation in [start, end].
    Returns a list of agent row dicts. Tolerates list or wrapped JSON."""
    url = f"{pc.base_url}{pc.transactions_path}"
    params = {"workstationId": workstation_id,
              "start": _iso(start), "end": _iso(end)}
    attempts = max(1, pc.poll_retry_attempts)
    last_exc = None
    for i in range(attempts):
        try:
            data = http_get(url, params, pc.request_timeout_sec)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get("transactions") or data.get("data") or []
            return []
        except Exception as exc:  # transient agent error (e.g. 500) -> retry
            last_exc = exc
            if i + 1 < attempts:
                import time
                log.warning("tillshield poll: ws %s fetch attempt %d/%d "
                            "failed (%s); retrying", workstation_id, i + 1,
                            attempts, exc)
                time.sleep(pc.poll_retry_backoff_sec * (i + 1))
    raise last_exc


# ---------------------------------------------------------------------------
# Poll cycle
# ---------------------------------------------------------------------------

def _empty_summary() -> dict:
    s = {"workstations": 0, "rows_seen": 0, "events_inserted": 0,
         "cases_created": 0}
    for r in IGNORE_REASONS:
        s[r] = 0
    return s


def _get_state(session: Session, source_system: str,
               workstation_id: str) -> IntegrationPollState:
    st = session.execute(
        select(IntegrationPollState).where(
            IntegrationPollState.source_system == source_system,
            IntegrationPollState.workstation_id == workstation_id,
        )).scalar_one_or_none()
    if st is None:
        st = IntegrationPollState(
            source_system=source_system, workstation_id=workstation_id,
            ignored_counts={})
        session.add(st)
        session.flush()
    return st


def _txn_dt(row: dict) -> Optional[datetime]:
    """The transaction start time as **naive UTC**.

    The TillShield agent sends Dubai-local (``+04:00``) tz-aware
    timestamps; the poll cursor, ``now`` and the stored ``pos_event_at``
    are all naive UTC. Returning naive UTC here keeps cursor comparison /
    sorting from mixing aware and naive datetimes (which raises
    ``TypeError``) and matches how the event is persisted."""
    try:
        dt = agent_row_to_tillshield(row).transaction_date
    except Exception:
        return None
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def poll_once(session: Session,
              *,
              cfg=None,
              pc: Optional[TillShieldPollConfig] = None,
              http_get: Optional[HttpGet] = None,
              now: Optional[datetime] = None) -> dict:
    """Run one polling cycle across all allow-listed workstations.

    Pure orchestration: fetch -> classify -> ingest qualifying rows ->
    advance per-workstation cursor. Errors for one workstation are
    isolated (recorded on its state row) and never abort the others.
    """
    if cfg is None:
        from app.config import load_config
        cfg = load_config()
    if pc is None:
        pc = load_poll_config(cfg)
    if http_get is None:
        http_get = _default_http_get
    if now is None:
        now = datetime.now()

    summary = _empty_summary()
    if not pc.enabled or not pc.allowed_workstation_ids:
        return summary

    from pos.tillshield import ingest_tillshield_batch

    def _resolve_cam(store_id, terminal_id):
        return resolve_camera_for_pos_event(store_id, terminal_id, cfg)

    for ws in pc.allowed_workstation_ids:
        summary["workstations"] += 1
        st = _get_state(session, pc.source_system, ws)
        st.last_poll_at = now
        if st.last_txn_at:
            # Re-query a window behind the cursor so late-landing rows are
            # caught; idempotency dedupes the overlap.
            start = st.last_txn_at - timedelta(seconds=pc.poll_overlap_seconds)
        else:
            start = now - timedelta(seconds=pc.poll_lookback_seconds)
        end = now + timedelta(seconds=1)
        try:
            rows = fetch_workstation_rows(pc, ws, start, end, http_get=http_get)
        except Exception as exc:
            st.last_error = f"fetch failed: {exc}"
            log.warning("tillshield poll: workstation %s fetch failed: %s",
                        ws, exc)
            # Commit (not just flush) so the SQLite write lock is released
            # immediately rather than held for the whole poll cycle —
            # otherwise the background poller starves interactive writes
            # (reprocess, config edits) and they hit "database is locked".
            session.commit()
            continue

        rows = sorted(rows, key=lambda r: (_txn_dt(r) or now))
        ignored = dict(st.ignored_counts or {})
        qualifying: list[TillShieldTransaction] = []
        cursor_at, cursor_id = st.last_txn_at, st.last_txn_id

        for row in rows:
            summary["rows_seen"] += 1
            st.rows_seen = (st.rows_seen or 0) + 1
            # Advance the cursor past every row we have looked at.
            dt = _txn_dt(row)
            if dt is not None and (cursor_at is None or dt >= cursor_at):
                cursor_at = dt
                cursor_id = str((row.get("transaction") or {})
                                .get("transactionId") or cursor_id or "")
            decision = classify_row(row, pc, cfg)
            if decision.accept and decision.txn is not None:
                qualifying.append(decision.txn)
            elif decision.reason:
                ignored[decision.reason] = ignored.get(decision.reason, 0) + 1
                summary[decision.reason] += 1

        if qualifying:
            batch = TillShieldBatch(source_system=pc.source_system,
                                    events=qualifying)
            result = ingest_tillshield_batch(
                session, cfg=cfg, batch=batch, source_ip="pos_poller",
                resolve_camera=_resolve_cam)
            inserted = result.get("events_inserted", 0)
            created = result.get("cases_created", 0)
            summary["events_inserted"] += inserted
            summary["cases_created"] += created
            st.events_inserted = (st.events_inserted or 0) + inserted
            st.cases_created = (st.cases_created or 0) + created
            # Defense-in-depth: any unmapped that slipped through ingest.
            slipped = result.get("ignored_unmapped_workstation_events", 0)
            if slipped:
                ignored["ignored_unmapped_workstation_events"] = (
                    ignored.get("ignored_unmapped_workstation_events", 0)
                    + slipped)
                summary["ignored_unmapped_workstation_events"] += slipped

        st.ignored_counts = ignored
        st.last_txn_at = cursor_at
        st.last_txn_id = cursor_id
        st.last_error = None
        st.last_success_at = now
        # Commit per workstation so the SQLite write lock is released
        # between workstations instead of being held for the entire poll
        # cycle. This keeps the background poller from starving
        # interactive writes (reprocess, config edits) which otherwise
        # fail with "database is locked".
        session.commit()

    log.info(
        "tillshield poll: workstations=%d rows=%d inserted=%d cases=%d "
        "ignored(non_return=%d non_negative=%d unconfigured=%d unmapped=%d)",
        summary["workstations"], summary["rows_seen"],
        summary["events_inserted"], summary["cases_created"],
        summary["ignored_non_return_events"],
        summary["ignored_non_negative_events"],
        summary["ignored_unconfigured_workstation_events"],
        summary["ignored_unmapped_workstation_events"])
    return summary


def backfill_range(session: Session,
                   start_local: datetime,
                   end_local: datetime,
                   *,
                   cfg=None,
                   pc: Optional[TillShieldPollConfig] = None,
                   http_get: Optional[HttpGet] = None,
                   workstation_ids: Optional[list[str]] = None,
                   advance_cursor: bool = False) -> dict:
    """One-time historical backfill over an explicit [start_local,
    end_local] window (NVR/agent local time), independent of the poll
    cursor. Same SCO checkout policy + idempotent ingest as the live
    poller, so re-running never duplicates cases. By default the poll
    cursor is left untouched (so the live poller keeps its own position).
    """
    if cfg is None:
        from app.config import load_config
        cfg = load_config()
    if pc is None:
        pc = load_poll_config(cfg)
    if http_get is None:
        http_get = _default_http_get
    wss = workstation_ids or pc.allowed_workstation_ids

    from pos.tillshield import ingest_tillshield_batch

    def _resolve_cam(store_id, terminal_id):
        return resolve_camera_for_pos_event(store_id, terminal_id, cfg)

    summary = _empty_summary()
    summary["range"] = [start_local.isoformat(), end_local.isoformat()]
    for ws in wss:
        summary["workstations"] += 1
        rows = fetch_workstation_rows(pc, ws, start_local, end_local,
                                      http_get=http_get)
        qualifying: list[TillShieldTransaction] = []
        for row in rows:
            summary["rows_seen"] += 1
            d = classify_row(row, pc, cfg)
            if d.accept and d.txn is not None:
                qualifying.append(d.txn)
            elif d.reason:
                summary[d.reason] += 1
        if qualifying:
            batch = TillShieldBatch(source_system=pc.source_system,
                                    events=qualifying)
            result = ingest_tillshield_batch(
                session, cfg=cfg, batch=batch, source_ip="pos_backfill",
                resolve_camera=_resolve_cam)
            summary["events_inserted"] += result.get("events_inserted", 0)
            summary["cases_created"] += result.get("cases_created", 0)
        if advance_cursor and rows:
            st = _get_state(session, pc.source_system, ws)
            mx = max((_txn_dt(r) for r in rows if _txn_dt(r)), default=None)
            if mx and (st.last_txn_at is None or mx > st.last_txn_at):
                st.last_txn_at = mx
        session.flush()
    log.info("tillshield backfill [%s..%s]: rows=%d inserted=%d cases=%d",
             start_local, end_local, summary["rows_seen"],
             summary["events_inserted"], summary["cases_created"])
    return summary


def read_status(session: Session,
                source_system: str = SOURCE_SYSTEM) -> dict:
    """Aggregate poller state for the status endpoint."""
    rows = session.execute(
        select(IntegrationPollState).where(
            IntegrationPollState.source_system == source_system,
        ).order_by(IntegrationPollState.workstation_id.asc())
    ).scalars().all()

    def _iso_or_none(dt):
        return dt.isoformat() if dt else None

    cumulative = {"rows_seen": 0, "events_inserted": 0, "cases_created": 0}
    cum_ignored: dict[str, int] = {}
    workstations = []
    last_poll = None
    last_success = None
    last_error = None
    for st in rows:
        cumulative["rows_seen"] += st.rows_seen or 0
        cumulative["events_inserted"] += st.events_inserted or 0
        cumulative["cases_created"] += st.cases_created or 0
        for k, v in (st.ignored_counts or {}).items():
            cum_ignored[k] = cum_ignored.get(k, 0) + v
        if st.last_poll_at and (last_poll is None or st.last_poll_at > last_poll):
            last_poll = st.last_poll_at
        if st.last_success_at and (
                last_success is None or st.last_success_at > last_success):
            last_success = st.last_success_at
        if st.last_error:
            last_error = st.last_error
        workstations.append({
            "workstation_id": st.workstation_id,
            "last_txn_at": _iso_or_none(st.last_txn_at),
            "last_txn_id": st.last_txn_id,
            "last_poll_at": _iso_or_none(st.last_poll_at),
            "last_success_at": _iso_or_none(st.last_success_at),
            "last_error": st.last_error,
            "rows_seen": st.rows_seen or 0,
            "events_inserted": st.events_inserted or 0,
            "cases_created": st.cases_created or 0,
            "ignored_counts": st.ignored_counts or {},
        })
    cumulative["ignored_counts"] = cum_ignored
    return {
        "source_system": source_system,
        "last_poll_at": _iso_or_none(last_poll),
        "last_successful_poll_at": _iso_or_none(last_success),
        "last_error": last_error,
        "cumulative": cumulative,
        "workstations": workstations,
    }


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class PollWorker:
    """Daemon thread that runs ``poll_once`` every ``interval`` seconds.

    Failure-isolated: a poll exception is logged and the loop continues.
    ``stop()`` signals the thread and joins it for clean shutdown."""

    def __init__(self, *, interval: int, source_system: str = SOURCE_SYSTEM):
        self.interval = max(1, int(interval))
        self.source_system = source_system
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="tillshield-poller", daemon=True)
        self._thread.start()
        log.info("tillshield poller started (interval=%ss)", self.interval)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        log.info("tillshield poller stopped")

    def _run(self) -> None:
        from db.session import get_sessionmaker
        while not self._stop.is_set():
            try:
                SM = get_sessionmaker()
                with SM() as s:
                    poll_once(s)
                    s.commit()
            except Exception:
                log.exception("tillshield poll cycle failed")
            self._stop.wait(self.interval)
