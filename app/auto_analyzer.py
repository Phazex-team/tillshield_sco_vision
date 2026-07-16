"""Background worker that analyses newly-opened cases.

Flow gap this closes: POS ingest (push endpoint + TillShield poller) only
*creates* cases with ``status="OPEN"``. The analysis step that builds the
video window + runs perception + the VLM (``app.case_runner.analyze_case``)
was only ever invoked by the manual reprocess endpoint. So a freshly
opened case had no video / no perception / no verdict until an operator
clicked "reprocess" — which is why many POS cases showed no video at all.

This worker periodically claims OPEN cases and submits them to the shared
reprocess pool (a single worker, so GPU/model work stays serialised with
manual reprocesses). Claiming flips ``OPEN -> REPROCESSING`` in one
committed step, so a case is never picked up twice and a crash leaves it
in REPROCESSING (recoverable) rather than silently lost.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional


log = logging.getLogger(__name__)


def queue_open_cases(limit: int = 10) -> int:
    """Claim up to ``limit`` OPEN cases and submit them for analysis.

    Returns the number of cases queued this call. Imports the reprocess
    pool lazily to avoid an import cycle with the cases router.
    """
    from app.api.cases import _REPROCESS_POOL, _run_reprocess, register_queued
    from db.models import Case
    from db.session import get_sessionmaker

    SM = get_sessionmaker()
    claimed: list[tuple[str, dict]] = []
    with SM() as s:
        rows = (s.query(Case)
                .filter(Case.status == "OPEN")
                .order_by(Case.opened_at.asc())
                .limit(limit)
                .all())
        for c in rows:
            before = {"status": c.status, "outcome": c.outcome}
            c.status = "REPROCESSING"  # claim so the next cycle skips it
            # Register as in-flight BEFORE the commit so the periodic reaper
            # never sees this row as an orphan in the window between commit
            # and pool submit (which would double-process it).
            register_queued(c.id)
            claimed.append((c.id, before))
        s.commit()

    for case_id, before in claimed:
        _REPROCESS_POOL.submit(_run_reprocess, case_id, before)
    return len(claimed)


class AutoAnalyzer:
    """Daemon thread that drains OPEN cases into the reprocess pool."""

    def __init__(self, interval: int = 30, batch: int = 10,
                 *, hang_timeout_sec: float = 900.0,
                 on_hang: str = "restart",
                 reaper_grace_sec: float = 180.0):
        self.interval = max(5, int(interval))
        self.batch = max(1, int(batch))
        # Watchdog: a job running longer than this is treated as wedged
        # (the perception GPU stage has no timeout and can't be cancelled).
        self.hang_timeout_sec = max(60.0, float(hang_timeout_sec))
        # "restart": quarantine the poison case then os._exit so the
        # container restarts with a warm compile cache + clear queue (the
        # only way to actually free the stuck single worker). "alert":
        # quarantine + log only, leave recovery to an operator.
        self.on_hang = on_hang if on_hang in ("restart", "alert") else "restart"
        # A REPROCESSING row that is neither queued nor active must be
        # orphaned for at least this long before the reaper reopens it —
        # avoids racing a just-claimed case.
        self.reaper_grace_sec = max(30.0, float(reaper_grace_sec))
        self._orphan_since: dict[str, float] = {}
        self._last_hang_handled: Optional[str] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._recover_stale()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="auto-analyzer", daemon=True)
        self._thread.start()
        log.info("auto-analyzer started (interval=%ss, batch=%s)",
                 self.interval, self.batch)

    @staticmethod
    def _recover_stale() -> None:
        """Reset orphaned REPROCESSING cases to OPEN at startup.

        The reprocess pool's work queue is in-memory, so a case that was
        claimed (OPEN -> REPROCESSING) but not finished before a restart
        is lost. At startup the pool is always empty, so any REPROCESSING
        case is by definition an orphan — flip it back to OPEN so it gets
        re-analysed instead of being stuck forever.
        """
        from db.models import Case
        from db.session import get_sessionmaker
        SM = get_sessionmaker()
        with SM() as s:
            rows = s.query(Case).filter(Case.status == "REPROCESSING").all()
            for c in rows:
                c.status = "OPEN"
            n = len(rows)
            s.commit()
        if n:
            log.info("auto-analyzer reset %d stale REPROCESSING case(s) "
                     "to OPEN", n)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        log.info("auto-analyzer stopped")

    def _run(self) -> None:
        while not self._stop.is_set():
            # Watchdog first: if the single worker is wedged, detect it
            # before trying to claim more work behind it.
            try:
                self._check_hang()
            except Exception:
                log.exception("auto-analyzer hang check failed")
            try:
                self._reap_orphans()
            except Exception:
                log.exception("auto-analyzer orphan reaper failed")
            try:
                # Don't pile the whole OPEN backlog into the pool at once:
                # claim more only when the pool has nearly drained. This
                # bounds how many cases sit in REPROCESSING (nicer UI) and
                # limits orphans if the app restarts mid-backlog.
                if _pool_backlog() < 2:
                    n = queue_open_cases(self.batch)
                    if n:
                        log.info("auto-analyzer queued %d open case(s)", n)
            except Exception:
                log.exception("auto-analyzer cycle failed")
            self._stop.wait(self.interval)

    def _check_hang(self) -> None:
        """If the active reprocess job has exceeded the hang timeout, the
        single worker is wedged (perception GPU stage can't be cancelled).
        Quarantine the poison case so it never re-wedges, then either exit
        for a self-healing restart or alert, per ``on_hang``."""
        from app.api.cases import check_reprocess_hang, quarantine_case
        hang = check_reprocess_hang(self.hang_timeout_sec)
        if not hang:
            self._last_hang_handled = None
            return
        case_id, elapsed = hang["case_id"], hang["elapsed_sec"]
        if case_id == self._last_hang_handled:
            return  # already handled this wedge; don't spam every cycle
        self._last_hang_handled = case_id
        log.critical("reprocess worker WEDGED on case %s for %.0fs (> %.0fs "
                     "timeout); single-worker queue is blocked. Quarantining "
                     "the case (-> CLOSED/REVIEW).", case_id, elapsed,
                     self.hang_timeout_sec)
        quarantine_case(case_id, elapsed)
        if self.on_hang == "restart":
            log.critical("on_reprocess_hang=restart: exiting so the container "
                         "restarts with a warm compile cache and a clear "
                         "queue (the wedged worker thread cannot be killed "
                         "in-process).")
            self._exit_for_restart()

    @staticmethod
    def _exit_for_restart() -> None:
        """Exit so the container restarts (``restart: unless-stopped``).

        ``os._exit`` only kills THIS process. That is enough in prod, where
        the app IS pid 1 (``scripts/run_app.py``). Under the dev override the
        app runs as a CHILD of the ``uvicorn --reload`` supervisor, which is
        pid 1 — killing the child left the container "Up" with a dead app,
        no restart, and every queued case stranded in REPROCESSING until
        someone noticed. Observed 2026-07-16: a 13-line/344s case wedged, the
        worker exited, and the API stayed down until manually recreated.

        So when we are not pid 1, signal pid 1 first and let it take the
        container down. Guarded on /.dockerenv: outside a container pid 1 is
        the host init, which must NEVER be signalled (``start.sh`` runs the
        app straight on the host).
        """
        import os
        import signal

        if os.getpid() != 1 and os.path.exists("/.dockerenv"):
            log.critical("not pid 1 (uvicorn --reload child): signalling pid 1 "
                         "so the container actually exits and restarts.")
            try:
                os.kill(1, signal.SIGTERM)
                return          # let pid 1 tear us down cleanly
            except Exception:
                log.exception("could not signal pid 1; exiting this process "
                              "only (container may stay up with a dead app)")
        os._exit(1)

    def _reap_orphans(self) -> None:
        """Reset REPROCESSING cases that are NOT in-flight in this process
        (queued or active) back to OPEN, once they've been orphaned for at
        least ``reaper_grace_sec``. Unlike the startup-only ``_recover_stale``,
        this runs every cycle so orphans self-heal without a restart. It is
        race-safe: a genuinely queued/active case is in ``in_flight_ids`` and
        is never touched."""
        from app.api.cases import in_flight_ids
        from db.models import Case
        from db.session import get_sessionmaker

        inflight = in_flight_ids()
        now = time.monotonic()
        SM = get_sessionmaker()
        with SM() as s:
            rows = (s.query(Case)
                    .filter(Case.status == "REPROCESSING")
                    .all())
            current_orphans = {c.id for c in rows if c.id not in inflight}
            # Forget rows that are no longer orphaned.
            for cid in list(self._orphan_since):
                if cid not in current_orphans:
                    del self._orphan_since[cid]
            reset = 0
            for c in rows:
                if c.id not in current_orphans:
                    continue
                first = self._orphan_since.setdefault(c.id, now)
                if now - first >= self.reaper_grace_sec:
                    c.status = "OPEN"
                    self._orphan_since.pop(c.id, None)
                    reset += 1
            if reset:
                s.commit()
        if reset:
            log.warning("auto-analyzer reaped %d orphaned REPROCESSING "
                        "case(s) -> OPEN", reset)


def _pool_backlog() -> int:
    """Best-effort count of work waiting in the reprocess pool."""
    try:
        from app.api.cases import _REPROCESS_POOL
        return _REPROCESS_POOL._work_queue.qsize()  # noqa: SLF001
    except Exception:
        return 0


def load_auto_analyze_config(cfg) -> tuple[bool, int, int]:
    """Return (enabled, interval_sec, batch) from ``reasoning`` config.

    Defaults: enabled, every 30s, 10 cases per cycle. Opt out by setting
    ``reasoning.auto_analyze_enabled: false`` in config.yaml.
    """
    reasoning = (cfg.raw.get("reasoning") if cfg else None) or {}
    enabled = bool(reasoning.get("auto_analyze_enabled", True))
    interval = int(reasoning.get("auto_analyze_interval_sec", 30) or 30)
    batch = int(reasoning.get("auto_analyze_batch", 10) or 10)
    return enabled, interval, batch


def load_reprocess_guard_config(cfg) -> tuple[float, str, float]:
    """Return (hang_timeout_sec, on_hang, reaper_grace_sec) from the
    ``reasoning`` config.

    Defaults: a job is 'wedged' after 900s; on a wedge we RESTART (quarantine
    the poison case, then exit so the container restarts with a warm compile
    cache and a clear queue — the only way to free the stuck single worker);
    orphaned REPROCESSING rows are reaped after 180s. Set
    ``reasoning.on_reprocess_hang: alert`` to quarantine + log without exiting.
    """
    reasoning = (cfg.raw.get("reasoning") if cfg else None) or {}
    hang = float(reasoning.get("reprocess_hang_timeout_sec", 900) or 900)
    on_hang = str(reasoning.get("on_reprocess_hang", "restart") or "restart")
    grace = float(reasoning.get("stale_reprocess_reaper_grace_sec", 180) or 180)
    return hang, on_hang, grace
