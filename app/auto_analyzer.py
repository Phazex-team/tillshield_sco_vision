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
from typing import Optional


log = logging.getLogger(__name__)


def queue_open_cases(limit: int = 10) -> int:
    """Claim up to ``limit`` OPEN cases and submit them for analysis.

    Returns the number of cases queued this call. Imports the reprocess
    pool lazily to avoid an import cycle with the cases router.
    """
    from app.api.cases import _REPROCESS_POOL, _run_reprocess
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
            claimed.append((c.id, before))
        s.commit()

    for case_id, before in claimed:
        _REPROCESS_POOL.submit(_run_reprocess, case_id, before)
    return len(claimed)


class AutoAnalyzer:
    """Daemon thread that drains OPEN cases into the reprocess pool."""

    def __init__(self, interval: int = 30, batch: int = 10):
        self.interval = max(5, int(interval))
        self.batch = max(1, int(batch))
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
