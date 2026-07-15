"""Reprocess-queue reliability: job registry, hang watchdog, config.

Pins the contracts that keep a single hung analysis from wedging the
whole single-worker reprocess pool:
  * the in-flight registry tells apart queued / active / orphan cases,
  * the hang check fires only after the timeout,
  * config parses with safe defaults + overrides.
The DB-backed orphan reaper is exercised by the API/integration tests.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.api.cases as cases  # noqa: E402
from app.auto_analyzer import load_reprocess_guard_config  # noqa: E402


def _reset_registry():
    with cases._JOBS_LOCK:
        cases._QUEUED_IDS.clear()
        cases._QUARANTINED.clear()
        cases._ACTIVE = None


def test_registry_queued_active_orphan_transitions():
    _reset_registry()
    cases.register_queued("c1")
    assert cases.in_flight_ids() == {"c1"}
    assert cases.active_job() is None

    cases._register_started("c1")
    assert cases.active_job()["case_id"] == "c1"
    assert cases.in_flight_ids() == {"c1"}       # active still in-flight

    cases._register_done("c1")
    assert cases.active_job() is None
    assert cases.in_flight_ids() == set()          # -> a c1 REPROCESSING row
    #                                                would now be an orphan


def test_check_reprocess_hang_only_after_timeout():
    _reset_registry()
    cases.register_queued("c2")
    cases._register_started("c2")
    # Not hung yet.
    assert cases.check_reprocess_hang(10_000) is None
    # Backdate the start so it looks long-running.
    with cases._JOBS_LOCK:
        cases._ACTIVE["started"] = time.monotonic() - 1000
    hang = cases.check_reprocess_hang(900)
    assert hang and hang["case_id"] == "c2" and hang["elapsed_sec"] >= 1000
    cases._register_done("c2")
    assert cases.check_reprocess_hang(1) is None   # no active job


def test_config_defaults_and_overrides():
    empty = SimpleNamespace(raw={})
    hang, on_hang, grace = load_reprocess_guard_config(empty)
    assert (hang, on_hang, grace) == (900.0, "restart", 180.0)

    custom = SimpleNamespace(raw={"reasoning": {
        "reprocess_hang_timeout_sec": 600,
        "on_reprocess_hang": "alert",
        "stale_reprocess_reaper_grace_sec": 60,
    }})
    hang, on_hang, grace = load_reprocess_guard_config(custom)
    assert (hang, on_hang, grace) == (600.0, "alert", 60.0)
