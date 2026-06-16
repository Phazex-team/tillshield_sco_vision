"""Tests for the TillShield POS-agent poller: policy, workstation
routing, checkpoint/idempotency, and config validation. All local and
offline — the POS agent HTTP layer is injected as a fake."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_session(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'t.sqlite'}")
    import db.session as s
    s._ENGINE = None
    s._SESSION_FACTORY = None
    s.init_schema()
    return s.get_sessionmaker()


def _cfg(*, cameras=None, allowed=("52",), ws_map=None,
         require_negative=True, poll_enabled=True, return_types=None,
         pos_timezone=None):
    from app.config import AppConfig
    cameras = list(cameras if cameras is not None
                   else [{"id": "cam_return_01"}])
    ws_map = dict(ws_map if ws_map is not None else {"52": "cam_return_01"})
    ts = {
        "poll_enabled": poll_enabled,
        "base_url": "http://localhost:8081",
        "transactions_path": "/pos/data/transactions",
        "poll_every_seconds": 300,
        "request_timeout_sec": 30,
        "require_negative_amount": require_negative,
        "allowed_workstation_ids": list(allowed),
        "workstation_camera_map": ws_map,
        "return_event_types": list(return_types or ["RETURN", "REFUND"]),
    }
    if pos_timezone is not None:
        ts["pos_timezone"] = pos_timezone
    raw = {"cameras": cameras, "integrations": {"tillshield": ts}}
    return AppConfig(raw=raw, cameras=cameras, settings={}, models={},
                     observability={})


def _row(txn_id, ws, ttype="RETURN", amount=-100.0,
         date="2026-06-15T00:01:17", store="2270"):
    return {
        "_meta": {"sourceFile": "X"},
        "items": [],
        "summary": {"totalAmount": amount, "totalItems": 1},
        "transaction": {
            "transactionId": txn_id, "transactionDate": date,
            "transactionType": ttype, "storeId": store,
            "workstationId": ws, "operatorId": "op1", "cashierName": "c",
            "currency": "AED", "transactionEndDate": date,
        },
    }


def _http(rows_by_ws):
    def _get(url, params, timeout):
        return list(rows_by_ws.get(params["workstationId"], []))
    return _get


NOW = datetime(2026, 6, 16, 12, 0, 0)


# ---------------------------------------------------------------------------
# 1. Policy (classify_row)
# ---------------------------------------------------------------------------

def test_policy_accepts_return_negative_allowed_mapped():
    from pos.tillshield_poll import classify_row, load_poll_config
    cfg = _cfg()
    d = classify_row(_row("R1", "52", amount=-100.0), load_poll_config(cfg), cfg)
    assert d.accept is True
    assert d.camera_id == "cam_return_01"


def test_policy_non_return_ignored():
    from pos.tillshield_poll import classify_row, load_poll_config
    cfg = _cfg()
    d = classify_row(_row("S1", "52", ttype="SALE", amount=55.9),
                     load_poll_config(cfg), cfg)
    assert d.accept is False
    assert d.reason == "ignored_non_return_events"


def test_policy_non_negative_ignored():
    from pos.tillshield_poll import classify_row, load_poll_config
    cfg = _cfg()
    d = classify_row(_row("R2", "52", ttype="RETURN", amount=100.0),
                     load_poll_config(cfg), cfg)
    assert d.accept is False
    assert d.reason == "ignored_non_negative_events"


def test_policy_unconfigured_workstation_ignored():
    from pos.tillshield_poll import classify_row, load_poll_config
    cfg = _cfg(allowed=("52",))
    # Row from workstation 99 which is not in the allowlist.
    d = classify_row(_row("R3", "99", amount=-100.0),
                     load_poll_config(cfg), cfg)
    assert d.accept is False
    assert d.reason == "ignored_unconfigured_workstation_events"


def test_policy_mapped_to_unknown_camera_ignored():
    from pos.tillshield_poll import classify_row, load_poll_config
    # Workstation allow-listed but mapped to a camera not in cameras:.
    cfg = _cfg(cameras=[{"id": "cam_return_01"}], allowed=("52",),
               ws_map={"52": "ghost_cam"})
    d = classify_row(_row("R4", "52", amount=-100.0),
                     load_poll_config(cfg), cfg)
    assert d.accept is False
    assert d.reason == "ignored_unmapped_workstation_events"


# ---------------------------------------------------------------------------
# 2. poll_once integration
# ---------------------------------------------------------------------------

def test_poll_creates_case_with_correct_camera(tmp_path, monkeypatch):
    SM = _fresh_session(tmp_path, monkeypatch)
    from pos.tillshield_poll import poll_once
    cfg = _cfg(allowed=("52",), ws_map={"52": "cam_return_01"})
    http = _http({"52": [_row("R1", "52", amount=-100.0)]})
    with SM() as s:
        summary = poll_once(s, cfg=cfg, http_get=http, now=NOW)
        s.commit()
    assert summary["cases_created"] == 1
    assert summary["events_inserted"] == 1
    from db.models import Case
    with SM() as s:
        cases = s.query(Case).all()
    assert len(cases) == 1
    assert cases[0].camera_id == "cam_return_01"


def test_poll_unmapped_workstation_does_not_default_to_first_camera(
        tmp_path, monkeypatch):
    """A return-counter row whose workstation has no camera mapping must
    NOT be routed to the first/default camera (cam_first)."""
    SM = _fresh_session(tmp_path, monkeypatch)
    from pos.tillshield_poll import poll_once
    # cam_first is the first configured camera. Workstation 52 is allowed
    # but maps to a camera id absent from cameras: -> no route.
    cfg = _cfg(cameras=[{"id": "cam_first"}, {"id": "cam_return_01"}],
               allowed=("52",), ws_map={"52": "ghost_cam"})
    http = _http({"52": [_row("R1", "52", amount=-100.0)]})
    with SM() as s:
        summary = poll_once(s, cfg=cfg, http_get=http, now=NOW)
        s.commit()
    assert summary["cases_created"] == 0
    assert summary["ignored_unmapped_workstation_events"] == 1
    from db.models import Case
    with SM() as s:
        assert s.query(Case).count() == 0  # never sent to cam_first


def test_poll_checkpoint_avoids_duplicates_across_runs(tmp_path, monkeypatch):
    SM = _fresh_session(tmp_path, monkeypatch)
    from pos.tillshield_poll import poll_once
    from db.models import Case, IntegrationPollState
    cfg = _cfg(allowed=("52",))
    http = _http({"52": [_row("R1", "52", amount=-100.0,
                              date="2026-06-15T00:01:17")]})
    with SM() as s:
        first = poll_once(s, cfg=cfg, http_get=http, now=NOW)
        s.commit()
    with SM() as s:
        second = poll_once(s, cfg=cfg, http_get=http, now=NOW)
        s.commit()
    assert first["cases_created"] == 1
    # Same row re-fetched on the 2nd cycle -> app-side idempotency drops it.
    assert second["cases_created"] == 0
    assert second["events_inserted"] == 0
    with SM() as s:
        assert s.query(Case).count() == 1
        st = s.query(IntegrationPollState).filter_by(
            workstation_id="52").one()
        # Cursor advanced to the consumed transaction.
        assert st.last_txn_id == "R1"
        assert st.last_txn_at is not None


def test_poll_mixed_feed_counts_by_reason(tmp_path, monkeypatch):
    SM = _fresh_session(tmp_path, monkeypatch)
    from pos.tillshield_poll import poll_once
    cfg = _cfg(allowed=("52",))
    rows = [
        _row("R1", "52", ttype="RETURN", amount=-100.0),   # qualifies
        _row("S1", "52", ttype="SALE", amount=55.9),       # non-return
        _row("R2", "52", ttype="RETURN", amount=20.0),     # non-negative
    ]
    http = _http({"52": rows})
    with SM() as s:
        summary = poll_once(s, cfg=cfg, http_get=http, now=NOW)
        s.commit()
    assert summary["cases_created"] == 1
    assert summary["ignored_non_return_events"] == 1
    assert summary["ignored_non_negative_events"] == 1
    assert summary["rows_seen"] == 3


def test_poll_agent_unavailable_does_not_crash(tmp_path, monkeypatch):
    SM = _fresh_session(tmp_path, monkeypatch)
    from pos.tillshield_poll import poll_once
    cfg = _cfg(allowed=("52",))

    def _boom(url, params, timeout):
        raise ConnectionError("agent down")

    with SM() as s:
        summary = poll_once(s, cfg=cfg, http_get=_boom, now=NOW)
        s.commit()
    assert summary["cases_created"] == 0
    from db.models import IntegrationPollState
    with SM() as s:
        st = s.query(IntegrationPollState).filter_by(
            workstation_id="52").one()
        assert st.last_error and "agent down" in st.last_error


# ---------------------------------------------------------------------------
# 2b. Timezone normalisation (pos_event_at -> naive UTC)
# ---------------------------------------------------------------------------

def test_poll_naive_local_transaction_date_stored_as_utc(tmp_path, monkeypatch):
    """A naive local transactionDate is converted to naive UTC so it
    correlates with UTC CCTV segments (Asia/Dubai is UTC+4)."""
    SM = _fresh_session(tmp_path, monkeypatch)
    from pos.tillshield_poll import poll_once
    cfg = _cfg(allowed=("52",), pos_timezone="Asia/Dubai")
    http = _http({"52": [_row("R1", "52", amount=-100.0,
                              date="2026-06-16T14:30:00")]})  # naive local
    with SM() as s:
        poll_once(s, cfg=cfg, http_get=http, now=datetime(2026, 6, 17))
        s.commit()
    from db.models import PosEvent
    with SM() as s:
        ev = s.query(PosEvent).filter_by(transaction_id="R1").one()
    assert ev.pos_event_at == datetime(2026, 6, 16, 10, 30, 0)  # 14:30 - 4h


def test_poll_offset_timezone_form(tmp_path, monkeypatch):
    SM = _fresh_session(tmp_path, monkeypatch)
    from pos.tillshield_poll import poll_once
    cfg = _cfg(allowed=("52",), pos_timezone="+04:00")
    http = _http({"52": [_row("R1", "52", amount=-100.0,
                              date="2026-06-16T14:30:00")]})
    with SM() as s:
        poll_once(s, cfg=cfg, http_get=http, now=datetime(2026, 6, 17))
        s.commit()
    from db.models import PosEvent
    with SM() as s:
        ev = s.query(PosEvent).filter_by(transaction_id="R1").one()
    assert ev.pos_event_at == datetime(2026, 6, 16, 10, 30, 0)


def test_tz_aware_transaction_date_normalised_to_utc():
    from datetime import timezone as _tz
    from pos.tillshield import normalise_to_pos_events
    from pos.tillshield_schemas import TillShieldTransaction
    txn = TillShieldTransaction(
        transaction_id="A", transaction_date="2026-06-16T14:30:00+04:00",
        transaction_type="RETURN", store_id="2270", workstation_id="52",
        total_amount=-10.0)
    # pos_tz is irrelevant when the timestamp is already tz-aware.
    out = normalise_to_pos_events(txn, pos_tz=None)
    assert out[0].pos_event_at == datetime(2026, 6, 16, 10, 30, 0)


# ---------------------------------------------------------------------------
# 3. Status
# ---------------------------------------------------------------------------

def test_status_reports_counts(tmp_path, monkeypatch):
    SM = _fresh_session(tmp_path, monkeypatch)
    from pos.tillshield_poll import poll_once, read_status
    cfg = _cfg(allowed=("52",))
    http = _http({"52": [_row("R1", "52", amount=-100.0)]})
    with SM() as s:
        poll_once(s, cfg=cfg, http_get=http, now=NOW)
        s.commit()
    with SM() as s:
        status = read_status(s)
    assert status["cumulative"]["cases_created"] == 1
    assert status["last_successful_poll_at"] is not None
    assert any(w["workstation_id"] == "52" for w in status["workstations"])


# ---------------------------------------------------------------------------
# 4. Config validation
# ---------------------------------------------------------------------------

def test_validate_flags_workstation_without_camera_mapping():
    from pos.tillshield_poll import validate_poll_config
    cfg = _cfg(allowed=("52", "53"), ws_map={"52": "cam_return_01"})
    issues = validate_poll_config(cfg)
    assert any("'53'" in i and "workstation_camera_map" in i for i in issues)


def test_validate_flags_mapped_camera_absent():
    from pos.tillshield_poll import validate_poll_config
    cfg = _cfg(cameras=[{"id": "cam_return_01"}], allowed=("52",),
               ws_map={"52": "ghost_cam"})
    issues = validate_poll_config(cfg)
    assert any("ghost_cam" in i for i in issues)


def test_validate_clean_config_has_no_issues():
    from pos.tillshield_poll import validate_poll_config
    cfg = _cfg(cameras=[{"id": "cam_return_01"}], allowed=("52",),
               ws_map={"52": "cam_return_01"})
    assert validate_poll_config(cfg) == []


def test_validate_disabled_poll_is_noop():
    from pos.tillshield_poll import validate_poll_config
    # Misconfigured but disabled -> no issues.
    cfg = _cfg(allowed=("52", "53"), ws_map={}, poll_enabled=False)
    assert validate_poll_config(cfg) == []
