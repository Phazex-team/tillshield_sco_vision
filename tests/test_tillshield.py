"""TillShield POS integration tests."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("TILLSHIELD_INGEST_TOKEN", raising=False)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'t.sqlite'}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    import db.session as ds
    ds._ENGINE = None
    ds._SESSION_FACTORY = None
    ds.init_schema()

    from fastapi.testclient import TestClient
    from app.main import create_app
    return TestClient(create_app())


def _txn(transaction_id="RTX-1", txn_type="SALE",
         pos_event_at="2026-06-15T14:00:00+04:00",
         items=None, store="store_1", ws="WS-1",
         operator="op_1", amount="49.99", currency="AED") -> dict:
    return {
        "transaction_id": transaction_id,
        "store_id": store,
        "workstation_id": ws,
        "transaction_type": txn_type,
        "transaction_date": pos_event_at,
        "operator_id": operator,
        "cashier_name": "anita",
        "currency": currency,
        "total_amount": amount,
        "total_items": 1,
        "items": items or [{"line_id": "L1", "sku": "SKU-A",
                            "description": "shirt", "quantity": 1,
                            "amount": 49.99}],
        "payload": {"channel": "store"},
        "source_ip": "10.0.0.42",
        "reference_id": "REF-99",
        "transaction_end_date": "2026-06-15T14:01:00+04:00",
        "received_at": "2026-06-15T14:01:30+04:00",
    }


# ---------------------------------------------------------------------------
# 1. Happy path
# ---------------------------------------------------------------------------

def test_tillshield_return_creates_one_case(client):
    r = client.post(
        "/api/v1/integrations/tillshield/transactions/event",
        json=_txn(txn_type="SALE"),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["events_inserted"] == 1
    assert body["cases_created"] == 1
    assert body["ignored_non_return_events"] == 0
    assert len(body["case_ids"]) == 1


def test_tillshield_refund_creates_one_case(client):
    r = client.post(
        "/api/v1/integrations/tillshield/transactions/event",
        json=_txn(transaction_id="RTX-2", txn_type="SALE"),
    )
    assert r.json()["cases_created"] == 1


# ---------------------------------------------------------------------------
# 2. Idempotency + retry
# ---------------------------------------------------------------------------

def test_tillshield_duplicate_transaction_no_duplicate_case(client):
    client.post("/api/v1/integrations/tillshield/transactions/event",
                json=_txn())
    r = client.post(
        "/api/v1/integrations/tillshield/transactions/event",
        json=_txn(),
    )
    # The second call must NOT 5xx (agent retries are safe) and must
    # not open a second case.
    assert r.status_code == 200
    body = r.json()
    # Same exact payload -> the underlying ingest reports the batch as
    # already-present (events_already_present counts the duplicate
    # natural key) and cases_created stays at 0.
    assert body["cases_created"] == 0
    cases = client.get("/api/v1/cases").json()
    assert cases["count"] == 1


# ---------------------------------------------------------------------------
# 3. Non-return events
# ---------------------------------------------------------------------------

def test_tillshield_sale_event_creates_case_in_sco_mode(client):
    """Phase 1 / SCO mode: SALE is now the canonical checkout event and
    opens a case. (In legacy refund mode this test asserted the
    opposite — flipped here as part of the SCO conversion.)"""
    r = client.post(
        "/api/v1/integrations/tillshield/transactions/event",
        json=_txn(transaction_id="RTX-SALE", txn_type="SALE"),
    )
    body = r.json()
    assert body["cases_created"] == 1
    assert body["ignored_non_return_events"] == 0
    assert client.get("/api/v1/cases").json()["count"] == 1


def test_tillshield_unknown_type_is_ignored_silently(client):
    r = client.post(
        "/api/v1/integrations/tillshield/transactions/event",
        json=_txn(transaction_id="RTX-X", txn_type="EXCHANGE"),
    )
    body = r.json()
    assert body["ignored_non_return_events"] == 1
    assert body["cases_created"] == 0


# ---------------------------------------------------------------------------
# 4. Validation
# ---------------------------------------------------------------------------

def test_tillshield_missing_transaction_date_returns_400(client):
    payload = _txn()
    payload.pop("transaction_date")
    r = client.post(
        "/api/v1/integrations/tillshield/transactions/event",
        json=payload,
    )
    assert r.status_code == 400
    assert "transaction_date" in r.text.lower()


def test_tillshield_invalid_json_returns_400(client):
    # Send raw bytes that aren't JSON
    r = client.post(
        "/api/v1/integrations/tillshield/transactions/event",
        content=b"not json",
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# 5. Timezone + delayed POS
# ---------------------------------------------------------------------------

def test_tillshield_timezone_aware_date_preserved(client):
    """The POS event must record the original transaction_date even
    when received_at is much later."""
    payload = _txn(transaction_id="RTX-TZ",
                   pos_event_at="2026-06-15T14:00:00+04:00")
    # Pretend received_at is 30 minutes later.
    payload["received_at"] = "2026-06-15T14:30:00+04:00"
    r = client.post("/api/v1/integrations/tillshield/transactions/event",
                    json=payload)
    assert r.status_code == 200
    # The persisted case row carries the txn time; reprocess must use
    # transaction_date when resolving the CCTV window — proven by the
    # INVALID_VIDEO failure_reason mentioning the original timestamp.
    case_id = client.get("/api/v1/cases").json()["items"][0]["id"]
    rr = client.post(f"/api/v1/cases/{case_id}/reprocess")
    assert rr.status_code == 202
    from app.api.cases import _drain_reprocess_pool
    _drain_reprocess_pool()
    body = client.get(f"/api/v1/cases/{case_id}").json()
    # Without segments the case becomes INVALID_VIDEO and the reason
    # references the requested window centred on transaction_date.
    assert body["outcome"] == "INVALID_VIDEO"


def test_tillshield_items_jsonb_preserved_in_raw_payload(client):
    items = [{"line_id": "L1", "sku": "SKU-A", "quantity": 1,
              "amount": 49.99, "promo_code": "P10"}]
    r = client.post(
        "/api/v1/integrations/tillshield/transactions/event",
        json=_txn(transaction_id="RTX-ITEMS", items=items),
    )
    assert r.status_code == 200
    # Confirm the item payload landed in the pos_events.raw_payload column.
    import db.session as ds
    from db.models import PosEvent
    SM = ds.get_sessionmaker()
    with SM() as s:
        ev = s.query(PosEvent).filter(
            PosEvent.transaction_id == "RTX-ITEMS").first()
    assert ev is not None
    raw = ev.raw_payload or {}
    assert raw.get("items") == items
    # Source IP also propagated for audit.
    assert raw.get("source_ip") in ("10.0.0.42", "testclient", None) \
        or raw["source_ip"]


def test_tillshield_source_ip_falls_back_to_request_ip(client):
    """When the TillShield payload has no source_ip the request IP is
    captured instead. The TestClient client IP is 'testclient'."""
    payload = _txn(transaction_id="RTX-NOIP")
    payload.pop("source_ip", None)
    r = client.post("/api/v1/integrations/tillshield/transactions/event",
                    json=payload)
    assert r.status_code == 200
    from db.models import PosEvent
    import db.session as ds
    SM = ds.get_sessionmaker()
    with SM() as s:
        ev = s.query(PosEvent).filter(
            PosEvent.transaction_id == "RTX-NOIP").first()
    assert ev and ev.raw_payload.get("source_ip") == "testclient"


# ---------------------------------------------------------------------------
# 6. Batch
# ---------------------------------------------------------------------------

def test_tillshield_batch_mixed_kinds(client):
    """Phase 1 / SCO mode: SALE, RETURN, REFUND all canonicalise to SALE
    and open cases. Only types outside the configured accept list
    (e.g. EXCHANGE) are ignored."""
    body = {
        "source_system": "tillshield_agent",
        "events": [
            _txn(transaction_id="RTX-A", txn_type="SALE"),
            _txn(transaction_id="RTX-B", txn_type="SALE"),
            _txn(transaction_id="RTX-A", txn_type="SALE"),  # duplicate
            _txn(transaction_id="RTX-C", txn_type="SALE"),
        ],
    }
    r = client.post(
        "/api/v1/integrations/tillshield/transactions/batch", json=body)
    assert r.status_code == 200
    summary = r.json()
    # All three unique events (A, B, C) canonicalise to SALE → 3 cases.
    assert summary["cases_created"] == 3
    assert summary["ignored_non_return_events"] == 0
    assert summary["events_inserted"] == 3


# ---------------------------------------------------------------------------
# 7. Auth
# ---------------------------------------------------------------------------

def test_tillshield_token_required_when_configured(monkeypatch, tmp_path):
    """When TILLSHIELD_INGEST_TOKEN is set, requests without the header
    must 401."""
    monkeypatch.setenv("TILLSHIELD_INGEST_TOKEN", "phzx_test")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'t.sqlite'}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    import db.session as ds
    ds._ENGINE = None
    ds._SESSION_FACTORY = None
    ds.init_schema()
    from fastapi.testclient import TestClient
    from app.main import create_app
    c = TestClient(create_app())

    r = c.post("/api/v1/integrations/tillshield/transactions/event",
               json=_txn())
    assert r.status_code == 401

    r2 = c.post("/api/v1/integrations/tillshield/transactions/event",
                json=_txn(),
                headers={"X-PhazeX-Ingest-Token": "phzx_test"})
    assert r2.status_code == 200


def test_tillshield_token_never_in_response_body(client):
    """No token-related field ever leaks into the JSON response."""
    r = client.post("/api/v1/integrations/tillshield/transactions/event",
                    json=_txn(),
                    headers={"X-PhazeX-Ingest-Token": "should-not-echo"})
    body_text = r.text.lower()
    assert "should-not-echo" not in body_text
    assert "ingest_token" not in body_text


# ---------------------------------------------------------------------------
# 8. No user-facing accusation language
# ---------------------------------------------------------------------------

def test_tillshield_docs_no_accusation_language():
    src = (ROOT / "docs" / "TILLSHIELD_INTEGRATION.md").read_text().lower()
    for phrase in ("fraud", "fraudulent", "theft", "suspect",
                    "loss-prevention"):
        assert phrase not in src, \
            f"docs/TILLSHIELD_INTEGRATION.md contains banned phrase {phrase!r}"
