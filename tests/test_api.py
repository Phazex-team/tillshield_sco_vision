"""API surface tests.

Uses FastAPI's TestClient + a per-test SQLite database so the full
transaction-led flow (POS ingest -> case -> reprocess -> evidence
package -> reviewer action -> audit log) is exercised end-to-end
without loading any real models.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'t.sqlite'}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))

    # Reset DB singleton before any code imports it.
    import db.session as ds
    ds._ENGINE = None
    ds._SESSION_FACTORY = None
    ds.init_schema()

    # Stub the VLM chain so the test never tries to load real models.
    from app import case_runner
    def _stub_perception(session, case, window):
        return {"tracks": [{"track_id": "t1", "label": "bag"}],
                "keyframes": [], "ocr": [],
                "obstructed": False,
                "limitations": []}
    def _stub_vlm(session, case, window):
        return {
            "provider": "stub",
            "model_name": "stub-vlm",
            "parsed": {
                "handover_occurred": True,
                "physical_item_presented": True,
                "receipt_visible": True,
                "items_observed": ["bag"],
                "narrative": "stubbed",
                "confidence": "high",
                "obstructed": False,
                "camera_view_clear": True,
                "limitations": [],
            },
            "latency_ms": 12,
            "error": None,
        }

    real_analyze = case_runner.analyze_case
    def patched_analyze(s, case_id, perception_runner=None, vlm_runner=None,
                        prompt_version="return_review_v1", **kwargs):
        # Accept (and ignore) the window-widening kwargs the reprocess path
        # now threads through (pre_roll_sec / post_roll_sec).
        return real_analyze(s, case_id,
                            perception_runner=perception_runner
                                or _stub_perception,
                            vlm_runner=vlm_runner or _stub_vlm,
                            prompt_version=prompt_version)
    monkeypatch.setattr("app.api.cases.analyze_case", patched_analyze)

    from fastapi.testclient import TestClient
    from app.main import create_app
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# Health + memory
# ---------------------------------------------------------------------------

def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_memory(client):
    r = client.get("/api/v1/memory")
    assert r.status_code == 200
    body = r.json()
    for key in ("total_gb", "used_gb", "available_gb", "state",
                "inference_allowed", "loaded_providers"):
        assert key in body


# ---------------------------------------------------------------------------
# POS ingest endpoints
# ---------------------------------------------------------------------------

def _event(transaction_id: str = "txn-1", line_id: str = "L1") -> dict:
    return {
        "store_id": "store_1",
        "terminal_id": "t1",
        "transaction_id": transaction_id,
        "line_id": line_id,
        "event_type": "SALE",
        "pos_event_at": "2026-06-15T14:00:00",
        "sku": "SKU-1",
        "amount": 49.99,
        "currency": "AED",
    }


def test_pos_single_event_creates_case(client):
    r = client.post("/api/v1/pos/returns/event", json=_event())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["events_inserted"] == 1
    assert body["cases_created"] == 1


def test_pos_event_idempotent(client):
    client.post("/api/v1/pos/returns/event", json=_event())
    r = client.post("/api/v1/pos/returns/event", json=_event())
    assert r.json()["duplicate_batch"] is True


def test_pos_batch(client):
    body = {
        "source_system": "pos_v1",
        "store_id": "store_1",
        "events": [_event("txn-A", "L1"), _event("txn-B", "L1")],
    }
    r = client.post("/api/v1/pos/returns/batch", json=body)
    assert r.status_code == 200
    assert r.json()["events_inserted"] == 2
    assert r.json()["cases_created"] == 2


# ---------------------------------------------------------------------------
# Cases list / detail / reprocess / evidence
# ---------------------------------------------------------------------------

def test_cases_list_after_ingest(client):
    client.post("/api/v1/pos/returns/event", json=_event())
    r = client.get("/api/v1/cases")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["items"][0]["status"] == "OPEN"


def test_case_reprocess_invalid_video_when_no_segments(client):
    """No video segments are seeded; reprocess must produce
    INVALID_VIDEO with a structured reason."""
    client.post("/api/v1/pos/returns/event", json=_event())
    cases = client.get("/api/v1/cases").json()["items"]
    case_id = cases[0]["id"]
    r = client.post(f"/api/v1/cases/{case_id}/reprocess")
    assert r.status_code in (200, 202)
    from app.api.cases import _drain_reprocess_pool
    _drain_reprocess_pool()
    body = client.get(f"/api/v1/cases/{case_id}").json()
    assert body["outcome"] == "INVALID_VIDEO"
    assert "no overlapping" in body.get("invalid_reason", "").lower()


def test_case_evidence_package_404_then_present(client):
    client.post("/api/v1/pos/returns/event", json=_event())
    case_id = client.get("/api/v1/cases").json()["items"][0]["id"]
    r = client.get(f"/api/v1/cases/{case_id}/evidence-package")
    assert r.status_code == 404
    client.post(f"/api/v1/cases/{case_id}/reprocess")
    from app.api.cases import _drain_reprocess_pool
    _drain_reprocess_pool()
    r2 = client.get(f"/api/v1/cases/{case_id}/evidence-package")
    assert r2.status_code == 200
    body = r2.json()
    assert body["case"]["outcome"] == "INVALID_VIDEO"
    assert body["audit"]["package_sha256"]


# ---------------------------------------------------------------------------
# Review actions write audit log
# ---------------------------------------------------------------------------

def test_review_action_records_audit(client, tmp_path):
    client.post("/api/v1/pos/returns/event", json=_event())
    case_id = client.get("/api/v1/cases").json()["items"][0]["id"]
    r = client.post(
        f"/api/v1/cases/{case_id}/review-actions",
        json={"reviewer_id": "user_1",
              "action": "verified_physical_return",
              "outcome": "VERIFIED",
              "notes": "manual approve",
              "labels": {"item_visible": True}},
    )
    assert r.status_code == 201
    assert r.json()["outcome"] == "VERIFIED"
    assert r.json()["status"] == "CLOSED"

    # Audit log row recorded.
    import db.session as ds
    from db.models import AuditLog
    SM = ds.get_sessionmaker()
    with SM() as s:
        rows = s.query(AuditLog).filter(
            AuditLog.action == "case.review_action").all()
    assert rows
    assert rows[0].entity_id == case_id


def test_review_action_rejects_bad_action(client):
    client.post("/api/v1/pos/returns/event", json=_event())
    case_id = client.get("/api/v1/cases").json()["items"][0]["id"]
    r = client.post(
        f"/api/v1/cases/{case_id}/review-actions",
        json={"action": "fraud_confirmed", "outcome": "VERIFIED"},
    )
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Video coverage
# ---------------------------------------------------------------------------

def test_segments_coverage_empty(client):
    r = client.get("/api/v1/video/segments/coverage",
                   params={"camera_id": "cam_01",
                           "start_at": "2026-06-15T14:00:00",
                           "end_at": "2026-06-15T14:05:00"})
    assert r.status_code == 200
    assert r.json()["coverage_ratio"] == 0.0


def test_segments_coverage_rejects_inverted_window(client):
    r = client.get("/api/v1/video/segments/coverage",
                   params={"camera_id": "cam_01",
                           "start_at": "2026-06-15T14:05:00",
                           "end_at": "2026-06-15T14:00:00"})
    assert r.status_code == 400
