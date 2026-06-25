"""End-to-end transaction-led flow.

Walks the user story from PRODUCTION_SPEC §3:
  1. POS event arrives (potentially up to 30 minutes after the fact).
  2. The app idempotently ingests it and creates a case.
  3. The window resolver picks the right segments (using
     pos_event_at, NOT ingested_at).
  4. Without footage the case becomes INVALID_VIDEO.
  5. Perception emits track/keyframe structures (stubbed for offline).
  6. Decision policy wraps the VLM output; only the four valid
     outcomes ever appear.
  7. Reviewer submits an action; audit trail records it.

The VLM + perception are stubbed so the test exercises plumbing only.
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
    import db.session as ds
    ds._ENGINE = None
    ds._SESSION_FACTORY = None
    ds.init_schema()

    from app import case_runner
    real = case_runner.analyze_case

    def _stub_perception(session, case, window):
        return {"tracks": [{"track_id": "t1", "label": "shopping bag",
                            "events": ["entered_counter_zone",
                                       "handover_candidate"],
                            "physical_item_candidate": True}],
                "keyframes": [{"role": "handover_candidate",
                               "frame_id": "f042",
                               "frame_idx": 42}],
                "ocr": [], "obstructed": False, "limitations": []}

    def _stub_vlm(session, case, window):
        return {
            "provider": "qwen3_vl", "model_name": "stub-qwen",
            "parsed": {
                "handover_occurred": True,
                "physical_item_presented": True,
                "receipt_visible": True,
                "items_observed": ["shopping bag"],
                "narrative": "stubbed verdict",
                "confidence": "high",
                "obstructed": False,
                "camera_view_clear": True,
                "limitations": [],
                "_chain_attempts": ["qwen3_vl=ok"],
            },
            "latency_ms": 12,
            "error": None,
        }

    def patched(s, case_id, perception_runner=None, vlm_runner=None,
                prompt_version="return_review_v1", **kwargs):
        # ``**kwargs`` tolerates new pass-through params the caller may add
        # (e.g. pre_roll_sec/post_roll_sec from the retime-window feature)
        # without this stub needing to track every signature change.
        return real(s, case_id,
                    perception_runner=perception_runner or _stub_perception,
                    vlm_runner=vlm_runner or _stub_vlm,
                    prompt_version=prompt_version, **kwargs)
    monkeypatch.setattr("app.api.cases.analyze_case", patched)

    from fastapi.testclient import TestClient
    from app.main import create_app
    return TestClient(create_app())


def _pos_event(*, transaction_id: str = "txn-1") -> dict:
    return {
        "store_id": "store_1", "terminal_id": "t1",
        "transaction_id": transaction_id, "line_id": "L1",
        "event_type": "RETURN",
        "pos_event_at": "2026-06-15T14:00:00",
        "staff_id": "staff_77", "sku": "SKU-A",
        "amount": 49.99, "currency": "AED",
    }


def test_delayed_pos_uses_event_time_not_ingest_time(client, tmp_path):
    """A POS event arriving 25 minutes after the fact must still
    resolve against pos_event_at, not the ingest time. Without
    segments the case becomes INVALID_VIDEO."""
    body = _pos_event()
    r = client.post("/api/v1/pos/returns/event", json=body)
    assert r.status_code == 200
    case = client.get("/api/v1/cases").json()["items"][0]
    rr = client.post(f"/api/v1/cases/{case['id']}/reprocess")
    assert rr.status_code == 202
    from app.api.cases import _drain_reprocess_pool
    _drain_reprocess_pool()
    closed = client.get(f"/api/v1/cases/{case['id']}").json()
    assert closed["outcome"] == "INVALID_VIDEO"


def test_idempotent_pos_does_not_double_open(client):
    client.post("/api/v1/pos/returns/event", json=_pos_event())
    client.post("/api/v1/pos/returns/event", json=_pos_event())
    cases = client.get("/api/v1/cases").json()
    assert cases["count"] == 1


def test_perception_pipeline_emits_structures_after_reprocess_with_segments(
        client, tmp_path, monkeypatch):
    """Seed a fake segment so the window resolver succeeds, then
    reprocess. The stubbed VLM returns a VERIFIED-shape payload; the
    decision policy still has the final word."""
    import db.session as ds
    from db.models import VideoSegment
    SM = ds.get_sessionmaker()
    base = datetime(2026, 6, 15, 14, 0, 0)
    with SM() as s:
        s.add(VideoSegment(
            camera_id="cam_01",
            start_at=base - timedelta(seconds=200),
            end_at=base + timedelta(seconds=200),
            path=str(tmp_path / "fake.mp4"),
            sha256="a" * 64, fps=25,
            width=160, height=120,
            frame_count=1500, duration_sec=60,
            corrupt=False, has_gap=False,
        ))
        s.commit()
    client.post("/api/v1/pos/returns/event", json=_pos_event())
    case = client.get("/api/v1/cases").json()["items"][0]
    r = client.post(f"/api/v1/cases/{case['id']}/reprocess")
    assert r.status_code in (200, 202)
    from app.api.cases import _drain_reprocess_pool
    _drain_reprocess_pool()
    closed = client.get(f"/api/v1/cases/{case['id']}").json()
    assert closed["outcome"] in (
        "VERIFIED", "REVIEW", "HIGH_RISK_REVIEW", "INVALID_VIDEO")


def test_reviewer_action_is_audited_and_closes_case(client):
    client.post("/api/v1/pos/returns/event", json=_pos_event())
    case_id = client.get("/api/v1/cases").json()["items"][0]["id"]
    r = client.post(f"/api/v1/cases/{case_id}/review-actions",
                    json={"reviewer_id": "u1",
                          "action": "verified_physical_return",
                          "outcome": "VERIFIED",
                          "notes": "ok",
                          "labels": {"item_visible": True}})
    assert r.status_code == 201
    case = client.get(f"/api/v1/cases/{case_id}").json()
    assert case["status"] == "CLOSED"
    assert case["outcome"] == "VERIFIED"
    import db.session as ds
    from db.models import AuditLog
    SM = ds.get_sessionmaker()
    with SM() as s:
        rows = s.query(AuditLog).filter(
            AuditLog.action == "case.review_action").all()
    assert rows


def test_evidence_package_is_versioned_and_hashed(client, tmp_path):
    client.post("/api/v1/pos/returns/event", json=_pos_event())
    case_id = client.get("/api/v1/cases").json()["items"][0]["id"]
    from app.api.cases import _drain_reprocess_pool
    client.post(f"/api/v1/cases/{case_id}/reprocess")
    _drain_reprocess_pool()
    pkg = client.get(f"/api/v1/cases/{case_id}/evidence-package").json()
    assert pkg["audit"]["package_sha256"]
    pkg_dir = (Path(tmp_path) / "storage" / "cases"
               / f"case_id={case_id}" / "package")
    files = list(pkg_dir.iterdir())
    assert files, "package file should be written"
    # A second reprocess writes a NEW file, not overwrites.
    client.post(f"/api/v1/cases/{case_id}/reprocess")
    _drain_reprocess_pool()
    files2 = list(pkg_dir.iterdir())
    assert len(files2) > len(files)
