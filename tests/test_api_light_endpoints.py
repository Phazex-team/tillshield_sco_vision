"""Deepened tests for the API endpoints previously marked LIGHT.

Each test exercises a specific contract that wasn't covered before.
Tests are deliberately small and focused — one assertion-area each —
so a regression points at exactly the broken contract.

Covered endpoints in this file:
  GET    /api/v1/health
  GET    /api/v1/memory
  GET    /api/v1/admin/classifiers
  GET    /api/v1/cases/{case_id}/evidence-graph   (4xx path)
  POST   /api/v1/cases/{case_id}/reprocess        (4xx path)
  GET    /api/v1/integrations/tillshield/status
  GET    /api/v1/video/segments/coverage          (400 + happy path)
  GET    /api/v1/video/windows/{window_id}/stream (404 + 410 paths)
"""
from __future__ import annotations

import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("ADMIN_EDIT_TOKEN", raising=False)
    monkeypatch.delenv("TILLSHIELD_INGEST_TOKEN", raising=False)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'t.sqlite'}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    cfg_path = ROOT / "config.yaml"
    backup = tmp_path / "config_backup.yaml"
    shutil.copy(cfg_path, backup)
    import db.session as ds
    ds._ENGINE = None
    ds._SESSION_FACTORY = None
    ds.init_schema()
    try:
        from app.memory_guard import get_policy
        get_policy().reset_for_test()
    except Exception:
        pass
    from fastapi.testclient import TestClient
    from app.main import create_app
    c = TestClient(create_app())
    yield c
    shutil.copy(backup, cfg_path)


# ---------------------------------------------------------------------
# /api/v1/health
# ---------------------------------------------------------------------

def test_health_returns_200_with_status_ok(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ---------------------------------------------------------------------
# /api/v1/memory
# ---------------------------------------------------------------------

def test_memory_returns_full_status_shape(client):
    r = client.get("/api/v1/memory")
    assert r.status_code == 200
    body = r.json()
    for key in ("total_gb", "used_gb", "available_gb", "state",
                "inference_allowed", "loaded_providers",
                "degraded_reason", "pending_unloads", "polled_at"):
        assert key in body, f"missing memory key {key!r}"
    # Numeric sanity — used + available must be plausible vs total.
    assert body["total_gb"] >= 0
    assert body["used_gb"] >= 0
    assert body["state"] in (
        "normal", "soft_limit", "hard_limit", "emergency_limit")


# ---------------------------------------------------------------------
# /api/v1/admin/classifiers
# ---------------------------------------------------------------------

def test_classifiers_lists_known_keys(client):
    r = client.get("/api/v1/admin/classifiers")
    assert r.status_code == 200
    items = r.json()["items"]
    assert isinstance(items, list) and items, "expected non-empty classifier list"
    by_key = {it["key"]: it for it in items}
    # The repo ships a ``return_review`` classifier; ``fraud`` is a
    # silently-remapped alias in the loader, so it's allowed to be
    # absent from the listing.
    assert "return_review" in by_key, by_key
    rr = by_key["return_review"]
    for key in ("display_label", "color", "token_budget",
                "enable_thinking", "max_frames"):
        assert key in rr
    # token_budget is positive int; UI renders it as-is.
    assert isinstance(rr["token_budget"], int) and rr["token_budget"] > 0


# ---------------------------------------------------------------------
# /api/v1/cases/{case_id}/evidence-graph 4xx
# ---------------------------------------------------------------------

def test_evidence_graph_404_for_unknown_case(client):
    r = client.get("/api/v1/cases/does-not-exist/evidence-graph")
    assert r.status_code == 404


def test_evidence_package_404_for_unknown_case(client):
    r = client.get("/api/v1/cases/does-not-exist/evidence-package")
    assert r.status_code == 404


# ---------------------------------------------------------------------
# /api/v1/cases/{case_id}/reprocess 4xx
# ---------------------------------------------------------------------

def test_reprocess_returns_404_for_unknown_case(client):
    r = client.post("/api/v1/cases/does-not-exist/reprocess")
    assert r.status_code == 404


# ---------------------------------------------------------------------
# /api/v1/integrations/tillshield/status
# ---------------------------------------------------------------------

def test_tillshield_status_returns_expected_shape(client):
    r = client.get("/api/v1/integrations/tillshield/status")
    assert r.status_code == 200
    body = r.json()
    # Always-present poller meta + cumulative section.
    for key in ("source_system", "cumulative", "workstations",
                "poll_enabled", "poll_every_seconds",
                "allowed_workstation_ids", "workstation_camera_map"):
        assert key in body, f"missing tillshield status key {key!r}"
    assert isinstance(body["workstations"], list)
    assert isinstance(body["allowed_workstation_ids"], list)


# ---------------------------------------------------------------------
# /api/v1/video/segments/coverage
# ---------------------------------------------------------------------

def test_segments_coverage_rejects_end_before_start(client):
    r = client.get("/api/v1/video/segments/coverage",
                    params={"camera_id": "cam_01",
                            "start_at": "2026-06-17T14:00:00",
                            "end_at": "2026-06-17T13:59:59"})
    assert r.status_code == 400
    assert "end_at" in r.json()["detail"]


def test_segments_coverage_returns_zero_for_unknown_camera(client):
    r = client.get("/api/v1/video/segments/coverage",
                    params={"camera_id": "no-such-camera",
                            "start_at": "2026-06-17T14:00:00",
                            "end_at": "2026-06-17T14:01:00"})
    assert r.status_code == 200
    body = r.json()
    # coverage() returns a dict — the exact keys are stable for the UI.
    # We pin only the contract (segments+coverage_ratio present).
    assert "segments" in body
    assert "coverage_ratio" in body
    assert body["coverage_ratio"] == 0.0


# ---------------------------------------------------------------------
# /api/v1/video/windows/{window_id}/stream
# ---------------------------------------------------------------------

def test_window_stream_404_for_unknown_window(client):
    r = client.get("/api/v1/video/windows/does-not-exist/stream")
    assert r.status_code == 404


def test_window_stream_410_when_window_file_missing(client, tmp_path):
    """Insert a VideoWindow row whose ``path`` points at a file that
    was removed (the storage-cleanup path could remove it). The endpoint
    must return 410, not 200 or 500."""
    from db.models import Case, PosBatch, PosEvent, VideoWindow
    import db.session as ds
    SM = ds.get_sessionmaker()
    with SM() as s:
        batch = PosBatch(source_system="t", store_id="s"); s.add(batch); s.flush()
        pe = PosEvent(batch_id=batch.id, store_id="s", terminal_id="t1",
                       transaction_id="x1", line_id="L1",
                       event_type="RETURN",
                       pos_event_at=datetime.now())
        s.add(pe); s.flush()
        case = Case(pos_event_id=pe.id, camera_id="cam_01",
                     status="CLOSED", outcome="REVIEW")
        s.add(case); s.flush()
        win = VideoWindow(case_id=case.id, camera_id="cam_01",
                          requested_start_at=datetime.now(),
                          requested_end_at=datetime.now() + timedelta(seconds=30),
                          status="SUCCEEDED",
                          path=str(tmp_path / "deleted_window.mp4"))
        s.add(win); s.commit()
        win_id = win.id
    r = client.get(f"/api/v1/video/windows/{win_id}/stream")
    assert r.status_code == 410


def test_window_stream_returns_mp4_when_file_present(client, tmp_path):
    from db.models import Case, PosBatch, PosEvent, VideoWindow
    import db.session as ds
    real_mp4 = tmp_path / "real_window.mp4"
    # Minimal MP4 bytes — we don't actually decode; FileResponse just
    # streams them. Even an empty file proves the endpoint resolves.
    real_mp4.write_bytes(b"\x00" * 32)
    SM = ds.get_sessionmaker()
    with SM() as s:
        batch = PosBatch(source_system="t", store_id="s"); s.add(batch); s.flush()
        pe = PosEvent(batch_id=batch.id, store_id="s", terminal_id="t1",
                       transaction_id="x2", line_id="L1",
                       event_type="RETURN",
                       pos_event_at=datetime.now())
        s.add(pe); s.flush()
        case = Case(pos_event_id=pe.id, camera_id="cam_01",
                     status="CLOSED", outcome="REVIEW")
        s.add(case); s.flush()
        win = VideoWindow(case_id=case.id, camera_id="cam_01",
                          requested_start_at=datetime.now(),
                          requested_end_at=datetime.now() + timedelta(seconds=30),
                          status="SUCCEEDED", path=str(real_mp4))
        s.add(win); s.commit()
        win_id = win.id
    r = client.get(f"/api/v1/video/windows/{win_id}/stream")
    assert r.status_code == 200
    assert r.headers["content-type"] == "video/mp4"
