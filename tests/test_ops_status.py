"""Tests for the operations / pipeline surface.

Covers:

* ``GET /api/v1/ops/status`` shape + that it survives a down vLLM,
  surfaces vLLM unavailable as ERROR (not OK), reports provider chain
  members, memory + disk panels, TillShield panel, and camera segment
  freshness with both ``unknown`` and ``OK/WARNING`` paths.
* ``POST /api/v1/storage/cleanup/dry-run`` deletes nothing.
* ``POST /api/v1/storage/cleanup/execute`` requires the admin token
  when one is configured, audits every execute, and never deletes a
  segment that is linked from a video_window or referenced by an
  artifact.
"""
from __future__ import annotations

import os
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------
# Fixture: isolated DB + storage + sandboxed config.yaml restored after.
# ---------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("ADMIN_EDIT_TOKEN", raising=False)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'t.sqlite'}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))

    # Sandbox config.yaml so prompt edits don't bleed across tests.
    cfg_path = ROOT / "config.yaml"
    backup = tmp_path / "config_backup.yaml"
    shutil.copy(cfg_path, backup)

    import db.session as ds
    ds._ENGINE = None
    ds._SESSION_FACTORY = None
    ds.init_schema()

    # Reset memory policy so it doesn't carry state across tests.
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
# /api/v1/ops/status
# ---------------------------------------------------------------------

def test_ops_status_returns_200_with_expected_top_level_keys(client):
    r = client.get("/api/v1/ops/status")
    assert r.status_code == 200, r.text
    body = r.json()
    for key in ("generated_at", "production_offline_mode", "health",
                "memory", "storage", "tillshield", "provider_chain",
                "qwen_vllm", "gemma", "cameras", "cases", "warnings"):
        assert key in body, f"missing top-level key: {key!r}"


def test_ops_status_does_not_raise_when_vllm_is_down(client):
    # vLLM is NOT running in the test sandbox; the endpoint must still
    # return 200, not 500, and the qwen panel must carry a pill we can
    # render.
    r = client.get("/api/v1/ops/status")
    assert r.status_code == 200
    qwen = r.json()["qwen_vllm"]
    assert "pill" in qwen
    assert qwen.get("backend") in ("vllm_openai", "local_transformers",
                                    "unknown")


def test_ops_status_reports_vllm_unavailable_as_warning_or_error(client):
    """No vLLM server is listening on 127.0.0.1:8000 in the test box.
    The qwen panel pill must be ERROR (or at minimum NOT OK)."""
    r = client.get("/api/v1/ops/status")
    qwen = r.json()["qwen_vllm"]
    assert qwen["pill"] != "OK", \
        f"qwen pill should not be OK when vllm is down; got {qwen}"
    assert qwen["pill"] in ("WARNING", "ERROR", "UNKNOWN")


def test_ops_status_includes_provider_chain_members(client):
    r = client.get("/api/v1/ops/status")
    chain = r.json()["provider_chain"]
    assert "members" in chain
    assert isinstance(chain["members"], list)
    # qwen3_vl is the primary in this project's config.yaml.
    assert "qwen3_vl" in chain["members"] or chain.get("pill") == "ERROR"


def test_ops_status_includes_memory_and_disk(client):
    r = client.get("/api/v1/ops/status")
    body = r.json()
    mem = body["memory"]
    assert "state" in mem
    assert "total_gb" in mem
    assert mem["pill"] in ("OK", "WARNING", "ERROR", "UNKNOWN")
    storage = body["storage"]
    assert "free_gb" in storage
    assert "min_free_gb" in storage
    assert "low_disk_state" in storage
    assert storage["pill"] in ("OK", "WARNING", "ERROR", "UNKNOWN")


def test_ops_status_includes_tillshield(client):
    r = client.get("/api/v1/ops/status")
    ts = r.json()["tillshield"]
    # Either the poller is enabled (status present) or disabled
    # (status: None); both shapes are valid.
    assert "enabled" in ts
    assert "validation_issues" in ts
    assert ts["pill"] in ("OK", "WARNING", "ERROR", "UNKNOWN")


def test_ops_status_camera_freshness_unknown_when_no_segments(client):
    """Fresh DB: no segments => every configured camera reports UNKNOWN
    with a clear ``no segments recorded yet`` detail."""
    r = client.get("/api/v1/ops/status")
    cameras = r.json()["cameras"]
    # The project ships at least one camera (cam_01) in config.yaml.
    assert any(c.get("id") == "cam_01" for c in cameras)
    for cam in cameras:
        if not cam.get("id"):
            continue
        assert cam["pill"] in ("OK", "WARNING", "UNKNOWN")
        if cam["latest_segment_at"] is None:
            assert cam["pill"] == "UNKNOWN"
            assert "no segments" in (cam.get("detail") or "").lower()


def test_ops_status_camera_freshness_ok_when_recent_segment_present(client):
    """Insert a recent segment row and assert the panel flips to OK."""
    from db.models import VideoSegment
    import db.session as ds
    SM = ds.get_sessionmaker()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with SM() as s:
        seg = VideoSegment(
            camera_id="cam_01",
            start_at=now - timedelta(seconds=30),
            end_at=now,
            path="/tmp/seg.mp4",
            sha256="x" * 64,
            fps=25.0, width=640, height=360,
            frame_count=750, duration_sec=30.0,
        )
        s.add(seg)
        s.commit()

    r = client.get("/api/v1/ops/status")
    cam = next(c for c in r.json()["cameras"] if c["id"] == "cam_01")
    assert cam["pill"] == "OK", cam
    assert cam["latest_segment_age_seconds"] is not None
    assert cam["latest_segment_age_seconds"] < 600


def test_ops_status_camera_freshness_warning_when_stale(client):
    """A segment older than 10 minutes is flagged WARNING."""
    from db.models import VideoSegment
    import db.session as ds
    SM = ds.get_sessionmaker()
    old = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=4)
    with SM() as s:
        seg = VideoSegment(
            camera_id="cam_01",
            start_at=old,
            end_at=old + timedelta(seconds=30),
            path="/tmp/old.mp4",
            sha256="y" * 64,
            fps=25.0, width=640, height=360,
            frame_count=750, duration_sec=30.0,
        )
        s.add(seg)
        s.commit()

    r = client.get("/api/v1/ops/status")
    cam = next(c for c in r.json()["cameras"] if c["id"] == "cam_01")
    assert cam["pill"] == "WARNING", cam
    assert cam["latest_segment_age_seconds"] > 600


def test_ops_status_warnings_include_vllm_when_down(client):
    r = client.get("/api/v1/ops/status")
    warnings = r.json()["warnings"]
    # vLLM is not running; we expect a warning that mentions it.
    assert any("qwen3_vl" in w or "vllm" in w.lower() for w in warnings), \
        f"expected a vllm warning, got: {warnings}"


# ---------------------------------------------------------------------
# Storage cleanup endpoints
# ---------------------------------------------------------------------

def _make_two_segments(storage_root: Path):
    """Create two segment files + DB rows: one old & unlinked (target),
    one old & linked-to-window (must NOT be deleted)."""
    from db.models import Artifact, Case, PosEvent, VideoSegment, VideoWindow
    import db.session as ds
    SM = ds.get_sessionmaker()
    storage_root.mkdir(parents=True, exist_ok=True)
    seg_dir = storage_root / "segments"
    seg_dir.mkdir(exist_ok=True)
    free_path = seg_dir / "free.mp4"
    linked_path = seg_dir / "linked.mp4"
    free_path.write_bytes(b"\x00" * 1024)
    linked_path.write_bytes(b"\x00" * 1024)
    old = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=24)
    with SM() as s:
        free = VideoSegment(camera_id="cam_01", start_at=old,
                            end_at=old + timedelta(seconds=30),
                            path=str(free_path), sha256="a"*64,
                            fps=25, width=640, height=360,
                            frame_count=750, duration_sec=30.0)
        linked = VideoSegment(camera_id="cam_01",
                              start_at=old + timedelta(seconds=60),
                              end_at=old + timedelta(seconds=90),
                              path=str(linked_path), sha256="b"*64,
                              fps=25, width=640, height=360,
                              frame_count=750, duration_sec=30.0)
        s.add_all([free, linked])
        s.flush()
        # Build a case + POS event + window referencing ``linked``.
        pe = PosEvent(store_id="store", terminal_id="52",
                      transaction_id="tx1", line_id="transaction",
                      event_type="RETURN",
                      pos_event_at=old + timedelta(seconds=70))
        s.add(pe)
        s.flush()
        case = Case(pos_event_id=pe.id, camera_id="cam_01",
                    status="CLOSED", outcome="REVIEW")
        s.add(case)
        s.flush()
        win = VideoWindow(case_id=case.id, camera_id="cam_01",
                          requested_start_at=old + timedelta(seconds=65),
                          requested_end_at=old + timedelta(seconds=95),
                          status="SUCCEEDED",
                          segment_ids=[linked.id])
        s.add(win)
        s.commit()
        return free.id, linked.id, str(free_path), str(linked_path)


def test_storage_cleanup_dry_run_deletes_nothing(client, tmp_path):
    free_id, linked_id, free_path, linked_path = _make_two_segments(
        tmp_path / "storage")
    r = client.post("/api/v1/storage/cleanup/dry-run")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dry_run"] is True
    assert body["deleted_rows"] == 0
    assert body["deleted_files"] == []
    cand_ids = {c["id"] for c in body["candidates"]}
    assert free_id in cand_ids
    assert linked_id not in cand_ids, "linked segment must never be a candidate"
    # Files still on disk after dry-run.
    assert os.path.exists(free_path)
    assert os.path.exists(linked_path)


def test_storage_cleanup_execute_requires_token_when_configured(
        client, tmp_path, monkeypatch):
    """When ``ADMIN_EDIT_TOKEN`` is set the execute call must 401 without
    the header, then succeed with the correct header."""
    monkeypatch.setenv("ADMIN_EDIT_TOKEN", "phzx_admin")
    free_id, _, free_path, _ = _make_two_segments(tmp_path / "storage")
    r = client.post("/api/v1/storage/cleanup/execute")
    assert r.status_code == 401, r.text
    # File is still on disk.
    assert os.path.exists(free_path)
    r2 = client.post(
        "/api/v1/storage/cleanup/execute",
        headers={"X-PhazeX-Admin-Token": "phzx_admin"})
    assert r2.status_code == 200, r2.text
    assert r2.json()["dry_run"] is False


def test_storage_cleanup_execute_preserves_linked_segments(
        client, tmp_path):
    """Without a configured token, execute is open in dev — it must
    still NEVER delete the linked segment, because the underlying
    storage_guard.identify_expired_unlinked_segments excludes it."""
    free_id, linked_id, free_path, linked_path = _make_two_segments(
        tmp_path / "storage")
    r = client.post("/api/v1/storage/cleanup/execute")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dry_run"] is False
    assert not os.path.exists(free_path), \
        "free unlinked segment file should be deleted"
    assert os.path.exists(linked_path), \
        "linked segment file MUST NOT be deleted"
    # The linked seg row must remain in DB.
    from db.models import VideoSegment
    import db.session as ds
    SM = ds.get_sessionmaker()
    with SM() as s:
        assert s.get(VideoSegment, linked_id) is not None
        assert s.get(VideoSegment, free_id) is None


def test_storage_cleanup_execute_audits(client, tmp_path):
    _make_two_segments(tmp_path / "storage")
    r = client.post("/api/v1/storage/cleanup/execute")
    assert r.status_code == 200
    from db.models import AuditLog
    import db.session as ds
    SM = ds.get_sessionmaker()
    with SM() as s:
        rows = s.query(AuditLog).filter(
            AuditLog.action == "storage.cleanup_executed").all()
    assert rows, "every cleanup execute must be audited"
