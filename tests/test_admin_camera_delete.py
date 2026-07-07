"""DELETE /api/v1/admin/cameras/{id} — safe removal + hot-apply reporting.

Pins:
* delete removes the camera from config.yaml and from GET /admin/cameras.
* a camera still mapped to a POS workstation is refused (409) unless
  clear_workstation_mappings=true, which clears the mapping.
* 404 for an unknown camera; token required when configured.
* the response carries a runtime block (config_written + app applied).
* deleting an unrelated camera leaves workstation 57 -> cam_return_01 intact.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("ADMIN_EDIT_TOKEN", raising=False)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'t.sqlite'}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    # Isolate the recorder heartbeat so the runtime report is deterministic.
    monkeypatch.setenv("RECORDER_STATE_PATH", str(tmp_path / "rec_state.json"))

    cfg_path = ROOT / "config.yaml"
    backup = tmp_path / "config_backup.yaml"
    shutil.copy(cfg_path, backup)

    import db.session as ds
    ds._ENGINE = None
    ds._SESSION_FACTORY = None
    ds.init_schema()

    from fastapi.testclient import TestClient
    from app.main import create_app
    yield TestClient(create_app())

    shutil.copy(backup, cfg_path)


def _cfg():
    return yaml.safe_load((ROOT / "config.yaml").read_text()) or {}


def _ids():
    return [c["id"] for c in _cfg()["cameras"]]


def test_delete_unmapped_camera_hot_applies(client):
    # Create a spare camera (no workstation), then delete it.
    client.post("/api/v1/admin/cameras",
                json={"camera_id": "cam_tmp", "rtsp_url": "rtsp://h/s"})
    assert "cam_tmp" in _ids()
    r = client.delete("/api/v1/admin/cameras/cam_tmp")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["deleted"] is True
    assert body["runtime"]["config_written"] is True
    assert body["runtime"]["app"]["applied"] is True
    assert "cam_tmp" not in _ids()
    # And gone from the live list endpoint.
    ids = [i["camera_id"] for i in client.get("/api/v1/admin/cameras").json()["items"]]
    assert "cam_tmp" not in ids


def test_delete_mapped_camera_refused_without_flag(client):
    # cam_return_01 ships mapped to workstation 57 -> must be refused.
    r = client.delete("/api/v1/admin/cameras/cam_return_01")
    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert detail["workstations"]  # lists the blocking workstation(s)
    # Still present — nothing removed.
    assert "cam_return_01" in _ids()


def test_delete_mapped_camera_with_clear_flag(client):
    prior_ws = _cfg()["integrations"]["tillshield"]["workstation_camera_map"]
    mapped = [ws for ws, cam in prior_ws.items() if cam == "cam_return_01"]
    assert mapped, "precondition: cam_return_01 should be workstation-mapped"

    r = client.delete("/api/v1/admin/cameras/cam_return_01"
                      "?clear_workstation_mappings=true")
    assert r.status_code == 200, r.text
    assert set(r.json()["cleared_workstations"]) == set(mapped)
    assert "cam_return_01" not in _ids()
    ws_map = _cfg()["integrations"]["tillshield"]["workstation_camera_map"]
    for ws in mapped:
        assert ws not in ws_map


def test_delete_unknown_camera_404(client):
    assert client.delete("/api/v1/admin/cameras/nope").status_code == 404


def test_delete_requires_token_when_configured(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_EDIT_TOKEN", "phzx_admin")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'t.sqlite'}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("RECORDER_STATE_PATH", str(tmp_path / "rec_state.json"))
    cfg_path = ROOT / "config.yaml"
    backup = tmp_path / "config_backup.yaml"
    shutil.copy(cfg_path, backup)
    import db.session as ds
    ds._ENGINE = None
    ds._SESSION_FACTORY = None
    ds.init_schema()
    from fastapi.testclient import TestClient
    from app.main import create_app
    c = TestClient(create_app())
    try:
        c.post("/api/v1/admin/cameras",
               json={"camera_id": "cam_tok", "rtsp_url": "rtsp://h/s"},
               headers={"X-PhazeX-Admin-Token": "phzx_admin"})
        assert c.delete("/api/v1/admin/cameras/cam_tok").status_code == 401
        r = c.delete("/api/v1/admin/cameras/cam_tok",
                     headers={"X-PhazeX-Admin-Token": "phzx_admin"})
        assert r.status_code == 200, r.text
    finally:
        shutil.copy(backup, cfg_path)


def test_delete_preserves_production_workstation_mapping(client):
    # Deleting an unrelated spare camera must NOT disturb 57 -> cam_return_01.
    before = _cfg()["integrations"]["tillshield"]["workstation_camera_map"]
    client.post("/api/v1/admin/cameras",
                json={"camera_id": "cam_spare", "rtsp_url": "rtsp://h/s"})
    client.delete("/api/v1/admin/cameras/cam_spare")
    after = _cfg()["integrations"]["tillshield"]["workstation_camera_map"]
    assert after == before
    assert after.get("57") == "cam_return_01"
