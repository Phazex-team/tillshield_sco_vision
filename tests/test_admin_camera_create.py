"""Add-camera endpoint tests.

Pins for ``POST /api/v1/admin/cameras``:

* A new camera is written to ``config.yaml`` with the shipped shape and
  shows up in GET /admin/cameras immediately.
* Duplicate ``camera_id`` -> 409; missing ``camera_id`` / ``rtsp_url`` -> 400/422.
* An optional ``workstation_id`` maps the workstation to the new camera,
  reassigning it away from any camera it previously pointed at.
* When ``ADMIN_EDIT_TOKEN`` is set the endpoint requires the header.
* Every successful create lands in the audit log as ``admin.camera_created``.
* The existing PATCH edit flow still works on the created camera.
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

    # Sandbox config.yaml — the endpoint rewrites it in place; restore after.
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


def test_create_camera_persists_and_lists(client):
    r = client.post("/api/v1/admin/cameras", json={
        "camera_id": "cam_new", "rtsp_url": "rtsp://u:p@host:554/s",
        "name": "Test Camera",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["camera_id"] == "cam_new"
    assert body["created"]["name"] == "Test Camera"

    # Written to config with the seeded shape.
    cam = next(c for c in _cfg()["cameras"] if c["id"] == "cam_new")
    assert cam["rtsp_url"] == "rtsp://u:p@host:554/s"
    assert cam["classifier"] == "sco_checkout"
    assert cam["token_budget"] == 1120
    assert cam["cooldown_sec"] == 30
    assert cam["zones"] == {} and cam["prompts"] == {}

    # Visible via the listing endpoint.
    ids = [i["camera_id"] for i in client.get("/api/v1/admin/cameras").json()["items"]]
    assert "cam_new" in ids


def test_create_camera_defaults_name_to_id(client):
    r = client.post("/api/v1/admin/cameras",
                    json={"camera_id": "cam_x", "rtsp_url": "rtsp://h/s"})
    assert r.status_code == 200, r.text
    assert r.json()["created"]["name"] == "cam_x"


def test_duplicate_camera_id_conflicts(client):
    # cam_return_01 is the shipped production camera in config.yaml.
    r = client.post("/api/v1/admin/cameras",
                    json={"camera_id": "cam_return_01", "rtsp_url": "rtsp://h/s"})
    assert r.status_code == 409, r.text


def test_missing_rtsp_url_rejected(client):
    r = client.post("/api/v1/admin/cameras",
                    json={"camera_id": "cam_y", "rtsp_url": "   "})
    assert r.status_code == 400, r.text


def test_missing_camera_id_rejected(client):
    # Pydantic requires camera_id -> 422; blank string -> 400.
    assert client.post("/api/v1/admin/cameras",
                       json={"rtsp_url": "rtsp://h/s"}).status_code == 422
    assert client.post("/api/v1/admin/cameras",
                       json={"camera_id": "  ", "rtsp_url": "rtsp://h/s"}
                       ).status_code == 400


def test_create_with_workstation_mapping(client):
    r = client.post("/api/v1/admin/cameras", json={
        "camera_id": "cam_ws", "rtsp_url": "rtsp://h/s", "workstation_id": "99",
    })
    assert r.status_code == 200, r.text
    ts = _cfg()["integrations"]["tillshield"]
    assert ts["workstation_camera_map"]["99"] == "cam_ws"
    assert "99" in [str(x) for x in ts["allowed_workstation_ids"]]
    # And reported on the created camera's listing row.
    row = next(i for i in client.get("/api/v1/admin/cameras").json()["items"]
               if i["camera_id"] == "cam_ws")
    assert row["workstation_id"] == "99"


def test_workstation_reassignment_reported(client):
    # Workstation 57 ships mapped to the production camera; creating a
    # camera that claims it must report the takeover and drop the old map.
    prior = _cfg()["integrations"]["tillshield"]["workstation_camera_map"]["57"]
    r = client.post("/api/v1/admin/cameras", json={
        "camera_id": "cam_takeover", "rtsp_url": "rtsp://h/s",
        "workstation_id": "57",
    })
    assert r.status_code == 200, r.text
    assert r.json()["workstation_reassigned_from"] == prior
    ws_map = _cfg()["integrations"]["tillshield"]["workstation_camera_map"]
    assert ws_map["57"] == "cam_takeover"


def test_requires_token_when_configured(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_EDIT_TOKEN", "phzx_admin")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'t.sqlite'}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
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
        body = {"camera_id": "cam_tok", "rtsp_url": "rtsp://h/s"}
        assert c.post("/api/v1/admin/cameras", json=body).status_code == 401
        r = c.post("/api/v1/admin/cameras", json=body,
                   headers={"X-PhazeX-Admin-Token": "phzx_admin"})
        assert r.status_code == 200, r.text
    finally:
        shutil.copy(backup, cfg_path)


def test_create_writes_audit_log(client):
    client.post("/api/v1/admin/cameras",
                json={"camera_id": "cam_audit", "rtsp_url": "rtsp://h/s"})
    from db.models import AuditLog
    import db.session as ds
    SM = ds.get_sessionmaker()
    with SM() as s:
        rows = s.query(AuditLog).filter(
            AuditLog.action == "admin.camera_created").all()
    assert rows and rows[0].entity_id == "cam_audit"


def test_created_camera_is_editable(client):
    client.post("/api/v1/admin/cameras",
                json={"camera_id": "cam_edit", "rtsp_url": "rtsp://h/s"})
    # The existing PATCH edit flow must work on the new camera.
    r = client.patch("/api/v1/admin/cameras/cam_edit",
                     json={"name": "Renamed", "rtsp_url": "rtsp://h2/s2",
                           "workstation_id": ""})
    assert r.status_code == 200, r.text
    cam = next(c for c in _cfg()["cameras"] if c["id"] == "cam_edit")
    assert cam["name"] == "Renamed" and cam["rtsp_url"] == "rtsp://h2/s2"
