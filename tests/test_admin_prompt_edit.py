"""Local prompt editor tests.

Pins:

* PATCH /admin/prompts/{camera_id} updates the per-camera prompt
  overrides in ``config.yaml`` and surfaces them through GET /admin/prompts.
* Accusation language is rejected before disk is touched.
* When ``ADMIN_EDIT_TOKEN`` is set, the endpoint requires the header.
* Every successful edit lands in the audit log.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("ADMIN_EDIT_TOKEN", raising=False)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'t.sqlite'}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))

    # Copy config.yaml so the editor writes against a sandbox copy that
    # gets restored after the test.
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

    # Restore.
    shutil.copy(backup, cfg_path)


def test_patch_updates_prompts_visible_via_get(client):
    r = client.patch("/api/v1/admin/prompts/cam_return_01",
                     json={"gemma_user": "Describe what you see only."})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "gemma_user" in body["updated_fields"]
    g = client.get("/api/v1/admin/prompts",
                   params={"camera_id": "cam_return_01"}).json()["items"][0]
    assert g["gemma_user"] == "Describe what you see only."


def test_patch_rejects_accusation_language(client):
    r = client.patch("/api/v1/admin/prompts/cam_return_01",
                     json={"gemma_system": "Determine fraud now."})
    assert r.status_code == 400
    body = r.json()["detail"]
    assert "rejected_phrases" in body
    assert "gemma_system" in body["rejected_phrases"]


def test_patch_404_for_unknown_camera(client):
    r = client.patch("/api/v1/admin/prompts/nope",
                     json={"gemma_system": "Be safe."})
    assert r.status_code == 404


def test_patch_requires_token_when_configured(monkeypatch, tmp_path):
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
        r = c.patch("/api/v1/admin/prompts/cam_return_01",
                    json={"gemma_user": "ok"})
        assert r.status_code == 401
        r2 = c.patch("/api/v1/admin/prompts/cam_return_01",
                     json={"gemma_user": "ok"},
                     headers={"X-PhazeX-Admin-Token": "phzx_admin"})
        assert r2.status_code == 200
    finally:
        shutil.copy(backup, cfg_path)


def test_patch_writes_audit_log(client):
    client.patch("/api/v1/admin/prompts/cam_return_01",
                 json={"gemma_system": "Describe only."})
    from db.models import AuditLog
    import db.session as ds
    SM = ds.get_sessionmaker()
    with SM() as s:
        rows = s.query(AuditLog).filter(
            AuditLog.action == "admin.prompt_update").all()
    assert rows
    row = rows[0]
    assert row.entity_id == "cam_return_01"
    assert "prompts" in (row.after_json or {})
