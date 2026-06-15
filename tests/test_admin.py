"""Admin (read-only) endpoint tests."""
from __future__ import annotations

import sys
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
    from fastapi.testclient import TestClient
    from app.main import create_app
    return TestClient(create_app())


def test_admin_config_redacts_secrets(client):
    r = client.get("/api/v1/admin/config")
    assert r.status_code == 200
    body = r.json()
    # rtsp_url values must be redacted.
    for cam in body.get("cameras", []):
        if "rtsp_url" in cam:
            assert cam["rtsp_url"] == "***redacted***"
    assert "models" in body
    assert "sam2" in body["models"]


def test_admin_classifiers_lists_review_safe_entries(client):
    r = client.get("/api/v1/admin/classifiers")
    assert r.status_code == 200
    keys = {c["key"] for c in r.json()["items"]}
    assert "return_review" in keys


def test_admin_prompts_returns_effective_text_and_no_accusation(client):
    r = client.get("/api/v1/admin/prompts")
    assert r.status_code == 200
    items = r.json()["items"]
    assert items
    for item in items:
        assert item["gemma_system"]
        assert item["safety_violation"] == [], (
            f"safety violation on camera {item['camera_id']!r}: "
            f"{item['safety_violation']}"
        )


def test_admin_prompts_camera_id_filter(client):
    r = client.get("/api/v1/admin/prompts", params={"camera_id": "cam_01"})
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 1
    assert items[0]["camera_id"] == "cam_01"


def test_admin_prompts_unknown_camera_404(client):
    r = client.get("/api/v1/admin/prompts", params={"camera_id": "nope"})
    assert r.status_code == 404


def test_legacy_index_html_redirects_to_review(client):
    r = client.get("/index.html")
    assert r.status_code == 200
    src = r.text
    assert "review.html" in src
    assert "removed" in src.lower()
