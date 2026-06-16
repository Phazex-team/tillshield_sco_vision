"""Fresh-repo startup must not 500 on the first request.

Simulates a brand-new deploy: temp DATABASE_URL pointing at a
non-existent SQLite file, no init_schema run beforehand, then asks the
app for /health, /storage/disk, /cases. None of them may 500.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_fresh_sqlite_api_smoke(tmp_path, monkeypatch):
    """No init_schema before TestClient creation — the app factory
    must do it itself."""
    db_path = tmp_path / "fresh.sqlite"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))

    import db.session as ds
    ds._ENGINE = None
    ds._SESSION_FACTORY = None
    # Intentionally do NOT call init_schema here — app must.

    from fastapi.testclient import TestClient
    from app.main import create_app
    client = TestClient(create_app())

    health = client.get("/api/v1/health")
    assert health.status_code == 200

    disk = client.get("/api/v1/storage/disk")
    assert disk.status_code == 200, disk.text
    body = disk.json()
    assert "free_bytes" in body
    assert "low_disk_state" in body
    assert body["expired_unlinked_segments"] == 0

    cases = client.get("/api/v1/cases")
    assert cases.status_code == 200, cases.text
    assert cases.json() == {"items": [], "count": 0}


def test_run_app_script_initialises_schema(tmp_path, monkeypatch):
    """`scripts/run_app.py` (when called via its main entry point with
    --skip-checks so the test doesn't open a port) must call
    init_schema before returning into uvicorn."""
    db_path = tmp_path / "fresh.sqlite"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))

    import db.session as ds
    ds._ENGINE = None
    ds._SESSION_FACTORY = None

    # Stub uvicorn.run so the test doesn't actually serve.
    import scripts.run_app as ra
    called = {}
    def _stub_run(app, **kw):
        called["app"] = app
        called["kw"] = kw
    monkeypatch.setattr("uvicorn.run", _stub_run)
    monkeypatch.setattr(sys, "argv",
                        ["run_app.py", "--skip-checks",
                         "--host", "127.0.0.1", "--port", "0"])
    rc = ra.main()
    assert rc == 0
    # The DB file was created by init_schema.
    assert db_path.exists()
