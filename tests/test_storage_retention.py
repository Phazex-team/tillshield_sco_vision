"""Storage retention + cleanup + low-disk guard tests."""
from __future__ import annotations

import os
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _fresh_session(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'t.sqlite'}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    import db.session as s
    s._ENGINE = None
    s._SESSION_FACTORY = None
    s.init_schema()
    return s.get_sessionmaker()


def _make_file(path: Path, content: bytes = b"video bytes") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return str(path)


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _insert_segment(session, *, camera_id="cam_01", start_at, end_at,
                    path: str) -> str:
    from db.models import VideoSegment
    row = VideoSegment(camera_id=camera_id, start_at=start_at,
                       end_at=end_at, path=path, sha256="a" * 64,
                       fps=25, width=160, height=120,
                       frame_count=100, duration_sec=4)
    session.add(row)
    session.flush()
    return row.id


def test_identify_expired_unlinked_segments(tmp_path, monkeypatch):
    SM = _fresh_session(tmp_path, monkeypatch)
    from app.storage_guard import identify_expired_unlinked_segments
    base = _now()
    with SM() as s:
        # OLD unlinked -> deletable
        old_path = _make_file(tmp_path / "storage" / "old.mp4")
        old_id = _insert_segment(s,
            start_at=base - timedelta(hours=24),
            end_at=base - timedelta(hours=23),
            path=old_path)
        # FRESH unlinked -> NOT deletable
        fresh_path = _make_file(tmp_path / "storage" / "fresh.mp4")
        _insert_segment(s, start_at=base - timedelta(minutes=10),
                        end_at=base - timedelta(minutes=5),
                        path=fresh_path)
        s.commit()
    with SM() as s:
        expired = identify_expired_unlinked_segments(s)
    assert len(expired) == 1
    assert expired[0].id == old_id


def test_linked_segment_is_preserved_even_when_expired(tmp_path, monkeypatch):
    SM = _fresh_session(tmp_path, monkeypatch)
    from app.storage_guard import identify_expired_unlinked_segments
    from db.models import Case, VideoWindow
    base = _now()
    with SM() as s:
        # Case + window referencing the segment
        case = Case(camera_id="cam_01", status="OPEN")
        s.add(case); s.flush()
        seg_path = _make_file(tmp_path / "storage" / "linked.mp4")
        seg_id = _insert_segment(s,
            start_at=base - timedelta(hours=24),
            end_at=base - timedelta(hours=23),
            path=seg_path)
        s.add(VideoWindow(case_id=case.id, camera_id="cam_01",
                          requested_start_at=base, requested_end_at=base,
                          segment_ids=[seg_id], status="SUCCEEDED"))
        s.commit()
    with SM() as s:
        expired = identify_expired_unlinked_segments(s)
    assert expired == [], \
        "linked segment must NOT be eligible for cleanup"


def test_run_cleanup_dry_run_deletes_nothing(tmp_path, monkeypatch):
    SM = _fresh_session(tmp_path, monkeypatch)
    from app.storage_guard import run_cleanup
    base = _now()
    with SM() as s:
        path = _make_file(tmp_path / "storage" / "old.mp4")
        _insert_segment(s,
            start_at=base - timedelta(hours=24),
            end_at=base - timedelta(hours=23),
            path=path)
        s.commit()
    with SM() as s:
        report = run_cleanup(s, execute=False)
        s.commit()
    assert report["dry_run"] is True
    assert report["candidates"]
    assert Path(path).exists(), "dry-run must not delete files"


def test_run_cleanup_execute_deletes_files_and_rows(tmp_path, monkeypatch):
    SM = _fresh_session(tmp_path, monkeypatch)
    from app.storage_guard import run_cleanup
    from db.models import VideoSegment
    base = _now()
    with SM() as s:
        path = _make_file(tmp_path / "storage" / "old.mp4")
        _insert_segment(s,
            start_at=base - timedelta(hours=24),
            end_at=base - timedelta(hours=23),
            path=path)
        s.commit()
    with SM() as s:
        report = run_cleanup(s, execute=True)
        s.commit()
    assert report["dry_run"] is False
    assert report["deleted_rows"] == 1
    assert not Path(path).exists()
    with SM() as s:
        assert s.query(VideoSegment).count() == 0


def test_evidence_package_artifacts_preserved_by_cleanup(tmp_path,
                                                          monkeypatch):
    """Artifacts (keyframe / package files) must never be touched even
    when their associated case is old."""
    SM = _fresh_session(tmp_path, monkeypatch)
    from app.storage_guard import run_cleanup
    from db.models import Artifact, Case
    base = _now()
    case_dir = tmp_path / "storage" / "cases" / "case_id=test" / "package"
    case_dir.mkdir(parents=True)
    art_path = case_dir / "pkg_old.json"
    art_path.write_bytes(b'{"x":1}')
    with SM() as s:
        c = Case(id="test", camera_id="cam_01", status="CLOSED",
                 opened_at=base - timedelta(days=1),
                 closed_at=base - timedelta(days=1))
        s.add(c); s.flush()
        s.add(Artifact(case_id="test", artifact_type="PACKAGE",
                       uri=str(art_path), sha256="a"*64,
                       mime_type="application/json"))
        s.commit()
    with SM() as s:
        report = run_cleanup(s, execute=True)
        s.commit()
    assert art_path.exists()
    assert report["deleted_rows"] == 0


def test_low_disk_state_blocks_recorder_but_keeps_api(tmp_path, monkeypatch):
    """The recorder pauses writes when low_disk_state() is True; the
    API/reviewer UI does not consult the guard."""
    SM = _fresh_session(tmp_path, monkeypatch)
    import video.segment_recorder as sr
    monkeypatch.setattr(sr, "_disk_too_low", lambda: True)

    # The recorder helper that does the buffering loop is exercised
    # through start()/stop(). We stub RTSPReader to feed frames and
    # confirm that no segments land while the guard is high.
    import numpy as np

    class _FakeReader:
        def __init__(self, *_a, **_k): pass
        def frames(self, stop_evt):
            n = 0
            while not stop_evt.is_set() and n < 40:
                yield np.full((120, 160, 3), n, dtype=np.uint8)
                n += 1
                time.sleep(0.02)
        def close(self): pass

    monkeypatch.setattr("rtsp_reader.RTSPReader", _FakeReader)
    cfg = sr.RecorderConfig(camera_id="cam_low",
                            storage_root=tmp_path / "storage",
                            rtsp_url="rtsp://test", fps=25,
                            width=160, height=120,
                            segment_duration_sec=1)
    rec = sr.SegmentRecorder(cfg, session_factory=SM)
    rec.start()
    time.sleep(1.2)
    rec.stop()

    # No segments persisted while low_disk_state was True.
    from db.models import VideoSegment
    with SM() as s:
        assert s.query(VideoSegment).count() == 0

    # API is unaffected — instantiate it cheaply.
    from fastapi.testclient import TestClient
    from app.main import create_app
    client = TestClient(create_app())
    r = client.get("/api/v1/health")
    assert r.status_code == 200


def test_disk_status_endpoint_includes_required_fields(tmp_path, monkeypatch):
    SM = _fresh_session(tmp_path, monkeypatch)
    from app.storage_guard import disk_status
    with SM() as s:
        status = disk_status(s)
    for k in ("storage_root", "total_bytes", "used_bytes",
              "free_bytes", "free_gb", "min_free_gb",
              "low_disk_state", "retention_hours",
              "oldest_raw_segment_at", "expired_unlinked_segments"):
        assert k in status


def test_cleanup_script_dry_run(tmp_path, monkeypatch):
    """The CLI script must report candidates without deleting on
    --dry-run."""
    SM = _fresh_session(tmp_path, monkeypatch)
    base = _now()
    with SM() as s:
        path = _make_file(tmp_path / "storage" / "old.mp4")
        _insert_segment(s,
            start_at=base - timedelta(hours=24),
            end_at=base - timedelta(hours=23),
            path=path)
        s.commit()

    import subprocess
    env = os.environ.copy()
    env["DATABASE_URL"] = f"sqlite:///{tmp_path}/t.sqlite"
    env["STORAGE_ROOT"] = str(tmp_path / "storage")
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.run(
        [sys.executable, "scripts/cleanup_storage.py", "--dry-run"],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
    )
    assert proc.returncode == 0
    assert "dry_run" in proc.stdout
    assert Path(path).exists()
