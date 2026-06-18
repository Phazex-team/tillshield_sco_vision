"""Pipeline-tab camera preview — backend endpoint + UI surface.

This pins:
  * GET /api/v1/video/cameras/{camera_id}/preview-frame
      - 404 for unknown camera
      - 404 for known camera with no segment row yet
      - 410 when the segment file is gone from disk
      - 200 returns image_url + width + height + source + segment_id
      - latest segment is selected when multiple exist
      - Cache-Control: no-store stamped on BOTH success and error paths
      - response never leaks rtsp_url or the on-disk segment path
  * Reviewer UI Pipeline tab
      - Preview action rendered per camera row
      - fetches the documented endpoint, not an arbitrary URL
      - explicit "Preview only — no model inference." label
      - no RTSP / WebSocket / MJPEG / HLS / start/stop / restart controls
      - auto-refresh defaults to OFF and the polling floor is >= 5000ms
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("ADMIN_EDIT_TOKEN", raising=False)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'t.sqlite'}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
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
    return TestClient(create_app()), tmp_path


def _seed_segment(tmp_path, *, camera_id="cam_01",
                   width=320, height=240, duration_sec=2,
                   start_at=None) -> str:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg required to synthesise an MP4 segment")
    storage_root = tmp_path / "storage"
    seg_dir = storage_root / "segments" / camera_id
    seg_dir.mkdir(parents=True, exist_ok=True)
    # Distinct filename per call so successive segments live alongside.
    n = len(list(seg_dir.glob("seg_*.mp4")))
    seg_path = seg_dir / f"seg_{n:04d}.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi",
         "-i", f"color=c=black:s={width}x{height}:d={duration_sec}",
         "-r", "25", "-pix_fmt", "yuv420p", str(seg_path)],
        check=True, capture_output=True,
    )
    import hashlib
    h = hashlib.sha256()
    with open(seg_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    from db.models import VideoSegment
    import db.session as ds
    SM = ds.get_sessionmaker()
    if start_at is None:
        start_at = datetime.now(timezone.utc).replace(tzinfo=None) \
            - timedelta(hours=1)
    with SM() as s:
        seg = VideoSegment(
            camera_id=camera_id, start_at=start_at,
            end_at=start_at + timedelta(seconds=duration_sec),
            path=str(seg_path), sha256=h.hexdigest(),
            fps=25, width=width, height=height,
            frame_count=25 * duration_sec, duration_sec=float(duration_sec),
        )
        s.add(seg); s.commit()
        return seg.id


# ---------------------------------------------------------------------
# Backend contracts
# ---------------------------------------------------------------------

def test_preview_404_for_unknown_camera_has_no_store(client):
    c, _ = client
    r = c.get("/api/v1/video/cameras/nope/preview-frame")
    assert r.status_code == 404
    assert r.headers.get("cache-control") == "no-store"
    detail = r.json()["detail"]
    assert "rtsp://" not in detail
    assert "/" not in detail or "camera" in detail  # no raw filesystem path


def test_preview_404_when_no_segment_row(client):
    c, _ = client
    r = c.get("/api/v1/video/cameras/cam_01/preview-frame")
    assert r.status_code == 404
    assert r.headers.get("cache-control") == "no-store"
    body = r.json()
    assert "no local segment available" in body["detail"]


def test_preview_returns_latest_segment_metadata(client):
    c, tmp = client
    seg_id = _seed_segment(tmp, camera_id="cam_01",
                            width=320, height=240, duration_sec=2)
    r = c.get("/api/v1/video/cameras/cam_01/preview-frame")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["camera_id"] == "cam_01"
    assert body["source"] == "latest_segment"
    assert body["width"] == 320 and body["height"] == 240
    assert body["segment_id"] == seg_id
    assert body["image_url"].startswith("data:image/jpeg;base64,")
    assert body["captured_at"]


def test_preview_success_has_cache_control_no_store(client):
    c, tmp = client
    _seed_segment(tmp)
    r = c.get("/api/v1/video/cameras/cam_01/preview-frame")
    assert r.status_code == 200
    assert r.headers.get("cache-control") == "no-store"


def test_preview_picks_latest_segment_when_multiple_exist(client):
    """Two segments — the newer ``start_at`` must win even if it was
    inserted first."""
    c, tmp = client
    base = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=4)
    older = _seed_segment(tmp, camera_id="cam_01", width=160, height=120,
                            start_at=base)
    newer = _seed_segment(tmp, camera_id="cam_01", width=320, height=240,
                            start_at=base + timedelta(hours=2))
    r = c.get("/api/v1/video/cameras/cam_01/preview-frame")
    assert r.status_code == 200
    body = r.json()
    assert body["segment_id"] == newer
    assert body["segment_id"] != older
    assert (body["width"], body["height"]) == (320, 240)


def test_preview_410_when_segment_file_missing(client):
    c, tmp = client
    import os
    _seed_segment(tmp, camera_id="cam_01")
    # Delete the file on disk; leave the DB row.
    from db.models import VideoSegment
    import db.session as ds
    SM = ds.get_sessionmaker()
    with SM() as s:
        seg = s.query(VideoSegment).filter_by(camera_id="cam_01").first()
        os.remove(seg.path)
    r = c.get("/api/v1/video/cameras/cam_01/preview-frame")
    assert r.status_code == 410
    assert r.headers.get("cache-control") == "no-store"
    assert "no longer on disk" in r.json()["detail"]


def test_preview_response_does_not_leak_rtsp_url_or_path(client):
    c, tmp = client
    _seed_segment(tmp, camera_id="cam_01")
    r = c.get("/api/v1/video/cameras/cam_01/preview-frame")
    assert r.status_code == 200
    blob = r.text
    assert "rtsp://" not in blob
    assert "rtsp_url" not in blob
    body = r.json()
    assert "path" not in body
    assert "rtsp_url" not in body
    # The segments live under STORAGE_ROOT/segments/... — make sure
    # no such substring leaks (success body carries data URL only).
    storage_root = str(tmp / "storage")
    assert storage_root not in blob


def test_preview_endpoint_takes_only_camera_id_no_arbitrary_input(client):
    """The route accepts a single path parameter. Extra query params are
    ignored — there is no place to inject an arbitrary URL or path."""
    c, _ = client
    # Adding query params does NOT change behavior: still 404 for unknown.
    r = c.get("/api/v1/video/cameras/nope/preview-frame",
               params={"path": "/etc/passwd",
                        "url": "http://example.com/img.jpg"})
    assert r.status_code == 404


# ---------------------------------------------------------------------
# UI / static surface
# ---------------------------------------------------------------------

def _html() -> str:
    return (ROOT / "static" / "review.html").read_text()


def test_pipeline_tab_has_preview_button_per_camera_row():
    src = _html()
    # The renderer produces a Preview button on each row keyed by the
    # camera id. Look for the data-attribute + the button label.
    assert 'data-camera-preview="${escapeHtml(camId)}"' in src
    assert ">Preview</button>" in src


def test_preview_fetches_documented_endpoint_only():
    src = _html()
    # Exact template literal that hits the routed endpoint.
    assert ("/video/cameras/${encodeURIComponent(camId)}/preview-frame"
            in src)
    # No alternative URL inputs anywhere — confirms the operator can't
    # talk the UI into a different endpoint.
    assert "preview-url" not in src
    assert "preview_url" not in src


def test_preview_panel_has_disclaimer_about_no_inference():
    src = _html()
    assert "Preview only" in src
    assert "no model inference" in src


def test_preview_panel_has_no_rtsp_input():
    src = _html()
    panel_start = src.index('id="camera-preview-panel"')
    panel_end = src.index("</section>", panel_start)
    panel = src[panel_start:panel_end].lower()
    assert "rtsp_url" not in panel
    assert "rtsp://" not in panel
    # And no input that lets the operator paste a URL.
    assert re.search(
        r'<input[^>]*placeholder="[^"]*url', panel,
        flags=re.IGNORECASE) is None


def test_preview_does_not_add_streaming_or_process_controls():
    src = _html().lower()
    # No live-stream / sub-stream tech anywhere.
    for forbidden in (
        "websocket", "new websocket", "mjpeg",
        "application/vnd.apple.mpegurl", ".m3u8",
        "mediasource",
    ):
        assert forbidden not in src, \
            f"unexpected live-streaming primitive in UI: {forbidden!r}"
    # No process-control buttons for the recorder/cameras here either.
    for forbidden in (
        "restart recorder", "reload recorder",
        "start camera", "stop camera",
        "restart camera", "reload camera",
    ):
        assert forbidden not in src, \
            f"UI must not pretend to control external processes: {forbidden!r}"


def test_preview_autorefresh_default_off_and_interval_floor_5s():
    src = _html()
    # The checkbox is in the UI and starts unchecked (no ``checked`` attr).
    pattern = re.compile(
        r'<input[^>]*id="camera-preview-autorefresh"[^>]*>',
        flags=re.IGNORECASE)
    m = pattern.search(src)
    assert m is not None, "auto-refresh checkbox missing"
    assert "checked" not in m.group(0).lower(), (
        "auto-refresh must default OFF")
    # Polling floor at 5000ms — both as a named constant and as the
    # ``Math.max`` clamp inside the timer-restart helper.
    assert "CAMERA_PREVIEW_INTERVAL_MS = 5000" in src
    assert re.search(
        r"Math\.max\(CAMERA_PREVIEW_INTERVAL_MS,\s*5000\)", src), (
        "polling floor must clamp to >= 5000ms")


def test_preview_panel_closes_and_clears_timer_on_close():
    """The Close button (and switching cameras) must stop the
    auto-refresh interval. We pin this by inspecting the source."""
    src = _html()
    assert "function closeCameraPreview" in src
    close_start = src.index("function closeCameraPreview")
    close_end = src.index("}", close_start)
    body = src[close_start:close_end]
    assert "_stopCameraPreviewTimer()" in body
    # ``camera-preview-close`` button is bound to the close function.
    assert 'getElementById("camera-preview-close").onclick = closeCameraPreview' \
        in src
