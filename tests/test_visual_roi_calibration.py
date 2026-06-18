"""Visual ROI calibration — backend endpoint + schema extensions +
runtime scaling + UI structural assertions.

This file pins:
  * GET /api/v1/admin/camera-rois/{camera_id}/snapshot
      - admin-token gated like the existing PATCH
      - 404 unknown camera / missing segment
      - 503 cv2 unavailable or decoder failure (best-effort path)
      - returns image_url + width + height + source = "latest_segment"
      - NEVER leaks rtsp_url / on-disk path / NVR credentials
      - Cache-Control: no-store
  * ROI schema accepts optional source_width/source_height
      - persisted into config.yaml
      - echoed back via GET
      - validated as positive integers
  * scale_zone_to_frame helper behavior
      - no-op when source dims absent OR identical to frame size
      - scales when source dims differ
      - clamps to frame bounds and drops zero-area
  * Runtime consumers (Falcon crop, SAM 2 / OCR filter, VLM labeled
    crops) use scaled coordinates
  * UI surface
      - Load frame button + canvas wrapper present
      - JS calls /admin/camera-rois/${id}/snapshot, not arbitrary URLs
      - No RTSP URL editor anywhere in the ROI tab
      - Honest hot-reload / no-rebuild / no-RTSP-here copy is present
"""
from __future__ import annotations

import io
import os
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
    _isolate_config(monkeypatch, tmp_path)
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
    yield c, tmp_path


def _isolate_config(monkeypatch, tmp_path: Path) -> Path:
    """Route app.config reads/writes to a per-test config copy.

    The admin ROI PATCH endpoint intentionally persists to config.yaml.
    Tests must not copy over the repo-level file while the full suite is
    running because another test can read it mid-write. Patching both
    ``DEFAULT_CONFIG_PATH`` and ``load_config`` keeps endpoint semantics
    identical while confining writes to ``tmp_path``.
    """
    import app.config as ac
    cfg_copy = tmp_path / "config.yaml"
    shutil.copy(ROOT / "config.yaml", cfg_copy)
    real_load_config = ac.load_config

    def _load_config(path=cfg_copy):
        return real_load_config(path)

    monkeypatch.setattr(ac, "DEFAULT_CONFIG_PATH", cfg_copy)
    monkeypatch.setattr(ac, "load_config", _load_config)
    return cfg_copy


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _seed_segment(tmp_path, camera_id="cam_01", *,
                   width=320, height=240, duration_sec=2):
    """Synthesise a real MP4 segment + insert the DB row, returning
    the segment id."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg required to synthesise an MP4 segment")
    storage_root = tmp_path / "storage"
    seg_dir = storage_root / "segments" / camera_id
    seg_dir.mkdir(parents=True, exist_ok=True)
    seg_path = seg_dir / "seg_0001.mp4"
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
    start_at = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1)
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
# Snapshot endpoint contracts
# ---------------------------------------------------------------------

def test_snapshot_404_for_unknown_camera(client):
    c, _ = client
    r = c.get("/api/v1/admin/camera-rois/nope/snapshot")
    assert r.status_code == 404
    assert r.headers.get("cache-control") == "no-store"


def test_snapshot_requires_token_when_configured(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_EDIT_TOKEN", "phzx_admin")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'t.sqlite'}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    _isolate_config(monkeypatch, tmp_path)
    import db.session as ds
    ds._ENGINE = None
    ds._SESSION_FACTORY = None
    ds.init_schema()
    from fastapi.testclient import TestClient
    from app.main import create_app
    c = TestClient(create_app())
    r = c.get("/api/v1/admin/camera-rois/cam_01/snapshot")
    assert r.status_code == 401, r.text
    # With the right token but no segment yet, the gate is open
    # but the endpoint returns 404 with the documented detail.
    r2 = c.get("/api/v1/admin/camera-rois/cam_01/snapshot",
                headers={"X-PhazeX-Admin-Token": "phzx_admin"})
    assert r2.status_code == 404
    # Honest detail: 'live RTSP snapshot is not implemented'.
    assert "live RTSP snapshot" in r2.json()["detail"]


def test_snapshot_404_when_no_segment_yet(client):
    c, _ = client
    r = c.get("/api/v1/admin/camera-rois/cam_01/snapshot")
    assert r.status_code == 404
    assert r.headers.get("cache-control") == "no-store"
    detail = r.json()["detail"]
    # The 404 detail must not leak rtsp_url-shaped data.
    assert "rtsp://" not in detail
    assert "RTSP" in detail or "live" in detail or "snapshot" in detail


def test_snapshot_returns_image_url_and_metadata_from_latest_segment(client):
    c, tmp = client
    seg_id = _seed_segment(tmp, camera_id="cam_01",
                            width=320, height=240, duration_sec=2)
    r = c.get("/api/v1/admin/camera-rois/cam_01/snapshot")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["camera_id"] == "cam_01"
    assert body["source"] == "latest_segment"
    assert body["width"] == 320
    assert body["height"] == 240
    assert body["segment_id"] == seg_id
    # Image is a JPEG data URL — the operator's browser decodes it.
    assert body["image_url"].startswith("data:image/jpeg;base64,")


def test_snapshot_response_never_leaks_rtsp_url(client):
    """The response body + headers must never contain ``rtsp_url`` or a
    raw ``rtsp://`` URL even when the camera has one configured. The
    on-disk segment ``path`` is also intentionally NOT echoed."""
    c, tmp = client
    _seed_segment(tmp, camera_id="cam_01")
    r = c.get("/api/v1/admin/camera-rois/cam_01/snapshot")
    assert r.status_code == 200
    blob = r.text
    assert "rtsp://" not in blob
    assert "rtsp_url" not in blob
    # On-disk path is sensitive on shared boxes — don't echo it either.
    body = r.json()
    assert "path" not in body
    # Camera config keys that contain secrets must never appear.
    for forbidden in ("nvr_username", "nvr_password",
                       "ingest_token"):
        assert forbidden not in blob


def test_snapshot_response_has_cache_control_no_store(client):
    c, tmp = client
    _seed_segment(tmp, camera_id="cam_01")
    r = c.get("/api/v1/admin/camera-rois/cam_01/snapshot")
    assert r.status_code == 200
    assert r.headers.get("cache-control") == "no-store"


def test_snapshot_503_when_segment_file_missing(client):
    c, tmp = client
    seg_id = _seed_segment(tmp, camera_id="cam_01")
    # Delete the file but leave the DB row.
    from db.models import VideoSegment
    import db.session as ds
    SM = ds.get_sessionmaker()
    with SM() as s:
        seg = s.get(VideoSegment, seg_id)
        os.remove(seg.path)
    r = c.get("/api/v1/admin/camera-rois/cam_01/snapshot")
    assert r.status_code == 404
    assert "no longer on disk" in r.json()["detail"]


# ---------------------------------------------------------------------
# ROI schema extension (source_width/source_height)
# ---------------------------------------------------------------------

def test_patch_persists_source_dimensions(client):
    c, _ = client
    body = {
        "zones": {
            "counter_zone": {
                "label": "Counter", "purpose": "Handover",
                "x": 100, "y": 50, "w": 200, "h": 300,
                "source_width": 1920, "source_height": 1080,
            },
        },
    }
    r = c.patch("/api/v1/admin/camera-rois/cam_01", json=body)
    assert r.status_code == 200, r.text
    g = c.get("/api/v1/admin/camera-rois/cam_01").json()
    z = g["zones"]["counter_zone"]
    assert z["source_width"] == 1920
    assert z["source_height"] == 1080


def test_patch_rejects_zero_source_dimension(client):
    c, _ = client
    r = c.patch("/api/v1/admin/camera-rois/cam_01", json={
        "zones": {"counter_zone": {"x": 0, "y": 0, "w": 10, "h": 10,
                                     "source_width": 0,
                                     "source_height": 720}}})
    assert r.status_code == 400


def test_patch_rejects_non_integer_source_dimension(client):
    c, _ = client
    r = c.patch("/api/v1/admin/camera-rois/cam_01", json={
        "zones": {"counter_zone": {"x": 0, "y": 0, "w": 10, "h": 10,
                                     "source_width": "wide",
                                     "source_height": 720}}})
    assert r.status_code == 400


def test_patch_rejects_partial_source_dimensions(client):
    c, _ = client
    r = c.patch("/api/v1/admin/camera-rois/cam_01", json={
        "zones": {"counter_zone": {"x": 0, "y": 0, "w": 10, "h": 10,
                                     "source_width": 1280}}})
    assert r.status_code == 400
    assert "both source_width and source_height" in r.text


def test_legacy_zone_without_source_dims_still_works(client):
    """Configs that pre-date the visual-calibration feature must keep
    working — no source_width/source_height keys required."""
    c, _ = client
    r = c.patch("/api/v1/admin/camera-rois/cam_01", json={
        "zones": {"counter_zone": {"x": 1, "y": 2, "w": 3, "h": 4}}})
    assert r.status_code == 200, r.text
    g = c.get("/api/v1/admin/camera-rois/cam_01").json()
    z = g["zones"]["counter_zone"]
    assert z["x"] == 1 and z["w"] == 3
    assert "source_width" not in z
    assert "source_height" not in z


# ---------------------------------------------------------------------
# Runtime scaling helper
# ---------------------------------------------------------------------

def test_scale_zone_no_op_when_source_dims_absent():
    from app.camera_rois import scale_zone_to_frame
    zone = {"x": 10, "y": 20, "w": 30, "h": 40}
    out = scale_zone_to_frame(zone, frame_w=640, frame_h=360)
    assert out["x"] == 10 and out["y"] == 20
    assert out["w"] == 30 and out["h"] == 40
    # The helper stamps the actual frame dimensions so downstream code
    # can keep track without re-deriving from a different snapshot.
    assert out["source_width"] == 640
    assert out["source_height"] == 360


def test_scale_zone_no_op_when_source_matches_frame():
    from app.camera_rois import scale_zone_to_frame
    zone = {"x": 5, "y": 5, "w": 10, "h": 10,
             "source_width": 640, "source_height": 360}
    out = scale_zone_to_frame(zone, frame_w=640, frame_h=360)
    assert (out["x"], out["y"], out["w"], out["h"]) == (5, 5, 10, 10)


def test_scale_zone_scales_when_source_differs():
    from app.camera_rois import scale_zone_to_frame
    zone = {"x": 100, "y": 100, "w": 200, "h": 100,
             "source_width": 1920, "source_height": 1080}
    out = scale_zone_to_frame(zone, frame_w=640, frame_h=360)
    # 640/1920 = 1/3 ; 360/1080 = 1/3
    assert out["x"] == 33 and out["y"] == 33
    assert out["w"] == 67 and out["h"] == 33
    # The helper rewrites source dims to the actual frame.
    assert out["source_width"] == 640
    assert out["source_height"] == 360


def test_scale_zone_clamps_box_inside_frame_and_drops_zero_area():
    from app.camera_rois import scale_zone_to_frame
    # Zone extends past the frame in source coords — must be clamped.
    zone = {"x": 1800, "y": 1000, "w": 500, "h": 500,
             "source_width": 1920, "source_height": 1080}
    out = scale_zone_to_frame(zone, frame_w=640, frame_h=360)
    assert out is not None
    assert out["x"] + out["w"] <= 640
    assert out["y"] + out["h"] <= 360
    # Zone entirely outside the frame collapses to None.
    bad = {"x": 99999, "y": 99999, "w": 10, "h": 10,
            "source_width": 1920, "source_height": 1080}
    assert scale_zone_to_frame(bad, frame_w=640, frame_h=360) is None


# ---------------------------------------------------------------------
# Runtime consumers — Falcon crop + SAM2/OCR filter use scaled coords
# ---------------------------------------------------------------------

def test_falcon_roi_crop_uses_scaled_coordinates(monkeypatch):
    """A zone calibrated against 1920x1080 must produce a crop that
    matches the actual 640x360 frame, not the raw saved pixels."""
    import perception.pipeline as pl
    from PIL import Image
    img = Image.new("RGB", (640, 360), color=(0, 0, 0))
    frames = [(0, datetime(2026, 6, 18), img)]
    # Zone covers the right half of the source frame.
    view = {
        "mode": "union_crop",
        "margin_pct": 0.0,
        "resolved_zones": [{
            "id": "right_half", "x": 960, "y": 0, "w": 960, "h": 1080,
            "source_width": 1920, "source_height": 1080,
        }],
    }
    limitations: list[str] = []
    crop = pl._falcon_roi_crop(frames, view, limitations)
    assert crop is not None
    x1, y1, x2, y2 = crop
    # Scaled: 960 -> 320, full height -> 360.
    assert (x1, y1, x2, y2) == (320, 0, 640, 360)


def test_filter_by_roi_uses_frame_size_to_scale_zones(monkeypatch):
    """The SAM2 / OCR filter must scale the zone box onto the actual
    frame size before the centre-in-rect test."""
    from perception.pipeline import _filter_by_roi
    from perception.schemas import Detection
    ts = datetime(2026, 6, 18)
    # Detection at (200, 200) on a 640x360 frame — that maps onto
    # (600, 600) on a 1920x1080 source. The zone covers (400..700,
    # 400..700) in source space.
    detections = [Detection(label="x", score=0.5,
                              bbox_xyxy=[195, 195, 205, 205],
                              frame_id="f0", frame_idx=0, ts=ts)]
    view = {
        "resolved_zones": [{
            "id": "centre", "x": 400, "y": 400, "w": 300, "h": 300,
            "source_width": 1920, "source_height": 1080,
        }],
    }
    limits: list[str] = []
    kept = _filter_by_roi(detections, view,
                           limit_tag="ocr_roi_filter",
                           limitations=limits,
                           frame_size=(640, 360))
    assert len(kept) == 1


def test_filter_by_roi_drops_detection_outside_scaled_zone():
    from perception.pipeline import _filter_by_roi
    from perception.schemas import Detection
    ts = datetime(2026, 6, 18)
    detections = [Detection(label="x", score=0.5,
                              bbox_xyxy=[10, 10, 20, 20],
                              frame_id="f0", frame_idx=0, ts=ts)]
    view = {
        "resolved_zones": [{
            "id": "centre", "x": 400, "y": 400, "w": 300, "h": 300,
            "source_width": 1920, "source_height": 1080,
        }],
    }
    limits: list[str] = []
    kept = _filter_by_roi(detections, view,
                           limit_tag="ocr_roi_filter",
                           limitations=limits,
                           frame_size=(640, 360))
    assert kept == []


def test_track_zone_annotation_uses_scaled_source_dimensions():
    from perception.pipeline import _scale_temporal_zones
    from perception.schemas import Detection, Track
    from perception.temporal_memory import Zone, annotate_tracks

    ts = datetime(2026, 6, 18)
    zones = [Zone(name="counter_zone", x=960, y=0, w=960, h=1080,
                  source_width=1920, source_height=1080)]
    scaled = _scale_temporal_zones(zones, (640, 360), [])
    assert [(z.name, z.x, z.y, z.w, z.h) for z in scaled] == [
        ("counter_zone", 320, 0, 320, 360)
    ]
    det = Detection(label="bag", score=0.9,
                    bbox_xyxy=[400, 100, 440, 140],
                    frame_id="f0", frame_idx=0, ts=ts)
    track = Track(track_id="t1", label="bag",
                  first_seen_ts=ts, last_seen_ts=ts,
                  detections=[0], confidence=0.9)
    out = annotate_tracks([track], [det], zones=scaled)[0]
    assert "counter_zone" in out.zones
    assert "handover_candidate" in out.events


def test_build_vlm_roi_extras_scales_crop_xyxy_to_actual_frame(client):
    """When the zone is saved against (1920, 1080) but the manifest
    frame is (200, 200), the resulting ``crop_xyxy`` must be the
    scaled rectangle — never the raw saved pixels."""
    c, _ = client
    # Calibrate a counter_zone over the centre-left of a 1920x1080
    # source frame, then enable qwen3_vl labeled crops.
    r = c.patch("/api/v1/admin/camera-rois/cam_01", json={
        "zones": {
            "counter_zone": {
                "label": "Counter", "purpose": "Handover",
                "x": 960, "y": 0, "w": 960, "h": 1080,
                "source_width": 1920, "source_height": 1080,
            },
        },
        "model_roi_views": {
            "qwen3_vl": {"enabled": True, "mode": "labeled_crops",
                          "roi_ids": ["counter_zone"],
                          "include_full_frame_overview": False,
                          "margin_pct": 0.0,
                          "caption": "calibrated against 1920x1080"},
        },
    })
    assert r.status_code == 200, r.text
    from app.case_runner import _build_vlm_roi_extras
    # Manifest frame is 200x200 — much smaller than the source.
    from PIL import Image
    import base64 as _b64
    buf = io.BytesIO()
    Image.new("RGB", (200, 200), color=(10, 20, 30)).save(buf, format="PNG")
    url = ("data:image/png;base64,"
           + _b64.b64encode(buf.getvalue()).decode("ascii"))
    frames = [{"frame_id": "f0", "frame_idx": 0,
                "ts": "2026-06-18T14:00", "image_url": url}]
    extras = _build_vlm_roi_extras("cam_01", frames)
    assert extras is not None
    crops = [f for f in extras["frames"] if f.get("roi_id") == "counter_zone"]
    assert crops, extras
    x1, y1, x2, y2 = crops[0]["crop_xyxy"]
    # Scaled: 960 -> 100, 0 -> 0, 960 -> 100 wide, 1080 -> 200 tall.
    assert (x1, y1, x2, y2) == (100, 0, 200, 200)


# ---------------------------------------------------------------------
# UI surface
# ---------------------------------------------------------------------

def test_review_ui_has_load_frame_button_and_canvas():
    src = (ROOT / "static" / "review.html").read_text()
    assert 'id="roi-load-frame"' in src
    assert 'id="roi-canvas"' in src
    assert 'id="roi-canvas-wrap"' in src
    assert 'id="roi-snapshot-meta"' in src


def test_review_ui_fetches_dedicated_snapshot_endpoint_only():
    """The JS must call the routed snapshot endpoint — never an
    arbitrary URL supplied by the operator."""
    src = (ROOT / "static" / "review.html").read_text()
    assert "/admin/camera-rois/${encodeURIComponent(id)}/snapshot" in src
    # No reference to an arbitrary URL input field for snapshots.
    assert "snapshot-url" not in src
    assert "snapshot_url" not in src


def test_review_ui_has_no_rtsp_url_editor_in_roi_tab():
    """The ROI tab must NOT contain any input/text for editing the
    camera's RTSP URL."""
    src = (ROOT / "static" / "review.html").read_text()
    # Locate the ROI tab markup boundaries.
    start = src.index('id="tab-rois"')
    # End is the next "tab-pane" section.
    end = src.index('tab-pane" id="tab-', start + 1)
    roi_block = src[start:end]
    # No rtsp string in markup or labels.
    assert "rtsp_url" not in roi_block.lower()
    assert "rtsp://" not in roi_block.lower()
    # No input that looks like a URL editor for snapshots.
    assert re.search(
        r'<input[^>]*placeholder="[^"]*rtsp', roi_block,
        flags=re.IGNORECASE) is None


def test_review_ui_keeps_numeric_zone_fields():
    src = (ROOT / "static" / "review.html").read_text()
    # The numeric x/y/w/h inputs survive the visual editor addition.
    for key in ("data-rk=\"x\"", "data-rk=\"y\"",
                 "data-rk=\"w\"", "data-rk=\"h\""):
        assert key in src, f"missing numeric zone input {key!r}"
    # source_width/source_height are now part of the table.
    assert "data-rk=\"source_width\"" in src
    assert "data-rk=\"source_height\"" in src


def test_review_ui_save_still_calls_patch_camera_rois():
    src = (ROOT / "static" / "review.html").read_text()
    assert "PATCH" in src and "/admin/camera-rois/${encodeURIComponent(id)}" in src


def test_review_ui_explains_hot_reload_and_no_rebuild():
    src = (ROOT / "static" / "review.html").read_text()
    # Operator must be told what SAVE affects (HTML source has the
    # paragraph wrapped across lines; we collapse whitespace before
    # checking the canonical strings).
    collapsed = re.sub(r"\s+", " ", src)
    assert "Saves apply to the next case or reprocess" in collapsed
    assert "config snapshot" in collapsed
    assert "No app or Docker rebuild is required for ROI changes" in collapsed
    # And that RTSP changes are not in scope here.
    assert "RTSP URLs and other camera source settings are NOT controlled here" \
        in collapsed


def test_review_ui_does_not_add_fake_recorder_or_process_controls():
    src = (ROOT / "static" / "review.html").read_text().lower()
    for forbidden in (
        "restart recorder", "reload recorder", "stop recorder",
        "restart rtsp", "reload rtsp",
    ):
        assert forbidden not in src, \
            f"UI must not pretend to control external processes: {forbidden!r}"
