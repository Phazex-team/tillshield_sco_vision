"""Camera ROI feature tests.

Backend API:
  * GET /api/v1/admin/camera-rois returns the registry.
  * GET /api/v1/admin/camera-rois/{camera_id} returns one camera.
  * PATCH validates inputs and persists into config.yaml.
  * Admin token enforcement, audit log written.
  * Validation rejects bad ROI ids, negative dims, unknown model names,
    references to missing ROI ids.

Runtime:
  * Falcon ROI crop offsets bboxes back to full-frame coordinates.
  * SAM 2 and OCR filter by ROI centre semantics.
  * Qwen/Gemma manifest extras: labeled crops + caption + descriptors,
    user_prompt composed with the canonical default request.
  * Default-no-ROI config preserves prior behavior.
  * Decision policy still gates VERIFIED on perception tracks.
"""
from __future__ import annotations

import io
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------
# Fixture: isolated DB + storage + sandboxed config.yaml restored after
# ---------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("ADMIN_EDIT_TOKEN", raising=False)
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
# GET /api/v1/admin/camera-rois
# ---------------------------------------------------------------------

def test_get_camera_rois_returns_list(client):
    r = client.get("/api/v1/admin/camera-rois")
    assert r.status_code == 200
    body = r.json()
    assert "items" in body and isinstance(body["items"], list)
    ids = [c["camera_id"] for c in body["items"]]
    assert "cam_01" in ids
    cam = next(c for c in body["items"] if c["camera_id"] == "cam_01")
    assert "zones" in cam and "model_roi_views" in cam


def test_get_camera_rois_no_secret_fields(client):
    """The ROI API must not surface rtsp/nvr credentials."""
    r = client.get("/api/v1/admin/camera-rois")
    blob = r.text.lower()
    for forbidden in ("rtsp_url", "rtsp://", "nvr_username",
                       "nvr_password", "${nvr_username}",
                       "${nvr_password}"):
        assert forbidden not in blob, \
            f"ROI payload leaks secret-bearing field: {forbidden!r}"


def test_get_camera_rois_single(client):
    r = client.get("/api/v1/admin/camera-rois/cam_01")
    assert r.status_code == 200
    assert r.json()["camera_id"] == "cam_01"


def test_get_camera_rois_unknown_camera_404(client):
    r = client.get("/api/v1/admin/camera-rois/nope")
    assert r.status_code == 404


# ---------------------------------------------------------------------
# PATCH /api/v1/admin/camera-rois — happy path + persistence
# ---------------------------------------------------------------------

def test_patch_persists_zones_and_model_views(client):
    body = {
        "zones": {
            "customer_zone": {
                "label": "Customer area", "purpose": "Customer-side context",
                "x": 100, "y": 50, "w": 200, "h": 300,
            },
            "counter_zone": {
                "label": "Counter handover", "purpose": "Item handover",
                "x": 350, "y": 50, "w": 250, "h": 300,
            },
        },
        "model_roi_views": {
            "falcon": {"enabled": True, "roi_ids": ["customer_zone",
                                                     "counter_zone"],
                       "mode": "union_crop", "margin_pct": 0.05,
                       "caption": "Detect items + receipts."},
            "qwen3_vl": {"enabled": True,
                         "roi_ids": ["customer_zone", "counter_zone"],
                         "mode": "labeled_crops", "margin_pct": 0.08,
                         "include_full_frame_overview": True,
                         "caption": "Overview + customer/counter crops."},
        },
    }
    r = client.patch("/api/v1/admin/camera-rois/cam_01", json=body)
    assert r.status_code == 200, r.text
    payload = r.json()
    assert "zones" in payload["updated_keys"]
    assert payload["zones"]["customer_zone"]["x"] == 100
    assert payload["model_roi_views"]["qwen3_vl"]["caption"] \
        .startswith("Overview")
    # Round-trip via GET.
    g = client.get("/api/v1/admin/camera-rois/cam_01").json()
    assert g["zones"]["counter_zone"]["w"] == 250
    assert g["model_roi_views"]["falcon"]["roi_ids"] == [
        "customer_zone", "counter_zone"]


def test_patch_supports_multiple_roi_ids_per_model(client):
    """The ROI API must allow listing more than one ROI id per model."""
    body = {
        "zones": {
            "customer_zone": {"label": "Customer", "x": 0, "y": 0,
                              "w": 100, "h": 100},
            "counter_zone": {"label": "Counter", "x": 100, "y": 0,
                              "w": 100, "h": 100},
            "staff_zone": {"label": "Staff", "x": 200, "y": 0,
                            "w": 100, "h": 100},
        },
        "model_roi_views": {
            "qwen3_vl": {"enabled": True, "mode": "labeled_crops",
                          "roi_ids": ["customer_zone", "counter_zone",
                                       "staff_zone"]},
        },
    }
    r = client.patch("/api/v1/admin/camera-rois/cam_01", json=body)
    assert r.status_code == 200, r.text
    assigned = r.json()["model_roi_views"]["qwen3_vl"]["roi_ids"]
    assert assigned == ["customer_zone", "counter_zone", "staff_zone"]


def test_patch_writes_audit_log(client):
    client.patch("/api/v1/admin/camera-rois/cam_01",
                 json={"zones": {"counter_zone":
                                  {"label": "Counter", "x": 0, "y": 0,
                                   "w": 100, "h": 100}}})
    from db.models import AuditLog
    import db.session as ds
    SM = ds.get_sessionmaker()
    with SM() as s:
        rows = s.query(AuditLog).filter(
            AuditLog.action == "admin.camera_rois_update").all()
    assert rows


# ---------------------------------------------------------------------
# PATCH — validation
# ---------------------------------------------------------------------

def test_patch_rejects_unknown_top_level_keys(client):
    r = client.patch("/api/v1/admin/camera-rois/cam_01",
                     json={"foo": 1})
    assert r.status_code == 400
    assert "unknown" in r.json()["detail"]["error"].lower()


def test_patch_rejects_bad_roi_id(client):
    r = client.patch("/api/v1/admin/camera-rois/cam_01", json={
        "zones": {"!!bad!!": {"x": 0, "y": 0, "w": 10, "h": 10}}})
    assert r.status_code == 400
    assert "must match" in r.json()["detail"]["error"]


def test_patch_rejects_negative_dims(client):
    r = client.patch("/api/v1/admin/camera-rois/cam_01", json={
        "zones": {"counter_zone": {"x": -1, "y": 0, "w": 10, "h": 10}}})
    assert r.status_code == 400


def test_patch_rejects_zero_w(client):
    r = client.patch("/api/v1/admin/camera-rois/cam_01", json={
        "zones": {"counter_zone": {"x": 0, "y": 0, "w": 0, "h": 10}}})
    assert r.status_code == 400


def test_patch_rejects_unknown_model(client):
    r = client.patch("/api/v1/admin/camera-rois/cam_01", json={
        "zones": {"counter_zone":
                  {"x": 0, "y": 0, "w": 10, "h": 10}},
        "model_roi_views": {"made_up_model": {"enabled": True}},
    })
    assert r.status_code == 400


def test_patch_rejects_assignment_to_missing_roi_id(client):
    r = client.patch("/api/v1/admin/camera-rois/cam_01", json={
        "zones": {"counter_zone":
                  {"x": 0, "y": 0, "w": 10, "h": 10}},
        "model_roi_views": {"falcon":
                             {"enabled": True,
                              "roi_ids": ["does_not_exist"]}},
    })
    assert r.status_code == 400
    assert "unknown roi" in r.json()["detail"]["error"]


# ---------------------------------------------------------------------
# PATCH — token enforcement
# ---------------------------------------------------------------------

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
        body = {"zones": {"counter_zone":
                          {"x": 0, "y": 0, "w": 10, "h": 10}}}
        r = c.patch("/api/v1/admin/camera-rois/cam_01", json=body)
        assert r.status_code == 401
        r2 = c.patch("/api/v1/admin/camera-rois/cam_01", json=body,
                     headers={"X-PhazeX-Admin-Token": "phzx_admin"})
        assert r2.status_code == 200
    finally:
        shutil.copy(backup, cfg_path)


# ---------------------------------------------------------------------
# camera_rois module — helpers
# ---------------------------------------------------------------------

def test_camera_rois_helpers_compute_geometry():
    from app.camera_rois import (
        apply_margin, detection_inside_rois, union_bbox,
    )
    zones = [
        {"x": 10, "y": 10, "w": 90, "h": 90},
        {"x": 50, "y": 50, "w": 200, "h": 100},
    ]
    bbox = union_bbox(zones)
    assert bbox == (10, 10, 250, 150)
    # width 240 * 0.1 = 24 horizontal margin; height 140 * 0.1 = 14
    # vertical margin; result is clipped to image bounds (1000x1000).
    assert apply_margin(bbox, 0.1, 1000, 1000) == (0, 0, 274, 164)
    # Centre inside the first zone.
    assert detection_inside_rois([20, 20, 40, 40], zones)
    # Centre well outside.
    assert not detection_inside_rois([500, 500, 520, 520], zones)


# ---------------------------------------------------------------------
# Falcon ROI crop+offset
# ---------------------------------------------------------------------

class _FakeDet:
    def __init__(self, x1, y1, x2, y2, score=0.7):
        self.bbox_px = (x1, y1, x2, y2)
        self.score = score


class _FakeDetector:
    """Pretend Falcon detector that returns one detection at a known
    offset *inside the cropped image*. The wrapper must offset it back."""

    def detect(self, img, *, query):
        return None, [_FakeDet(5, 5, 15, 15)]


def test_falcon_client_offsets_bboxes_back_to_full_frame():
    from perception.falcon_client import FalconClient
    fc = FalconClient(model_path="/tmp")
    fc._detector = _FakeDetector()
    img = Image.new("RGB", (640, 480), color=(0, 0, 0))
    ts = datetime(2026, 6, 17, 14, 0, 0)
    detections = fc.detect_on_frames(
        [(0, ts, img)], query="x",
        categories={"item": "x"},
        roi_crop=(100, 200, 300, 400),
    )
    assert detections, "expected at least one detection"
    bx = detections[0].bbox_xyxy
    # Original fake bbox (5,5,15,15) + offset (100,200) = (105,205,115,215).
    assert bx == [105.0, 205.0, 115.0, 215.0]


def test_falcon_client_no_roi_crop_keeps_full_frame_bboxes():
    from perception.falcon_client import FalconClient
    fc = FalconClient(model_path="/tmp")
    fc._detector = _FakeDetector()
    img = Image.new("RGB", (640, 480), color=(0, 0, 0))
    ts = datetime(2026, 6, 17, 14, 0, 0)
    detections = fc.detect_on_frames(
        [(0, ts, img)], query="x", categories={"item": "x"})
    assert detections[0].bbox_xyxy == [5.0, 5.0, 15.0, 15.0]


# ---------------------------------------------------------------------
# SAM 2 + OCR ROI filter (pipeline-level)
# ---------------------------------------------------------------------

def test_filter_by_roi_drops_detections_outside_union():
    from perception.pipeline import _filter_by_roi
    from perception.schemas import Detection
    ts = datetime(2026, 6, 17, 14, 0, 0)
    detections = [
        Detection(label="item", score=0.6, bbox_xyxy=[20, 20, 40, 40],
                  frame_id="f0", frame_idx=0, ts=ts),
        Detection(label="item", score=0.6, bbox_xyxy=[500, 500, 520, 520],
                  frame_id="f0", frame_idx=0, ts=ts),
    ]
    view = {"resolved_zones": [{"x": 0, "y": 0, "w": 100, "h": 100}]}
    limits: list[str] = []
    kept = _filter_by_roi(detections, view,
                           limit_tag="ocr_roi_filter",
                           limitations=limits)
    assert len(kept) == 1 and kept[0].bbox_xyxy[0] == 20
    assert any("ocr_roi_filter" in l for l in limits)


def test_filter_by_roi_pass_through_when_view_missing():
    from perception.pipeline import _filter_by_roi
    from perception.schemas import Detection
    ts = datetime(2026, 6, 17, 14, 0, 0)
    detections = [
        Detection(label="item", score=0.6, bbox_xyxy=[20, 20, 40, 40],
                  frame_id="f0", frame_idx=0, ts=ts),
    ]
    limits: list[str] = []
    kept = _filter_by_roi(detections, None,
                           limit_tag="sam2_roi_filter",
                           limitations=limits)
    assert kept is detections  # untouched
    assert limits == []


# ---------------------------------------------------------------------
# Qwen/Gemma manifest extras (labeled crops + caption)
# ---------------------------------------------------------------------

def _make_data_url(width: int = 200, height: int = 200) -> str:
    img = Image.new("RGB", (width, height), color=(20, 60, 120))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    import base64
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def test_build_vlm_roi_extras_returns_none_when_no_view(client):
    # Sandbox config (cam_01 has no model_roi_views in default config).
    from app.case_runner import _build_vlm_roi_extras
    frames = [{"frame_id": "f0", "frame_idx": 0, "ts": "2026-06-17T14:00",
               "image_url": _make_data_url(160, 120)}]
    assert _build_vlm_roi_extras("cam_01", frames) is None


def test_build_vlm_roi_extras_appends_labeled_crops(client):
    """Configure a qwen3_vl labeled_crops view on cam_01 and assert the
    case_runner helper produces extra labeled-crop frames + a composed
    user prompt that contains both ROI guidance AND the canonical
    JSON request."""
    r = client.patch("/api/v1/admin/camera-rois/cam_01", json={
        "zones": {
            "counter_zone": {"label": "Counter", "purpose": "Handover",
                              "x": 10, "y": 10, "w": 80, "h": 80},
            "customer_zone": {"label": "Customer", "purpose": "Body",
                               "x": 100, "y": 10, "w": 60, "h": 80},
        },
        "model_roi_views": {
            "qwen3_vl": {"enabled": True, "mode": "labeled_crops",
                          "roi_ids": ["counter_zone", "customer_zone"],
                          "margin_pct": 0.0,
                          "include_full_frame_overview": True,
                          "caption": "Use overview + labeled crops."},
        },
    })
    assert r.status_code == 200, r.text
    from app.case_runner import _build_vlm_roi_extras
    from reasoning.providers.qwen3_vl import DEFAULT_USER_PROMPT
    frames = [{"frame_id": "f0", "frame_idx": 0, "ts": "2026-06-17T14:00",
               "image_url": _make_data_url(200, 200)}]
    extras = _build_vlm_roi_extras("cam_01", frames)
    assert extras is not None, "expected ROI extras when qwen view active"
    assert len(extras["extra_frames"]) == 2, extras["extra_frames"]
    labels = {f["roi_id"] for f in extras["extra_frames"]}
    assert labels == {"counter_zone", "customer_zone"}
    # Each crop frame must carry the full metadata contract.
    for f in extras["extra_frames"]:
        assert f["source_frame_id"] == "f0"
        assert "roi_label" in f and "crop_xyxy" in f
        assert f["image_url"].startswith("data:image/jpeg;base64,")
    # Prompt composed from ROI guidance + canonical JSON request.
    assert "Camera ROI views" in extras["user_prompt"]
    assert DEFAULT_USER_PROMPT in extras["user_prompt"]
    # Metadata descriptors are present.
    ids = {d["id"] for d in extras["roi_descriptors"]}
    assert ids == {"counter_zone", "customer_zone"}


def test_build_vlm_roi_extras_strips_overview_when_disabled(client):
    r = client.patch("/api/v1/admin/camera-rois/cam_01", json={
        "zones": {"counter_zone": {"label": "Counter", "x": 0, "y": 0,
                                    "w": 60, "h": 60}},
        "model_roi_views": {
            "qwen3_vl": {"enabled": True, "mode": "labeled_crops",
                          "roi_ids": ["counter_zone"],
                          "include_full_frame_overview": False,
                          "caption": "crops only"},
        },
    })
    assert r.status_code == 200, r.text
    from app.case_runner import _build_vlm_roi_extras
    frames = [{"frame_id": "f0", "frame_idx": 0, "ts": "2026-06-17T14:00",
               "image_url": _make_data_url(120, 120)}]
    extras = _build_vlm_roi_extras("cam_01", frames)
    assert extras is not None
    assert extras["include_full_frame_overview"] is False


# ---------------------------------------------------------------------
# Decision policy still gates VERIFIED on perception tracks
# ---------------------------------------------------------------------

def test_patch_model_only_succeeds_against_existing_zones(client):
    """Blocker 1: a PATCH that omits ``zones`` and only updates
    ``model_roi_views`` must succeed when the assignments reference
    ROI ids already saved on the camera. Previously this round-tripped
    a private ``_current_roi_ids`` key into the validator, which then
    rejected it as an unknown top-level key."""
    # Seed the camera with two ROIs.
    seed = {"zones": {
        "customer_zone": {"label": "Customer", "x": 0, "y": 0,
                           "w": 100, "h": 100},
        "counter_zone":  {"label": "Counter",  "x": 100, "y": 0,
                           "w": 100, "h": 100},
    }}
    r0 = client.patch("/api/v1/admin/camera-rois/cam_01", json=seed)
    assert r0.status_code == 200, r0.text

    # Now a model-only PATCH (no zones key) targeting both ROIs.
    body = {"model_roi_views": {
        "qwen3_vl": {"enabled": True, "mode": "labeled_crops",
                      "roi_ids": ["customer_zone", "counter_zone"],
                      "caption": "model-only update"},
    }}
    r = client.patch("/api/v1/admin/camera-rois/cam_01", json=body)
    assert r.status_code == 200, r.text
    payload = r.json()
    assert "model_roi_views" in payload["updated_keys"]
    assert payload["model_roi_views"]["qwen3_vl"]["roi_ids"] == [
        "customer_zone", "counter_zone"]
    assert payload["model_roi_views"]["qwen3_vl"]["caption"] == \
        "model-only update"


def test_patch_model_only_rejects_assignment_to_missing_roi_id(client):
    """The model-only path must still reject assignments referencing a
    ROI id that does not exist on the camera. This proves the kwarg
    plumbing actually delivers the current ROI registry to the
    validator rather than silently allowing anything."""
    r = client.patch("/api/v1/admin/camera-rois/cam_01", json={
        "model_roi_views": {"falcon":
                             {"enabled": True,
                              "mode": "union_crop",
                              "roi_ids": ["does_not_exist"]}},
    })
    assert r.status_code == 400
    assert "unknown roi" in r.json()["detail"]["error"]


def test_patch_still_rejects_client_supplied_private_helper_key(client):
    """The blocker fix relied on moving ``_current_roi_ids`` from a
    payload key to a function kwarg. Make sure a client who tries to
    inject it into the body is still rejected as an unknown top-level
    key — the strict public surface must not regress."""
    r = client.patch("/api/v1/admin/camera-rois/cam_01", json={
        "_current_roi_ids": ["counter_zone"],
        "model_roi_views": {"qwen3_vl": {"enabled": True}},
    })
    assert r.status_code == 400
    assert "_current_roi_ids" in r.json()["detail"]["error"]


def test_patch_rejects_falcon_filter_candidate_crops_mode(client):
    """Blocker 2: Falcon runtime only implements ``union_crop``. The
    UI/API must refuse ``filter_candidate_crops`` for Falcon so the
    saved config never lies about active behavior."""
    r = client.patch("/api/v1/admin/camera-rois/cam_01", json={
        "zones": {"counter_zone":
                  {"x": 0, "y": 0, "w": 10, "h": 10}},
        "model_roi_views": {"falcon":
                             {"enabled": True,
                              "mode": "filter_candidate_crops",
                              "roi_ids": ["counter_zone"]}},
    })
    assert r.status_code == 400
    assert "mode=" in r.json()["detail"]["error"]


def test_patch_rejects_sam2_union_crop_mode(client):
    """Blocker 2: SAM 2 runtime only implements
    ``filter_candidate_crops``."""
    r = client.patch("/api/v1/admin/camera-rois/cam_01", json={
        "zones": {"counter_zone":
                  {"x": 0, "y": 0, "w": 10, "h": 10}},
        "model_roi_views": {"sam2":
                             {"enabled": True,
                              "mode": "union_crop",
                              "roi_ids": ["counter_zone"]}},
    })
    assert r.status_code == 400
    assert "mode=" in r.json()["detail"]["error"]


def test_supported_modes_tightened_to_runtime_truth():
    """Direct assertion on the table operators see in /admin/camera-rois.
    Adding a mode here is a runtime commitment."""
    from app.camera_rois import SUPPORTED_MODES
    assert SUPPORTED_MODES["falcon"] == ("union_crop",)
    assert SUPPORTED_MODES["sam2"] == ("filter_candidate_crops",)
    assert SUPPORTED_MODES["ocr"] == ("filter_candidate_crops",)
    assert SUPPORTED_MODES["qwen3_vl"] == ("labeled_crops",)
    assert SUPPORTED_MODES["gemma"] == ("labeled_crops",)


def test_default_include_full_frame_overview_is_true_for_vlms():
    """Blocker 4: when the operator submits a VLM view without setting
    ``include_full_frame_overview``, the normalized view returned to the
    runtime/UI must default to True so the VLM is never blind outside
    the ROI crops."""
    from app.camera_rois import _normalise_model_views, VLM_MODELS
    out = _normalise_model_views({
        "qwen3_vl": {"enabled": True, "mode": "labeled_crops",
                      "roi_ids": ["counter_zone"]},
        "gemma":    {"enabled": True, "mode": "labeled_crops",
                      "roi_ids": ["counter_zone"]},
        "falcon":   {"enabled": True, "mode": "union_crop",
                      "roi_ids": ["counter_zone"]},
        "sam2":     {"enabled": True, "mode": "filter_candidate_crops",
                      "roi_ids": ["counter_zone"]},
        "ocr":      {"enabled": True, "mode": "filter_candidate_crops",
                      "roi_ids": ["counter_zone"]},
    })
    # VLMs default True.
    assert out["qwen3_vl"]["include_full_frame_overview"] is True
    assert out["gemma"]["include_full_frame_overview"] is True
    # Non-VLM models default False.
    assert out["falcon"]["include_full_frame_overview"] is False
    assert out["sam2"]["include_full_frame_overview"] is False
    assert out["ocr"]["include_full_frame_overview"] is False
    # The VLM membership set must match the runtime guarantee.
    assert VLM_MODELS == {"qwen3_vl", "gemma"}


def test_default_overview_true_round_trips_through_patch(client):
    """A saved view that omits ``include_full_frame_overview`` should
    surface as True for VLMs on the next GET — covering the persisted
    contract end-to-end, not just the normaliser."""
    r = client.patch("/api/v1/admin/camera-rois/cam_01", json={
        "zones": {"counter_zone":
                  {"label": "Counter", "x": 0, "y": 0,
                   "w": 100, "h": 100}},
        "model_roi_views": {
            "qwen3_vl": {"enabled": True, "mode": "labeled_crops",
                          "roi_ids": ["counter_zone"]},
            "gemma":    {"enabled": True, "mode": "labeled_crops",
                          "roi_ids": ["counter_zone"]},
        },
    })
    assert r.status_code == 200, r.text
    g = client.get("/api/v1/admin/camera-rois/cam_01").json()
    assert g["model_roi_views"]["qwen3_vl"]["include_full_frame_overview"] \
        is True
    assert g["model_roi_views"]["gemma"]["include_full_frame_overview"] \
        is True


def test_vlm_prompt_lists_exact_image_order_with_roi_metadata(client):
    """Blocker 3: the composed user prompt must declare each attached
    image's position with frame_id, AND for ROI crops also roi_id,
    roi_label, crop_xyxy, source_frame_id — in the SAME order the
    provider will attach them."""
    r = client.patch("/api/v1/admin/camera-rois/cam_01", json={
        "zones": {
            "counter_zone":  {"label": "Counter", "purpose": "Handover",
                               "x": 10, "y": 10, "w": 80, "h": 80},
            "customer_zone": {"label": "Customer", "purpose": "Body",
                               "x": 100, "y": 10, "w": 60, "h": 80},
        },
        "model_roi_views": {
            "qwen3_vl": {"enabled": True, "mode": "labeled_crops",
                          "roi_ids": ["counter_zone", "customer_zone"],
                          "include_full_frame_overview": True,
                          "caption": "Overview + crops."},
        },
    })
    assert r.status_code == 200, r.text

    from app.case_runner import _build_vlm_roi_extras
    frames = [
        {"frame_id": "f0", "frame_idx": 0, "ts": "2026-06-17T14:00",
         "image_url": _make_data_url(200, 200)},
        {"frame_id": "f1", "frame_idx": 1, "ts": "2026-06-17T14:00",
         "image_url": _make_data_url(200, 200)},
    ]
    extras = _build_vlm_roi_extras("cam_01", frames)
    assert extras is not None
    final = extras["frames"]
    user_prompt = extras["user_prompt"]

    # Provider attach order == the order this list enumerates.
    # 2 overview + crops for each source frame × 2 ROIs = 2 + 4 = 6.
    assert len(final) == 6
    assert final[0]["frame_id"] == "f0" and "roi_id" not in final[0]
    assert final[1]["frame_id"] == "f1" and "roi_id" not in final[1]
    # Crops come after overviews, grouped by source frame.
    roi_positions = [(i + 1, f) for i, f in enumerate(final) if "roi_id" in f]
    assert [f["roi_id"] for _, f in roi_positions] == [
        "counter_zone", "customer_zone", "counter_zone", "customer_zone"]

    # The prompt must declare the same order with the per-image metadata.
    assert "Attached images" in user_prompt
    assert "[1] overview" in user_prompt
    assert "frame_id=f0" in user_prompt
    assert "[2] overview" in user_prompt
    assert "frame_id=f1" in user_prompt
    for pos, f in roi_positions:
        line_marker = f"[{pos}] roi_crop"
        assert line_marker in user_prompt, \
            f"missing prompt line for position {pos}: {line_marker}"
        # Each ROI crop line must carry the full per-image identity.
        assert f"roi_id={f['roi_id']}" in user_prompt
        assert f"label={f['roi_label']!r}" in user_prompt
        assert f"crop_xyxy={list(f['crop_xyxy'])}" in user_prompt
        assert f"source_frame_id={f['source_frame_id']}" in user_prompt

    # The canonical JSON request is still present verbatim — the ROI
    # preamble does NOT replace the review-safe schema request.
    from reasoning.providers.qwen3_vl import DEFAULT_USER_PROMPT
    assert DEFAULT_USER_PROMPT in user_prompt


def test_decision_policy_still_gates_verified_after_roi_feature():
    """The ROI feature must not let a VLM caption decide outcomes.
    Repeat the K-series invariant: without a perception track that has
    physical_item_candidate=True, VLM-says-physical alone never
    upgrades a case to VERIFIED."""
    from reasoning.decision_policy import decide, summary_from_vlm
    vlm_parsed = {
        "handover_occurred": True,
        "physical_item_presented": True,
        "receipt_visible": True,
        "narrative": "handover visible",
        "confidence": "high",
        "obstructed": False,
        "camera_view_clear": True,
    }
    summary = summary_from_vlm(vlm_parsed, footage_valid=True,
                                obstructed=False, camera_gap=False,
                                perception_result={"tracks": []})
    assert decide(summary).outcome != "VERIFIED"
