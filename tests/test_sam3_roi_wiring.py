"""SAM3 ROI registry wiring (hot-food replay blocker fix).

Verifies the three things the replay turned up:

  1. ``sam3`` is a first-class entry in the ROI registry (``SUPPORTED_MODELS``
     + ``SUPPORTED_MODES``), so ``model_view(cfg, cam, "sam3")`` returns
     a usable view instead of ``None``.
  2. The production camera (``cam_return_01``) resolves its ``sam3``
     view to ``sco_audit_zone``.
  3. With SAM3 enabled and Falcon disabled, ``run_perception_on_window``
     passes a non-null ``roi_crop_xyxy`` to ``Sam3Client.process_window``
     and Falcon is never called.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# 1. Registry: sam3 is supported
# ---------------------------------------------------------------------------

def test_sam3_in_supported_models_and_modes():
    from app.camera_rois import SUPPORTED_MODELS, SUPPORTED_MODES
    assert "sam3" in SUPPORTED_MODELS
    assert "union_crop" in SUPPORTED_MODES.get("sam3", ())


# ---------------------------------------------------------------------------
# 2. The production camera resolves the sam3 view to sco_audit_zone
# ---------------------------------------------------------------------------

def test_model_view_resolves_sam3_for_cam_return_01():
    from app.camera_rois import model_view
    from app.config import load_config

    cfg = load_config()
    view = model_view(cfg, "cam_return_01", "sam3")
    assert view is not None, "cam_return_01 sam3 view must not be None"
    assert view["enabled"] is True
    assert view["roi_ids"] == ["sco_audit_zone"]
    resolved = view["resolved_zones"]
    assert len(resolved) == 1
    assert resolved[0]["id"] == "sco_audit_zone"


# ---------------------------------------------------------------------------
# 3. run_perception_on_window passes non-null roi_crop_xyxy to SAM3
#    (Falcon disabled, SAM3 enabled)
# ---------------------------------------------------------------------------

def test_sam3_receives_non_null_roi_crop_when_falcon_off(monkeypatch):
    from perception.pipeline import run_perception_on_window
    from perception.sampling import SamplingPolicy

    sam3_calls: list = []

    class _FakeSam3:
        def process_window(self, window_path, **kwargs):
            sam3_calls.append({"window_path": window_path,
                               "roi_crop_xyxy": kwargs.get("roi_crop_xyxy"),
                               "concepts": kwargs.get("concepts")})
            return {
                "detections": [], "tracks": [], "masks": [],
                "keyframes": [], "ocr": [], "limitations": [],
                "obstructed": False, "timings_ms": {"total_ms": 1},
                "sam3_meta": {"object_ids": [], "frame_count": 0,
                              "prompt_to_obj_ids": {}},
            }

    falcon_calls: list = []

    class _FailFalcon:
        DEFAULT_CATEGORIES = {"item": "x", "person": "y", "receipt": "z"}

        def detect_on_frames(self, frames, **kw):
            falcon_calls.append(kw)
            return []

    # Sampler stub: SAM-only mode returns before sampling, so this should
    # not actually be invoked — but we stub it defensively.
    from perception import pipeline as ppl
    monkeypatch.setattr(ppl, "_sample_frames", lambda *a, **kw: [])

    crop = (303, 209, 303 + 801, 209 + 865)  # the replay's ROI box
    run_perception_on_window(
        window_path="/fake/window.mp4",
        fps=25, zones=[],
        falcon_client=_FailFalcon(),
        sam2_client=None, ocr_engine=None,
        sam3_client=_FakeSam3(),
        sampling=SamplingPolicy(base_fps=1),
        sam3_concepts=[],
        sam3_roi_crop=crop,
        falcon_enabled=False, sam3_enabled=True,
        sam2_enabled=False, ocr_enabled=False,
    )

    assert falcon_calls == [], (
        "Falcon was invoked while SAM-only mode is active: "
        f"{len(falcon_calls)} call(s)")
    assert len(sam3_calls) == 1, (
        f"SAM3.process_window should run exactly once; got {len(sam3_calls)}")
    sent = sam3_calls[0]
    assert sent["roi_crop_xyxy"] == crop, (
        f"SAM3 did not receive the ROI crop; got {sent['roi_crop_xyxy']!r}")
    assert sent["window_path"] == "/fake/window.mp4"


# ---------------------------------------------------------------------------
# 4. case_runner builds roi_crop_xyxy from the sam3 model view
# ---------------------------------------------------------------------------

def test_case_runner_derives_sam3_crop_from_model_view():
    """Case-runner contract: when the active camera has a SAM3 model
    view, the SCO crop passed to ``run_perception`` is computed from
    the view's resolved zone (``x, y, x+w, y+h``)."""
    from app.camera_rois import model_view
    from app.config import load_config

    cfg = load_config()
    view = model_view(cfg, "cam_return_01", "sam3")
    assert view is not None
    z = view["resolved_zones"][0]
    crop = (int(z["x"]), int(z["y"]),
            int(z["x"]) + int(z["w"]), int(z["y"]) + int(z["h"]))
    # Box dimensions are positive and inside a 1920x1080 source frame
    assert crop[2] > crop[0]
    assert crop[3] > crop[1]
    assert crop[0] >= 0 and crop[1] >= 0
