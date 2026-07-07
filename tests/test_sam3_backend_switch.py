"""SAM-3 backend on/off switching + grouper compatibility.

Pins the council's experiment contract:
  * When Falcon is OFF and SAM-3 is ON, Falcon is NOT called.
  * SAM-3 runs on the video window with concept prompts and emits
    detections/tracks tagged sco_item_NNN / sco_generic_* so the
    existing grouper can consume them unchanged.
  * One SAM-3 identity carrying multiple concept labels collapses to
    a single canonical group via the grouper's overlap rule.
  * Two distinct SAM-3 identities surface as two distinct groups.
  * When Falcon is ON, the Falcon path remains functional (regression).
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE = datetime(2026, 6, 28, 18, 0, 30)


def _fake_sam3_window_result_one_identity_two_concepts():
    """SAM-3 returns one stable object id that gets attributed to both
    a POS-specific concept (sco_item_000) and a generic concept
    (sco_generic_products) — the duplicate-label situation we
    explicitly want the grouper to collapse."""
    det_pos = {
        "label": "sco_item_000", "score": 0.92,
        "bbox_xyxy": [400, 300, 520, 420],
        "frame_id": "frame_000010", "frame_idx": 10,
        "ts": (BASE + timedelta(seconds=2)).isoformat(),
        "sam3_object_id": 1, "query": "sco_item_000",
    }
    det_generic = {
        "label": "sco_generic_products", "score": 0.71,
        "bbox_xyxy": [402, 301, 522, 421],
        "frame_id": "frame_000030", "frame_idx": 30,
        "ts": (BASE + timedelta(seconds=4)).isoformat(),
        "sam3_object_id": 1, "query": "sco_generic_products",
    }
    return {
        "detections": [det_pos, det_generic],
        "tracks": [{
            "track_id": "sam3_obj_0001",
            "label": "sco_item_000",
            "first_seen_ts": (BASE + timedelta(seconds=2)).isoformat(),
            "last_seen_ts": (BASE + timedelta(seconds=4)).isoformat(),
            "detections": [0, 1],
            "zones": ["sco_audit_zone"],
            "events": [],
            "physical_item_candidate": True,
            "receipt_candidate": False,
            "confidence": 0.92,
            "sam3_object_id": 1,
        }],
        "masks": [], "keyframes": [], "ocr": [],
        "limitations": [], "obstructed": False, "timings_ms": {"total_ms": 1},
        "sam3_meta": {"object_ids": [1], "frame_count": 2,
                      "prompt_to_obj_ids": {"sco_item_000": [1]}},
    }


def _fake_sam3_window_result_two_identities_far_apart():
    det_pos = {
        "label": "sco_item_000", "score": 0.91,
        "bbox_xyxy": [400, 300, 520, 420],
        "frame_id": "frame_000010", "frame_idx": 10,
        "ts": (BASE + timedelta(seconds=2)).isoformat(),
        "sam3_object_id": 1, "query": "sco_item_000",
    }
    det_extra = {
        "label": "sco_generic_food_container", "score": 0.65,
        "bbox_xyxy": [900, 300, 1020, 420],
        "frame_id": "frame_000020", "frame_idx": 20,
        "ts": (BASE + timedelta(seconds=3)).isoformat(),
        "sam3_object_id": 2, "query": "sco_generic_food_container",
    }
    return {
        "detections": [det_pos, det_extra],
        "tracks": [
            {"track_id": "sam3_obj_0001", "label": "sco_item_000",
             "first_seen_ts": det_pos["ts"], "last_seen_ts": det_pos["ts"],
             "detections": [0], "zones": ["sco_audit_zone"],
             "events": [], "physical_item_candidate": True,
             "receipt_candidate": False, "confidence": 0.91,
             "sam3_object_id": 1},
            {"track_id": "sam3_obj_0002",
             "label": "sco_generic_food_container",
             "first_seen_ts": det_extra["ts"],
             "last_seen_ts": det_extra["ts"],
             "detections": [1], "zones": ["sco_audit_zone"],
             "events": [], "physical_item_candidate": True,
             "receipt_candidate": False, "confidence": 0.65,
             "sam3_object_id": 2},
        ],
        "masks": [], "keyframes": [], "ocr": [],
        "limitations": [], "obstructed": False, "timings_ms": {"total_ms": 1},
        "sam3_meta": {"object_ids": [1, 2], "frame_count": 2,
                      "prompt_to_obj_ids": {"sco_item_000": [1],
                                             "sco_generic_food_container": [2]}},
    }


# ---------------------------------------------------------------------------
# 1. SAM-only mode: Falcon is not called
# ---------------------------------------------------------------------------

def test_sam_only_mode_falcon_is_not_called(monkeypatch):
    from perception.pipeline import run_perception_on_window
    from perception.sampling import SamplingPolicy

    falcon_calls: list = []

    class _FakeFalcon:
        DEFAULT_CATEGORIES = {"item": "x", "person": "y", "receipt": "z"}

        def detect_on_frames(self, frames, **kw):
            falcon_calls.append((frames, kw))
            return []

    class _FakeSam3:
        def process_window(self, window_path, **kwargs):
            return _fake_sam3_window_result_one_identity_two_concepts()

    # Sampler not needed in the SAM-only path; perception returns the
    # SAM-3 result before reaching the sampling stage. Still, stub it
    # so the function doesn't try to open the file.
    from perception import pipeline as ppl
    monkeypatch.setattr(ppl, "_sample_frames", lambda *a, **kw: [])

    result = run_perception_on_window(
        window_path="/some/fake/window.mp4",
        fps=25, zones=[],
        falcon_client=_FakeFalcon(),       # provided but should not be called
        sam2_client=None, ocr_engine=None,
        sam3_client=_FakeSam3(),
        sampling=SamplingPolicy(base_fps=1),
        sam3_concepts=[],
        falcon_enabled=False,              # <-- Falcon OFF
        sam3_enabled=True,                 # <-- SAM-3 ON
        sam2_enabled=False, ocr_enabled=False,
    )

    # Falcon never touched
    assert falcon_calls == [], (
        "Falcon was invoked in SAM-only mode: "
        f"{len(falcon_calls)} call(s)")
    # SAM-3 result surfaced
    assert "limitations" in result
    assert "falcon_disabled_by_config" in result["limitations"]
    assert any(d.get("label") == "sco_item_000"
               for d in result["detections"])
    assert any(t.get("sam3_object_id") == 1 for t in result["tracks"])


# ---------------------------------------------------------------------------
# 2. Grouper collapses one SAM-3 identity carrying two concept labels
# ---------------------------------------------------------------------------

def test_grouper_collapses_one_sam3_identity_to_one_canonical_group():
    from perception.item_grouping import group_sco_items
    sam3_out = _fake_sam3_window_result_one_identity_two_concepts()
    groups = group_sco_items(
        sam3_out["detections"], sam3_out["tracks"],
        pos_basket=[{"description": "Biriyani Hot Food"}],
    )
    assert len(groups) == 1, f"expected single canonical group, got {groups}"
    g = groups[0]
    assert g["matched_pos_item"] == "Biriyani Hot Food"
    assert g["is_extra_candidate"] is False
    # The SAM-3 identity track id is preserved in the group's track_ids
    assert "sam3_obj_0001" in g["track_ids"]


def test_grouper_two_sam3_identities_become_two_groups():
    from perception.item_grouping import group_sco_items
    sam3_out = _fake_sam3_window_result_two_identities_far_apart()
    groups = group_sco_items(
        sam3_out["detections"], sam3_out["tracks"],
        pos_basket=[{"description": "Biriyani Hot Food"}],
    )
    assert len(groups) == 2
    matched = [g for g in groups if not g["is_extra_candidate"]]
    extras = [g for g in groups if g["is_extra_candidate"]]
    assert len(matched) == 1 and len(extras) == 1
    assert matched[0]["matched_pos_item"] == "Biriyani Hot Food"
    assert extras[0]["source_labels"] == ["sco_generic_food_container"]
    # Both SAM-3 identities preserved across groups
    assert "sam3_obj_0001" in matched[0]["track_ids"]
    assert "sam3_obj_0002" in extras[0]["track_ids"]


# ---------------------------------------------------------------------------
# 3. Falcon-enabled regression — Falcon path still works when Falcon ON
# ---------------------------------------------------------------------------

def test_falcon_enabled_path_unchanged(monkeypatch):
    """Drive the pipeline with Falcon ON, SAM-3 OFF — output is the
    legacy Falcon-detection list. Pins back-compat for the
    non-experiment runtime."""
    from perception.pipeline import run_perception_on_window
    from perception.sampling import SamplingPolicy
    from perception.schemas import Detection
    from datetime import datetime, timedelta
    from PIL import Image

    ts0 = datetime(2026, 6, 28, 18, 0, 0)
    img = Image.new("RGB", (640, 480), (0, 0, 0))

    class _FakeFalcon:
        DEFAULT_CATEGORIES = {"item": "x", "person": "y", "receipt": "z"}

        def detect_on_frames(self, frames, **kw):
            return [Detection(
                label="item", score=0.9, bbox_xyxy=[10, 10, 50, 50],
                frame_id="f0", frame_idx=0, ts=ts0,
            )]

    # Stub the sampler to return one synthetic frame.
    from perception import pipeline as ppl
    monkeypatch.setattr(
        ppl, "_sample_frames",
        lambda *a, **kw: [(0, ts0, img)],
    )

    result = run_perception_on_window(
        window_path="ignored", fps=25, zones=[],
        falcon_client=_FakeFalcon(),
        sam2_client=None, ocr_engine=None,
        sam3_client=None,
        sampling=SamplingPolicy(base_fps=1),
        falcon_enabled=True, sam3_enabled=False,
        sam2_enabled=False, ocr_enabled=False,
    )
    labels = [d["label"] for d in result["detections"]]
    assert "item" in labels
    # No SAM-3 metadata when sam3 is OFF
    assert "sam3_meta" not in result


# ---------------------------------------------------------------------------
# 4. A/B mode (Falcon ON + SAM-3 ON): both contribute to detections/tracks
# ---------------------------------------------------------------------------

def test_ab_mode_falcon_and_sam3_both_contribute(monkeypatch):
    from perception.pipeline import run_perception_on_window
    from perception.sampling import SamplingPolicy
    from perception.schemas import Detection
    from datetime import datetime
    from PIL import Image

    ts0 = datetime(2026, 6, 28, 18, 0, 0)
    img = Image.new("RGB", (640, 480), (0, 0, 0))

    class _FakeFalcon:
        DEFAULT_CATEGORIES = {"item": "x", "person": "y", "receipt": "z"}

        def detect_on_frames(self, frames, **kw):
            return [Detection(label="item", score=0.9,
                              bbox_xyxy=[10, 10, 50, 50],
                              frame_id="f0", frame_idx=0, ts=ts0)]

    class _FakeSam3:
        def process_window(self, window_path, **kwargs):
            return _fake_sam3_window_result_one_identity_two_concepts()

    from perception import pipeline as ppl
    monkeypatch.setattr(
        ppl, "_sample_frames", lambda *a, **kw: [(0, ts0, img)],
    )

    result = run_perception_on_window(
        window_path="ignored", fps=25, zones=[],
        falcon_client=_FakeFalcon(),
        sam2_client=None, ocr_engine=None,
        sam3_client=_FakeSam3(),
        sampling=SamplingPolicy(base_fps=1),
        falcon_enabled=True, sam3_enabled=True,
        sam2_enabled=False, ocr_enabled=False,
    )
    labels = {d["label"] for d in result["detections"]}
    # Falcon's default 'item' AND SAM-3's POS label are both present
    assert "item" in labels
    assert "sco_item_000" in labels
    assert "sam3_meta" in result
    assert result["sam3_meta"]["object_ids"] == [1]


# ---------------------------------------------------------------------------
# 5. Config-driven gating — when models.sam3.enabled is False, no SAM-3
# ---------------------------------------------------------------------------

def test_sam3_config_gate_off_skips_sam3(monkeypatch):
    from perception.pipeline import run_perception_on_window
    from perception.sampling import SamplingPolicy
    from datetime import datetime
    from PIL import Image

    sam3_calls: list = []

    class _FakeSam3:
        def process_window(self, window_path, **kwargs):
            sam3_calls.append(window_path)
            return _fake_sam3_window_result_one_identity_two_concepts()

    ts0 = datetime(2026, 6, 28, 18, 0, 0)
    img = Image.new("RGB", (640, 480), (0, 0, 0))
    from perception import pipeline as ppl
    monkeypatch.setattr(
        ppl, "_sample_frames", lambda *a, **kw: [(0, ts0, img)],
    )

    run_perception_on_window(
        window_path="ignored", fps=25, zones=[],
        falcon_client=None, sam2_client=None, ocr_engine=None,
        sam3_client=_FakeSam3(),
        sampling=SamplingPolicy(base_fps=1),
        falcon_enabled=False, sam3_enabled=False,  # both off
        sam2_enabled=False, ocr_enabled=False,
    )
    assert sam3_calls == [], "sam3.enabled=False should not invoke SAM-3"


# ---------------------------------------------------------------------------
# 6. Sam3Client has_capability gating
# ---------------------------------------------------------------------------

def test_sam3_client_has_capability_false_without_weights():
    from perception.sam3_client import Sam3Client
    c = Sam3Client(model_path="/definitely/not/a/path")
    assert c.has_capability() is False


def test_sam3_client_empty_concepts_returns_empty_result():
    from perception.sam3_client import Sam3Client
    c = Sam3Client(model_path=None)
    out = c.process_window("/anything.mp4", concepts=[])
    assert out["detections"] == []
    assert out["tracks"] == []
    assert "sam3_no_concepts" in out["limitations"]


# ---------------------------------------------------------------------------
# 7. Concept builder uses the SKU translator
# ---------------------------------------------------------------------------

def test_build_concepts_from_pos_includes_pos_and_generics():
    """Hot-food mode (default): POS-derived concepts come first; only
    food-container generics are appended; broad product/package terms
    are OFF unless explicitly enabled (they over-fire on POS hardware
    in the hot-food simulation)."""
    from perception.sam3_client import build_concepts_from_pos
    from types import SimpleNamespace
    pos = SimpleNamespace(raw_payload={"items": [
        {"description": "Biriyani Hot Food", "quantity": 1},
        {"description": "Curry Hot Food", "quantity": 1},
    ]})
    concepts = build_concepts_from_pos(pos)
    labels = [c.label for c in concepts]
    texts = [c.text for c in concepts]
    assert "sco_item_000" in labels
    assert "sco_item_001" in labels
    # food-container generics present
    assert "sco_generic_food_container" in labels
    assert "sco_generic_takeaway_container" in labels
    # broad noisy generics OFF by default
    assert "sco_generic_products" not in labels
    assert "sco_generic_packaging" not in labels
    # POS concepts come BEFORE generics
    assert (labels.index("sco_item_000")
            < labels.index("sco_generic_food_container"))
    # Hot-food phrasing was applied (container-shaped, not the raw SKU)
    assert any("container" in t.lower() and "rice" in t.lower()
               for t in texts), texts
    assert any("container" in t.lower() and "curry" in t.lower()
               for t in texts), texts


def test_build_concepts_opt_in_broad_generics_re_enables_product_terms():
    from perception.sam3_client import build_concepts_from_pos
    from types import SimpleNamespace
    pos = SimpleNamespace(raw_payload={"items": [
        {"description": "Biriyani Hot Food"}]})
    concepts = build_concepts_from_pos(pos, include_broad_generics=True)
    labels = [c.label for c in concepts]
    assert "sco_generic_products" in labels
    assert "sco_generic_packaging" in labels


# ---------------------------------------------------------------------------
# 8. Active-config integration: cam_return_01 has a sam3 model view
# ---------------------------------------------------------------------------

def test_real_config_cam_return_01_has_sam3_model_view():
    from app.config import load_config
    cfg = load_config()
    cam = next(c for c in cfg.cameras if c.get("id") == "cam_return_01")
    views = cam.get("model_roi_views") or {}
    sam3_view = views.get("sam3")
    assert sam3_view is not None, "cam_return_01 must define a sam3 model view"
    assert bool(sam3_view.get("enabled", False)) is True
    assert sam3_view.get("roi_ids") == ["sco_audit_zone"], (
        "sam3 view must target only sco_audit_zone")


def test_real_config_sam3_block_present_but_default_off():
    from app.config import load_config
    cfg = load_config()
    sam3_raw = (cfg.raw.get("models") or {}).get("sam3") or {}
    assert sam3_raw.get("name") == "facebook/sam3"
    # Default OFF — the experiment requires explicit opt-in.
    assert bool(sam3_raw.get("enabled", False)) is False
