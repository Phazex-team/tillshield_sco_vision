"""Perception pipeline unit tests.

All tests stub the Falcon + SAM 2 clients so no real model weights are
loaded.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def test_sampling_base_only():
    from perception.sampling import SamplingPolicy, plan_indices
    base = datetime(2026, 6, 15, 14, 0, 0)
    out = plan_indices(fps=25, frame_count=125, base_start_ts=base,
                       policy=SamplingPolicy(base_fps=1))
    # 125 frames @ 25 fps = 5s of video; base_fps 1 => ~5 sampling points.
    indices = [idx for idx, _ in out]
    assert len(indices) >= 5


def test_sampling_densifies_around_handover():
    from perception.sampling import SamplingPolicy, plan_indices
    base = datetime(2026, 6, 15, 14, 0, 0)
    centre = base + timedelta(seconds=5)
    out = plan_indices(fps=25, frame_count=250, base_start_ts=base,
                       policy=SamplingPolicy(base_fps=1, handover_fps=10,
                                             burst_pre_sec=2,
                                             burst_post_sec=2),
                       handover_candidates=[centre])
    assert len(out) >= 30  # base+handover densification


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

def _det(label, score, bbox, frame_idx, ts):
    from perception.schemas import Detection
    return Detection(label=label, score=score, bbox_xyxy=list(bbox),
                     frame_id=f"f{frame_idx:03d}", frame_idx=frame_idx,
                     ts=ts)


def test_tracker_creates_and_continues_track():
    from perception.tracker import Tracker
    base = datetime(2026, 6, 15, 14, 0, 0)
    dets = [
        _det("bag", 0.9, [100, 100, 200, 200], 0, base),
        _det("bag", 0.9, [110, 110, 210, 210], 1,
             base + timedelta(seconds=1)),
        _det("bag", 0.9, [120, 120, 220, 220], 2,
             base + timedelta(seconds=2)),
    ]
    t = Tracker(confirm_hits=2)
    t.update(dets)
    tracks = t.export()
    assert len(tracks) == 1
    assert tracks[0].label == "bag"


def test_tracker_separates_distinct_objects():
    from perception.tracker import Tracker
    base = datetime(2026, 6, 15, 14, 0, 0)
    dets = [
        _det("bag", 0.9, [0, 0, 50, 50], 0, base),
        _det("bag", 0.9, [500, 500, 600, 600], 0, base),
        _det("bag", 0.9, [5, 5, 55, 55], 1,
             base + timedelta(seconds=1)),
        _det("bag", 0.9, [505, 505, 605, 605], 1,
             base + timedelta(seconds=1)),
    ]
    t = Tracker()
    t.update(dets)
    tracks = t.export()
    assert len(tracks) == 2


# ---------------------------------------------------------------------------
# Temporal memory
# ---------------------------------------------------------------------------

def test_annotate_tracks_marks_handover_when_in_counter_zone():
    from perception.schemas import Track
    from perception.temporal_memory import Zone, annotate_tracks
    base = datetime(2026, 6, 15, 14, 0, 0)
    detections = [
        _det("shopping bag", 0.9, [100, 100, 150, 150], 0, base),
        _det("shopping bag", 0.9, [600, 100, 650, 150], 1,
             base + timedelta(seconds=1)),
    ]
    track = Track(track_id="t1", label="shopping bag",
                  first_seen_ts=base,
                  last_seen_ts=base + timedelta(seconds=1),
                  detections=[0, 1])
    zones = [Zone(name="customer_zone", x=0, y=0, w=300, h=400),
             Zone(name="counter_zone", x=500, y=0, w=400, h=400)]
    annotated = annotate_tracks([track], detections, zones=zones)
    t = annotated[0]
    assert "customer_zone" in t.zones
    assert "counter_zone" in t.zones
    assert "handover_candidate" in t.events
    assert t.physical_item_candidate is True


def test_annotate_tracks_does_not_mark_receipt_only_as_handover():
    from perception.schemas import Track
    from perception.temporal_memory import Zone, annotate_tracks
    base = datetime(2026, 6, 15, 14, 0, 0)
    detections = [
        _det("receipt", 0.9, [600, 100, 650, 150], 0, base),
    ]
    track = Track(track_id="t1", label="receipt",
                  first_seen_ts=base, last_seen_ts=base,
                  detections=[0])
    zones = [Zone(name="counter_zone", x=500, y=0, w=400, h=400)]
    annotated = annotate_tracks([track], detections, zones=zones)
    t = annotated[0]
    assert t.physical_item_candidate is False
    assert t.receipt_candidate is True
    assert "handover_candidate" not in t.events


# ---------------------------------------------------------------------------
# Keyframes
# ---------------------------------------------------------------------------

def test_select_keyframes_emits_first_and_final():
    from perception.keyframes import select_keyframes
    from perception.schemas import Track
    base = datetime(2026, 6, 15, 14, 0, 0)
    detections = [
        _det("bag", 0.9, [100, 100, 200, 200], 0, base),
        _det("bag", 0.9, [110, 110, 210, 210], 1,
             base + timedelta(seconds=1)),
        _det("bag", 0.9, [600, 600, 700, 700], 2,
             base + timedelta(seconds=2)),
    ]
    t = Track(track_id="t1", label="bag",
              first_seen_ts=base,
              last_seen_ts=base + timedelta(seconds=2),
              detections=[0, 1, 2],
              physical_item_candidate=True,
              events=["handover_candidate"])
    kfs = select_keyframes([t], detections)
    roles = {k.role for k in kfs}
    assert "first_appearance" in roles
    assert "final_state" in roles
    assert "handover_candidate" in roles


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class _StubFalcon:
    def __init__(self, detections):
        self._detections = detections

    def detect_on_frames(self, frames, *, query):
        return self._detections

    def _ensure_loaded(self):
        pass

    def unload(self):
        pass


class _StubSam2:
    def has_capability(self):
        return False

    def segment(self, *args, **kwargs):
        return []


def test_pipeline_with_stubbed_clients_returns_tracks_and_keyframes():
    from perception.pipeline import run_perception_on_window
    from perception.sampling import SamplingPolicy
    from perception.schemas import Detection
    from perception.temporal_memory import Zone

    base = datetime(2026, 6, 15, 14, 0, 0)
    detections = [
        Detection(label="shopping bag", score=0.9,
                  bbox_xyxy=[100, 100, 200, 200],
                  frame_id="f000", frame_idx=0, ts=base),
        Detection(label="shopping bag", score=0.9,
                  bbox_xyxy=[600, 100, 700, 200],
                  frame_id="f025", frame_idx=25,
                  ts=base + timedelta(seconds=1)),
    ]
    # Pipeline needs at least one frame from sampling; we bypass real
    # video decoding by passing window_path=None and pre-populating
    # detections via the stub. Since detection_on_frames just returns
    # whatever we hand it, the result is empty (no decoded frames). To
    # actually exercise the tracker path, we instead invoke the helper
    # functions directly here in addition to the pipeline-runs-clean test.
    result = run_perception_on_window(
        window_path=None, fps=25,
        zones=[Zone(name="counter_zone", x=500, y=0, w=400, h=400)],
        falcon_client=_StubFalcon(detections),
        sam2_client=_StubSam2(),
        sampling=SamplingPolicy(),
    )
    # With no window_path, the sampler returns no frames so detection
    # never runs — but the structure must still be well-formed.
    assert "limitations" in result
    assert "no_window_path" in result["limitations"]
    assert result["tracks"] == []
    assert result["keyframes"] == []


def test_pipeline_assembles_tracks_when_detections_provided_via_synthetic_frames(
        tmp_path, monkeypatch):
    """Exercise the tracker + temporal memory + keyframes glue by
    short-circuiting the sampler. We monkeypatch ``_sample_frames`` to
    return a known list and let the rest of the pipeline run."""
    from perception import pipeline as pl
    from perception.sampling import SamplingPolicy
    from perception.schemas import Detection
    from perception.temporal_memory import Zone

    base = datetime(2026, 6, 15, 14, 0, 0)
    detections = [
        Detection(label="shopping bag", score=0.9,
                  bbox_xyxy=[100, 100, 200, 200],
                  frame_id="f000", frame_idx=0, ts=base),
        Detection(label="shopping bag", score=0.9,
                  bbox_xyxy=[600, 100, 700, 200],
                  frame_id="f025", frame_idx=25,
                  ts=base + timedelta(seconds=1)),
        Detection(label="shopping bag", score=0.9,
                  bbox_xyxy=[600, 100, 700, 200],
                  frame_id="f050", frame_idx=50,
                  ts=base + timedelta(seconds=2)),
    ]
    fake_frames = [(0, base, object()), (25, base + timedelta(seconds=1),
                                          object()),
                   (50, base + timedelta(seconds=2), object())]

    class _Stub:
        def detect_on_frames(self, frames, *, query):
            return detections
        def _ensure_loaded(self):
            pass
        def unload(self):
            pass
        _detector = object()

    monkeypatch.setattr(pl, "_sample_frames",
                        lambda *a, **k: fake_frames)
    monkeypatch.setattr(pl, "run_ocr", lambda *a, **k: [])

    result = pl.run_perception_on_window(
        window_path="/tmp/fake.mp4", fps=25,
        zones=[Zone(name="counter_zone", x=500, y=0, w=400, h=400)],
        falcon_client=_Stub(),
        sam2_client=_StubSam2(),
        sampling=SamplingPolicy(),
    )
    assert result["tracks"], "expected at least one track"
    assert any(t["physical_item_candidate"] for t in result["tracks"])
    assert result["keyframes"], "expected at least one keyframe"
    assert "sam2_unavailable" in result["limitations"]
