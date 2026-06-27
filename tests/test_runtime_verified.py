"""Proof: the live ``analyze_case`` path can reach VERIFIED with
real perception track evidence — and stays REVIEW without it.

Also pins:

  * Runtime perception shape (``track_id``, not ``tracker_id``) is
    accepted by the decision policy.
  * ``config.yaml.storage.min_free_disk_gb`` matches the PRODUCTION_SPEC
    default of 100 — no doc/config drift.
"""
from __future__ import annotations

import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="ffmpeg required for window builder",
)


def _synthetic_frames(n: int, *, fps: int = 25,
                      start: datetime | None = None,
                      size: tuple[int, int] = (160, 120)):
    start = start or datetime(2026, 6, 15, 14, 0, 0)
    out = []
    dt = timedelta(seconds=1.0 / fps)
    for i in range(n):
        ts = start + dt * i
        frame = np.full((size[1], size[0], 3),
                        fill_value=(i * 5) % 255, dtype=np.uint8)
        out.append((ts, frame))
    return out


def _fresh_session(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'t.sqlite'}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    import db.session as s
    s._ENGINE = None
    s._SESSION_FACTORY = None
    s.init_schema()
    import pos.correlation as pc
    monkeypatch.setattr(pc, "PRE_ROLL_SEC", 2)
    monkeypatch.setattr(pc, "POST_ROLL_SEC", 2)
    return s.get_sessionmaker()


def _seed_segment(SM, storage_root, *, start_at, duration_sec=10):
    from video.segment_recorder import RecorderConfig, SegmentRecorder
    cfg = RecorderConfig(camera_id="cam_01",
                         storage_root=storage_root,
                         fps=25, width=160, height=120,
                         segment_duration_sec=duration_sec)
    rec = SegmentRecorder(cfg, session_factory=SM)
    rec.record_one_segment(
        _synthetic_frames(25 * duration_sec, start=start_at),
        start_at=start_at)


def _seed_pos(SM, *, pos_event_at):
    from db.models import Case
    from pos.ingest import ingest_batch
    from pos.schemas import PosBatchIn, PosEventIn
    with SM() as s:
        ingest_batch(s, PosBatchIn(
            source_system="test", store_id="store_1",
            received_at=pos_event_at,
            events=[PosEventIn(
                store_id="store_1", terminal_id="t1",
                transaction_id="txn-V", line_id="L1",
                event_type="SALE", pos_event_at=pos_event_at,
            )],
        ))
        s.commit()
        return s.query(Case).first().id


# ---------------------------------------------------------------------------
# 1. analyze_case with runtime-shaped track produces VERIFIED
# ---------------------------------------------------------------------------

def test_analyze_case_verified_with_real_perception_track(tmp_path,
                                                          monkeypatch):
    SM = _fresh_session(tmp_path, monkeypatch)
    storage = tmp_path / "storage"
    base = datetime(2026, 6, 15, 14, 0, 0)
    _seed_segment(SM, storage, start_at=base - timedelta(seconds=5))
    case_id = _seed_pos(SM, pos_event_at=base)

    # Runtime perception shape: emits ``track_id`` (not ``tracker_id``)
    # and matches the dataclass produced by perception.pipeline._t_dict.
    def _runtime_perception(session, case, window):
        return {
            "detections": [{"label": "shopping bag", "score": 0.9,
                            "bbox_xyxy": [10, 10, 50, 50],
                            "frame_id": "frame_000000",
                            "frame_idx": 0, "ts": base.isoformat()}],
            "tracks": [{
                "track_id": "track_0001",
                "label": "shopping bag",
                "first_seen_ts": base.isoformat(),
                "last_seen_ts": base.isoformat(),
                "detections": [0],
                "zones": ["customer_zone", "counter_zone"],
                "events": ["entered_counter_zone", "handover_candidate"],
                "physical_item_candidate": True,
                "receipt_candidate": False,
                "confidence": 0.9,
            }, {
                "track_id": "track_0002",
                "label": "person",
                "first_seen_ts": base.isoformat(),
                "last_seen_ts": base.isoformat(),
                "detections": [0],
                "zones": ["customer_zone"],
                "events": ["entered_customer_zone"],
                "physical_item_candidate": False,
                "receipt_candidate": False,
                "confidence": 0.9,
            }],
            "keyframes": [{"frame_id": "frame_000000", "frame_idx": 0,
                            "ts": base.isoformat(),
                            "role": "handover_candidate",
                            "track_id": "track_0001"}],
            "ocr": [], "obstructed": False, "limitations": [],
        }

    def _stub_vlm(session, case, window, manifest=None):
        return {
            "provider": "qwen3_vl", "model_name": "stub",
            "parsed": {
                "handover_occurred": True,
                "physical_item_presented": True,
                "receipt_visible": True,
                "items_observed": ["shopping bag"],
                "narrative": "clean handover observed",
                "confidence": "high",
                "obstructed": False,
                "camera_view_clear": True,
                "limitations": [],
            },
            "latency_ms": 5, "error": None,
        }

    from app.case_runner import analyze_case
    with SM() as s:
        # This test exercises the legacy REFUND flow (handover semantics,
        # customer_zone tracks, refund-shaped VLM output). The default
        # prompt in SCO mode is sco_basket_match_v1 — explicitly opt
        # into the legacy refund prompt so the legacy decision policy
        # runs and VERIFIED is reachable.
        result = analyze_case(s, case_id,
                              perception_runner=_runtime_perception,
                              vlm_runner=_stub_vlm,
                              prompt_version="return_review_v1")
    assert result["outcome"] == "VERIFIED", result


# ---------------------------------------------------------------------------
# 2. analyze_case with NO perception track stays REVIEW
# ---------------------------------------------------------------------------

def test_analyze_case_no_track_stays_review(tmp_path, monkeypatch):
    SM = _fresh_session(tmp_path, monkeypatch)
    storage = tmp_path / "storage"
    base = datetime(2026, 6, 15, 14, 0, 0)
    _seed_segment(SM, storage, start_at=base - timedelta(seconds=5))
    case_id = _seed_pos(SM, pos_event_at=base)

    def _no_tracks(session, case, window):
        return {"detections": [], "tracks": [],
                "keyframes": [], "ocr": [],
                "obstructed": False, "limitations": []}

    def _vlm_says_verified(session, case, window, manifest=None):
        return {
            "provider": "qwen3_vl", "model_name": "stub",
            "parsed": {
                "handover_occurred": True,
                "physical_item_presented": True,
                "receipt_visible": True,
                "items_observed": ["bag"],
                "confidence": "high",
                "obstructed": False,
                "camera_view_clear": True,
            },
            "latency_ms": 1, "error": None,
        }

    from app.case_runner import analyze_case
    with SM() as s:
        result = analyze_case(s, case_id,
                              perception_runner=_no_tracks,
                              vlm_runner=_vlm_says_verified)
    assert result["outcome"] != "VERIFIED"
    assert result["outcome"] == "REVIEW"


# ---------------------------------------------------------------------------
# 3. decision_policy directly accepts runtime track_id
# ---------------------------------------------------------------------------

def test_runtime_track_shape_accepted_by_policy():
    from reasoning.decision_policy import (
        OUTCOME_VERIFIED, decide, summary_from_vlm,
    )
    # Runtime perception emits ``track_id``; the policy must NOT require
    # the persisted ``tracker_id`` field name.
    perception = {"tracks": [{
        "track_id": "track_0001",
        "label": "shopping bag",
        "physical_item_candidate": True,
        "zones": ["counter_zone"],
        "events": ["entered_counter_zone", "handover_candidate"],
        "confidence": 0.9,
    }, {
        "track_id": "track_0002",
        "label": "person",
        "physical_item_candidate": False,
        "zones": ["customer_zone"],
        "events": ["entered_customer_zone"],
        "confidence": 0.9,
    }]}
    summary = summary_from_vlm(
        {"handover_occurred": True,
         "physical_item_presented": True,
         "receipt_visible": True,
         "confidence": "high"},
        footage_valid=True,
        perception_result=perception,
    )
    assert summary.physical_item_track is True
    assert summary.item_reaches_counter is True
    assert summary.customer_present is True
    assert decide(summary).outcome == OUTCOME_VERIFIED


def test_persisted_track_shape_still_accepted():
    """ORM/persisted shape uses ``tracker_id``. Both must work."""
    from reasoning.decision_policy import (
        OUTCOME_VERIFIED, decide, summary_from_vlm,
    )
    perception = {"tracks": [{
        "tracker_id": "track_0002",
        "label": "shopping bag",
        "physical_item_candidate": True,
        "zones": ["counter_zone"],
        "events": ["handover_candidate"],
        "confidence": 0.9,
    }, {
        "tracker_id": "track_0003",
        "label": "person",
        "physical_item_candidate": False,
        "zones": ["customer_zone"],
        "events": ["entered_customer_zone"],
        "confidence": 0.9,
    }]}
    summary = summary_from_vlm(
        {"handover_occurred": True,
         "physical_item_presented": True,
         "receipt_visible": True,
         "confidence": "high"},
        footage_valid=True,
        perception_result=perception,
    )
    assert decide(summary).outcome == OUTCOME_VERIFIED


# ---------------------------------------------------------------------------
# 4. config + docs alignment
# ---------------------------------------------------------------------------

def test_config_min_free_disk_gb_matches_spec_default():
    """PRODUCTION_SPEC says the default is 100 GiB. Config must match."""
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    storage = cfg.get("storage") or {}
    assert storage.get("min_free_disk_gb") == 100, (
        f"config.yaml.storage.min_free_disk_gb must be 100 to match "
        f"PRODUCTION_SPEC + models/README.md; got "
        f"{storage.get('min_free_disk_gb')}"
    )
