"""Phase 4 — perception.episode_selector tests.

Covers clean episode, multiple-overlapping ambiguity, long-continuous
ambiguity, no-activity fallback, anchor-outside-groups, and
person/zone filter behavior.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


WINDOW_START = datetime(2026, 6, 15, 14, 0, 0)
WINDOW_END = datetime(2026, 6, 15, 14, 5, 0)   # 5-minute POS window
POS_TIME = datetime(2026, 6, 15, 14, 2, 30)    # middle of window
ROI = "sco_audit_zone"


def _track(label, zone, t0_off, t1_off):
    return {
        "label": label,
        "zones": [zone] if isinstance(zone, str) else list(zone),
        "first_seen_ts": WINDOW_START + timedelta(seconds=t0_off),
        "last_seen_ts": WINDOW_START + timedelta(seconds=t1_off),
    }


def _select(tracks):
    from perception.episode_selector import select_sco_episode
    return select_sco_episode(
        tracks,
        pos_time=POS_TIME,
        roi_name=ROI,
        window_start=WINDOW_START,
        window_end=WINDOW_END,
    )


# ---------------------------------------------------------------------------
# 1. Clean episode
# ---------------------------------------------------------------------------

def test_single_clean_episode_returns_its_time_range():
    tracks = [
        _track("person", ROI, t0_off=130, t1_off=170),   # ~14:02:10..14:02:50
    ]
    ep = _select(tracks)
    assert ep.ambiguous is False
    assert ep.reason == "clean_episode"
    assert ep.start == WINDOW_START + timedelta(seconds=130)
    assert ep.end == WINDOW_START + timedelta(seconds=170)
    # Coverage = 40s / 300s
    assert 0.13 < ep.coverage_ratio < 0.14


def test_two_close_tracks_merge_into_one_episode():
    """Two person tracks 3s apart should merge (gap < MERGE_GAP_SEC=5s)."""
    tracks = [
        _track("person", ROI, t0_off=140, t1_off=145),
        _track("person", ROI, t0_off=148, t1_off=160),
    ]
    ep = _select(tracks)
    assert ep.ambiguous is False
    assert ep.reason == "clean_episode"
    assert ep.start == WINDOW_START + timedelta(seconds=140)
    assert ep.end == WINDOW_START + timedelta(seconds=160)


# ---------------------------------------------------------------------------
# 2. Multiple-groups ambiguity
# ---------------------------------------------------------------------------

def test_two_separate_groups_overlapping_pos_anchor_are_ambiguous():
    """Two distinct groups, both within POS anchor tolerance, → ambiguous."""
    tracks = [
        _track("person", ROI, t0_off=130, t1_off=140),  # group 1
        _track("person", ROI, t0_off=160, t1_off=175),  # group 2 (gap > 5s)
    ]
    ep = _select(tracks)
    assert ep.ambiguous is True
    assert ep.reason == "multiple_groups"
    # Fallback widens to the POS window
    assert ep.start == WINDOW_START
    assert ep.end == WINDOW_END


# ---------------------------------------------------------------------------
# 3. Long-continuous ambiguity
# ---------------------------------------------------------------------------

def test_long_continuous_episode_is_ambiguous():
    """A 120s person-in-zone block (> MAX_EPISODE_SEC=90s) is suspicious."""
    tracks = [
        _track("person", ROI, t0_off=90, t1_off=210),  # 120s span
    ]
    ep = _select(tracks)
    assert ep.ambiguous is True
    assert ep.reason == "long_continuous"
    assert ep.start == WINDOW_START
    assert ep.end == WINDOW_END


# ---------------------------------------------------------------------------
# 4. No-activity fallback
# ---------------------------------------------------------------------------

def test_no_person_tracks_in_zone_falls_back_unambiguously():
    # No tracks at all
    ep = _select([])
    assert ep.ambiguous is False
    assert ep.reason == "no_activity"
    assert ep.start == WINDOW_START
    assert ep.end == WINDOW_END
    assert ep.coverage_ratio == 0.0


def test_person_in_a_different_zone_is_ignored():
    tracks = [
        _track("person", "other_zone", t0_off=130, t1_off=170),
    ]
    ep = _select(tracks)
    assert ep.reason == "no_activity"


def test_item_track_in_zone_is_not_a_person():
    """sco_item_* / sco_generic_* tracks are NOT persons."""
    tracks = [
        _track("sco_generic_products", ROI, t0_off=130, t1_off=170),
        _track("sco_item_001", ROI, t0_off=130, t1_off=170),
    ]
    ep = _select(tracks)
    assert ep.reason == "no_activity"


# ---------------------------------------------------------------------------
# 5. Anchor outside groups
# ---------------------------------------------------------------------------

def test_person_in_zone_far_from_pos_anchor_does_not_qualify():
    """Person active at 14:00:00–14:00:10 but POS is 14:02:30. Tolerance
    is 30s, so far away → no group qualifies."""
    tracks = [
        _track("person", ROI, t0_off=0, t1_off=10),
    ]
    ep = _select(tracks)
    assert ep.reason == "anchor_outside_groups"
    assert ep.ambiguous is False
    assert ep.start == WINDOW_START
    assert ep.end == WINDOW_END


# ---------------------------------------------------------------------------
# 6. Person label variants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label", ["person", "Person", "PERSON",
                                    "customer_person", "person_hand"])
def test_person_label_variants_are_recognised(label):
    tracks = [_track(label, ROI, t0_off=140, t1_off=160)]
    ep = _select(tracks)
    assert ep.reason == "clean_episode"


@pytest.mark.parametrize("label", ["bag", "product", "receipt", "item",
                                    "sco_item_001"])
def test_non_person_labels_are_ignored(label):
    tracks = [_track(label, ROI, t0_off=140, t1_off=160)]
    ep = _select(tracks)
    assert ep.reason == "no_activity"


# ---------------------------------------------------------------------------
# 7. Track dataclass (not dict) input
# ---------------------------------------------------------------------------

def test_track_dataclass_input_also_works():
    from perception.schemas import Track
    tracks = [
        Track(track_id="t1", label="person",
              first_seen_ts=WINDOW_START + timedelta(seconds=140),
              last_seen_ts=WINDOW_START + timedelta(seconds=170),
              zones=[ROI]),
    ]
    from perception.episode_selector import select_sco_episode
    ep = select_sco_episode(tracks, pos_time=POS_TIME, roi_name=ROI,
                            window_start=WINDOW_START, window_end=WINDOW_END)
    assert ep.reason == "clean_episode"


# ---------------------------------------------------------------------------
# 8. to_dict serialisation
# ---------------------------------------------------------------------------

def test_to_dict_is_serialisable():
    import json
    tracks = [_track("person", ROI, t0_off=140, t1_off=160)]
    ep = _select(tracks)
    d = ep.to_dict()
    json.dumps(d)  # would raise if not JSON-serialisable
    assert d["reason"] == "clean_episode"
    assert d["ambiguous"] is False
    assert isinstance(d["start"], str) and isinstance(d["end"], str)
