"""SCO item de-duplication (v1.1) tests.

The grouper must:
  * fold a POS-specific track and a spatially-overlapping generic
    track into one canonical group with matched_pos_item populated
    (no spurious extra candidate),
  * keep a spatially-distinct generic track as its own extra
    candidate group,
  * never treat person/receipt tracks as items,
  * resolve matched_pos_item via the POS basket index encoded in
    the ``sco_item_NNN`` label.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Common bbox geometry: a tight ROI in a 1280x720 frame.
BX_BIRIYANI = [400, 300, 520, 420]            # 120x120 box
BX_BIRIYANI_NUDGED = [410, 305, 530, 425]     # heavy IoU with BX_BIRIYANI
BX_FAR_RIGHT = [900, 300, 1020, 420]          # disjoint from BX_BIRIYANI

BASE = datetime(2026, 6, 27, 14, 2, 30)


def _det(idx, label, bbox, *, score=0.85, ts_off=0):
    return {
        "label": label, "score": score,
        "bbox_xyxy": list(bbox),
        "frame_id": f"frame_{idx:06d}",
        "frame_idx": idx,
        "ts": (BASE + timedelta(seconds=ts_off)).isoformat(),
    }


def _track(track_id, label, det_indices, *, t0=0, t1=10, zones=("sco_audit_zone",)):
    return {
        "track_id": track_id, "label": label,
        "first_seen_ts": (BASE + timedelta(seconds=t0)).isoformat(),
        "last_seen_ts":  (BASE + timedelta(seconds=t1)).isoformat(),
        "detections": list(det_indices),
        "zones": list(zones),
    }


def _basket(*descs):
    return [{"description": d} for d in descs]


def _call(detections, tracks, basket=None):
    from perception.item_grouping import group_sco_items
    return group_sco_items(detections, tracks,
                            pos_basket=basket or _basket())


# ---------------------------------------------------------------------------
# 1. Overlapping POS + generic tracks fold into one canonical group
# ---------------------------------------------------------------------------

def test_overlapping_pos_and_generic_collapse_to_one_matched_group():
    detections = [
        _det(0, "sco_item_000", BX_BIRIYANI, score=0.91, ts_off=2),
        _det(1, "sco_generic_products", BX_BIRIYANI_NUDGED,
             score=0.62, ts_off=3),
    ]
    tracks = [
        _track("t_pos",     "sco_item_000",         [0], t0=2, t1=8),
        _track("t_generic", "sco_generic_products", [1], t0=3, t1=7),
    ]
    basket = _basket("Biriyani Hot Food", "Curry Hot Food")
    groups = _call(detections, tracks, basket=basket)

    assert len(groups) == 1, f"expected single canonical group, got {groups}"
    g = groups[0]
    assert g["matched_pos_item"] == "Biriyani Hot Food"
    assert g["matched_pos_index"] == 0
    assert g["is_extra_candidate"] is False
    # Both labels and both tracks recorded for audit
    assert set(g["source_labels"]) == {"sco_item_000", "sco_generic_products"}
    assert set(g["track_ids"]) == {"t_pos", "t_generic"}
    assert g["confidence"] == "high"  # multi-label corroboration


def test_overlapping_default_item_label_also_collapses():
    """The Falcon DEFAULT 'item' category can also fire on the same
    physical object — it must merge into the POS group, not show as
    extra."""
    detections = [
        _det(0, "sco_item_001", BX_BIRIYANI, score=0.88, ts_off=2),
        _det(1, "item",         BX_BIRIYANI_NUDGED, score=0.55, ts_off=3),
    ]
    tracks = [
        _track("t_pos",     "sco_item_001", [0], t0=2, t1=8),
        _track("t_default", "item",         [1], t0=3, t1=7),
    ]
    basket = _basket("Biriyani Hot Food", "Curry Hot Food")
    groups = _call(detections, tracks, basket=basket)

    assert len(groups) == 1
    assert groups[0]["matched_pos_item"] == "Curry Hot Food"
    assert "item" in groups[0]["source_labels"]


# ---------------------------------------------------------------------------
# 2. Spatially-separate generic track is an extra candidate
# ---------------------------------------------------------------------------

def test_generic_track_far_from_pos_track_is_extra_candidate():
    detections = [
        _det(0, "sco_item_000",         BX_BIRIYANI,  score=0.9, ts_off=2),
        _det(1, "sco_generic_products", BX_FAR_RIGHT, score=0.7, ts_off=3),
    ]
    tracks = [
        _track("t_pos",     "sco_item_000",         [0], t0=2, t1=8),
        _track("t_generic", "sco_generic_products", [1], t0=3, t1=7),
    ]
    basket = _basket("Biriyani Hot Food")
    groups = _call(detections, tracks, basket=basket)

    assert len(groups) == 2
    matched = [g for g in groups if not g["is_extra_candidate"]]
    extras = [g for g in groups if g["is_extra_candidate"]]
    assert len(matched) == 1 and len(extras) == 1
    assert matched[0]["matched_pos_item"] == "Biriyani Hot Food"
    assert matched[0]["source_labels"] == ["sco_item_000"]
    assert extras[0]["matched_pos_item"] is None
    assert extras[0]["source_labels"] == ["sco_generic_products"]
    assert extras[0]["confidence"] == "low"


# ---------------------------------------------------------------------------
# 3. Time-separated tracks (same bbox, far apart in time) do NOT merge
# ---------------------------------------------------------------------------

def test_time_separated_tracks_do_not_merge_even_if_bbox_overlaps():
    """A POS-matched item leaves the zone; a different generic item
    later occupies the same physical spot. They must not collapse
    because their time windows don't overlap within the gap."""
    detections = [
        _det(0, "sco_item_000",         BX_BIRIYANI, score=0.9, ts_off=2),
        _det(1, "sco_generic_products", BX_BIRIYANI_NUDGED, score=0.7,
             ts_off=40),
    ]
    tracks = [
        _track("t_pos",     "sco_item_000",         [0], t0=2,  t1=10),
        _track("t_generic", "sco_generic_products", [1], t0=35, t1=42),
    ]
    basket = _basket("Biriyani Hot Food")
    groups = _call(detections, tracks, basket=basket)

    assert len(groups) == 2
    extras = [g for g in groups if g["is_extra_candidate"]]
    assert len(extras) == 1
    assert extras[0]["source_labels"] == ["sco_generic_products"]


# ---------------------------------------------------------------------------
# 4. Person / receipt tracks are not items
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("noise_label", ["person", "Person", "PERSON",
                                          "receipt"])
def test_person_and_receipt_tracks_are_skipped(noise_label):
    detections = [
        _det(0, "sco_item_000", BX_BIRIYANI, score=0.9, ts_off=2),
        _det(1, noise_label,    BX_BIRIYANI_NUDGED, score=0.8, ts_off=3),
    ]
    tracks = [
        _track("t_pos",   "sco_item_000", [0], t0=2, t1=8),
        _track("t_noise", noise_label,    [1], t0=3, t1=7),
    ]
    basket = _basket("Biriyani Hot Food")
    groups = _call(detections, tracks, basket=basket)
    # Only the POS group survives; the noise track is dropped entirely.
    assert len(groups) == 1
    assert groups[0]["matched_pos_item"] == "Biriyani Hot Food"
    assert "t_noise" not in groups[0]["track_ids"]
    assert noise_label not in groups[0]["source_labels"]


# ---------------------------------------------------------------------------
# 5. Multiple POS items all surface
# ---------------------------------------------------------------------------

def test_two_distinct_pos_items_both_appear_with_distinct_bbox():
    detections = [
        _det(0, "sco_item_000", BX_BIRIYANI,  score=0.9, ts_off=2),
        _det(1, "sco_item_001", BX_FAR_RIGHT, score=0.9, ts_off=2),
    ]
    tracks = [
        _track("t_a", "sco_item_000", [0], t0=2, t1=8),
        _track("t_b", "sco_item_001", [1], t0=2, t1=8),
    ]
    basket = _basket("Biriyani Hot Food", "Curry Hot Food")
    groups = _call(detections, tracks, basket=basket)
    assert len(groups) == 2
    by_name = {g["matched_pos_item"]: g for g in groups}
    assert set(by_name.keys()) == {"Biriyani Hot Food", "Curry Hot Food"}
    for g in groups:
        assert g["is_extra_candidate"] is False


# ---------------------------------------------------------------------------
# 6. Empty input / malformed input is safe
# ---------------------------------------------------------------------------

def test_empty_inputs_return_empty_list():
    from perception.item_grouping import group_sco_items
    assert group_sco_items([], []) == []
    assert group_sco_items(None, None) == []  # type: ignore


def test_track_with_no_resolvable_detection_is_dropped():
    # det_indices points to a missing index → grouper should skip
    detections = [_det(0, "sco_item_000", BX_BIRIYANI, score=0.9)]
    tracks = [_track("t_pos", "sco_item_000", [0], t0=2, t1=8),
              _track("t_broken", "sco_generic_products", [99], t0=2, t1=8)]
    # Fall-back: t_broken's label doesn't match any detection's label,
    # so no representative bbox -> skipped silently.
    detections_minus_match = list(detections)
    groups = _call(detections_minus_match, tracks, _basket("Biriyani"))
    assert len(groups) == 1
    assert groups[0]["matched_pos_item"] == "Biriyani"


# ---------------------------------------------------------------------------
# 7. group_id is stable + sequential
# ---------------------------------------------------------------------------

def test_group_ids_are_stable_and_sequential():
    detections = [
        _det(0, "sco_item_000", BX_BIRIYANI, score=0.9),
        _det(1, "sco_item_001", BX_FAR_RIGHT, score=0.9),
        _det(2, "sco_generic_products", [0, 0, 50, 50], score=0.7,
             ts_off=2),  # extra candidate, disjoint
    ]
    tracks = [
        _track("t_a", "sco_item_000", [0]),
        _track("t_b", "sco_item_001", [1]),
        _track("t_x", "sco_generic_products", [2], t0=2, t1=5),
    ]
    basket = _basket("A", "B")
    groups = _call(detections, tracks, basket=basket)
    ids = [g["group_id"] for g in groups]
    assert ids == ["sco_group_001", "sco_group_002", "sco_group_003"]


# ---------------------------------------------------------------------------
# 8. Convenience splitters
# ---------------------------------------------------------------------------

def test_matched_and_extra_split_helpers():
    from perception.item_grouping import matched_groups, extra_groups
    detections = [
        _det(0, "sco_item_000",         BX_BIRIYANI,  score=0.9),
        _det(1, "sco_generic_products", BX_FAR_RIGHT, score=0.7, ts_off=2),
    ]
    tracks = [
        _track("t_a", "sco_item_000",         [0]),
        _track("t_x", "sco_generic_products", [1], t0=2, t1=5),
    ]
    groups = _call(detections, tracks, _basket("Biriyani"))
    assert len(matched_groups(groups)) == 1
    assert len(extra_groups(groups)) == 1
    assert matched_groups(groups)[0]["matched_pos_item"] == "Biriyani"
