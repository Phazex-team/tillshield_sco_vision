"""perception.container_merge tests.

Pins the SAM3 fragmentation-handling rules: same physical container
appearing under multiple SAM3 IDs across time merges into one
merged_group; truly co-existing containers stay separate.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


BASE = datetime(2026, 6, 28, 18, 0, 30)


def _g(gid, *, label, bbox, t0_off, t1_off, sam3_obj=None,
       confidence="medium", matched_pos=None, repr_score=0.8):
    return {
        "group_id": gid,
        "matched_pos_item": matched_pos,
        "matched_pos_index": None,
        "source_labels": [label],
        "track_ids": [f"sam3_obj_{sam3_obj:04d}"] if sam3_obj is not None
                     else [],
        "first_seen_ts": (BASE + timedelta(seconds=t0_off)).isoformat(),
        "last_seen_ts": (BASE + timedelta(seconds=t1_off)).isoformat(),
        "representative_bbox": list(bbox),
        "representative_frame_id": f"frame_{t0_off * 25:06d}",
        "_repr_score": repr_score,
        "confidence": confidence,
        "is_extra_candidate": matched_pos is None,
    }


def _d(*, sam3_obj, frame_idx, bbox, label="sco_generic_food_container",
       ts_off=0):
    return {
        "label": label, "score": 0.8, "bbox_xyxy": list(bbox),
        "frame_id": f"frame_{frame_idx:06d}", "frame_idx": frame_idx,
        "ts": (BASE + timedelta(seconds=ts_off)).isoformat(),
        "sam3_object_id": sam3_obj, "query": label,
    }


# ---------------------------------------------------------------------------
# R1 — only container family is considered
# ---------------------------------------------------------------------------

def test_bag_groups_are_not_merged_with_containers():
    from perception.container_merge import merge_sam3_containers
    groups = [
        _g("sco_group_001", label="sco_generic_food_container",
           bbox=[400, 300, 520, 420], t0_off=2, t1_off=5, sam3_obj=1),
        _g("sco_group_002", label="sco_generic_bag",
           bbox=[400, 300, 520, 420], t0_off=10, t1_off=12, sam3_obj=2),
    ]
    res = merge_sam3_containers(groups, [])
    # Container survives merging pool; bag is appended back as-is.
    assert any(g["group_id"] == "sco_group_001" for g in res.merged_groups)
    assert any(g["group_id"] == "sco_group_002" for g in res.merged_groups)
    assert res.count_min == 1 and res.count_max == 1


# ---------------------------------------------------------------------------
# R2 — simultaneous visibility prevents merge
# ---------------------------------------------------------------------------

def test_two_containers_co_visible_in_same_frame_stay_separate():
    from perception.container_merge import merge_sam3_containers
    bbox_a = [400, 300, 520, 420]
    bbox_b = [800, 300, 920, 420]   # spatially separate
    groups = [
        _g("sco_group_001", label="sco_generic_food_container",
           bbox=bbox_a, t0_off=2, t1_off=5, sam3_obj=1),
        _g("sco_group_002", label="sco_generic_food_container",
           bbox=bbox_b, t0_off=2, t1_off=5, sam3_obj=2),
    ]
    # Both fire in frame 50 with separated bboxes → conflict.
    detections = [
        _d(sam3_obj=1, frame_idx=50, bbox=bbox_a, ts_off=2),
        _d(sam3_obj=2, frame_idx=50, bbox=bbox_b, ts_off=2),
    ]
    res = merge_sam3_containers(groups, detections)
    assert len(res.merged_groups) == 2
    assert res.count_min == 2 and res.count_max == 2
    assert res.fragmentation_suspected is False
    assert any("never merged" in a for a in res.merge_audit)


# ---------------------------------------------------------------------------
# R3 — temporal/spatial continuity → merge
# ---------------------------------------------------------------------------

def test_two_temporally_disjoint_similar_containers_merge():
    """Same hand-held container appearing as two SAM3 IDs across an
    occlusion gap should fold into one merged container."""
    from perception.container_merge import merge_sam3_containers
    groups = [
        _g("sco_group_001", label="sco_generic_food_container",
           bbox=[400, 300, 520, 420], t0_off=2, t1_off=5, sam3_obj=1),
        _g("sco_group_002", label="sco_generic_plastic_food_box",
           bbox=[420, 305, 540, 425], t0_off=8, t1_off=12, sam3_obj=2),
    ]
    # No simultaneous detections — gap of 3s, similar size + position.
    res = merge_sam3_containers(groups, [])
    assert res.count_min == 1, res.merged_groups
    assert res.count_max == 2
    assert res.fragmentation_suspected is True
    assert res.count_confidence in ("medium", "low")
    merged = [g for g in res.merged_groups if "+" in g["group_id"]]
    assert len(merged) == 1
    assert set(merged[0]["track_ids"]) == {"sam3_obj_0001", "sam3_obj_0002"}
    assert merged[0]["raw_group_count"] == 2


def test_one_container_fragmented_into_many_sam3_ids_merges():
    from perception.container_merge import merge_sam3_containers
    groups = [
        _g(f"sco_group_{i:03d}", label="sco_generic_food_container",
           bbox=[400 + i * 5, 300, 520 + i * 5, 420],
           t0_off=2 + i * 3, t1_off=4 + i * 3,
           sam3_obj=i + 1)
        for i in range(4)
    ]
    res = merge_sam3_containers(groups, [])
    assert res.count_min == 1
    assert res.count_max == 4
    assert res.fragmentation_suspected is True


# ---------------------------------------------------------------------------
# R3 negative — wildly different sizes don't merge
# ---------------------------------------------------------------------------

def test_very_different_sizes_do_not_merge():
    from perception.container_merge import merge_sam3_containers
    groups = [
        _g("sco_group_001", label="sco_generic_food_container",
           bbox=[400, 300, 410, 310], t0_off=2, t1_off=5, sam3_obj=1),    # 10x10
        _g("sco_group_002", label="sco_generic_food_container",
           bbox=[400, 300, 600, 500], t0_off=8, t1_off=12, sam3_obj=2),   # 200x200
    ]
    res = merge_sam3_containers(groups, [])
    assert res.count_min == 2
    assert res.fragmentation_suspected is False


def test_temporal_gap_too_large_does_not_merge():
    from perception.container_merge import merge_sam3_containers
    groups = [
        _g("sco_group_001", label="sco_generic_food_container",
           bbox=[400, 300, 520, 420], t0_off=2, t1_off=5, sam3_obj=1),
        # 30s gap > MAX_FRAGMENT_GAP_SEC (8s)
        _g("sco_group_002", label="sco_generic_food_container",
           bbox=[420, 305, 540, 425], t0_off=35, t1_off=40, sam3_obj=2),
    ]
    res = merge_sam3_containers(groups, [])
    assert res.count_min == 2


def test_spatially_far_groups_do_not_merge():
    from perception.container_merge import merge_sam3_containers
    groups = [
        _g("sco_group_001", label="sco_generic_food_container",
           bbox=[100, 100, 200, 200], t0_off=2, t1_off=5, sam3_obj=1),
        # Far to the right (>250px center distance)
        _g("sco_group_002", label="sco_generic_food_container",
           bbox=[1000, 100, 1100, 200], t0_off=8, t1_off=12, sam3_obj=2),
    ]
    res = merge_sam3_containers(groups, [])
    assert res.count_min == 2


# ---------------------------------------------------------------------------
# R4 — missed_container_possible
# ---------------------------------------------------------------------------

def test_missed_container_possible_when_count_max_below_pos_size():
    from perception.container_merge import merge_sam3_containers
    groups = [
        _g("sco_group_001", label="sco_generic_food_container",
           bbox=[400, 300, 520, 420], t0_off=2, t1_off=5, sam3_obj=1,
           matched_pos="Biriyani Hot Food"),
    ]
    res = merge_sam3_containers(groups, [], pos_basket_size=2,
                                  episode_coverage_ratio=0.5)
    assert res.missed_container_possible is True
    assert res.count_confidence in ("medium", "low")


def test_missed_flag_not_set_when_coverage_too_low():
    from perception.container_merge import merge_sam3_containers
    res = merge_sam3_containers([], [], pos_basket_size=2,
                                  episode_coverage_ratio=0.01)
    assert res.missed_container_possible is False


# ---------------------------------------------------------------------------
# Empty / malformed
# ---------------------------------------------------------------------------

def test_empty_input_returns_low_confidence_empty():
    from perception.container_merge import merge_sam3_containers
    res = merge_sam3_containers([], [])
    assert res.merged_groups == []
    assert res.count_min == 0 and res.count_max == 0
    assert res.count_confidence == "low"
    assert res.fragmentation_suspected is False


def test_no_container_groups_returns_zero_with_low_conf():
    from perception.container_merge import merge_sam3_containers
    groups = [
        _g("sco_group_001", label="sco_generic_bag",
           bbox=[400, 300, 520, 420], t0_off=2, t1_off=5, sam3_obj=1),
    ]
    res = merge_sam3_containers(groups, [])
    assert res.count_min == 0
    # bag still surfaced for audit
    assert any(g["group_id"] == "sco_group_001" for g in res.merged_groups)


# ---------------------------------------------------------------------------
# Council acceptance: SAM3 3-IDs vs POS qty 2
# ---------------------------------------------------------------------------

def test_council_fragmentation_scenario_three_ids_two_pos():
    """3 raw SAM3 container groups, all similar size/aspect, no
    co-existence in any frame → merged into 1 or 2, with
    fragmentation_suspected=True and count_confidence<high.
    Policy can then NOT flag a confident mismatch."""
    from perception.container_merge import merge_sam3_containers
    groups = [
        _g("sco_group_001", label="sco_generic_food_container",
           bbox=[746, 512, 824, 575], t0_off=2, t1_off=5, sam3_obj=3),
        _g("sco_group_002", label="sco_generic_food_container",
           bbox=[750, 515, 828, 580], t0_off=7, t1_off=10, sam3_obj=4),
        _g("sco_group_003", label="sco_generic_plastic_food_box",
           bbox=[760, 520, 838, 585], t0_off=12, t1_off=15, sam3_obj=5),
    ]
    res = merge_sam3_containers(groups, [], pos_basket_size=2,
                                  episode_coverage_ratio=0.5)
    assert res.count_min == 1
    assert res.count_max == 3
    assert res.fragmentation_suspected is True
    assert res.count_confidence != "high"
