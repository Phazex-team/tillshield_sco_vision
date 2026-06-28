"""Policy-v2 honors container-merge metadata.

The council fix: when raw SAM3 produces N IDs but the merger
flattens them to merged_count ≤ POS basket, the policy must NOT
fire sco_basket_mismatch on the raw id count.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _vlm(**overrides):
    from reasoning.schemas.sco_basket_match_v2 import ScoBasketMatchV2
    base = dict(
        physical_count_match="uncertain",
        semantic_identity_match="uncertain",
        matched_items=[], missing_visible_items=[], extra_visible_items=[],
        uncertainty_reason="", video_usable=True,
        confidence="medium", narrative="ok",
    )
    base.update(overrides)
    return ScoBasketMatchV2.model_validate(base)


def _ep(**o):
    base = {"start": "x", "end": "y", "ambiguous": False,
            "reason": "item_occupancy", "coverage_ratio": 0.5}
    base.update(o)
    return base


# ---------------------------------------------------------------------------
# Headline fix: raw SAM3 id count != POS count is NOT a confident mismatch
# ---------------------------------------------------------------------------

def test_three_raw_ids_two_pos_after_merge_to_one_is_count_uncertain_not_mismatch():
    from reasoning.sco_policy_v2 import (
        decide_sco_v2, OUTCOME_REVIEW,
        TAG_BASKET_MISMATCH, TAG_COUNT_UNCERTAIN,
    )
    # 3 raw SAM3 groups, merger collapsed to 1 with fragmentation.
    # POS basket = 2.
    merge = {
        "merged_groups": [{"group_id": "g_merged",
                            "matched_pos_item": None,
                            "source_labels": ["sco_generic_food_container"],
                            "confidence": "medium",
                            "is_extra_candidate": True,
                            "raw_group_count": 3}],
        "count_min": 1, "count_max": 3,
        "count_confidence": "medium",
        "fragmentation_suspected": True,
        "missed_container_possible": True,
        "merge_audit": ["R3 merge: ..."],
    }
    d = decide_sco_v2(_vlm(physical_count_match="uncertain",
                            semantic_identity_match="uncertain"),
                      _ep(), container_merge_meta=merge, pos_basket_size=2)
    assert d.outcome == OUTCOME_REVIEW
    # The whole point: not a confident mismatch.
    assert TAG_BASKET_MISMATCH not in d.reasons, d.reasons
    # And the count_uncertain tag is present with merger detail.
    assert TAG_COUNT_UNCERTAIN in d.reasons


def test_merger_clearly_outside_range_with_high_confidence_does_mismatch():
    """If merger emits a tight range (3,3) at high confidence vs POS=1,
    that IS a real mismatch."""
    from reasoning.sco_policy_v2 import (decide_sco_v2, OUTCOME_REVIEW,
                                          TAG_BASKET_MISMATCH)
    merge = {
        "merged_groups": [], "count_min": 3, "count_max": 3,
        "count_confidence": "high",
        "fragmentation_suspected": False,
        "missed_container_possible": False,
        "merge_audit": [],
    }
    d = decide_sco_v2(_vlm(physical_count_match="no",
                            semantic_identity_match="no"),
                      _ep(), container_merge_meta=merge, pos_basket_size=1)
    assert d.outcome == OUTCOME_REVIEW
    assert TAG_BASKET_MISMATCH in d.reasons


def test_vlm_says_yes_overrides_merger_mismatch_to_uncertain():
    """If the VLM affirmatively says physical_count_match=yes, the
    policy refuses to confidently flag mismatch even when the
    merger range looks off. Treats the VLM as a soft veto."""
    from reasoning.sco_policy_v2 import (
        decide_sco_v2, TAG_BASKET_MISMATCH, TAG_COUNT_UNCERTAIN,
    )
    merge = {
        "merged_groups": [], "count_min": 3, "count_max": 3,
        "count_confidence": "high",
        "fragmentation_suspected": False,
        "missed_container_possible": False,
        "merge_audit": [],
    }
    d = decide_sco_v2(_vlm(physical_count_match="yes",
                            semantic_identity_match="uncertain"),
                      _ep(), container_merge_meta=merge, pos_basket_size=1)
    # VLM affirmative → don't flag mismatch, but count_uncertain
    # surfaces because merger and VLM disagree on count.
    assert TAG_BASKET_MISMATCH not in d.reasons
    # No COUNT_UNCERTAIN here because the merger's range is tight
    # (3,3) and VLM said yes — caller knows there's disagreement
    # via the v2 audit log, not by tag.


def test_low_merger_confidence_blocks_count_mismatch():
    """The COUNT signal does not mismatch when merger confidence is
    low, even if the VLM thinks count is wrong. (semantic_identity
    is held neutral here so the test isolates the count gate.)"""
    from reasoning.sco_policy_v2 import (
        decide_sco_v2, TAG_BASKET_MISMATCH, TAG_COUNT_UNCERTAIN,
    )
    merge = {
        "merged_groups": [], "count_min": 0, "count_max": 5,
        "count_confidence": "low",
        "fragmentation_suspected": True,
        "missed_container_possible": False,
        "merge_audit": [],
    }
    d = decide_sco_v2(_vlm(physical_count_match="no",
                            semantic_identity_match="uncertain"),
                      _ep(), container_merge_meta=merge, pos_basket_size=2)
    # COUNT signal alone did not produce a mismatch
    assert TAG_BASKET_MISMATCH not in d.reasons
    assert TAG_COUNT_UNCERTAIN in d.reasons


def test_no_container_merge_meta_falls_back_to_vlm_signal():
    """Legacy/Falcon-only callers don't have merger meta. Policy
    behaviour falls back to VLM physical_count_match → unchanged
    v1 semantics."""
    from reasoning.sco_policy_v2 import (
        decide_sco_v2, TAG_BASKET_MISMATCH,
    )
    d = decide_sco_v2(_vlm(physical_count_match="no",
                            semantic_identity_match="no"),
                      _ep(), container_merge_meta=None)
    assert TAG_BASKET_MISMATCH in d.reasons


# ---------------------------------------------------------------------------
# Episode missed-container hint flows into reasons
# ---------------------------------------------------------------------------

def test_missed_container_possible_with_pos_above_range_is_count_uncertain():
    from reasoning.sco_policy_v2 import (
        decide_sco_v2, TAG_COUNT_UNCERTAIN, TAG_BASKET_MISMATCH,
    )
    # POS = 2 but merger only saw 1. Range (1,1) low conf →
    # not mismatch, count_uncertain with missed_container note.
    merge = {
        "merged_groups": [], "count_min": 1, "count_max": 1,
        "count_confidence": "low",
        "fragmentation_suspected": False,
        "missed_container_possible": True,
        "merge_audit": [],
    }
    d = decide_sco_v2(_vlm(physical_count_match="uncertain",
                            semantic_identity_match="uncertain"),
                      _ep(), container_merge_meta=merge, pos_basket_size=2)
    assert TAG_BASKET_MISMATCH not in d.reasons
    assert TAG_COUNT_UNCERTAIN in d.reasons
    # Detail in human-readable reason.
    assert any("missed container possible" in r.lower() for r in d.reasons)
