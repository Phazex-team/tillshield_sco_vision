"""sco_policy_v2 tests — physical_count vs semantic_identity, closed-container path."""
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
        physical_count_match="yes",
        semantic_identity_match="yes",
        matched_items=[
            {"pos_item": "X", "group_id": "sco_group_001",
             "visible_count_class": "one"},
        ],
        missing_visible_items=[],
        extra_visible_items=[],
        uncertainty_reason="",
        video_usable=True,
        confidence="high",
        narrative="ok",
    )
    base.update(overrides)
    return ScoBasketMatchV2.model_validate(base)


def _ep(**o):
    base = {"start": "x", "end": "y", "ambiguous": False,
            "reason": "clean_episode", "coverage_ratio": 0.5}
    base.update(o)
    return base


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_all_gates_clean_yields_verified():
    from reasoning.sco_policy_v2 import (decide_sco_v2, OUTCOME_VERIFIED,
                                          TAG_BASKET_MATCH)
    d = decide_sco_v2(_vlm(), _ep())
    assert d.outcome == OUTCOME_VERIFIED
    assert TAG_BASKET_MATCH in d.reasons
    assert d.policy_version == "sco_v2"


# ---------------------------------------------------------------------------
# The headline fix: closed-container uncertainty is REVIEW + identity tag,
# NOT a basket_mismatch
# ---------------------------------------------------------------------------

def test_closed_container_case_yields_identity_uncertain_review():
    """The council's hot-food scenario. SAM3 sees 2 takeaway
    containers. VLM cannot tell biriyani vs curry inside closed
    boxes. v2 must produce REVIEW with sco_identity_uncertain —
    not sco_basket_mismatch."""
    from reasoning.sco_policy_v2 import (
        decide_sco_v2, OUTCOME_REVIEW,
        TAG_IDENTITY_UNCERTAIN, TAG_BASKET_MISMATCH,
    )
    vlm = _vlm(
        physical_count_match="yes",      # 2 visible, 2 POS — fine
        semantic_identity_match="uncertain",
        # NO mismatch — closed containers
        uncertainty_reason="items inside closed takeaway containers",
        confidence="high",
        narrative="Two takeaway containers visible; contents not legible.",
    )
    d = decide_sco_v2(vlm, _ep())
    assert d.outcome == OUTCOME_REVIEW
    assert TAG_IDENTITY_UNCERTAIN in d.reasons
    # The critical anti-regression: do NOT call this a mismatch
    assert TAG_BASKET_MISMATCH not in d.reasons, (
        "closed-container case must not be tagged as basket_mismatch")


def test_closed_container_does_not_flag_missing_when_identity_uncertain():
    """Even if the VLM reports missing_visible_items (shouldn't, but
    legacy providers might), the policy must suppress the missing tag
    when semantic_identity_match=uncertain — otherwise closed
    containers would always get sco_missing_items false flags."""
    from reasoning.sco_policy_v2 import decide_sco_v2, TAG_MISSING_ITEMS
    vlm = _vlm(
        physical_count_match="yes",
        semantic_identity_match="uncertain",
        missing_visible_items=[
            {"pos_item": "Biriyani Hot Food", "reason": "closed container"},
            {"pos_item": "Curry Hot Food", "reason": "closed container"},
        ],
        uncertainty_reason="items inside closed takeaway containers",
    )
    d = decide_sco_v2(vlm, _ep())
    assert TAG_MISSING_ITEMS not in d.reasons


def test_semantic_identity_no_is_still_a_basket_mismatch():
    """A real semantic contradiction (POS says electronics, you see
    food) is still a mismatch. We only soften the *uncertain* case."""
    from reasoning.sco_policy_v2 import (decide_sco_v2, OUTCOME_REVIEW,
                                          TAG_BASKET_MISMATCH)
    d = decide_sco_v2(_vlm(semantic_identity_match="no"), _ep())
    assert d.outcome == OUTCOME_REVIEW
    assert TAG_BASKET_MISMATCH in d.reasons


# ---------------------------------------------------------------------------
# Physical count
# ---------------------------------------------------------------------------

def test_physical_count_no_yields_basket_mismatch():
    from reasoning.sco_policy_v2 import (decide_sco_v2, OUTCOME_REVIEW,
                                          TAG_BASKET_MISMATCH)
    d = decide_sco_v2(_vlm(physical_count_match="no"), _ep())
    assert d.outcome == OUTCOME_REVIEW
    assert TAG_BASKET_MISMATCH in d.reasons


def test_physical_count_uncertain_yields_count_uncertain_tag():
    from reasoning.sco_policy_v2 import (decide_sco_v2, OUTCOME_REVIEW,
                                          TAG_COUNT_UNCERTAIN,
                                          TAG_BASKET_MISMATCH)
    d = decide_sco_v2(_vlm(physical_count_match="uncertain"), _ep())
    assert d.outcome == OUTCOME_REVIEW
    assert TAG_COUNT_UNCERTAIN in d.reasons
    # And NOT mismatch — uncertain is its own category
    assert TAG_BASKET_MISMATCH not in d.reasons


# ---------------------------------------------------------------------------
# Other gates
# ---------------------------------------------------------------------------

def test_video_unusable_returns_invalid_video():
    from reasoning.sco_policy_v2 import (decide_sco_v2, OUTCOME_INVALID_VIDEO,
                                          TAG_BAD_FOOTAGE)
    d = decide_sco_v2(_vlm(video_usable=False), _ep())
    assert d.outcome == OUTCOME_INVALID_VIDEO
    assert TAG_BAD_FOOTAGE in d.reasons


def test_ambiguous_episode_yields_review():
    from reasoning.sco_policy_v2 import (decide_sco_v2, OUTCOME_REVIEW,
                                          TAG_EPISODE_AMBIGUOUS)
    d = decide_sco_v2(_vlm(), _ep(ambiguous=True, reason="multiple_groups"))
    assert d.outcome == OUTCOME_REVIEW
    assert TAG_EPISODE_AMBIGUOUS in d.reasons


def test_low_episode_coverage_yields_review():
    from reasoning.sco_policy_v2 import (decide_sco_v2, TAG_EPISODE_SHORT)
    d = decide_sco_v2(_vlm(), _ep(coverage_ratio=0.01))
    assert TAG_EPISODE_SHORT in d.reasons


def test_low_confidence_yields_review():
    from reasoning.sco_policy_v2 import (decide_sco_v2, TAG_LOW_CONFIDENCE)
    d = decide_sco_v2(_vlm(confidence="low"), _ep())
    assert TAG_LOW_CONFIDENCE in d.reasons


def test_extras_with_count_match_still_flag_extras():
    from reasoning.sco_policy_v2 import (decide_sco_v2, OUTCOME_REVIEW,
                                          TAG_EXTRA_CANDIDATES)
    d = decide_sco_v2(_vlm(
        physical_count_match="yes",
        extra_visible_items=[{"group_id": "sco_group_005",
                              "description": "unknown box"}],
    ), _ep())
    assert d.outcome == OUTCOME_REVIEW
    assert TAG_EXTRA_CANDIDATES in d.reasons


# ---------------------------------------------------------------------------
# Closed-container suppression (post-replay hot-food fix)
# ---------------------------------------------------------------------------

def test_closed_container_with_extras_does_not_flag_extras():
    """Hot-food replay: SAM3 fragmented 2 takeaway containers into 3
    raw groups. VLM said count matches POS basket and identity is
    uncertain (closed boxes). The two unmatched generic container
    groups must NOT be promoted to sco_extra_candidates — they are
    perception artefacts, not POS-vs-video mismatches."""
    from reasoning.sco_policy_v2 import (
        decide_sco_v2, OUTCOME_REVIEW,
        TAG_EXTRA_CANDIDATES, TAG_MISSING_ITEMS, TAG_BASKET_MISMATCH,
        TAG_IDENTITY_UNCERTAIN,
    )
    vlm = _vlm(
        physical_count_match="yes",
        semantic_identity_match="uncertain",
        extra_visible_items=[
            {"group_id": "sco_group_002", "description": "takeaway container"},
            {"group_id": "sco_group_003", "description": "takeaway container"},
        ],
        uncertainty_reason="items inside closed takeaway containers",
        narrative="Two takeaway containers visible; contents not legible.",
    )
    d = decide_sco_v2(vlm, _ep())
    assert d.outcome == OUTCOME_REVIEW
    assert TAG_IDENTITY_UNCERTAIN in d.reasons
    assert TAG_EXTRA_CANDIDATES not in d.reasons, (
        "closed-container case must not be tagged as extra candidates")
    assert TAG_MISSING_ITEMS not in d.reasons
    assert TAG_BASKET_MISMATCH not in d.reasons


def test_closed_container_with_merger_range_does_not_flag_mismatch():
    """Same scenario but with container_merge_meta: merger reports a
    2-3 range (fragmentation suspected). Policy must still NOT emit
    basket_mismatch / extra_candidates / missing_items — only the
    identity_uncertain (+ optional count_uncertain) tags."""
    from reasoning.sco_policy_v2 import (
        decide_sco_v2, OUTCOME_REVIEW,
        TAG_BASKET_MISMATCH, TAG_EXTRA_CANDIDATES, TAG_MISSING_ITEMS,
        TAG_IDENTITY_UNCERTAIN, TAG_COUNT_UNCERTAIN,
    )
    vlm = _vlm(
        physical_count_match="yes",
        semantic_identity_match="uncertain",
        extra_visible_items=[
            {"group_id": "sco_group_002", "description": "takeaway container"},
        ],
        uncertainty_reason="items inside closed takeaway containers",
        narrative="Two takeaway containers visible; contents not legible.",
    )
    merge_meta = {
        "count_min": 2, "count_max": 3,
        "count_confidence": "medium",
        "fragmentation_suspected": True,
        "missed_container_possible": False,
    }
    d = decide_sco_v2(vlm, _ep(),
                      container_merge_meta=merge_meta,
                      pos_basket_size=2)
    assert d.outcome == OUTCOME_REVIEW
    assert TAG_IDENTITY_UNCERTAIN in d.reasons
    assert TAG_COUNT_UNCERTAIN in d.reasons
    assert TAG_BASKET_MISMATCH not in d.reasons
    assert TAG_EXTRA_CANDIDATES not in d.reasons
    assert TAG_MISSING_ITEMS not in d.reasons


def test_genuine_semantic_mismatch_still_flags_basket_mismatch():
    """Regression guard: identity=no is still a mismatch, suppression
    only fires when identity=uncertain."""
    from reasoning.sco_policy_v2 import (
        decide_sco_v2, OUTCOME_REVIEW,
        TAG_BASKET_MISMATCH, TAG_EXTRA_CANDIDATES,
    )
    vlm = _vlm(
        physical_count_match="yes",
        semantic_identity_match="no",
        extra_visible_items=[{"group_id": "g", "description": "bottle"}],
        uncertainty_reason="VLM observed electronics; POS says food",
    )
    d = decide_sco_v2(vlm, _ep())
    assert d.outcome == OUTCOME_REVIEW
    assert TAG_BASKET_MISMATCH in d.reasons
    # extras still flagged here because suppression is gated on
    # identity=="uncertain" only
    assert TAG_EXTRA_CANDIDATES in d.reasons


def test_genuine_extras_with_identity_yes_still_flag_extras():
    """Genuine extras (identity=yes, count=yes, but VLM lists an
    extra it confidently identified) must still flag extras."""
    from reasoning.sco_policy_v2 import (
        decide_sco_v2, OUTCOME_REVIEW, TAG_EXTRA_CANDIDATES,
    )
    vlm = _vlm(
        physical_count_match="yes",
        semantic_identity_match="yes",
        extra_visible_items=[
            {"group_id": "g", "description": "chocolate bar"},
        ],
    )
    d = decide_sco_v2(vlm, _ep())
    assert d.outcome == OUTCOME_REVIEW
    assert TAG_EXTRA_CANDIDATES in d.reasons


def test_no_vlm_output_yields_review():
    from reasoning.sco_policy_v2 import decide_sco_v2, TAG_NO_VLM
    d = decide_sco_v2(None, _ep())
    assert TAG_NO_VLM in d.reasons


# ---------------------------------------------------------------------------
# Sanity: never HIGH_RISK_REVIEW
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("count", ["yes", "no", "uncertain"])
@pytest.mark.parametrize("ident", ["yes", "no", "uncertain"])
@pytest.mark.parametrize("conf", ["high", "medium", "low"])
@pytest.mark.parametrize("video", [True, False])
@pytest.mark.parametrize("ambig", [True, False])
def test_never_emits_high_risk_review(count, ident, conf, video, ambig):
    from reasoning.sco_policy_v2 import (decide_sco_v2, OUTCOME_VERIFIED,
                                          OUTCOME_REVIEW,
                                          OUTCOME_INVALID_VIDEO)
    d = decide_sco_v2(
        _vlm(physical_count_match=count, semantic_identity_match=ident,
             confidence=conf, video_usable=video),
        _ep(ambiguous=ambig, coverage_ratio=0.5),
    )
    assert d.outcome in {OUTCOME_VERIFIED, OUTCOME_REVIEW,
                          OUTCOME_INVALID_VIDEO}
    assert d.outcome != "HIGH_RISK_REVIEW"
