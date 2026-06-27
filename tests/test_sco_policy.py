"""Phase 6 — SCO decision policy tests.

Asserts the strict VERIFIED gates, the REVIEW catchments, the
INVALID_VIDEO outcome, and crucially that no input combination ever
produces HIGH_RISK_REVIEW.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vlm(**overrides):
    from reasoning.schemas.sco_basket_match import ScoBasketMatch
    base = dict(
        basket_match="yes",
        matched=[{"pos_item": "X", "visible_count_class": "one"}],
        missing=[],
        extras=[],
        video_usable=True,
        confidence="high",
        narrative="ok",
    )
    base.update(overrides)
    return ScoBasketMatch.model_validate(base)


def _ep(**overrides):
    base = dict(
        start="2026-06-15T14:02:10",
        end="2026-06-15T14:02:50",
        ambiguous=False,
        reason="clean_episode",
        coverage_ratio=0.50,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. VERIFIED — only with all five gates clean
# ---------------------------------------------------------------------------

def test_all_gates_clean_yields_verified():
    from reasoning.sco_policy import decide_sco, OUTCOME_VERIFIED, TAG_BASKET_MATCH
    d = decide_sco(_vlm(), _ep())
    assert d.outcome == OUTCOME_VERIFIED
    assert TAG_BASKET_MATCH in d.reasons


# ---------------------------------------------------------------------------
# 2. INVALID_VIDEO — wins over everything else
# ---------------------------------------------------------------------------

def test_video_unusable_returns_invalid_video():
    from reasoning.sco_policy import (decide_sco, OUTCOME_INVALID_VIDEO,
                                       TAG_BAD_FOOTAGE)
    d = decide_sco(_vlm(video_usable=False), _ep())
    assert d.outcome == OUTCOME_INVALID_VIDEO
    assert TAG_BAD_FOOTAGE in d.reasons


# ---------------------------------------------------------------------------
# 3. Each individual failing gate → REVIEW, never VERIFIED
# ---------------------------------------------------------------------------

def test_ambiguous_episode_yields_review():
    from reasoning.sco_policy import (decide_sco, OUTCOME_REVIEW,
                                       TAG_EPISODE_AMBIGUOUS)
    d = decide_sco(_vlm(), _ep(ambiguous=True, reason="multiple_groups"))
    assert d.outcome == OUTCOME_REVIEW
    assert TAG_EPISODE_AMBIGUOUS in d.reasons


def test_low_episode_coverage_yields_review():
    from reasoning.sco_policy import (decide_sco, OUTCOME_REVIEW,
                                       TAG_EPISODE_SHORT)
    d = decide_sco(_vlm(), _ep(coverage_ratio=0.01))
    assert d.outcome == OUTCOME_REVIEW
    assert TAG_EPISODE_SHORT in d.reasons


def test_low_confidence_yields_review():
    from reasoning.sco_policy import (decide_sco, OUTCOME_REVIEW,
                                       TAG_LOW_CONFIDENCE)
    d = decide_sco(_vlm(confidence="low"), _ep())
    assert d.outcome == OUTCOME_REVIEW
    assert TAG_LOW_CONFIDENCE in d.reasons


def test_basket_mismatch_yields_review():
    from reasoning.sco_policy import (decide_sco, OUTCOME_REVIEW,
                                       TAG_BASKET_MISMATCH)
    d = decide_sco(_vlm(basket_match="no"), _ep())
    assert d.outcome == OUTCOME_REVIEW
    assert TAG_BASKET_MISMATCH in d.reasons


def test_basket_uncertain_yields_review():
    from reasoning.sco_policy import (decide_sco, OUTCOME_REVIEW,
                                       TAG_BASKET_MISMATCH)
    d = decide_sco(_vlm(basket_match="uncertain"), _ep())
    assert d.outcome == OUTCOME_REVIEW
    assert TAG_BASKET_MISMATCH in d.reasons


def test_missing_items_yield_review_with_tag():
    from reasoning.sco_policy import (decide_sco, OUTCOME_REVIEW,
                                       TAG_MISSING_ITEMS)
    d = decide_sco(_vlm(missing=[{"pos_item": "MILK", "reason": "obscured"}]),
                   _ep())
    assert d.outcome == OUTCOME_REVIEW
    assert TAG_MISSING_ITEMS in d.reasons


def test_extra_candidates_yield_review_with_tag():
    from reasoning.sco_policy import (decide_sco, OUTCOME_REVIEW,
                                       TAG_EXTRA_CANDIDATES)
    d = decide_sco(_vlm(extras=[{"visible_item": "unknown box",
                                   "note": "not on bill"}]), _ep())
    assert d.outcome == OUTCOME_REVIEW
    assert TAG_EXTRA_CANDIDATES in d.reasons


# ---------------------------------------------------------------------------
# 4. Combined failures stack tags
# ---------------------------------------------------------------------------

def test_multiple_failures_stack_multiple_tags():
    from reasoning.sco_policy import (decide_sco, OUTCOME_REVIEW,
                                       TAG_BASKET_MISMATCH,
                                       TAG_EXTRA_CANDIDATES,
                                       TAG_LOW_CONFIDENCE)
    d = decide_sco(_vlm(basket_match="no", confidence="low",
                         extras=[{"visible_item": "x", "note": "y"}]),
                   _ep())
    assert d.outcome == OUTCOME_REVIEW
    for tag in (TAG_BASKET_MISMATCH, TAG_EXTRA_CANDIDATES, TAG_LOW_CONFIDENCE):
        assert tag in d.reasons


# ---------------------------------------------------------------------------
# 5. Missing VLM → REVIEW with TAG_NO_VLM (defensive)
# ---------------------------------------------------------------------------

def test_no_vlm_output_yields_review():
    from reasoning.sco_policy import decide_sco, OUTCOME_REVIEW, TAG_NO_VLM
    d = decide_sco(None, _ep())
    assert d.outcome == OUTCOME_REVIEW
    assert TAG_NO_VLM in d.reasons


def test_no_episode_meta_treated_as_low_coverage_and_not_ambiguous():
    from reasoning.sco_policy import decide_sco, OUTCOME_REVIEW, TAG_EPISODE_SHORT
    # When episode_meta is None: ambiguous defaults False, coverage defaults 0
    d = decide_sco(_vlm(), None)
    assert d.outcome == OUTCOME_REVIEW
    assert TAG_EPISODE_SHORT in d.reasons


# ---------------------------------------------------------------------------
# 6. Sanity: never HIGH_RISK_REVIEW under any input combination
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("video", [True, False])
@pytest.mark.parametrize("ambig", [True, False])
@pytest.mark.parametrize("cov", [0.0, 0.04, 0.2, 0.95])
@pytest.mark.parametrize("conf", ["high", "medium", "low"])
@pytest.mark.parametrize("match", ["yes", "no", "uncertain"])
@pytest.mark.parametrize("miss", [0, 2])
@pytest.mark.parametrize("ext", [0, 2])
def test_never_emits_high_risk_review(video, ambig, cov, conf, match,
                                       miss, ext):
    from reasoning.sco_policy import (decide_sco, OUTCOME_VERIFIED,
                                       OUTCOME_REVIEW, OUTCOME_INVALID_VIDEO)
    vlm = _vlm(
        video_usable=video, confidence=conf, basket_match=match,
        missing=[{"pos_item": str(i), "reason": "x"} for i in range(miss)],
        extras=[{"visible_item": str(i), "note": "y"} for i in range(ext)],
    )
    ep = _ep(ambiguous=ambig, coverage_ratio=cov)
    d = decide_sco(vlm, ep)
    assert d.outcome in {OUTCOME_VERIFIED, OUTCOME_REVIEW,
                         OUTCOME_INVALID_VIDEO}
    assert d.outcome != "HIGH_RISK_REVIEW"


# ---------------------------------------------------------------------------
# 7. Policy version surfaced
# ---------------------------------------------------------------------------

def test_decisions_carry_sco_policy_version():
    from reasoning.sco_policy import decide_sco, SCO_POLICY_VERSION
    d = decide_sco(_vlm(), _ep())
    assert d.policy_version == SCO_POLICY_VERSION
