"""Decision policy must be conservative-by-construction.

These tests pin the contract: ambiguous evidence MUST yield ``REVIEW``
(never HIGH_RISK_REVIEW), only valid outcomes are ever produced, and
the VLM cannot upgrade a case beyond what the structured evidence
supports.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasoning.decision_policy import (  # noqa: E402
    OUTCOME_HIGH_RISK_REVIEW,
    OUTCOME_INVALID_VIDEO,
    OUTCOME_REVIEW,
    OUTCOME_VERIFIED,
    VALID_OUTCOMES,
    EvidenceSummary,
    decide,
    summary_from_vlm,
)


def test_outcomes_are_constrained():
    summaries = [
        EvidenceSummary(footage_valid=False),
        EvidenceSummary(footage_valid=True, physical_item_track=True,
                        item_reaches_counter=True, vlm_confidence="high"),
        EvidenceSummary(footage_valid=True, receipt_visible=True,
                        vlm_confidence="high"),
        EvidenceSummary(footage_valid=True, obstructed=True),
        EvidenceSummary(footage_valid=True, vlm_confidence="low"),
        EvidenceSummary(footage_valid=True,
                        contradictions=["vlm says no, tracks say yes"]),
    ]
    for s in summaries:
        decision = decide(s)
        assert decision.outcome in VALID_OUTCOMES


def test_invalid_video_short_circuits():
    d = decide(EvidenceSummary(footage_valid=False))
    assert d.outcome == OUTCOME_INVALID_VIDEO


def test_verified_requires_high_or_medium_confidence_and_clean_track():
    # All ingredients present, high confidence -> VERIFIED.
    d = decide(EvidenceSummary(
        footage_valid=True,
        physical_item_track=True,
        item_reaches_counter=True,
        vlm_confidence="high",
    ))
    assert d.outcome == OUTCOME_VERIFIED

    # Same scene, low VLM confidence -> downgrades.
    d2 = decide(EvidenceSummary(
        footage_valid=True,
        physical_item_track=True,
        item_reaches_counter=True,
        vlm_confidence="low",
    ))
    assert d2.outcome == OUTCOME_REVIEW


def test_receipt_only_clean_window_is_high_risk_review():
    d = decide(EvidenceSummary(
        footage_valid=True,
        receipt_visible=True,
        physical_item_track=False,
        obstructed=False,
        camera_gap=False,
        vlm_confidence="high",
    ))
    assert d.outcome == OUTCOME_HIGH_RISK_REVIEW


def test_receipt_only_obstructed_is_review_not_high_risk():
    d = decide(EvidenceSummary(
        footage_valid=True,
        receipt_visible=True,
        physical_item_track=False,
        obstructed=True,
        vlm_confidence="high",
    ))
    assert d.outcome == OUTCOME_REVIEW


def test_receipt_only_low_confidence_downgrades_to_review():
    d = decide(EvidenceSummary(
        footage_valid=True,
        receipt_visible=True,
        physical_item_track=False,
        vlm_confidence="low",
    ))
    assert d.outcome == OUTCOME_REVIEW


def test_contradictions_force_review_even_with_strong_signals():
    d = decide(EvidenceSummary(
        footage_valid=True,
        physical_item_track=True,
        item_reaches_counter=True,
        receipt_visible=True,
        vlm_confidence="high",
        contradictions=["track says appeared but VLM denies"],
    ))
    assert d.outcome == OUTCOME_REVIEW


def test_camera_gap_forces_review():
    d = decide(EvidenceSummary(
        footage_valid=True,
        physical_item_track=True,
        item_reaches_counter=True,
        camera_gap=True,
        vlm_confidence="high",
    ))
    assert d.outcome == OUTCOME_REVIEW


def test_legacy_vlm_payload_adapter():
    parsed = {
        "handover_occurred": True,
        "item_count": 1,
        "items_handed_over": ["shirt"],
        "receipt_visible": True,
        "physical_item_presented": True,
        "confidence": "high",
    }
    summary = summary_from_vlm(parsed, footage_valid=True)
    d = decide(summary)
    assert d.outcome == OUTCOME_VERIFIED


def test_policy_never_emits_fraud_label():
    """No code path may use the word FRAUD in the outcome enum."""
    for s in [
        EvidenceSummary(footage_valid=False),
        EvidenceSummary(footage_valid=True, receipt_visible=True,
                        vlm_confidence="high"),
        EvidenceSummary(footage_valid=True, vlm_confidence="low"),
    ]:
        d = decide(s)
        assert "FRAUD" not in d.outcome
        assert "ACCUSE" not in d.outcome.upper()
