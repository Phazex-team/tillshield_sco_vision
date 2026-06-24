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
    # All ingredients present (incl. VLM-confirmed handover), high
    # confidence -> VERIFIED.
    d = decide(EvidenceSummary(
        footage_valid=True,
        physical_item_track=True,
        item_reaches_counter=True,
        vlm_handover=True,
        vlm_confidence="high",
    ))
    assert d.outcome == OUTCOME_VERIFIED

    # Same scene, low VLM confidence -> downgrades.
    d2 = decide(EvidenceSummary(
        footage_valid=True,
        physical_item_track=True,
        item_reaches_counter=True,
        vlm_handover=True,
        vlm_confidence="low",
    ))
    assert d2.outcome == OUTCOME_REVIEW


def test_item_track_without_vlm_handover_stays_review():
    # Counter clutter / staff-only refund: an item track reaches the
    # counter but the VLM reports NO handover -> must NOT be VERIFIED.
    d = decide(EvidenceSummary(
        footage_valid=True,
        physical_item_track=True,
        item_reaches_counter=True,
        vlm_handover=False,
        vlm_confidence="high",
    ))
    assert d.outcome == OUTCOME_REVIEW


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


def test_legacy_vlm_payload_adapter_needs_track_evidence_to_verify():
    """Track-gating contract: VLM alone never produces VERIFIED. The
    legacy ``physical_item_presented`` field is now a self-reported
    hint; a real ``perception_result`` track must independently
    confirm it for the policy to upgrade."""
    parsed = {
        "handover_occurred": True,
        "item_count": 1,
        "items_handed_over": ["shirt"],
        "receipt_visible": True,
        "physical_item_presented": True,
        "confidence": "high",
    }
    # Without perception evidence the policy must NOT verify.
    summary_no_track = summary_from_vlm(parsed, footage_valid=True)
    assert decide(summary_no_track).outcome == OUTCOME_REVIEW

    # With independent perception evidence, the policy may verify.
    perception = {
        "tracks": [{
            "tracker_id": "track_0001",
            "label": "shirt",
            "physical_item_candidate": True,
            "zones": ["counter_zone"],
            "events": ["entered_counter_zone", "handover_candidate"],
            "confidence": 0.9,
        }]
    }
    summary_with_track = summary_from_vlm(parsed, footage_valid=True,
                                          perception_result=perception)
    assert decide(summary_with_track).outcome == OUTCOME_VERIFIED


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
