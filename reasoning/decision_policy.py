"""Deterministic decision policy.

This module wraps every VLM output. The VLM does **not** decide; it
explains. The policy below produces one of:

    VERIFIED            track evidence proves a legitimate return
    REVIEW              ambiguity, low confidence, model disagreement
    HIGH_RISK_REVIEW    receipt-only with a clean unobstructed window
    INVALID_VIDEO       footage missing/corrupt/not enough coverage

There is **no** "FRAUD" outcome. Conservative-by-construction: when in
doubt, the policy degrades to REVIEW, never up to HIGH_RISK_REVIEW.
"""
from __future__ import annotations

from dataclasses import dataclass, field


POLICY_VERSION = "v1"

OUTCOME_VERIFIED = "VERIFIED"
OUTCOME_REVIEW = "REVIEW"
OUTCOME_HIGH_RISK_REVIEW = "HIGH_RISK_REVIEW"
OUTCOME_INVALID_VIDEO = "INVALID_VIDEO"
VALID_OUTCOMES = {
    OUTCOME_VERIFIED,
    OUTCOME_REVIEW,
    OUTCOME_HIGH_RISK_REVIEW,
    OUTCOME_INVALID_VIDEO,
}


@dataclass
class EvidenceSummary:
    """Summary fed into the policy. The perception/pipeline produces this
    from the evidence graph; the VLM output is an *advisory* input only."""
    footage_valid: bool
    physical_item_track: bool = False
    item_reaches_counter: bool = False
    receipt_visible: bool = False
    obstructed: bool = False
    camera_gap: bool = False
    vlm_confidence: str = "low"          # high / medium / low
    vlm_outcome_hint: str = ""           # advisory only
    contradictions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class PolicyDecision:
    outcome: str
    risk_score: float
    reasons: list[str]
    policy_version: str = POLICY_VERSION


def decide(summary: EvidenceSummary) -> PolicyDecision:
    reasons: list[str] = []

    if not summary.footage_valid:
        reasons.append("footage invalid or missing")
        return PolicyDecision(OUTCOME_INVALID_VIDEO, 0.0, reasons)

    # Any contradiction or low-confidence VLM result → REVIEW. The VLM
    # cannot upgrade a case; it can only downgrade it.
    if summary.contradictions:
        reasons.append("model produced contradictions: "
                       + ", ".join(summary.contradictions))
        return PolicyDecision(OUTCOME_REVIEW, 0.5, reasons)

    confidence = (summary.vlm_confidence or "low").lower()
    low_confidence = confidence not in {"high", "medium"}

    # Clear-window receipt-only is the strongest *non-accusatory* signal
    # we ever emit. It still goes to a human reviewer.
    if summary.receipt_visible and not summary.physical_item_track \
            and not summary.obstructed and not summary.camera_gap:
        reasons.append("receipt visible without physical item track "
                       "in an unobstructed window")
        if low_confidence:
            # Don't escalate when the underlying model isn't confident.
            reasons.append("downgraded to REVIEW: VLM confidence low")
            return PolicyDecision(OUTCOME_REVIEW, 0.6, reasons)
        return PolicyDecision(OUTCOME_HIGH_RISK_REVIEW, 0.85, reasons)

    if summary.physical_item_track and summary.item_reaches_counter \
            and not summary.obstructed and not summary.camera_gap \
            and not low_confidence:
        reasons.append("item track appears from customer side and reaches "
                       "counter/staff with sufficient VLM confidence")
        return PolicyDecision(OUTCOME_VERIFIED, 0.1, reasons)

    if summary.obstructed or summary.camera_gap:
        reasons.append("scene obstructed or camera gap; cannot resolve")
        return PolicyDecision(OUTCOME_REVIEW, 0.5, reasons)

    if low_confidence:
        reasons.append("VLM confidence below threshold")
        return PolicyDecision(OUTCOME_REVIEW, 0.5, reasons)

    reasons.append("no decisive evidence; defaulting to human review")
    return PolicyDecision(OUTCOME_REVIEW, 0.5, reasons)


def summary_from_vlm(parsed: dict, *, footage_valid: bool,
                     obstructed: bool | None = None,
                     camera_gap: bool = False,
                     perception_result: dict | None = None
                     ) -> EvidenceSummary:
    """Adapt a VLM ``reason`` payload to an ``EvidenceSummary``.

    Reads the review-safe schema (``physical_item_presented``,
    ``receipt_visible``, ``obstructed``, ``camera_view_clear``,
    ``limitations``) when present and falls back to the legacy schema
    fields so this adapter stays robust across model upgrades.

    **Track gating (PRODUCTION_SPEC §11):** ``physical_item_track`` is
    derived ONLY from the perception layer's persisted tracks (passed
    in via ``perception_result``). The VLM's ``physical_item_presented``
    flag is captured separately as ``vlm_says_physical_item`` but it can
    never set ``physical_item_track=True`` on its own. Without an
    independent track-level signal, the policy refuses to upgrade a
    case to ``VERIFIED`` even when the VLM claims a clean handover.
    """
    if not isinstance(parsed, dict):
        parsed = {}

    confidence = str(parsed.get("confidence", "low")).lower()
    items = (parsed.get("items_observed")
             or parsed.get("items_handed_over")
             or [])
    receipt_visible = bool(parsed.get("receipt_visible", False))
    vlm_says_physical_item = bool(parsed.get("physical_item_presented",
                                              bool(items)))

    # Obstruction: explicit caller override > model self-report >
    # inverse of ``camera_view_clear`` > limitations heuristic.
    if obstructed is None:
        if "obstructed" in parsed:
            obstructed = bool(parsed.get("obstructed"))
        elif "camera_view_clear" in parsed:
            obstructed = not bool(parsed.get("camera_view_clear"))
        else:
            obstructed = False

    # ---- Track-gated physical_item_track --------------------------
    # Independent perception evidence is the ONLY source that sets
    # ``physical_item_track``. Both the runtime perception_result dict
    # and any pre-projected ``tracks`` block on the parsed VLM payload
    # (e.g. when the caller has already attached structured perception)
    # are inspected — but the VLM's own narrative claim is never
    # promoted into a track.
    perception_tracks: list = []
    if perception_result and isinstance(perception_result, dict):
        perception_tracks = list(perception_result.get("tracks") or [])
    physical_item_track = any(
        bool(t.get("physical_item_candidate"))
        and t.get("tracker_id")  # require a real tracker id
        for t in perception_tracks
        if isinstance(t, dict)
    )
    item_reaches_counter = any(
        bool(t.get("physical_item_candidate"))
        and any(z in (t.get("zones") or [])
                for z in ("counter_zone", "staff_zone"))
        and any(e in (t.get("events") or [])
                for e in ("handover_candidate",
                          "entered_counter_zone",
                          "entered_staff_zone"))
        for t in perception_tracks
        if isinstance(t, dict)
    )

    contradictions: list[str] = []
    # If the VLM claims a physical item but perception found no
    # qualifying track, flag the disagreement so the policy downgrades
    # to REVIEW. This is the core safety contract.
    if vlm_says_physical_item and not physical_item_track:
        contradictions.append(
            "vlm claims physical item presented but no track evidence")

    limitations = parsed.get("limitations") or []
    notes = [str(s)[:240] for s in limitations] if isinstance(limitations,
                                                              list) else []

    return EvidenceSummary(
        footage_valid=footage_valid,
        physical_item_track=physical_item_track,
        item_reaches_counter=item_reaches_counter,
        receipt_visible=receipt_visible,
        obstructed=bool(obstructed),
        camera_gap=camera_gap,
        vlm_confidence=confidence,
        vlm_outcome_hint=(
            "VERIFIED" if (physical_item_track and receipt_visible)
            else "REVIEW"
        ),
        contradictions=contradictions,
        notes=notes,
    )
