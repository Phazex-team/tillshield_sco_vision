"""SCO basket-match decision policy (Phase 6).

Replaces the refund-shaped ``reasoning.decision_policy.decide`` for SCO
cases. The legacy module stays on disk so it can be re-activated if/
when multi-scenario routing returns.

Inputs:
  * ``ScoBasketMatch`` — parsed VLM structured output.
  * ``episode_meta`` — dict from ``perception.episode_selector``.

Output:
  ``PolicyDecision`` re-using the legacy dataclass so the case_runner
  persistence path is unchanged.

Outcome is one of: ``VERIFIED``, ``REVIEW``, ``INVALID_VIDEO``.
``HIGH_RISK_REVIEW`` is reserved for v2 and is NEVER emitted.

Strict VERIFIED gates (all must pass):
  1. video_usable is True.
  2. episode is not ambiguous.
  3. episode coverage_ratio >= MIN_EPISODE_COVERAGE.
  4. VLM confidence is "high" or "medium".
  5. basket_match == "yes" AND missing == [] AND extras == [].

Any single failure → REVIEW with machine-readable risk_reasons tags.
"""
from __future__ import annotations

from typing import Optional

from reasoning.decision_policy import (
    OUTCOME_INVALID_VIDEO,
    OUTCOME_REVIEW,
    OUTCOME_VERIFIED,
    PolicyDecision,
    POLICY_VERSION as LEGACY_POLICY_VERSION,
)
from reasoning.schemas.sco_basket_match import ScoBasketMatch


# Bump independently from the legacy POLICY_VERSION so audit history
# can tell which policy was active for a given case.
SCO_POLICY_VERSION = "sco_v1"

# Minimum fraction of the POS window the episode must cover before we
# trust it as a clean customer episode. Tunable in config later.
MIN_EPISODE_COVERAGE: float = 0.05

# Tags emitted into PolicyDecision.reasons. Keep them stable — the
# reviewer UI and audit log filter on them.
TAG_BASKET_MATCH = "sco_basket_match"
TAG_BASKET_MISMATCH = "sco_basket_mismatch"
TAG_MISSING_ITEMS = "sco_missing_items"
TAG_EXTRA_CANDIDATES = "sco_extra_candidates"
TAG_EPISODE_AMBIGUOUS = "sco_episode_ambiguous"
TAG_EPISODE_SHORT = "sco_episode_short"
TAG_LOW_CONFIDENCE = "sco_low_confidence"
TAG_BAD_FOOTAGE = "sco_bad_footage"
TAG_NO_VLM = "sco_no_vlm_output"


def decide_sco(vlm: Optional[ScoBasketMatch],
               episode_meta: Optional[dict] = None,
               *,
               min_episode_coverage: float = MIN_EPISODE_COVERAGE,
               ) -> PolicyDecision:
    """Map VLM basket-match + episode metadata to a case outcome.

    Defensive: missing VLM output → REVIEW with TAG_NO_VLM (never
    crashes). Outcome always within {VERIFIED, REVIEW, INVALID_VIDEO}.
    """
    if vlm is None:
        return PolicyDecision(
            outcome=OUTCOME_REVIEW, risk_score=0.5,
            reasons=[TAG_NO_VLM, "no VLM output available"],
            policy_version=SCO_POLICY_VERSION,
        )

    reasons: list[str] = []
    tags: list[str] = []

    # ---- Gate 1: video usable --------------------------------------
    if not vlm.video_usable:
        tags.append(TAG_BAD_FOOTAGE)
        reasons = tags + ["VLM reports video unusable"]
        return PolicyDecision(
            outcome=OUTCOME_INVALID_VIDEO, risk_score=0.0,
            reasons=reasons, policy_version=SCO_POLICY_VERSION,
        )

    # ---- Gate 2: episode not ambiguous -----------------------------
    ambiguous = bool((episode_meta or {}).get("ambiguous"))
    if ambiguous:
        tags.append(TAG_EPISODE_AMBIGUOUS)
        reasons.append(
            f"episode ambiguous "
            f"({(episode_meta or {}).get('reason', '?')})"
        )

    # ---- Gate 3: episode coverage ----------------------------------
    coverage = float((episode_meta or {}).get("coverage_ratio") or 0.0)
    if coverage < min_episode_coverage:
        tags.append(TAG_EPISODE_SHORT)
        reasons.append(
            f"episode coverage {coverage:.2f} below floor "
            f"{min_episode_coverage:.2f}"
        )

    # ---- Gate 4: VLM confidence ------------------------------------
    if vlm.confidence not in {"high", "medium"}:
        tags.append(TAG_LOW_CONFIDENCE)
        reasons.append(f"VLM confidence is {vlm.confidence!r}")

    # ---- Gate 5: basket match + no missing/extras ------------------
    if vlm.basket_match == "no":
        tags.append(TAG_BASKET_MISMATCH)
        reasons.append("VLM reports basket does not match POS bill")
    elif vlm.basket_match == "uncertain":
        # uncertain is itself a REVIEW signal (NOT a verified match)
        tags.append(TAG_BASKET_MISMATCH)
        reasons.append("VLM uncertain about basket match")
    if vlm.missing:
        tags.append(TAG_MISSING_ITEMS)
        reasons.append(
            f"{len(vlm.missing)} POS item(s) not visibly confirmed"
        )
    if vlm.extras:
        tags.append(TAG_EXTRA_CANDIDATES)
        reasons.append(
            f"{len(vlm.extras)} extra candidate(s) visible "
            "(possibly not on POS bill)"
        )

    if not tags:
        # All five gates clean → VERIFIED.
        return PolicyDecision(
            outcome=OUTCOME_VERIFIED, risk_score=0.1,
            reasons=[TAG_BASKET_MATCH,
                     "POS basket fully visually accounted for in a clean "
                     "customer episode at high/medium VLM confidence"],
            policy_version=SCO_POLICY_VERSION,
        )

    # Any failing gate → REVIEW. The tag list is the machine-readable
    # cause; the human-readable strings follow. Order matters for the
    # reviewer UI filter chips.
    risk = _risk_from_tags(tags)
    return PolicyDecision(
        outcome=OUTCOME_REVIEW, risk_score=risk,
        reasons=tags + reasons,
        policy_version=SCO_POLICY_VERSION,
    )


def _risk_from_tags(tags: list[str]) -> float:
    """Coarse risk score so the reviewer UI can sort newest-cases-first
    by perceived severity. Never used for the outcome itself."""
    if TAG_BASKET_MISMATCH in tags and TAG_EXTRA_CANDIDATES in tags:
        return 0.75
    if TAG_BASKET_MISMATCH in tags or TAG_MISSING_ITEMS in tags:
        return 0.6
    if TAG_EPISODE_AMBIGUOUS in tags:
        return 0.5
    if TAG_LOW_CONFIDENCE in tags or TAG_EPISODE_SHORT in tags:
        return 0.4
    return 0.5
