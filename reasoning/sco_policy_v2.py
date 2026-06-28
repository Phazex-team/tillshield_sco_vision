"""SCO basket-match decision policy v2.

Companion to the v1 policy in ``reasoning.sco_policy``. v1 is kept
on disk and continues to consume ``ScoBasketMatch``. v2 consumes
``ScoBasketMatchV2`` which splits physical_count_match from
semantic_identity_match.

Outcome space remains the same: ``VERIFIED``, ``REVIEW``,
``INVALID_VIDEO``. ``HIGH_RISK_REVIEW`` is still never emitted.

Key semantic change vs v1:

  v1 mapped any non-"yes" basket_match → REVIEW with
  ``sco_basket_mismatch``. For SAM3+VLM SCO cases where the items
  are inside closed takeaway containers, that produced a
  confidently-wrong "mismatch" verdict.

  v2 distinguishes:
    * physical_count_match    — used to flag count mismatches
                                (true REVIEW signal).
    * semantic_identity_match — when uncertain (closed containers,
                                obscured labels, etc.) the policy
                                emits ``sco_identity_uncertain`` —
                                NOT ``sco_basket_mismatch``. Both
                                still REVIEW, but the reviewer UI
                                and audit log can tell the two
                                apart.

VERIFIED gates (all must pass):
  1. video_usable is True.
  2. episode is not ambiguous.
  3. episode coverage_ratio >= MIN_EPISODE_COVERAGE.
  4. VLM confidence is "high" or "medium".
  5. physical_count_match == "yes".
  6. semantic_identity_match == "yes".
  7. missing_visible_items == [] AND extra_visible_items == [].

Any single failure → REVIEW with stable machine-readable tags.
"""
from __future__ import annotations

from typing import Optional

from reasoning.decision_policy import (
    OUTCOME_INVALID_VIDEO,
    OUTCOME_REVIEW,
    OUTCOME_VERIFIED,
    PolicyDecision,
)
from reasoning.schemas.sco_basket_match_v2 import ScoBasketMatchV2


SCO_POLICY_VERSION_V2 = "sco_v2"

MIN_EPISODE_COVERAGE: float = 0.05

# Stable tags. v2-specific ones come last so v1 callers' filter
# chips don't break.
TAG_BASKET_MATCH = "sco_basket_match"
TAG_BASKET_MISMATCH = "sco_basket_mismatch"          # physical count mismatch
TAG_IDENTITY_UNCERTAIN = "sco_identity_uncertain"    # v2: semantic-only
TAG_COUNT_UNCERTAIN = "sco_count_uncertain"          # v2: count-only
TAG_MISSING_ITEMS = "sco_missing_items"
TAG_EXTRA_CANDIDATES = "sco_extra_candidates"
TAG_EPISODE_AMBIGUOUS = "sco_episode_ambiguous"
TAG_EPISODE_SHORT = "sco_episode_short"
TAG_LOW_CONFIDENCE = "sco_low_confidence"
TAG_BAD_FOOTAGE = "sco_bad_footage"
TAG_NO_VLM = "sco_no_vlm_output"


def decide_sco_v2(vlm: Optional[ScoBasketMatchV2],
                  episode_meta: Optional[dict] = None,
                  *,
                  min_episode_coverage: float = MIN_EPISODE_COVERAGE,
                  container_merge_meta: Optional[dict] = None,
                  pos_basket_size: Optional[int] = None,
                  ) -> PolicyDecision:
    """Decide the SCO outcome.

    ``container_merge_meta`` (optional, post-SAM3): when present,
    physical-count gates honour the merged count range
    (``count_min``..``count_max``) and the ``fragmentation_suspected``
    / ``missed_container_possible`` flags. Without it (Falcon-only,
    legacy callers), physical-count behaviour is unchanged from
    v1 of this policy.
    """
    if vlm is None:
        return PolicyDecision(
            outcome=OUTCOME_REVIEW, risk_score=0.5,
            reasons=[TAG_NO_VLM, "no VLM output available"],
            policy_version=SCO_POLICY_VERSION_V2,
        )

    tags: list[str] = []
    reasons: list[str] = []

    # ---- Gate 1: video usable ----
    if not vlm.video_usable:
        return PolicyDecision(
            outcome=OUTCOME_INVALID_VIDEO, risk_score=0.0,
            reasons=[TAG_BAD_FOOTAGE, "VLM reports video unusable"],
            policy_version=SCO_POLICY_VERSION_V2,
        )

    # ---- Gate 2: episode not ambiguous ----
    if bool((episode_meta or {}).get("ambiguous")):
        tags.append(TAG_EPISODE_AMBIGUOUS)
        reasons.append(
            f"episode ambiguous "
            f"({(episode_meta or {}).get('reason', '?')})")

    # ---- Gate 3: episode coverage ----
    coverage = float((episode_meta or {}).get("coverage_ratio") or 0.0)
    if coverage < min_episode_coverage:
        tags.append(TAG_EPISODE_SHORT)
        reasons.append(
            f"episode coverage {coverage:.2f} below floor "
            f"{min_episode_coverage:.2f}")

    # ---- Gate 4: VLM confidence ----
    if vlm.confidence not in {"high", "medium"}:
        tags.append(TAG_LOW_CONFIDENCE)
        reasons.append(f"VLM confidence is {vlm.confidence!r}")

    # ---- Gate 5: physical count ----
    # When container-merge metadata is present, the merger's count
    # range + confidence is the ground truth for the COUNT signal.
    # The VLM's physical_count_match is treated as a secondary
    # opinion: it can downgrade certainty but cannot upgrade a
    # "wide-range / fragmented" merger result to a confident match.
    if container_merge_meta is not None:
        cmin = container_merge_meta.get("count_min")
        cmax = container_merge_meta.get("count_max")
        conf = (container_merge_meta.get("count_confidence")
                or "low").lower()
        frag = bool(container_merge_meta.get("fragmentation_suspected"))
        missed = bool(container_merge_meta.get("missed_container_possible"))
        if pos_basket_size is not None and pos_basket_size >= 0 \
                and isinstance(cmin, int) and isinstance(cmax, int):
            # Only flag a CONFIDENT mismatch when:
            #   * the merger has medium/high confidence in its count,
            #   * AND the POS basket size is OUTSIDE the merger's
            #     count range by more than 1 in either direction,
            #   * AND the VLM did not actively contradict the
            #     mismatch claim by saying physical_count_match=yes.
            # Otherwise this is count uncertainty, not mismatch.
            outside_range = (pos_basket_size + 1 < cmin
                             or pos_basket_size - 1 > cmax)
            merger_confident = conf in ("high", "medium")
            vlm_contradicts = (vlm.physical_count_match == "yes")
            if outside_range and merger_confident and not vlm_contradicts:
                tags.append(TAG_BASKET_MISMATCH)
                reasons.append(
                    f"physical item count {cmin}-{cmax} does not "
                    f"match POS basket size {pos_basket_size} "
                    f"(merger_confidence={conf})")
            elif cmax > cmin or frag or conf == "low":
                tags.append(TAG_COUNT_UNCERTAIN)
                detail = []
                if cmax > cmin:
                    detail.append(
                        f"merged count range {cmin}-{cmax}")
                if frag:
                    detail.append("fragmentation suspected")
                if missed:
                    detail.append("missed container possible")
                if conf == "low":
                    detail.append("merger confidence low")
                reasons.append("physical item count uncertain: "
                               + "; ".join(detail))
        else:
            # No POS size or no usable range → defer to VLM signal.
            if vlm.physical_count_match == "no":
                tags.append(TAG_BASKET_MISMATCH)
                reasons.append(
                    "physical item count does not match POS basket")
            elif vlm.physical_count_match == "uncertain":
                tags.append(TAG_COUNT_UNCERTAIN)
                reasons.append("physical item count uncertain")
    else:
        # Legacy / Falcon-only callers: physical-count signal comes
        # purely from the VLM (the v1 policy behaviour for this gate).
        if vlm.physical_count_match == "no":
            tags.append(TAG_BASKET_MISMATCH)
            reasons.append("physical item count does not match POS basket")
        elif vlm.physical_count_match == "uncertain":
            tags.append(TAG_COUNT_UNCERTAIN)
            reasons.append("physical item count uncertain")

    # ---- Gate 6: semantic identity ----
    # The critical v2 split. semantic_identity_match=uncertain is
    # NOT a mismatch — it's an honest "we can't tell which is which".
    if vlm.semantic_identity_match == "no":
        # A real semantic contradiction (e.g. POS says electronics, you
        # see only food) still tags as mismatch.
        tags.append(TAG_BASKET_MISMATCH)
        reasons.append("VLM observed a semantic contradiction with POS")
    elif vlm.semantic_identity_match == "uncertain":
        tags.append(TAG_IDENTITY_UNCERTAIN)
        ur = (vlm.uncertainty_reason or "items not visually identifiable")
        reasons.append(f"identity uncertain: {ur}")

    # ---- Gate 7: missing / extras ----
    # Critical: do NOT flag missing items when semantic identity is
    # uncertain — the VLM has been told to leave matched_items
    # populated and missing empty under that condition.
    if vlm.missing_visible_items \
            and vlm.semantic_identity_match != "uncertain":
        tags.append(TAG_MISSING_ITEMS)
        reasons.append(
            f"{len(vlm.missing_visible_items)} POS item(s) "
            "not visibly confirmed")
    if vlm.extra_visible_items:
        tags.append(TAG_EXTRA_CANDIDATES)
        reasons.append(
            f"{len(vlm.extra_visible_items)} extra candidate(s) "
            "visible (not on POS bill)")

    if not tags:
        return PolicyDecision(
            outcome=OUTCOME_VERIFIED, risk_score=0.1,
            reasons=[TAG_BASKET_MATCH,
                     "POS basket visually accounted for; physical count "
                     "and semantic identity both match at high/medium "
                     "VLM confidence"],
            policy_version=SCO_POLICY_VERSION_V2,
        )

    return PolicyDecision(
        outcome=OUTCOME_REVIEW, risk_score=_risk_from_tags(tags),
        reasons=tags + reasons,
        policy_version=SCO_POLICY_VERSION_V2,
    )


def _risk_from_tags(tags: list[str]) -> float:
    # The closed-container case (identity uncertain only) is the
    # LOWEST-risk REVIEW outcome — it's honest uncertainty, not a
    # signal of mismatch.
    if TAG_BASKET_MISMATCH in tags and TAG_EXTRA_CANDIDATES in tags:
        return 0.75
    if TAG_BASKET_MISMATCH in tags or TAG_MISSING_ITEMS in tags:
        return 0.6
    if TAG_EPISODE_AMBIGUOUS in tags:
        return 0.5
    if TAG_LOW_CONFIDENCE in tags or TAG_EPISODE_SHORT in tags:
        return 0.4
    if TAG_IDENTITY_UNCERTAIN in tags or TAG_COUNT_UNCERTAIN in tags:
        return 0.35
    return 0.5
