"""SCO basket-match prompt v2 — separates physical count from semantic identity.

What changed from v1 (sco_basket_match):

  v1 conflated two questions:
    1. "How many physical items are visible?"
    2. "Are they the POS-listed items?"

  When a SAM-3 backend returned 2 stable food-container identities
  for a POS basket of {Biriyani, Curry}, the v1 prompt asked the
  VLM to tag each POS line as matched/missing/extra. Gemma — looking
  at two closed takeaway containers — said "both POS items missing"
  AND "two extras visible". That's a false mismatch verdict for
  cases where the physical count plausibly satisfies the basket
  but the contents are obscured.

  v2 makes the two questions explicit:
    * ``physical_count_match``  — does the COUNT of distinct items
      visible plausibly match the POS basket size?
    * ``semantic_identity_match`` — can the contents of each
      visible item be tied to a POS line?

  Hard rule for closed containers: when items are inside opaque
  packaging (takeaway boxes, bags, wrapped food) and POS items
  are described by their contents (biriyani, curry), the VLM
  MUST set semantic_identity_match=uncertain — NOT "no". The
  policy treats uncertain as REVIEW, never as an accusatory
  mismatch.

  The prompt also receives canonical SAM-3 groups (one per
  physical identity) and the count of those groups, and is
  forbidden from re-collapsing identities by label class.
"""
from __future__ import annotations

from typing import Iterable, Optional


PROMPT_VERSION = "sco_basket_match_v2"


_SYSTEM_PROMPT = (
    "You are a visual auditor reviewing a short Self-Checkout (SCO) "
    "video clip alongside the POS bill for the same transaction. You "
    "do NOT decide outcomes. A separate deterministic policy turns "
    "your structured report into the case outcome.\n\n"
    "Hard rules:\n"
    "  * Never use the words fraud, theft, suspect, dishonest, "
    "scanned, or unscanned. The video cannot show a scan event.\n"
    "  * You will be shown CANONICAL ITEM GROUPS — one per physical "
    "identity that the tracker has already de-duplicated across "
    "frames and detector labels. Treat the group count as the "
    "authoritative count of distinct physical items in the scene. "
    "Do NOT collapse two physical groups just because they share a "
    "label class (e.g. two takeaway containers are still TWO).\n"
    "  * Separate two questions:\n"
    "      (a) physical_count_match — does the COUNT of distinct "
    "          visible items plausibly satisfy the POS basket size?\n"
    "      (b) semantic_identity_match — can you tell which POS "
    "          line each visible item is?\n"
    "  * If items are inside closed/opaque containers (takeaway "
    "boxes, food packaging, bags) and POS items are described by "
    "their contents you cannot see, set semantic_identity_match to "
    "\"uncertain\" — NOT \"no\". Only use \"no\" when you can see "
    "actual semantic contradiction (e.g. POS says milk bottle and "
    "you see a t-shirt).\n"
    "  * Quantity reporting is presence-only. Report \"one\" or "
    "\"multiple\" per visible item; never invent integer counts.\n"
    "  * Output ONLY the JSON object the user turn requests. No "
    "markdown, no preamble."
)


_USER_TEMPLATE = """\
POS bill items for this transaction:
{pos_items_block}

Canonical item groups (tracker-deduplicated; each group = ONE physical item):
{groups_block}

Selected customer episode (anchor: POS transaction time):
{episode_block}

Compare what is VISIBLE in the audit-zone frames to the POS bill above.
Treat the canonical group count as the authoritative physical count.

Return EXACTLY this JSON shape, no extra keys, no trailing text:

{{
  "physical_count_match": "yes" | "no" | "uncertain",
  "semantic_identity_match": "yes" | "no" | "uncertain",
  "matched_items": [
    {{"pos_item": "<as on bill>", "group_id": "<sco_group_NNN or null>",
      "visible_count_class": "one"|"multiple"|"uncertain"}}
  ],
  "missing_visible_items": [
    {{"pos_item": "<as on bill>",
      "reason": "<short reason not visible in episode>"}}
  ],
  "extra_visible_items": [
    {{"group_id": "<sco_group_NNN>",
      "description": "<short description of what you see>"}}
  ],
  "uncertainty_reason": "<short reason if either *_match is uncertain, else empty>",
  "video_usable": true | false,
  "confidence": "high" | "medium" | "low",
  "narrative": "<one short factual sentence>"
}}

Hard rules to obey when filling the schema:

  * physical_count_match:
      yes        — visible group count plausibly satisfies POS basket size
                   (allow ±1 for occluded edges).
      no         — group count is clearly far off (e.g. POS=5, visible=1).
      uncertain  — video is unusable, episode ambiguous, or count is
                   borderline and you cannot tell.

  * semantic_identity_match:
      yes        — every POS line is visually tied to a specific
                   canonical group (visible label, distinguishing shape,
                   etc.).
      no         — you see a clear contradiction (e.g. POS says
                   electronics, you see only food).
      uncertain  — items are inside closed containers, wrapping, or
                   visually indistinguishable groups, and you cannot
                   tell which POS line maps to which group. THIS IS
                   THE EXPECTED VALUE for takeaway food containers,
                   wrapped clothing, etc.

  * matched_items: only fill when you can credibly tie a POS line to
    a specific canonical group_id (semantic_identity_match in
    {{"yes","uncertain"}} when the link is plausible).

  * missing_visible_items: only items you specifically looked for and
    could not see in the canonical groups. If everything is in
    closed containers, do NOT mark contents as missing — set
    semantic_identity_match="uncertain" instead.

  * extra_visible_items: ONLY for canonical groups marked
    is_extra_candidate=true in the input AND clearly distinct from any
    POS line. Do not invent extras by re-counting matched groups.

  * uncertainty_reason: one short sentence when either *_match is
    "uncertain". Common reasons: "items inside closed takeaway
    containers", "episode flagged ambiguous", "video too low
    resolution to read labels".

  * Never write the word "scanned" or "unscanned" in narrative.
"""


def build_user_prompt_v2(*,
                          basket: Optional[Iterable[dict]],
                          canonical_groups: Optional[Iterable[dict]] = None,
                          episode_meta: Optional[dict] = None) -> str:
    return _USER_TEMPLATE.format(
        pos_items_block=_format_items_block(basket),
        groups_block=_format_groups_block(canonical_groups or []),
        episode_block=_format_episode_block(episode_meta or {}),
    )


def build_system_prompt_v2() -> str:
    return _SYSTEM_PROMPT


def build_sco_prompts_v2(*,
                          basket: Optional[Iterable[dict]] = None,
                          canonical_groups: Optional[Iterable[dict]] = None,
                          episode_meta: Optional[dict] = None
                          ) -> tuple[str, str]:
    return (build_system_prompt_v2(),
            build_user_prompt_v2(basket=basket,
                                  canonical_groups=canonical_groups,
                                  episode_meta=episode_meta))


# ---------------------------------------------------------------------------
# Formatters — separate physical-count rendering from POS items
# ---------------------------------------------------------------------------

def _format_items_block(basket: Optional[Iterable[dict]]) -> str:
    items = list(basket or [])
    if not items:
        return "  (POS bill has no line items — basket is empty.)"
    lines = [f"  POS basket size: {len(items)} line(s).",
             f"  POS items:"]
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        desc = (it.get("description") or it.get("name")
                or it.get("item_description") or it.get("sku") or "")
        qty = it.get("quantity") or it.get("qty")
        line = f"    {i + 1}. {desc or '(no description)'}"
        if qty is not None:
            line += f"  (POS qty: {qty})"
        lines.append(line)
    return "\n".join(lines)


def _format_groups_block(groups: Iterable[dict]) -> str:
    glist = list(groups or [])
    if not glist:
        return ("  (No canonical groups produced — the tracker saw no "
                "items in the audit zone.)")
    matched = [g for g in glist if not g.get("is_extra_candidate")]
    extras = [g for g in glist if g.get("is_extra_candidate")]
    lines = [
        f"  Total canonical groups: {len(glist)} (this is the "
        f"authoritative physical count).",
        f"  Of these, {len(matched)} were tentatively matched to a "
        f"POS line by spatial overlap and {len(extras)} are "
        f"extra-candidate (no POS link from perception).",
        "",
        f"  Tentatively-matched groups ({len(matched)}):",
    ]
    if matched:
        for g in matched:
            pos = g.get("matched_pos_item") or "(unknown)"
            labels = ", ".join(g.get("source_labels") or [])
            conf = g.get("confidence") or "low"
            lines.append(
                f"    - {g.get('group_id', '?')}: pos_item={pos!r}; "
                f"source_labels=[{labels}]; perception_confidence={conf}"
            )
    else:
        lines.append("    (none)")
    lines.append(f"  Extra-candidate groups ({len(extras)}):")
    if extras:
        for g in extras:
            labels = ", ".join(g.get("source_labels") or [])
            conf = g.get("confidence") or "low"
            lines.append(
                f"    - {g.get('group_id', '?')}: "
                f"source_labels=[{labels}]; perception_confidence={conf}"
            )
    else:
        lines.append("    (none)")
    return "\n".join(lines)


def _format_episode_block(episode: dict) -> str:
    if not episode:
        return ("  (No episode metadata. Reason from full POS window — "
                "treat this as ambiguous.)")
    return "\n".join([
        f"  episode start:     {episode.get('start', '<unknown>')}",
        f"  episode end:       {episode.get('end', '<unknown>')}",
        f"  ambiguous:         {bool(episode.get('ambiguous'))}",
        f"  reason:            {episode.get('reason', '<unknown>')}",
        f"  coverage_ratio:    {episode.get('coverage_ratio', 0):.2f}",
    ])
