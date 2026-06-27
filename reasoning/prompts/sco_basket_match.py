"""SCO basket-match prompt builder (Phase 5).

Composes a system + user prompt pair from:
  * POS basket items (from PosEvent.raw_payload["items"])
  * Falcon evidence summary (matched / unmatched / candidate counts)
  * Episode metadata (from perception.episode_selector)

The VLM is asked to compare what's VISIBLE to what the POS bill SAYS.
Hard rules baked into the system prompt:
  * Never claim "scanned" / "unscanned" — that's behaviour, not visible.
  * Never accuse fraud.
  * Quantity is informational only — v1 is presence consistency.
  * If unsure, downgrade confidence; never invent.
"""
from __future__ import annotations

from typing import Iterable, Optional


PROMPT_VERSION = "sco_basket_match_v1"


_SYSTEM_PROMPT = (
    "You are a visual auditor reviewing a short Self-Checkout (SCO) "
    "video clip alongside the POS bill for the same transaction. You "
    "do NOT decide outcomes. A separate deterministic policy turns "
    "your structured report into the case outcome.\n\n"
    "Rules:\n"
    "  * Never use the words fraud, theft, suspect, dishonest, "
    "scanned, or unscanned. The video cannot show a scan event.\n"
    "  * Compare visible items in the audit zone with the POS bill. "
    "Report what visibly matches, what's visibly missing, and what's "
    "visibly extra.\n"
    "  * Quantity reporting in v1 is presence-only. If you see more "
    "than two of the same item class, report \"multiple\" rather "
    "than a number.\n"
    "  * If the episode is flagged ambiguous, downgrade confidence to "
    "at most \"medium\" and explain the ambiguity in your narrative.\n"
    "  * If the video is unusable, set basket_match=\"uncertain\" and "
    "say so in narrative.\n"
    "  * Output ONLY the JSON object the user turn requests. No "
    "markdown, no preamble."
)


_USER_TEMPLATE = """\
POS bill items for this transaction:
{pos_items_block}

Falcon detector evidence summary (single pass on the wide POS window):
{falcon_summary_block}

Selected customer episode (anchor: POS transaction time):
{episode_block}

Compare what is VISIBLE in the audit-zone frames to the POS bill above.

Return EXACTLY this JSON shape, no extra keys, no trailing text:

{{
  "basket_match": "yes" | "no" | "uncertain",
  "matched": [
    {{"pos_item": "<as listed on bill>", "visible_count_class": "one"|"multiple"|"uncertain"}}
  ],
  "missing": [
    {{"pos_item": "<as listed on bill>", "reason": "<short reason it could not be confirmed visually>"}}
  ],
  "extras": [
    {{"visible_item": "<short description>", "note": "<why this looks extra>"}}
  ],
  "video_usable": true | false,
  "confidence": "high" | "medium" | "low",
  "narrative": "<one short factual sentence>"
}}

Hard rules to obey:
  * basket_match = "yes" ONLY if every POS item is visibly accounted for
    AND no extras seen AND the episode is not ambiguous.
  * basket_match = "uncertain" when video is unusable, episode is
    ambiguous, or you cannot tell.
  * basket_match = "no" when you see a clear mismatch (something missing
    or something extra) AND the video is usable.
  * Never write the word "scanned" or "unscanned" in narrative.
"""


def build_user_prompt(*,
                      basket: Optional[Iterable[dict]],
                      falcon_summary: Optional[dict] = None,
                      episode_meta: Optional[dict] = None) -> str:
    """Render the user prompt with POS items + Falcon summary + episode."""
    return _USER_TEMPLATE.format(
        pos_items_block=_format_items_block(basket),
        falcon_summary_block=_format_falcon_block(falcon_summary or {}),
        episode_block=_format_episode_block(episode_meta or {}),
    )


def build_system_prompt() -> str:
    return _SYSTEM_PROMPT


def build_sco_prompts(*,
                      basket: Optional[Iterable[dict]] = None,
                      falcon_summary: Optional[dict] = None,
                      episode_meta: Optional[dict] = None
                      ) -> tuple[str, str]:
    """Convenience: returns (system_prompt, user_prompt)."""
    return (build_system_prompt(),
            build_user_prompt(basket=basket,
                              falcon_summary=falcon_summary,
                              episode_meta=episode_meta))


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _format_items_block(basket: Optional[Iterable[dict]]) -> str:
    items = list(basket or [])
    if not items:
        return "  (POS bill has no line items — basket is empty.)"
    lines = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        desc = (it.get("description") or it.get("name")
                or it.get("item_description") or it.get("sku") or "")
        qty = it.get("quantity") or it.get("qty")
        line = f"  {i + 1}. {desc or '(no description)'}"
        if qty is not None:
            line += f"  (POS qty: {qty})"
        lines.append(line)
    return "\n".join(lines) if lines else "  (no readable items)"


def _format_falcon_block(falcon_summary: dict) -> str:
    if not falcon_summary:
        return ("  (Falcon evidence summary not provided. Reason about "
                "visible content from frames alone.)")
    matched = falcon_summary.get("matched_count")
    unmatched = falcon_summary.get("unmatched_count")
    generic = falcon_summary.get("generic_candidate_count")
    queries = falcon_summary.get("queries_run") or []
    lines = []
    if matched is not None:
        lines.append(f"  matched POS-item detections: {matched}")
    if unmatched is not None:
        lines.append(f"  unmatched POS items (no detection): {unmatched}")
    if generic is not None:
        lines.append(f"  generic-product candidate detections "
                     f"(possible extras): {generic}")
    if queries:
        shown = ", ".join(str(q) for q in list(queries)[:6])
        more = ""
        if len(queries) > 6:
            more = f" (+{len(queries) - 6} more)"
        lines.append(f"  queries Falcon ran: {shown}{more}")
    return "\n".join(lines) if lines \
        else "  (Falcon ran but produced no summary fields.)"


def _format_episode_block(episode: dict) -> str:
    if not episode:
        return ("  (No episode metadata. Reason from full POS window — "
                "treat this as ambiguous.)")
    parts = [
        f"  episode start:     {episode.get('start', '<unknown>')}",
        f"  episode end:       {episode.get('end', '<unknown>')}",
        f"  ambiguous:         {bool(episode.get('ambiguous'))}",
        f"  reason:            {episode.get('reason', '<unknown>')}",
        f"  coverage_ratio:    {episode.get('coverage_ratio', 0):.2f}",
    ]
    return "\n".join(parts)
