"""Chunked VLM analysis: partition frames + aggregate per-chunk verdicts.

Why this exists
---------------
The VLM can only read ~64 frames per call, but a transaction window (now
tx_start -> tx_end) can be several minutes. Cramming the whole window into
one sparse 64-frame call (a) overflows the model's context and (b) samples
so thinly that a customer can arrive and leave between two frames. So we
watch the window in short **overlapping chunks**, each a small in-budget
call at a higher effective frame rate, then combine the chunk verdicts.

Aggregation is deliberately **observational**, not a naive per-chunk
basket match: any single chunk sees only PART of the transaction, so its
"missing items" list is meaningless on its own. Instead we UNION what the
chunks positively saw (matched + extra items, customer presence) and only
call a basket item "missing" if NO chunk ever saw it.

The guard
---------
The combined verdict may not contradict what a chunk plainly saw: if any
chunk observed a customer, the aggregate ``customer_present`` is True even
if most chunks showed an empty counter. Disagreement is flagged for audit.

This module is pure data-shaping (no I/O, no model calls) so it is fully
unit-testable.
"""
from __future__ import annotations

import json
from typing import Optional


def partition_frames(frames: list,
                     chunk_frames: int = 40,
                     overlap: int = 4) -> list[list]:
    """Split an ordered frame list into overlapping chunks.

    Each chunk has at most ``chunk_frames`` frames and shares ``overlap``
    frames with the previous chunk so an event on a chunk boundary is seen
    by both neighbours (no blind spot at the seams). A list that already
    fits in one chunk is returned as a single chunk.
    """
    n = len(frames)
    if chunk_frames <= 0 or n <= chunk_frames:
        return [list(frames)] if frames else []
    step = max(1, chunk_frames - max(0, overlap))
    chunks: list[list] = []
    i = 0
    while i < n:
        chunks.append(list(frames[i:i + chunk_frames]))
        if i + chunk_frames >= n:
            break
        i += step
    return chunks


def _item_key(it) -> str:
    if isinstance(it, dict):
        v = it.get("pos_item") or it.get("description") or it.get("name") or ""
    else:
        v = str(it or "")
    return v.strip().lower()


def aggregate_chunk_verdicts(chunk_dicts: list[dict],
                             basket_descriptions: Optional[list[str]] = None
                             ) -> dict:
    """Combine per-chunk SCO verdicts into one transaction verdict.

    * ``customer_present`` — OR across chunks (the guard).
    * ``matched_items`` / ``extra_visible_items`` — union of what any chunk
      positively saw.
    * ``missing_visible_items`` — basket items NO chunk matched (computed
      against the union, never per-chunk). Falls back to the union of the
      chunk "missing" lists (minus anything matched) when the basket is
      not supplied.
    * ``physical_count_match`` — "no" when there is a confirmed missing or
      extra item; otherwise the representative chunk's value.
    * ``narrative`` — distinct chunk narratives joined.
    * ``_chunked`` — audit block: chunk count + whether chunks disagreed on
      customer presence (guard fired).
    """
    dicts = chunk_dicts or []
    valid = [c for c in dicts
             if isinstance(c, dict) and not c.get("error")]
    if not valid:
        # Every chunk errored — surface the first so the failure is visible.
        return (dicts or [{}])[0] or {}

    # Each chunk dict is an ``_adapt_vlm_result`` envelope with the SCO
    # verdict inside ``["parsed"]`` (production), OR a flat verdict dict
    # (unit tests). Aggregate over the VERDICTS either way, and re-wrap the
    # result in the same shape so downstream recording (which reads
    # ``["parsed"]``) preserves the aggregate + the _chunked block.
    _enveloped = isinstance(valid[0].get("parsed"), dict)

    def _verdict(c: dict) -> dict:
        p = c.get("parsed")
        v = p if isinstance(p, dict) else c
        # A chunk sometimes lands the whole SCO verdict as a JSON *string*
        # in ``narrative`` instead of parsed fields — recover it so the
        # union sees the real matched/extra items and the narrative reads
        # as prose, not a JSON blob.
        nv = v.get("narrative")
        if isinstance(nv, str) and nv.lstrip()[:1] in ("{", "["):
            try:
                recovered = json.loads(nv)
            except Exception:
                recovered = None
            if isinstance(recovered, dict) and (
                    "matched_items" in recovered
                    or "physical_count_match" in recovered):
                merged = dict(v)
                merged.update(recovered)
                # Keep an inner narrative sentence if present, else blank
                # (never the raw JSON string).
                inner = recovered.get("narrative")
                merged["narrative"] = inner if isinstance(inner, str) \
                    and not inner.lstrip()[:1] in ("{", "[") else ""
                return merged
        return v

    verdicts = [_verdict(c) for c in valid]

    matched, extra = [], []
    seen_matched, seen_extra = set(), set()
    for c in verdicts:
        for it in (c.get("matched_items") or []):
            k = _item_key(it)
            if k and k not in seen_matched:
                seen_matched.add(k)
                matched.append(it)
        for it in (c.get("extra_visible_items") or []):
            k = _item_key(it)
            if k and k not in seen_extra:
                seen_extra.add(k)
                extra.append(it)

    presents = [bool(c.get("customer_present")) for c in verdicts]
    customer_present = any(presents)
    disagreement = (True in presents) and (False in presents)

    missing = []
    if basket_descriptions is not None:
        for d in basket_descriptions:
            if _item_key(d) not in seen_matched:
                missing.append({"pos_item": d,
                                "reason": "not visible in any chunk"})
    else:
        seen_missing = set()
        for c in verdicts:
            for it in (c.get("missing_visible_items") or []):
                k = _item_key(it)
                if k and k not in seen_matched and k not in seen_missing:
                    seen_missing.add(k)
                    missing.append(it)

    # Representative chunk for the residual match verdicts: the one that saw
    # the most (customer present, then most matched items).
    rep = max(verdicts, key=lambda c: (bool(c.get("customer_present")),
                                    len(c.get("matched_items") or [])))
    physical = "no" if (missing or extra) else rep.get("physical_count_match")

    narratives = []
    for c in verdicts:
        nv = (c.get("narrative") or "").strip()
        if nv and nv not in narratives:
            narratives.append(nv)

    verdict = dict(rep)
    verdict.update({
        "customer_present": customer_present,
        "matched_items": matched,
        "extra_visible_items": extra,
        "missing_visible_items": missing,
        "physical_count_match": physical,
        "video_usable": any(bool(c.get("video_usable")) for c in verdicts),
        "narrative": " | ".join(narratives),
        "_chunked": {
            "chunks": len(verdicts),
            "customer_present_disagreement": disagreement,
            "guard_applied_customer_present": disagreement and customer_present,
        },
    })
    if _enveloped:
        # Re-wrap in the _adapt_vlm_result envelope (provider/model_name/…)
        # with the aggregated verdict in ["parsed"] so downstream recording
        # (which reads ["parsed"] -> output_json) keeps the aggregate +
        # the _chunked audit block.
        env = {k: v for k, v in valid[0].items() if k != "parsed"}
        env["parsed"] = verdict
        return env
    return verdict
