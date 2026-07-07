"""Chunked-analysis core: frame partitioning + verdict aggregation + guard."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasoning.chunk_aggregate import (aggregate_chunk_verdicts,  # noqa: E402
                                       partition_frames)


# --- partition_frames -------------------------------------------------------

def test_partition_single_chunk_when_small():
    frames = list(range(30))
    assert partition_frames(frames, chunk_frames=40) == [frames]


def test_partition_overlapping_chunks():
    frames = list(range(100))
    chunks = partition_frames(frames, chunk_frames=40, overlap=4)
    assert all(len(c) <= 40 for c in chunks)
    # overlap: each chunk after the first repeats the tail of the previous
    assert chunks[0][-4:] == chunks[1][:4]
    # full coverage — every frame appears in some chunk
    assert set(f for c in chunks for f in c) == set(frames)


def test_partition_empty():
    assert partition_frames([], chunk_frames=40) == []


# --- aggregation + guard ----------------------------------------------------

def test_guard_customer_present_if_any_chunk_saw_one():
    chunks = [
        {"customer_present": False, "matched_items": []},
        {"customer_present": False, "matched_items": []},
        {"customer_present": True, "matched_items": [{"pos_item": "Milk"}]},
    ]
    agg = aggregate_chunk_verdicts(chunks)
    # The guard: even though 2/3 chunks saw no one, the combined verdict
    # must NOT claim there was no customer.
    assert agg["customer_present"] is True
    assert agg["_chunked"]["customer_present_disagreement"] is True
    assert agg["_chunked"]["guard_applied_customer_present"] is True


def test_matched_and_extra_are_unioned():
    chunks = [
        {"customer_present": True,
         "matched_items": [{"pos_item": "Milk"}],
         "extra_visible_items": [{"description": "Gum"}]},
        {"customer_present": True,
         "matched_items": [{"pos_item": "Bread"}],
         "extra_visible_items": [{"description": "Gum"}]},  # dup
    ]
    agg = aggregate_chunk_verdicts(chunks)
    names = sorted(i["pos_item"] for i in agg["matched_items"])
    assert names == ["Bread", "Milk"]
    assert len(agg["extra_visible_items"]) == 1  # deduped


def test_missing_only_if_no_chunk_saw_it():
    basket = ["Milk", "Bread", "Eggs"]
    chunks = [
        {"customer_present": True, "matched_items": [{"pos_item": "Milk"}]},
        {"customer_present": True, "matched_items": [{"pos_item": "Bread"}]},
    ]
    agg = aggregate_chunk_verdicts(chunks, basket_descriptions=basket)
    missing = [m["pos_item"] for m in agg["missing_visible_items"]]
    # Milk + Bread seen across chunks -> only Eggs is missing.
    assert missing == ["Eggs"]
    assert agg["physical_count_match"] == "no"  # a confirmed missing item


def test_all_seen_no_extras_is_not_forced_no():
    basket = ["Milk"]
    chunks = [{"customer_present": True, "matched_items": [{"pos_item": "Milk"}],
               "physical_count_match": "yes"}]
    agg = aggregate_chunk_verdicts(chunks, basket_descriptions=basket)
    assert agg["missing_visible_items"] == []
    assert agg["physical_count_match"] == "yes"


def test_all_chunks_failed_returns_first():
    chunks = [{"error": "boom"}, {"error": "boom2"}]
    agg = aggregate_chunk_verdicts(chunks)
    assert agg.get("error") == "boom"


def test_narrative_joins_distinct():
    chunks = [
        {"customer_present": True, "narrative": "Customer scans milk."},
        {"customer_present": False, "narrative": "Empty counter."},
        {"customer_present": False, "narrative": "Empty counter."},  # dup
    ]
    agg = aggregate_chunk_verdicts(chunks)
    assert "Customer scans milk." in agg["narrative"]
    assert agg["narrative"].count("Empty counter.") == 1
