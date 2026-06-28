"""sco_basket_match_v2 prompt + schema tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def test_prompt_v2_renders_basket_groups_and_episode():
    from reasoning.prompts.sco_basket_match_v2 import build_user_prompt_v2
    out = build_user_prompt_v2(
        basket=[{"description": "Biriyani Hot Food"},
                {"description": "Curry Hot Food"}],
        canonical_groups=[
            {"group_id": "sco_group_001", "matched_pos_item": None,
             "source_labels": ["sco_generic_food_container"],
             "confidence": "low", "is_extra_candidate": True},
            {"group_id": "sco_group_002", "matched_pos_item": None,
             "source_labels": ["sco_generic_plastic_food_box"],
             "confidence": "low", "is_extra_candidate": True},
        ],
        episode_meta={"start": "x", "end": "y", "ambiguous": False,
                       "reason": "item_occupancy",
                       "coverage_ratio": 0.5},
    )
    assert "POS basket size: 2 line(s)" in out
    assert "Biriyani Hot Food" in out and "Curry Hot Food" in out
    assert "Total canonical groups: 2" in out
    assert "sco_group_001" in out and "sco_group_002" in out
    assert "item_occupancy" in out


def test_prompt_v2_schema_keys_match_pydantic_schema():
    from reasoning.prompts.sco_basket_match_v2 import build_user_prompt_v2
    out = build_user_prompt_v2(basket=[], canonical_groups=[])
    for k in ("physical_count_match", "semantic_identity_match",
              "matched_items", "missing_visible_items",
              "extra_visible_items", "uncertainty_reason",
              "video_usable", "confidence", "narrative"):
        assert f'"{k}"' in out, f"prompt v2 missing {k!r}"


def test_system_prompt_v2_distinguishes_count_vs_identity():
    from reasoning.prompts.sco_basket_match_v2 import build_system_prompt_v2
    sys_p = build_system_prompt_v2().lower()
    assert "physical_count_match" in sys_p
    assert "semantic_identity_match" in sys_p
    # The closed-container rule
    assert "closed/opaque containers" in sys_p \
        or "closed" in sys_p
    # The anti-collapse rule
    assert "do not collapse two physical groups" in sys_p


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_v2_schema_validates_well_formed_uncertainty_case():
    from reasoning.schemas.sco_basket_match_v2 import (
        ScoBasketMatchV2, parse_or_fallback_v2,
    )
    obj = parse_or_fallback_v2({
        "physical_count_match": "yes",
        "semantic_identity_match": "uncertain",
        "matched_items": [
            {"pos_item": "Biriyani Hot Food", "group_id": "sco_group_001",
             "visible_count_class": "one"},
            {"pos_item": "Curry Hot Food", "group_id": "sco_group_002",
             "visible_count_class": "one"},
        ],
        "missing_visible_items": [],
        "extra_visible_items": [],
        "uncertainty_reason": "items inside closed takeaway containers",
        "video_usable": True,
        "confidence": "high",
        "narrative": "Two takeaway containers visible; contents not legible.",
    })
    assert isinstance(obj, ScoBasketMatchV2)
    assert obj.physical_count_match == "yes"
    assert obj.semantic_identity_match == "uncertain"
    assert obj.confidence == "high"
    assert len(obj.matched_items) == 2


@pytest.mark.parametrize("raw,expected", [
    ("yes", "yes"), ("YES", "yes"), ("match", "yes"),
    ("no", "no"), ("MISMATCH", "no"),
    ("uncertain", "uncertain"), ("dunno", "uncertain"), (None, "uncertain"),
])
def test_v2_normalises_tri_state(raw, expected):
    from reasoning.schemas.sco_basket_match_v2 import parse_or_fallback_v2
    obj = parse_or_fallback_v2({"physical_count_match": raw,
                                  "semantic_identity_match": raw,
                                  "video_usable": True})
    assert obj.physical_count_match == expected
    assert obj.semantic_identity_match == expected


def test_v2_garbage_input_returns_uncertain_low():
    from reasoning.schemas.sco_basket_match_v2 import parse_or_fallback_v2
    obj = parse_or_fallback_v2({"completely": "wrong"})
    assert obj.physical_count_match == "uncertain"
    assert obj.semantic_identity_match == "uncertain"
    assert obj.confidence == "low"


def test_v1_shape_is_shimmed_to_v2():
    """When a provider still emits the v1 keys, the v2 parser must
    not crash. Physical_count_match takes the v1 basket_match value;
    semantic_identity_match degrades to uncertain (v1 didn't split
    the two questions)."""
    from reasoning.schemas.sco_basket_match_v2 import parse_or_fallback_v2
    obj = parse_or_fallback_v2({
        "basket_match": "no",
        "matched": [{"pos_item": "X", "visible_count_class": "one"}],
        "missing": [],
        "extras": [{"visible_item": "Y", "note": "n/a"}],
        "video_usable": True, "confidence": "high",
        "narrative": "ok",
    })
    assert obj.physical_count_match == "no"
    assert obj.semantic_identity_match == "uncertain"
    assert obj.confidence == "high"
    assert obj.matched_items[0].pos_item == "X"
    assert obj.extra_visible_items[0].description == "Y"
    assert "v1 schema" in obj.uncertainty_reason.lower()
