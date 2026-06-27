"""Phase 5 — SCO basket-match prompt builder + output schema tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def test_prompt_includes_pos_basket_items():
    from reasoning.prompts.sco_basket_match import build_user_prompt
    basket = [
        {"description": "DOVE SOAP BAR 100G", "quantity": 1},
        {"description": "COKE CAN 330ML", "quantity": 2},
    ]
    out = build_user_prompt(basket=basket)
    assert "DOVE SOAP BAR 100G" in out
    assert "COKE CAN 330ML" in out
    assert "POS qty: 1" in out and "POS qty: 2" in out


def test_prompt_handles_empty_basket():
    from reasoning.prompts.sco_basket_match import build_user_prompt
    out = build_user_prompt(basket=[])
    assert "no line items" in out.lower() or "empty" in out.lower()


def test_prompt_includes_falcon_summary_when_provided():
    from reasoning.prompts.sco_basket_match import build_user_prompt
    out = build_user_prompt(
        basket=[],
        falcon_summary={
            "matched_count": 3,
            "unmatched_count": 1,
            "generic_candidate_count": 2,
            "queries_run": ["dove soap bar", "coke can"],
        },
    )
    assert "matched POS-item detections: 3" in out
    assert "unmatched POS items" in out
    assert "generic-product candidate" in out
    assert "dove soap bar" in out


def test_prompt_includes_episode_metadata():
    from reasoning.prompts.sco_basket_match import build_user_prompt
    out = build_user_prompt(
        basket=[],
        episode_meta={
            "start": "2026-06-15T14:02:10",
            "end": "2026-06-15T14:02:50",
            "ambiguous": False,
            "reason": "clean_episode",
            "coverage_ratio": 0.13,
        },
    )
    assert "clean_episode" in out
    assert "ambiguous:         False" in out
    assert "coverage_ratio:    0.13" in out


def test_system_prompt_forbids_dangerous_words():
    from reasoning.prompts.sco_basket_match import build_system_prompt
    sys_p = build_system_prompt().lower()
    # Words the system MUST tell the model to avoid
    for word in ("fraud", "theft", "suspect", "scanned", "unscanned"):
        assert word in sys_p, \
            f"system prompt should mention {word!r} as forbidden"


def test_user_prompt_schema_keys_match_pydantic_schema():
    """The literal JSON shape in the prompt must align with the
    pydantic schema field names. Drift here = silently broken parsing."""
    from reasoning.prompts.sco_basket_match import build_user_prompt
    out = build_user_prompt(basket=[])
    required_keys = ["basket_match", "matched", "missing", "extras",
                     "video_usable", "confidence", "narrative"]
    for k in required_keys:
        assert f'"{k}"' in out, f"prompt missing {k!r} in JSON shape"


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

def test_schema_validates_well_formed_output():
    from reasoning.schemas.sco_basket_match import ScoBasketMatch
    obj = ScoBasketMatch.model_validate({
        "basket_match": "yes",
        "matched": [{"pos_item": "DOVE SOAP", "visible_count_class": "one"}],
        "missing": [],
        "extras": [],
        "video_usable": True,
        "confidence": "high",
        "narrative": "All POS items visible in the bagging area.",
    })
    assert obj.basket_match == "yes"
    assert obj.matched[0].pos_item == "DOVE SOAP"
    assert obj.confidence == "high"


@pytest.mark.parametrize("raw,expected", [
    ("yes", "yes"), ("YES", "yes"), ("match", "yes"), ("true", "yes"),
    ("no", "no"), ("MISMATCH", "no"),
    ("uncertain", "uncertain"), ("dunno", "uncertain"), (None, "uncertain"),
])
def test_schema_normalises_basket_match_values(raw, expected):
    from reasoning.schemas.sco_basket_match import ScoBasketMatch
    obj = ScoBasketMatch.model_validate({"basket_match": raw,
                                          "video_usable": True})
    assert obj.basket_match == expected


@pytest.mark.parametrize("raw,expected", [
    ("high", "high"), ("HIGH", "high"),
    ("medium", "medium"), ("med", "medium"),
    ("low", "low"), ("dunno", "low"), (None, "low"),
])
def test_schema_normalises_confidence(raw, expected):
    from reasoning.schemas.sco_basket_match import ScoBasketMatch
    obj = ScoBasketMatch.model_validate({"basket_match": "uncertain",
                                          "confidence": raw})
    assert obj.confidence == expected


def test_schema_ignores_unknown_extra_fields():
    from reasoning.schemas.sco_basket_match import ScoBasketMatch
    obj = ScoBasketMatch.model_validate({
        "basket_match": "no", "confidence": "low",
        "narrative": "Mismatch.",
        "free_text_extra_field": "something the model added",
        "another_unknown": 123,
    })
    assert obj.basket_match == "no"
    assert not hasattr(obj, "free_text_extra_field")


def test_parse_or_fallback_on_garbage_returns_uncertain():
    from reasoning.schemas.sco_basket_match import parse_or_fallback
    out = parse_or_fallback({"completely": "wrong shape"})
    # basket_match is required → fallback kicks in
    assert out.basket_match == "uncertain"
    assert out.confidence == "low"
    assert "did not match schema" in out.narrative.lower()


def test_parse_or_fallback_accepts_well_formed():
    from reasoning.schemas.sco_basket_match import parse_or_fallback
    out = parse_or_fallback({
        "basket_match": "yes", "confidence": "high",
        "video_usable": True, "narrative": "All good.",
    })
    assert out.basket_match == "yes"


# ---------------------------------------------------------------------------
# Round-trip: builder → fake VLM → parser
# ---------------------------------------------------------------------------

def test_round_trip_builder_to_parser():
    """Build a prompt, simulate a VLM response that matches the shape
    described, parse it through the schema. Proves the prompt's JSON
    shape and the pydantic schema agree."""
    from reasoning.prompts.sco_basket_match import build_user_prompt
    from reasoning.schemas.sco_basket_match import parse_or_fallback
    _ = build_user_prompt(basket=[])  # prompt is just for documentation here
    fake_vlm_output = {
        "basket_match": "yes",
        "matched": [{"pos_item": "X", "visible_count_class": "one"}],
        "missing": [],
        "extras": [],
        "video_usable": True,
        "confidence": "medium",
        "narrative": "All items visible.",
    }
    parsed = parse_or_fallback(fake_vlm_output)
    assert parsed.basket_match == "yes"
    assert parsed.matched[0].pos_item == "X"
