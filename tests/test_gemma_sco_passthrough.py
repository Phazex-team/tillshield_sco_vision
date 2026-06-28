"""Gemma SCO schema-passthrough regression.

The legacy ``gemma_reasoner._parse_json`` projects everything onto
refund fields (``handover_occurred`` / ``items_handed_over`` /
``item_count`` / ...). When the active prompt is
``sco_basket_match_v1``, the model's JSON has SCO keys
(``basket_match`` / ``matched`` / ``missing`` / ``extras`` /
``video_usable`` / ``confidence`` / ``narrative``). Projection drops
those, the SCO schema parser sees no ``basket_match`` and falls back
to uncertain/low, and the SCO policy emits REVIEW with
``sco_low_confidence`` for the wrong reason.

These tests pin:
  1. SCO prompt → GemmaProvider returns the SCO dict verbatim.
  2. The SCO dict round-trips through ``parse_or_fallback`` and
     ``decide_sco`` → REVIEW with ``sco_extra_candidates``
     (the substantive reason), not ``sco_low_confidence`` /
     schema fallback.
  3. Refund prompt (or missing prompt_version) → Gemma still emits
     the legacy refund shape. Back-compat preserved.
"""
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCO_VLM_JSON = {
    "basket_match": "no",
    "matched": [
        {"pos_item": "Biriyani Hot Food", "visible_count_class": "one"},
        {"pos_item": "Curry Hot Food", "visible_count_class": "one"},
    ],
    "missing": [],
    "extras": [
        {"visible_item": "white container",
         "note": "additional container visible"},
    ],
    "video_usable": True,
    "confidence": "high",
    "narrative": "Two POS items are visible plus one extra candidate.",
}


REFUND_VLM_JSON = {
    "handover_occurred": True,
    "items_handed_over": ["bag"],
    "item_count": 1,
    "customer_description": "tall customer in red shirt",
    "narrative": "Customer handed bag across counter.",
    "confidence": "high",
    "flag_for_review": False,
}


def _data_url_frame(w: int = 32, h: int = 24) -> dict:
    img = Image.new("RGB", (w, h), (10, 20, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return {
        "frame_id": "frame_000000",
        "frame_idx": 0,
        "ts": "2026-06-15T14:02:30",
        "image_url": "data:image/jpeg;base64,"
                     + base64.b64encode(buf.getvalue()).decode("ascii"),
    }


def _manifest(prompt_version):
    from reasoning.providers.base import EvidenceManifest
    meta = {}
    if prompt_version is not None:
        meta["prompt_version"] = prompt_version
    return EvidenceManifest(
        case_id="c1", camera_id="cam_01",
        window_start_ts="2026-06-15T14:02:00",
        window_end_ts="2026-06-15T14:03:00",
        frames=[_data_url_frame()],
        tracks=[], ocr=[],
        system_prompt="(test system prompt)",
        user_prompt="(test user prompt)",
        metadata=meta,
    )


def _provider_with_canned_text(canned_text: str):
    """Build a GemmaProvider whose underlying HTTP call returns canned_text.

    We stub the lazy-built client by hand to avoid spinning up the
    GemmaVideoReasoner constructor's HTTP wiring.
    """
    from reasoning.providers.gemma import GemmaProvider
    from gemma_reasoner import GemmaVideoReasoner

    provider = GemmaProvider(model_name="stub", enabled=True,
                              vllm_url="http://stub")
    # Build a real GemmaVideoReasoner so its `reason()` body is exercised,
    # but stub the HTTP POST so no network is touched.
    real_client = GemmaVideoReasoner(
        model_name="stub", vllm_url="http://stub",
        request_timeout_sec=1.0, request_retries=1, request_retry_backoff_sec=0.0,
    )
    real_client._post_chat = MagicMock(return_value=canned_text)  # type: ignore
    provider._client_cache = real_client
    return provider, real_client


# ---------------------------------------------------------------------------
# 1. SCO passthrough — keys survive
# ---------------------------------------------------------------------------

def test_sco_prompt_keeps_basket_match_keys_in_parsed():
    provider, _ = _provider_with_canned_text(json.dumps(SCO_VLM_JSON))
    result = provider.analyze_evidence(
        _manifest(prompt_version="sco_basket_match_v1"))

    assert result.error is None, result.error
    parsed = result.parsed
    # SCO keys are preserved verbatim
    for key in ("basket_match", "matched", "missing", "extras",
                "video_usable", "confidence", "narrative"):
        assert key in parsed, f"SCO key {key!r} dropped from parsed dict"
    assert parsed["basket_match"] == "no"
    assert parsed["confidence"] == "high"
    assert parsed["matched"][0]["pos_item"] == "Biriyani Hot Food"
    assert parsed["extras"][0]["visible_item"] == "white container"
    # And the legacy refund refresher fields are NOT injected
    assert "handover_occurred" not in parsed
    assert "items_handed_over" not in parsed
    assert "item_count" not in parsed


def test_sco_passthrough_survives_markdown_fence_and_thinking_tags():
    """The model sometimes wraps JSON in ```json ... ``` or emits
    <think>...</think> preamble. The tolerant extractor must still
    pull out a clean SCO dict."""
    noisy = (
        "<think>I see two items and one extra container.</think>\n"
        "```json\n"
        + json.dumps(SCO_VLM_JSON)
        + "\n```\n"
    )
    provider, _ = _provider_with_canned_text(noisy)
    result = provider.analyze_evidence(
        _manifest(prompt_version="sco_basket_match_v1"))
    assert result.error is None
    assert result.parsed.get("basket_match") == "no"
    assert "extras" in result.parsed


# ---------------------------------------------------------------------------
# 2. SCO end-to-end: parse_or_fallback + decide_sco
# ---------------------------------------------------------------------------

def test_sco_passthrough_round_trips_to_decide_sco_review_with_extras():
    from reasoning.schemas.sco_basket_match import parse_or_fallback
    from reasoning.sco_policy import (
        decide_sco, OUTCOME_REVIEW, TAG_EXTRA_CANDIDATES,
        TAG_LOW_CONFIDENCE, TAG_BASKET_MISMATCH,
    )

    provider, _ = _provider_with_canned_text(json.dumps(SCO_VLM_JSON))
    result = provider.analyze_evidence(
        _manifest(prompt_version="sco_basket_match_v1"))

    sco_vlm = parse_or_fallback(result.parsed)
    episode = {
        "start": "2026-06-15T14:02:10", "end": "2026-06-15T14:02:50",
        "ambiguous": False, "reason": "clean_episode",
        "coverage_ratio": 0.50,
    }
    decision = decide_sco(sco_vlm, episode)
    assert decision.outcome == OUTCOME_REVIEW
    # The substantive reason: extra candidate + basket mismatch.
    assert TAG_EXTRA_CANDIDATES in decision.reasons, decision.reasons
    assert TAG_BASKET_MISMATCH in decision.reasons, decision.reasons
    # Specifically NOT a schema-fallback or low-confidence outcome.
    assert TAG_LOW_CONFIDENCE not in decision.reasons, (
        "low-confidence tag fired even though the model returned "
        "confidence=high — schema passthrough is not effective")
    assert sco_vlm.confidence == "high"
    assert sco_vlm.basket_match == "no"
    assert len(sco_vlm.extras) == 1
    assert sco_vlm.extras[0].visible_item == "white container"


# ---------------------------------------------------------------------------
# 3. Refund / missing prompt_version regression — legacy parser unchanged
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prompt_version", ["return_review_v1", None])
def test_refund_or_missing_prompt_uses_legacy_refund_parser(prompt_version):
    provider, _ = _provider_with_canned_text(json.dumps(REFUND_VLM_JSON))
    result = provider.analyze_evidence(_manifest(prompt_version=prompt_version))
    assert result.error is None
    parsed = result.parsed
    # Legacy projected refund shape is present
    for key in ("handover_occurred", "item_count", "items_handed_over",
                "customer_description", "flag_for_review",
                "people", "item_presented", "objects_detected"):
        assert key in parsed, (
            f"legacy refund parser dropped {key!r} for "
            f"prompt_version={prompt_version!r}")
    assert parsed["handover_occurred"] is True
    assert parsed["item_count"] == 1
    assert parsed["items_handed_over"] == ["bag"]


def test_refund_prompt_does_not_request_schema_passthrough():
    """Defensive check on the wire-level switch: the GemmaProvider
    must NOT pass schema_passthrough=True for refund prompts. We
    capture the kwargs forwarded to the underlying reason() call."""
    from reasoning.providers.gemma import GemmaProvider
    captured: dict = {}

    class _FakeClient:
        def reason(self, frames, **kwargs):
            captured.update(kwargs)
            return dict(REFUND_VLM_JSON)

    provider = GemmaProvider(model_name="stub", enabled=True)
    provider._client_cache = _FakeClient()
    provider.analyze_evidence(_manifest(prompt_version="return_review_v1"))
    assert captured.get("schema_passthrough") is False, (
        "refund prompt path must request the legacy refund parser "
        "(schema_passthrough=False)")


def test_sco_prompt_requests_schema_passthrough():
    from reasoning.providers.gemma import GemmaProvider
    captured: dict = {}

    class _FakeClient:
        def reason(self, frames, **kwargs):
            captured.update(kwargs)
            return dict(SCO_VLM_JSON)

    provider = GemmaProvider(model_name="stub", enabled=True)
    provider._client_cache = _FakeClient()
    provider.analyze_evidence(
        _manifest(prompt_version="sco_basket_match_v1"))
    assert captured.get("schema_passthrough") is True, (
        "SCO prompt path must request schema passthrough so the SCO "
        "keys survive the parser")


# ---------------------------------------------------------------------------
# 4. _parse_json_passthrough unit-level
# ---------------------------------------------------------------------------

def test_parse_json_passthrough_returns_dict_verbatim():
    from gemma_reasoner import _parse_json_passthrough
    out = _parse_json_passthrough(json.dumps(SCO_VLM_JSON))
    assert out == SCO_VLM_JSON


def test_parse_json_passthrough_extracts_from_noisy_wrapper():
    from gemma_reasoner import _parse_json_passthrough
    noisy = ("preamble text\n```json\n"
             + json.dumps({"basket_match": "yes", "confidence": "medium"})
             + "\n``` trailing text")
    out = _parse_json_passthrough(noisy)
    assert out == {"basket_match": "yes", "confidence": "medium"}


def test_parse_json_passthrough_on_empty_input_returns_unparsable_marker():
    from gemma_reasoner import _parse_json_passthrough
    out = _parse_json_passthrough("")
    assert out.get("_unparsable") is True
    assert out.get("confidence") == "low"


def test_parse_json_passthrough_on_unparseable_returns_unparsable_marker():
    from gemma_reasoner import _parse_json_passthrough
    out = _parse_json_passthrough("this is definitely not json")
    assert out.get("_unparsable") is True
