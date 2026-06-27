"""Phase 0.5 — FalconClient category merge correctness.

Verifies that:
  * `DEFAULT_CATEGORIES` (item, receipt, person) always survive a
    `detect_on_frames` call, even when the caller passes extra
    `categories`. Prior behavior silently dropped them.
  * Reserved keys (`item`, `person`, `receipt`) cannot be overwritten
    by a `categories=` payload — downstream consumers depend on them.
  * The `categories` parameter is plumbed through
    `perception.pipeline.run_perception_on_window` to
    `FalconClient.detect_on_frames`.
  * Backward-compatible `query=` semantics are preserved.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
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

def _mk_frame(idx: int = 0):
    img = Image.new("RGB", (640, 480), (0, 0, 0))
    return (idx, datetime(2026, 6, 15, 14, 0, 0), img)


def _capturing_falcon_client():
    """Build a FalconClient with the underlying detector replaced by a
    capture spy. Returns (client, captured_queries) where the spy
    records every (label, query_text) pair passed to detector.detect().
    """
    from perception.falcon_client import FalconClient

    client = FalconClient(model_name="stub")
    captured: list[tuple[str, str]] = []

    def _detect(img, *, query):
        captured.append(("__call__", query))
        return ([], [])  # ndarray-like result + empty detections list

    client._detector = MagicMock()
    client._detector.detect.side_effect = _detect
    return client, captured


# ---------------------------------------------------------------------------
# 1. Defaults survive when no categories are passed
# ---------------------------------------------------------------------------

def test_defaults_only_calls_every_default_category():
    client, captured = _capturing_falcon_client()
    client.detect_on_frames([_mk_frame()])

    queries = [q for _, q in captured]
    defaults = client.DEFAULT_CATEGORIES
    assert defaults["item"] in queries
    assert defaults["receipt"] in queries
    assert defaults["person"] in queries
    assert len(queries) == len(defaults), \
        "exactly the defaults should be queried when nothing is overridden"


# ---------------------------------------------------------------------------
# 2. Custom additions preserve defaults (the main bug fix)
# ---------------------------------------------------------------------------

def test_custom_categories_preserve_all_defaults():
    client, captured = _capturing_falcon_client()
    extras = {
        "sku_coke_can": "coke can, soda can",
        "sku_milk_bottle": "milk bottle",
        "sco_generic": "product, package, retail item, bag, box",
    }
    client.detect_on_frames([_mk_frame()], categories=extras)

    queries = [q for _, q in captured]
    defaults = client.DEFAULT_CATEGORIES
    # All defaults must still be queried
    assert defaults["item"] in queries, "default 'item' query lost"
    assert defaults["receipt"] in queries, "default 'receipt' query lost"
    assert defaults["person"] in queries, "default 'person' query lost — " \
        "person-detection silently disabled"
    # And all extras must be queried
    for v in extras.values():
        assert v in queries, f"custom category {v!r} not queried"


def test_generic_product_query_survives_alongside_pos_categories():
    """Specific test the user/council called out: the generic 'product'
    fallback must coexist with POS-derived SKU categories."""
    client, captured = _capturing_falcon_client()
    pos = {"sku_x": "milk bottle"}
    generic = {"sco_generic_products":
               "product, retail item, package, bottle, box, bag, clothing"}
    merged = {**pos, **generic}
    client.detect_on_frames([_mk_frame()], categories=merged)

    queries = [q for _, q in captured]
    assert generic["sco_generic_products"] in queries
    assert pos["sku_x"] in queries
    # And defaults
    assert client.DEFAULT_CATEGORIES["person"] in queries


# ---------------------------------------------------------------------------
# 3. Reserved keys cannot be overwritten
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reserved_key", ["item", "person", "receipt"])
def test_reserved_keys_cannot_be_overwritten_by_categories(reserved_key, caplog):
    client, captured = _capturing_falcon_client()
    spoof = "TOTALLY DIFFERENT QUERY THAT SHOULD BE REJECTED"
    with caplog.at_level(logging.WARNING):
        client.detect_on_frames(
            [_mk_frame()],
            categories={reserved_key: spoof},
        )

    queries = [q for _, q in captured]
    assert spoof not in queries, \
        f"reserved key {reserved_key!r} was overwritten — defaults compromised"
    assert client.DEFAULT_CATEGORIES[reserved_key] in queries, \
        f"default for reserved key {reserved_key!r} not preserved"
    # And a warning was logged
    assert any("refusing to overwrite reserved category" in r.message
               for r in caplog.records), \
        "expected a warning when categories= tries to overwrite a reserved key"


# ---------------------------------------------------------------------------
# 4. Back-compat: query= still works as before
# ---------------------------------------------------------------------------

def test_query_param_still_accepted_without_categories():
    """When categories=None and query=X, behavior must match the prior
    implementation: defaults are queried, query= is only used to provide
    the 'item' default if it isn't already present (it always is).
    """
    client, captured = _capturing_falcon_client()
    client.detect_on_frames([_mk_frame()], query="custom item phrase")

    queries = [q for _, q in captured]
    # Defaults present
    assert client.DEFAULT_CATEGORIES["item"] in queries
    assert client.DEFAULT_CATEGORIES["receipt"] in queries
    assert client.DEFAULT_CATEGORIES["person"] in queries


# ---------------------------------------------------------------------------
# 5. Pipeline plumbing — categories reach FalconClient
# ---------------------------------------------------------------------------

def test_pipeline_plumbs_categories_through_to_falcon_client(monkeypatch):
    """`run_perception_on_window(falcon_categories=...)` must forward the
    dict into `FalconClient.detect_on_frames(categories=...)`.
    """
    from perception import pipeline as ppl
    from perception.sampling import SamplingPolicy

    captured_kwargs = {}

    class FakeFalcon:
        DEFAULT_CATEGORIES = {"item": "x", "person": "y", "receipt": "z"}

        def detect_on_frames(self, frames, **kwargs):
            captured_kwargs.update(kwargs)
            return []  # no detections — short-circuits later stages

    # Force the sampler to return one synthetic frame so Falcon is invoked.
    def _fake_sample(*args, **kwargs):
        return [_mk_frame()]

    monkeypatch.setattr(ppl, "_sample_frames", _fake_sample)

    custom_categories = {
        "sku_a": "thing a",
        "sco_generic": "package, bag, bottle",
    }
    ppl.run_perception_on_window(
        window_path="ignored",
        fps=25,
        zones=[],
        falcon_client=FakeFalcon(),
        sam2_client=None,
        sampling=SamplingPolicy(base_fps=1),
        falcon_categories=custom_categories,
        sam2_enabled=False,
        ocr_enabled=False,
    )

    assert "categories" in captured_kwargs, \
        "run_perception_on_window did not forward categories= to FalconClient"
    assert captured_kwargs["categories"] == custom_categories, \
        "categories were modified in transit"
