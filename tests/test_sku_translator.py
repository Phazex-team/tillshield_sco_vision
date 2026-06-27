"""Phase 3 — perception.sku_translator tests.

Covers deterministic cleanup, brand preservation, override priority,
cache hit-on-repeat, and POS-basket → Falcon-categories build.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _reset_translator():
    from perception import sku_translator
    sku_translator.reset_for_tests()
    yield
    sku_translator.reset_for_tests()


# ---------------------------------------------------------------------------
# 1. Deterministic cleanup — strip noise, preserve brand
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected_contains,expected_absent", [
    ("DOVE BAR SOAP 100G WHITE", ["dove", "soap"], ["100", "white"]),
    ("COKE CAN 330ML", ["coke", "can"], ["330", "ml"]),
    ("PEPSI 6X330ML", ["pepsi", "can"], ["6", "330", "ml"]),
    ("ORANGE JUICE 1L", ["juice", "bottle"], ["1l", " 1 "]),
    ("MILK PACK OF 6", ["milk", "bottle"], ["pack of 6", " 6 "]),
    ("LARGE T-SHIRT BLACK", ["t-shirt"], ["large", "black"]),
])
def test_cleanup_strips_noise_preserves_brand(raw, expected_contains, expected_absent):
    from perception.sku_translator import cleanup
    out = cleanup(raw)
    for token in expected_contains:
        assert token in out, f"missing {token!r} from cleanup({raw!r}) → {out!r}"
    for token in expected_absent:
        assert token not in out, \
            f"noise token {token!r} survived cleanup({raw!r}) → {out!r}"


def test_cleanup_empty_input():
    from perception.sku_translator import cleanup
    assert cleanup("") == ""
    assert cleanup("   ") == ""


# ---------------------------------------------------------------------------
# 2. Overrides take priority
# ---------------------------------------------------------------------------

def test_overrides_take_priority(tmp_path):
    from perception import sku_translator
    overrides = tmp_path / "sku_overrides.yaml"
    overrides.write_text("SOMETHING WEIRD SKU 123: 'red toy car'\n")
    sku_translator.init_translator(overrides_path=str(overrides))
    assert sku_translator.translate("SOMETHING WEIRD SKU 123") == "red toy car"


# ---------------------------------------------------------------------------
# 3. Cache hit on repeat translate
# ---------------------------------------------------------------------------

def test_cache_hit_on_repeat_translate(tmp_path):
    from perception import sku_translator
    cache_path = tmp_path / "cache.json"
    sku_translator.init_translator(cache_path=str(cache_path))
    out1 = sku_translator.translate("Some Product Name 500G")
    out2 = sku_translator.translate("some product name 500g")
    assert out1 == out2
    # And the cache file got written
    assert cache_path.exists()
    saved = json.loads(cache_path.read_text())
    # Normalised key (uppercased, single-spaced)
    assert "SOME PRODUCT NAME 500G" in saved


# ---------------------------------------------------------------------------
# 4. Build Falcon categories from POS basket
# ---------------------------------------------------------------------------

def _pos_event(items):
    return SimpleNamespace(raw_payload={"items": items})


def test_build_falcon_categories_returns_generic_for_empty_basket():
    from perception.sku_translator import (
        build_falcon_categories_from_pos,
        GENERIC_PRODUCTS_KEY,
    )
    cats = build_falcon_categories_from_pos(_pos_event([]))
    assert list(cats.keys()) == [GENERIC_PRODUCTS_KEY]


def test_build_falcon_categories_returns_generic_for_missing_pos():
    from perception.sku_translator import (
        build_falcon_categories_from_pos,
        GENERIC_PRODUCTS_KEY,
    )
    assert GENERIC_PRODUCTS_KEY in build_falcon_categories_from_pos(None)


def test_build_falcon_categories_per_line_unique_keys_and_generic_present():
    from perception.sku_translator import (
        build_falcon_categories_from_pos,
        GENERIC_PRODUCTS_KEY,
    )
    items = [
        {"description": "DOVE SOAP BAR 100G", "sku": "111"},
        {"description": "COKE CAN 330ML", "sku": "222"},
        {"description": "ORANGE JUICE 1L", "sku": "333"},
    ]
    cats = build_falcon_categories_from_pos(_pos_event(items))
    # Generic catch-all is always present
    assert GENERIC_PRODUCTS_KEY in cats
    # Each line gets a unique key
    assert "sco_item_000" in cats
    assert "sco_item_001" in cats
    assert "sco_item_002" in cats
    # And the queries are the cleaned-up forms
    assert "soap" in cats["sco_item_000"]
    assert "coke" in cats["sco_item_001"]
    assert "juice" in cats["sco_item_002"]


def test_build_falcon_categories_skips_empty_descriptions():
    from perception.sku_translator import build_falcon_categories_from_pos
    items = [
        {"description": "DOVE SOAP 100G"},
        {"description": ""},
        {"sku": ""},
        {"name": "PEPSI CAN"},
    ]
    cats = build_falcon_categories_from_pos(_pos_event(items))
    # 0 (soap) and 3 (pepsi) survive; 1 and 2 don't
    assert "sco_item_000" in cats
    assert "sco_item_001" not in cats
    assert "sco_item_002" not in cats
    assert "sco_item_003" in cats


# ---------------------------------------------------------------------------
# 5. Reserved keys are never produced by the builder
# ---------------------------------------------------------------------------

def test_builder_never_produces_reserved_keys():
    from perception.falcon_client import FalconClient
    from perception.sku_translator import build_falcon_categories_from_pos
    items = [{"description": "Anything " + name.upper()}
             for name in FalconClient.RESERVED_CATEGORY_KEYS]
    cats = build_falcon_categories_from_pos(_pos_event(items))
    for reserved in FalconClient.RESERVED_CATEGORY_KEYS:
        assert reserved not in cats, \
            f"builder accidentally produced reserved key {reserved!r}"


# ---------------------------------------------------------------------------
# 6. End-to-end: build categories + pass to FalconClient → defaults survive,
#    POS items present, generic present
# ---------------------------------------------------------------------------

def test_end_to_end_sco_categories_merge_with_defaults():
    from datetime import datetime
    from unittest.mock import MagicMock
    from PIL import Image
    from perception.falcon_client import FalconClient
    from perception.sku_translator import (
        build_falcon_categories_from_pos,
        GENERIC_PRODUCTS_KEY,
    )

    client = FalconClient(model_name="stub")
    captured: list[str] = []

    def _detect(img, *, query):
        captured.append(query)
        return ([], [])

    client._detector = MagicMock()
    client._detector.detect.side_effect = _detect

    items = [
        {"description": "DOVE SOAP BAR 100G"},
        {"description": "COKE CAN 330ML"},
    ]
    cats = build_falcon_categories_from_pos(_pos_event(items))
    img = Image.new("RGB", (640, 480), 0)
    client.detect_on_frames([(0, datetime(2026, 6, 15, 0, 0, 0), img)],
                            categories=cats)

    # Defaults survive
    assert client.DEFAULT_CATEGORIES["person"] in captured
    assert client.DEFAULT_CATEGORIES["item"] in captured
    assert client.DEFAULT_CATEGORIES["receipt"] in captured
    # POS-derived queries flow through
    assert any("soap" in q for q in captured)
    assert any("coke" in q for q in captured)
    # Generic catch-all flows through
    assert cats[GENERIC_PRODUCTS_KEY] in captured
