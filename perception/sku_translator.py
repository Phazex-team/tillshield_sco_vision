"""POS SKU → Falcon visual-query translator (Phase 3).

POS line-item strings look like ``DOVE BAR SOAP 100G WHITE`` or
``COKE CAN 330ML 6X``. Falcon-Perception is an open-vocabulary
referring detector that matches natural-language phrases — so we
turn each line into a short visual phrase (``dove soap bar``,
``coke can``) before passing it down as a category.

v1 is deterministic + local:
  * STRIP size/UoM/noise tokens.
  * PRESERVE brand tokens — ``coke can`` is sharper than ``can``.
  * Optional operator overrides at ``config/sku_overrides.yaml``.
  * File-backed JSON cache at ``storage/sku_translator/cache.json``
    to keep repeated lookups cheap.
  * NO network. NO LLM. (Deferred to v2 if cleanup proves
    insufficient on real POS data.)
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from threading import Lock
from typing import Optional


log = logging.getLogger(__name__)


# Tokens that add noise without visual value: sizes, units, packaging
# qualifiers, neutral adjectives. Brand tokens are NOT in here.
_NOISE_PATTERNS: tuple[str, ...] = (
    # 100G, 330ML, 6 PCS, 12 PACK etc.
    r"\b\d+(?:\.\d+)?\s*(?:G|GR|GM|GRAMS?|KG|KGS|ML|L|LTR|LITRES?|OZ|LB|LBS|"
    r"CT|CTN|PCS?|PIECES?|PK|PCK|PACK|EA|EACH|DOZ|DOZEN)\b",
    # 6X330ML, 2X100
    r"\b\d+\s*X\s*\d+(?:\.\d+)?\s*[A-Z]*\b",
    # 100%, 2.5%
    r"\b\d+(?:\.\d+)?\s*%\b",
    # generic adjectives that confuse open-vocab matching
    r"\b(?:LARGE|SMALL|MEDIUM|REGULAR|MINI|JUMBO|GIANT|XL|XXL|XS|"
    r"WHITE|BLACK|BLUE|RED|YELLOW|GREEN|BROWN|GRAY|GREY)\b",
    # "PACK OF 6"
    r"\b(?:PACK|PKT|BOX|CAN|JAR|BTL|BOTTLE|TIN|BAG|TUB)\s+OF\s+\d+\b",
    # parenthesized notes
    r"\([^)]*\)",
)

# Suffix hints that sharpen open-vocab matching when the description
# only carries the product class (e.g. "MILK" → "milk bottle").
_VISUAL_SUFFIX_HINTS: dict[str, str] = {
    "soap": " bar",
    "shampoo": " bottle",
    "milk": " bottle",
    "coke": " can",
    "pepsi": " can",
    "soda": " can",
    "juice": " bottle",
    "water": " bottle",
    "chips": " packet",
    "biscuit": " packet",
    "cookies": " packet",
    "chocolate": " bar",
    "shirt": "",
    "t-shirt": "",
}


# Module-level lazy state. `init_translator` populates from disk.
_CACHE: dict[str, str] = {}
_CACHE_LOCK = Lock()
_CACHE_PATH: Optional[Path] = None
_OVERRIDES: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().upper()


def _load_overrides(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        import yaml
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return {_norm(str(k)): str(v) for k, v in data.items() if v}
    except Exception:
        log.exception("sku_translator: failed to load overrides at %s", path)
        return {}


def _load_cache_from_disk(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return {str(k): str(v) for k, v in (data or {}).items()}
    except Exception:
        log.exception("sku_translator: failed to load cache at %s", path)
        return {}


def _save_cache_to_disk(path: Path, cache: dict[str, str]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(cache, f, indent=2, sort_keys=True)
        tmp.replace(path)
    except Exception:
        log.exception("sku_translator: failed to save cache at %s", path)


def init_translator(*, cache_path: Optional[str] = None,
                    overrides_path: Optional[str] = None) -> None:
    """Load overrides + cache from disk. Safe to call repeatedly (idempotent).

    Either path may be ``None`` to leave that source untouched. Cache
    writes are best-effort — failures log and continue (the in-memory
    cache always works).
    """
    global _CACHE_PATH, _OVERRIDES
    if overrides_path is not None:
        _OVERRIDES = _load_overrides(Path(overrides_path))
    if cache_path is not None:
        _CACHE_PATH = Path(cache_path)
        with _CACHE_LOCK:
            _CACHE.update(_load_cache_from_disk(_CACHE_PATH))


def reset_for_tests() -> None:
    """Clear in-memory state. Test fixtures should call this."""
    global _CACHE_PATH, _OVERRIDES
    with _CACHE_LOCK:
        _CACHE.clear()
    _CACHE_PATH = None
    _OVERRIDES = {}


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

def cleanup(description: str) -> str:
    """Strip size/UoM/noise tokens but preserve brand. Returns a
    lowercased, single-line phrase suitable for Falcon."""
    if not description:
        return ""
    s = _norm(description)
    for pat in _NOISE_PATTERNS:
        s = re.sub(pat, "", s, flags=re.IGNORECASE)
    # Replace remaining punctuation (but keep hyphens for compound brands
    # like t-shirt) with whitespace.
    s = re.sub(r"[^\w\s\-&]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    if not s:
        return ""
    # Apply visual suffix hint if a known product word is in the phrase
    # and the suffix isn't already present.
    for word, hint in _VISUAL_SUFFIX_HINTS.items():
        if word in s:
            if hint and hint.strip() not in s:
                s = (s + hint).strip()
            break
    return s


def translate(description: str) -> str:
    """Translate one POS line item to a Falcon visual phrase.

    Order: explicit overrides → in-memory cache → deterministic cleanup.
    A successful cleanup result is cached (in-memory and on disk if
    init_translator was called with a cache_path).
    """
    key = _norm(description)
    if not key:
        return ""
    if _OVERRIDES and key in _OVERRIDES:
        return _OVERRIDES[key]
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            return cached
    out = cleanup(description)
    if out:
        with _CACHE_LOCK:
            _CACHE[key] = out
            if _CACHE_PATH is not None:
                _save_cache_to_disk(_CACHE_PATH, dict(_CACHE))
    return out


# ---------------------------------------------------------------------------
# Falcon categories from a POS basket
# ---------------------------------------------------------------------------

# The generic SCO catch-all. Without this, Falcon can never detect
# extras/unmatched items that aren't on the POS bill. The unique key
# prevents FalconClient's reserved-key gate from refusing it.
GENERIC_PRODUCTS_KEY = "sco_generic_products"
GENERIC_PRODUCTS_QUERY = ("product, retail item, package, bottle, "
                          "box, bag, clothing, can, packet, jar")


def build_falcon_categories_from_pos(pos_event) -> dict[str, str]:
    """Build a Falcon ``categories`` dict from a PosEvent's basket items.

    Output shape: ``{unique_key: natural_language_query}`` where keys
    never collide with FalconClient.RESERVED_CATEGORY_KEYS (item/person/
    receipt). Always includes a generic-product catch-all so the
    detector can surface extras that aren't on the POS bill.

    Empty / malformed input → returns just the generic catch-all.
    """
    cats: dict[str, str] = {GENERIC_PRODUCTS_KEY: GENERIC_PRODUCTS_QUERY}
    for i, it in enumerate(_items_from_pos(pos_event)):
        desc = (it.get("description") or it.get("name")
                or it.get("item_description") or it.get("sku") or "")
        if not desc:
            continue
        query = translate(str(desc))
        if not query:
            continue
        # Tag with a stable per-line key so audit downstream can map
        # detections back to POS line index without overwriting either
        # reserved keys or the generic catch-all.
        cats[f"sco_item_{i:03d}"] = query
    return cats


def _items_from_pos(pos_event) -> list[dict]:
    """Extract the basket items list from a PosEvent.raw_payload."""
    if pos_event is None:
        return []
    raw = getattr(pos_event, "raw_payload", None)
    if not isinstance(raw, dict):
        return []
    items = raw.get("items")
    if not isinstance(items, list):
        return []
    return [it for it in items if isinstance(it, dict)]
