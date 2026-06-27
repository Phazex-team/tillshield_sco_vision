"""Phase 1 — pos.event_normalizer tests.

Covers normalization rules, SCO-mode canonicalization, legacy fallback,
and case-opening-types resolution.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sco_cfg(aliases=("SALE", "SCO_SALE", "CHECKOUT"), canonical="SALE"):
    return SimpleNamespace(raw={
        "sco_checkout": {
            "accept_event_types": list(aliases),
            "canonical_event_type": canonical,
            "roi_name": "sco_audit_zone",
        }
    })


def _empty_cfg():
    return SimpleNamespace(raw={})


# ---------------------------------------------------------------------------
# Normalization rules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("SALE", "SALE"),
    ("sale", "SALE"),
    ("  sale  ", "SALE"),
    ("Sale", "SALE"),
    ("sco-sale", "SCO_SALE"),
    ("sco sale", "SCO_SALE"),
    ("SCO_SALE", "SCO_SALE"),
    ("check-out", "CHECK_OUT"),
])
def test_normalize_helper_canonicalises_case_whitespace_separators(raw, expected):
    from pos.event_normalizer import _normalize
    assert _normalize(raw) == expected


# ---------------------------------------------------------------------------
# SCO mode acceptance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw", [
    "SALE", "sale", "Sale", "sCo-sAlE", "checkout", " CHECKOUT ",
    "sco sale", "SCO-SALE",
])
def test_sco_mode_accepts_configured_aliases_and_returns_canonical(raw):
    from pos.event_normalizer import normalize_event_type
    cfg = _sco_cfg()
    assert normalize_event_type(raw, cfg) == "SALE"


@pytest.mark.parametrize("raw", [
    "RETURN", "REFUND", "REPLACEMENT", "SOMETHING_ELSE", "",
    None, "  ", "ZZZ",
])
def test_sco_mode_rejects_non_alias_types(raw):
    from pos.event_normalizer import normalize_event_type
    cfg = _sco_cfg()
    assert normalize_event_type(raw, cfg) is None


def test_sco_mode_case_opening_types_is_canonical_only():
    from pos.event_normalizer import case_opening_types
    cfg = _sco_cfg()
    assert case_opening_types(cfg) == {"SALE"}


def test_sco_mode_canonical_is_returned_normalized():
    from pos.event_normalizer import canonical_event_type
    cfg = _sco_cfg(canonical="sale-event")
    assert canonical_event_type(cfg) == "SALE_EVENT"


# ---------------------------------------------------------------------------
# Legacy fallback (no SCO config)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("RETURN", "RETURN"),
    ("refund", "REFUND"),
    ("Replacement", "REPLACEMENT"),
    ("SALE", None),       # SCO alias rejected in legacy mode
    ("CHECKOUT", None),
])
def test_legacy_fallback_accepts_refund_family(raw, expected):
    from pos.event_normalizer import normalize_event_type
    cfg = _empty_cfg()
    assert normalize_event_type(raw, cfg) == expected


def test_legacy_fallback_case_opening_types_is_return_refund():
    from pos.event_normalizer import case_opening_types
    cfg = _empty_cfg()
    assert case_opening_types(cfg) == {"RETURN", "REFUND"}


# ---------------------------------------------------------------------------
# Defensive: malformed cfg shouldn't crash the normaliser
# ---------------------------------------------------------------------------

def test_partial_sco_config_falls_back_to_legacy():
    """If sco_checkout is present but missing accept_event_types or
    canonical_event_type, fall back to legacy behavior rather than crash."""
    from pos.event_normalizer import normalize_event_type
    cfg = SimpleNamespace(raw={"sco_checkout": {
        "canonical_event_type": "SALE",  # but no accept_event_types
    }})
    # Falls back to legacy because the gate is incomplete
    assert normalize_event_type("RETURN", cfg) == "RETURN"
    assert normalize_event_type("SALE", cfg) is None
