"""Config-driven POS event-type normalization for SCO checkout.

Single source of truth for accepting / canonicalising incoming POS
transaction-type strings. Both the direct ingest endpoint
(``POST /api/v1/pos/returns/event``) and the TillShield adapter path
read from here so the two cannot drift apart.

Config shape (under top-level ``sco_checkout`` in ``config.yaml``):

    sco_checkout:
      accept_event_types: [SALE, SCO_SALE, CHECKOUT]
      canonical_event_type: SALE
      roi_name: sco_audit_zone
      sku_overrides_path: config/sku_overrides.yaml
      sku_cache_path: storage/sku_translator/cache.json

If the ``sco_checkout`` block is absent or malformed, the normaliser
accepts no event types. This SCO-only copy must not silently fall back
to return/refund semantics.
"""
from __future__ import annotations

from typing import Optional


def _normalize(raw: str) -> str:
    """Strip whitespace, uppercase, replace spaces and hyphens with underscores."""
    return raw.strip().upper().replace(" ", "_").replace("-", "_")


def _sco_cfg(cfg=None) -> dict:
    """Read the ``sco_checkout`` block from config. Returns ``{}`` if absent
    or if config loading fails (so the normaliser stays defensive)."""
    if cfg is None:
        try:
            from app.config import load_config
            cfg = load_config()
        except Exception:
            return {}
    if cfg is None:
        return {}
    raw = getattr(cfg, "raw", None) or {}
    return raw.get("sco_checkout") or {}


def accepted_aliases(cfg=None) -> set[str]:
    """Set of NORMALISED aliases the current deployment will accept as a
    checkout event. Excludes the legacy fallback set when SCO mode is on."""
    sco = _sco_cfg(cfg)
    aliases = sco.get("accept_event_types") or []
    return {_normalize(a) for a in aliases if a}


def canonical_event_type(cfg=None) -> Optional[str]:
    """Canonical event type to use INTERNALLY (one string, normalised)."""
    sco = _sco_cfg(cfg)
    c = sco.get("canonical_event_type")
    return _normalize(c) if c else None


def normalize_event_type(raw: Optional[str], cfg=None) -> Optional[str]:
    """Return the canonical event type for ``raw`` or ``None`` if rejected.

    Rules:
      * ``None`` / empty / non-string input → ``None``.
      * Normalisation is case-insensitive, strips whitespace, and
        replaces spaces and hyphens with underscores.
      * If the ``sco_checkout`` block is configured AND has both
        ``accept_event_types`` and ``canonical_event_type``: only
        configured aliases normalise (to the canonical); everything else
        returns ``None``. **Legacy refund types are NOT accepted in this
        mode.**
      * If the ``sco_checkout`` block is missing/incomplete: reject the
        event by returning ``None``.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    norm = _normalize(raw)
    aliases = accepted_aliases(cfg)
    canonical = canonical_event_type(cfg)
    if aliases and canonical:
        return canonical if norm in aliases else None
    return None


def case_opening_types(cfg=None) -> set[str]:
    """The set of event types that open a case in this deployment.

    SCO mode: ``{canonical_event_type}`` only.
    Missing/malformed SCO config: empty set.
    """
    canonical = canonical_event_type(cfg)
    if canonical:
        return {canonical}
    return set()
