"""SCO basket-match v2 output schema.

Mirrors the v2 prompt's JSON shape. Backward-compat: v1's
``ScoBasketMatch`` continues to ship in
``reasoning.schemas.sco_basket_match`` for callers that haven't
migrated. ``parse_or_fallback`` here always returns a v2 object;
the policy reads v2-specific fields.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


_TRI = Literal["yes", "no", "uncertain"]
_VISIBLE_COUNT = Literal["one", "multiple", "uncertain"]
_CONFIDENCE = Literal["high", "medium", "low"]


def _norm_tri(v):
    if not isinstance(v, str):
        return "uncertain"
    s = v.strip().lower()
    if s in ("yes", "true", "match", "matches", "ok"):
        return "yes"
    if s in ("no", "false", "mismatch", "differs"):
        return "no"
    return "uncertain"


def _norm_count(v):
    if not isinstance(v, str):
        return "uncertain"
    s = v.strip().lower()
    if s in ("one", "single", "1"):
        return "one"
    if s in ("multiple", "many", "several"):
        return "multiple"
    return "uncertain"


def _norm_conf(v):
    if not isinstance(v, str):
        return "low"
    s = v.strip().lower()
    if s in ("high", "h"):
        return "high"
    if s in ("medium", "med", "m"):
        return "medium"
    return "low"


class MatchedItemV2(BaseModel):
    model_config = ConfigDict(extra="ignore")
    pos_item: str = ""
    group_id: str = ""
    visible_count_class: _VISIBLE_COUNT = "uncertain"

    @field_validator("visible_count_class", mode="before")
    @classmethod
    def _v(cls, v): return _norm_count(v)

    @field_validator("pos_item", "group_id", mode="before")
    @classmethod
    def _str_or_empty(cls, v):
        # Real Gemma emits JSON `null` for group_id when it cannot
        # tie a POS line to a specific canonical group; pydantic
        # would otherwise fall back the whole schema. Coerce
        # None -> "".
        return "" if v is None else v


class MissingItemV2(BaseModel):
    model_config = ConfigDict(extra="ignore")
    pos_item: str = ""
    reason: str = ""

    @field_validator("pos_item", "reason", mode="before")
    @classmethod
    def _str_or_empty(cls, v):
        return "" if v is None else v


class ExtraItemV2(BaseModel):
    model_config = ConfigDict(extra="ignore")
    group_id: str = ""
    description: str = ""

    @field_validator("group_id", "description", mode="before")
    @classmethod
    def _str_or_empty(cls, v):
        return "" if v is None else v


class ScoBasketMatchV2(BaseModel):
    """v2 structured output the VLM must produce."""

    model_config = ConfigDict(extra="ignore")

    physical_count_match: _TRI
    semantic_identity_match: _TRI
    matched_items: list[MatchedItemV2] = Field(default_factory=list)
    missing_visible_items: list[MissingItemV2] = Field(default_factory=list)
    extra_visible_items: list[ExtraItemV2] = Field(default_factory=list)
    uncertainty_reason: str = ""
    video_usable: bool = True
    confidence: _CONFIDENCE = "low"
    narrative: str = ""

    @field_validator("physical_count_match", mode="before")
    @classmethod
    def _p(cls, v): return _norm_tri(v)

    @field_validator("semantic_identity_match", mode="before")
    @classmethod
    def _s(cls, v): return _norm_tri(v)

    @field_validator("confidence", mode="before")
    @classmethod
    def _c(cls, v): return _norm_conf(v)

    @field_validator("uncertainty_reason", "narrative",
                      mode="before")
    @classmethod
    def _str_or_empty(cls, v):
        # VLMs occasionally emit JSON null for optional strings.
        return "" if v is None else v


def parse_or_fallback_v2(raw: dict) -> ScoBasketMatchV2:
    """Strict-ish parse. Tolerant of unknown keys and missing required
    fields. Returns a low-confidence uncertain object on any
    structural failure so the policy stage sees a usable result.

    Also bridges from v1-shaped output (e.g. if a provider still
    emits the old ``basket_match``/``matched``/``missing``/``extras``
    keys): the v1 ``basket_match`` is mapped to BOTH
    ``physical_count_match`` and ``semantic_identity_match`` so the
    policy doesn't crash, with ``uncertainty_reason`` flagging the
    legacy shape.
    """
    if not isinstance(raw, dict):
        return ScoBasketMatchV2(
            physical_count_match="uncertain",
            semantic_identity_match="uncertain",
            confidence="low", video_usable=False,
            narrative="VLM output is not a JSON object.",
        )
    if "physical_count_match" not in raw and "basket_match" in raw:
        # Best-effort v1 → v2 shim. Don't claim semantic identity from
        # v1: it conflated the two questions, so we degrade to uncertain.
        legacy = raw.get("basket_match")
        raw = dict(raw)
        raw["physical_count_match"] = legacy
        raw["semantic_identity_match"] = "uncertain"
        raw["uncertainty_reason"] = (
            "VLM returned v1 schema; semantic identity downgraded "
            "to uncertain by parser shim."
        )
        # Best-effort field migration so matched_items / extras render.
        if "matched_items" not in raw:
            raw["matched_items"] = [
                {"pos_item": m.get("pos_item", ""), "group_id": "",
                 "visible_count_class": m.get("visible_count_class",
                                              "uncertain")}
                for m in (raw.get("matched") or []) if isinstance(m, dict)
            ]
        if "missing_visible_items" not in raw:
            raw["missing_visible_items"] = list(raw.get("missing") or [])
        if "extra_visible_items" not in raw:
            raw["extra_visible_items"] = [
                {"group_id": "",
                 "description": (e.get("visible_item") or "")}
                for e in (raw.get("extras") or []) if isinstance(e, dict)
            ]
    try:
        return ScoBasketMatchV2.model_validate(raw)
    except Exception:
        return ScoBasketMatchV2(
            physical_count_match="uncertain",
            semantic_identity_match="uncertain",
            confidence="low",
            narrative="VLM output did not match schema.",
        )
