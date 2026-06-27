"""SCO basket-match output schema (Phase 5).

Pydantic schema for the structured VLM response. Used to validate +
normalise the JSON the model emits so the policy stage downstream
can rely on shape (no defensive `isinstance` everywhere).

Tolerance choices for v1:
  * Unknown extra fields are ignored — not all providers will be
    perfectly disciplined.
  * Lower-cased enum values are accepted.
  * Missing list fields default to [].
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


_BASKET_MATCH = Literal["yes", "no", "uncertain"]
_CONFIDENCE = Literal["high", "medium", "low"]
_VISIBLE_COUNT = Literal["one", "multiple", "uncertain"]


class MatchedItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    pos_item: str = ""
    visible_count_class: _VISIBLE_COUNT = "uncertain"

    @field_validator("visible_count_class", mode="before")
    @classmethod
    def _norm_count(cls, v):
        if not isinstance(v, str):
            return "uncertain"
        s = v.strip().lower()
        if s in ("one", "single", "1"):
            return "one"
        if s in ("multiple", "many", "several"):
            return "multiple"
        return "uncertain"


class MissingItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    pos_item: str = ""
    reason: str = ""


class ExtraItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    visible_item: str = ""
    note: str = ""


class ScoBasketMatch(BaseModel):
    """Structured output the VLM must produce."""

    model_config = ConfigDict(extra="ignore")

    basket_match: _BASKET_MATCH
    matched: list[MatchedItem] = Field(default_factory=list)
    missing: list[MissingItem] = Field(default_factory=list)
    extras: list[ExtraItem] = Field(default_factory=list)
    video_usable: bool = True
    confidence: _CONFIDENCE = "low"
    narrative: str = ""

    @field_validator("basket_match", mode="before")
    @classmethod
    def _norm_match(cls, v):
        if not isinstance(v, str):
            return "uncertain"
        s = v.strip().lower()
        if s in ("yes", "true", "match", "matches", "ok"):
            return "yes"
        if s in ("no", "false", "mismatch", "differs"):
            return "no"
        return "uncertain"

    @field_validator("confidence", mode="before")
    @classmethod
    def _norm_conf(cls, v):
        if not isinstance(v, str):
            return "low"
        s = v.strip().lower()
        if s in ("high", "h"):
            return "high"
        if s in ("medium", "med", "m"):
            return "medium"
        return "low"


def parse_or_fallback(raw: dict) -> ScoBasketMatch:
    """Strict-ish parse: tolerant of unknown keys, defensive about
    missing required fields. Returns a low-confidence uncertain object
    on any structural failure so the policy stage sees a usable result.
    """
    try:
        return ScoBasketMatch.model_validate(raw or {})
    except Exception:
        return ScoBasketMatch(basket_match="uncertain", confidence="low",
                              narrative="VLM output did not match schema.")
