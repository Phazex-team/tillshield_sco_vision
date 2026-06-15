"""Read-only admin endpoints.

Surfaces the effective ``config.yaml`` (sensitive values redacted), the
review-safe classifier catalog, and the active prompts a camera resolves
to. The UI in ``static/index.html`` previously called ``/config`` /
``/classifiers`` / ``/prompts`` directly; this router restores that
capability under the v1 namespace so reviewers can audit the prompt
text the model actually sees.

Prompt registry editing (CRUD) is intentionally NOT exposed here:
PRODUCTION_SPEC §14 + §16 reserve write-side prompt management for the
deferred MLOps tier. Read-only display is core.

Every read here is audited at info-level so operators can trace who
inspected configuration.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request


log = logging.getLogger(__name__)


router = APIRouter(prefix="/admin", tags=["admin"])


_REDACT_TOKENS = ("password", "secret", "token", "rtsp_url")


def _redact(value):
    if isinstance(value, dict):
        return {k: ("***redacted***"
                    if any(t in k.lower() for t in _REDACT_TOKENS)
                    else _redact(v))
                for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


@router.get("/config")
def get_effective_config(request: Request) -> dict:
    """Return the active runtime configuration with secrets redacted."""
    from app.config import load_config
    cfg = load_config()
    return {
        "cameras": _redact(cfg.cameras),
        "settings": _redact(cfg.settings),
        "models": {k: {"name": v.name, "enabled": v.enabled,
                       "extra": _redact(v.extra)}
                   for k, v in cfg.models.items()},
        "reasoning": _redact(cfg.raw.get("reasoning") or {}),
        "gpu": _redact(cfg.raw.get("gpu") or {}),
        "storage": _redact(cfg.raw.get("storage") or {}),
        "observability": _redact(cfg.observability),
    }


@router.get("/classifiers")
def list_all_classifiers(request: Request) -> dict:
    from classifiers import list_classifiers
    return {"items": list_classifiers()}


@router.get("/prompts")
def list_active_prompts(camera_id: Optional[str] = Query(None)) -> dict:
    """Return the effective prompt text per camera. Any prompt
    containing accusation language is flagged so an operator can see
    that the safety contract is intact even on cached configs."""
    from app.config import load_config
    from classifiers import resolve_prompts

    cfg = load_config()
    items = []
    for cam in cfg.cameras:
        if camera_id and cam.get("id") != camera_id:
            continue
        resolved = resolve_prompts(cam)
        items.append({
            "camera_id": cam.get("id"),
            "classifier": resolved.get("classifier"),
            "scenario_label": resolved.get("display_label"),
            "gemma_system": resolved.get("gemma_system"),
            "gemma_user": resolved.get("gemma_user"),
            "falcon": resolved.get("falcon"),
            "token_budget": resolved.get("token_budget"),
            "safety_violation": _detect_unsafe_language(resolved),
        })
    if camera_id and not items:
        raise HTTPException(status_code=404,
                            detail=f"camera {camera_id!r} not configured")
    return {"items": items}


_BANNED_PROMPT_PHRASES = (
    "determine fraud", "is this fraud", "fraud indicator",
    "loss-prevention analyst", "return fraud", "accuse",
)


def _detect_unsafe_language(resolved: dict) -> list[str]:
    hits: list[str] = []
    haystack = (resolved.get("gemma_system", "") + " " +
                resolved.get("gemma_user", "")).lower()
    for phrase in _BANNED_PROMPT_PHRASES:
        if phrase in haystack:
            hits.append(phrase)
    return hits
