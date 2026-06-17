"""Admin endpoints — read effective config/prompts, and edit per-camera
prompt overrides locally.

Read endpoints:

  GET /admin/config       — effective ``config.yaml`` with secrets redacted
  GET /admin/classifiers  — review-safe classifier catalog
  GET /admin/prompts      — effective per-camera prompts with safety scan

Write endpoint (local prompt editing — PRODUCTION_SPEC §14 minimum):

  PATCH /admin/prompts/{camera_id}  — update gemma_system / gemma_user /
                                       falcon override text on a camera.

The PATCH endpoint is gated by the optional shared secret
``X-PhazeX-Admin-Token`` (env ``ADMIN_EDIT_TOKEN`` or
``config.yaml.admin.edit_token``). It rejects any submitted prompt that
contains accusation phrases so the safety contract cannot regress via
the UI. Every write is audited with before/after JSON.

The dedicated *prompt registry UI / approval workflow / experiment
tracking* surface remains explicitly deferred to the MLOps tier; the
PATCH endpoint is the local minimum required to operate the app
without hand-editing ``config.yaml``.
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


# Phrases that flag an effective prompt as actively unsafe (i.e. it
# instructs the model to make an accusation). These are deliberately
# multi-word so that the review-safe instruction "never use the words
# 'fraud', 'fraudulent', or 'theft'" itself does not trip the scan.
_BANNED_INSTRUCTION_PHRASES = (
    "determine fraud", "is this fraud", "fraud indicator",
    "loss-prevention analyst", "return fraud", "accuse",
    "fraud detected", "is fraudulent", "commit theft",
)

# Phrases that operator-submitted prompts must NOT contain at all,
# even as quoted examples. The PATCH endpoint refuses these so the
# review-safe contract cannot regress through a hand-edited prompt.
_BANNED_NEW_PROMPT_PHRASES = _BANNED_INSTRUCTION_PHRASES + (
    "fraudulent", "theft", "suspect ",
)


def _detect_unsafe_language(resolved: dict) -> list[str]:
    hits: list[str] = []
    haystack = (resolved.get("gemma_system", "") + " " +
                resolved.get("gemma_user", "")).lower()
    for phrase in _BANNED_INSTRUCTION_PHRASES:
        if phrase in haystack:
            hits.append(phrase)
    return hits


# ---------------------------------------------------------------------------
# Local prompt editor (PATCH)
# ---------------------------------------------------------------------------

from fastapi import Body, Header
from pydantic import BaseModel


class PromptOverrides(BaseModel):
    gemma_system: Optional[str] = None
    gemma_user: Optional[str] = None
    falcon: Optional[str] = None


def _admin_token() -> Optional[str]:
    import os
    from app.config import load_config
    env = os.environ.get("ADMIN_EDIT_TOKEN")
    if env:
        return env.strip() or None
    cfg = load_config()
    admin = cfg.raw.get("admin") or {}
    tok = admin.get("edit_token")
    return str(tok).strip() if tok else None


def _check_admin_token(token: Optional[str]) -> None:
    expected = _admin_token()
    if not expected:
        return
    if not token or token.strip() != expected:
        raise HTTPException(status_code=401,
                            detail="invalid or missing admin token")


def _scan_for_accusation(text: str) -> list[str]:
    low = (text or "").lower()
    return [p for p in _BANNED_NEW_PROMPT_PHRASES if p in low]


@router.patch("/prompts/{camera_id}")
def update_prompts(camera_id: str,
                   overrides: PromptOverrides,
                   request: Request,
                   x_phazex_admin_token: Optional[str] = Header(default=None),
                   ) -> dict:
    """Persist per-camera prompt overrides into ``config.yaml``.

    The body fields are optional; only the ones supplied are updated.
    The endpoint refuses any text containing accusation phrases."""
    _check_admin_token(x_phazex_admin_token)

    # Reject accusation language before touching disk.
    rejections: dict[str, list[str]] = {}
    for field in ("gemma_system", "gemma_user", "falcon"):
        v = getattr(overrides, field, None)
        if v is None:
            continue
        hits = _scan_for_accusation(v)
        if hits:
            rejections[field] = hits
    if rejections:
        raise HTTPException(
            status_code=400,
            detail={"rejected_phrases": rejections,
                    "reason": "prompt contains accusation language; "
                              "must not regress the review-safe contract"},
        )

    from pathlib import Path

    import yaml as _yaml

    from app import audit
    from app.config import DEFAULT_CONFIG_PATH, load_config
    from db.session import get_sessionmaker

    cfg = load_config()
    cam_idx = None
    for i, cam in enumerate(cfg.cameras):
        if cam.get("id") == camera_id:
            cam_idx = i
            break
    if cam_idx is None:
        raise HTTPException(status_code=404,
                            detail=f"camera {camera_id!r} not found")

    # Read raw YAML so we preserve formatting / comments order best-effort.
    raw_path = Path(DEFAULT_CONFIG_PATH)
    data = _yaml.safe_load(raw_path.read_text()) or {}
    cameras = data.get("cameras") or []
    target = None
    for c in cameras:
        if c.get("id") == camera_id:
            target = c
            break
    if target is None:
        raise HTTPException(status_code=404,
                            detail=f"camera {camera_id!r} not in config.yaml")
    before_prompts = dict(target.get("prompts") or {})
    new_prompts = dict(before_prompts)
    for field in ("gemma_system", "gemma_user", "falcon"):
        v = getattr(overrides, field, None)
        if v is not None:
            new_prompts[field] = v
    target["prompts"] = new_prompts
    raw_path.write_text(_yaml.safe_dump(data, sort_keys=False))

    SM = get_sessionmaker()
    with SM() as s:
        audit.record(
            s, action="admin.prompt_update",
            entity_type="camera", entity_id=camera_id,
            actor_type="admin_api",
            before={"prompts": before_prompts},
            after={"prompts": new_prompts},
            ip=(request.client.host if request.client else None),
            user_agent=request.headers.get("user-agent"),
        )
        s.commit()

    return {"camera_id": camera_id,
            "updated_fields": sorted(k for k, v in
                                     overrides.model_dump().items()
                                     if v is not None),
            "prompts": new_prompts}


# ---------------------------------------------------------------------------
# Camera ROIs (per-camera ROI registry + per-model view assignments)
# ---------------------------------------------------------------------------

@router.get("/camera-rois")
def list_camera_rois_endpoint() -> dict:
    """Return ROI registry + model assignments for every camera."""
    from app.camera_rois import list_camera_rois
    from app.config import load_config
    return {"items": list_camera_rois(load_config())}


@router.get("/camera-rois/{camera_id}")
def get_camera_rois(camera_id: str) -> dict:
    from app.camera_rois import describe_camera_rois
    from app.config import load_config
    cfg = load_config()
    for cam in cfg.cameras:
        if cam.get("id") == camera_id:
            return describe_camera_rois(cam)
    raise HTTPException(status_code=404,
                        detail=f"camera {camera_id!r} not configured")


@router.patch("/camera-rois/{camera_id}")
def update_camera_rois(camera_id: str,
                       payload: dict = Body(...),
                       request: Request = None,
                       x_phazex_admin_token: Optional[str] = Header(default=None),
                       ) -> dict:
    """Persist ROI registry + model view assignments for ``camera_id``.

    Payload (both keys optional, at least one required)::

        {
          "zones": {
            "counter_zone": {"label": "...", "purpose": "...",
                              "x": 0, "y": 0, "w": 100, "h": 100}
          },
          "model_roi_views": {
            "falcon":   {"enabled": true, "roi_ids": ["counter_zone"],
                          "mode": "union_crop", "margin_pct": 0.08,
                          "caption": "..."},
            "qwen3_vl": {"enabled": true, "roi_ids": ["counter_zone"],
                          "include_full_frame_overview": true,
                          "mode": "labeled_crops", "caption": "..."}
          }
        }

    Admin-token gated like ``PATCH /admin/prompts/{camera_id}``. Every
    successful write produces an ``AuditLog`` row with before/after.
    """
    _check_admin_token(x_phazex_admin_token)

    from pathlib import Path
    import yaml as _yaml

    from app import audit
    from app.camera_rois import (
        SUPPORTED_MODELS, describe_camera_rois, validate_roi_update,
    )
    from app.config import DEFAULT_CONFIG_PATH, load_config
    from db.session import get_sessionmaker

    cfg = load_config()
    target_cam = None
    for cam in cfg.cameras:
        if cam.get("id") == camera_id:
            target_cam = cam
            break
    if target_cam is None:
        raise HTTPException(
            status_code=404,
            detail=f"camera {camera_id!r} not found")

    # When PATCH does not include zones, model_roi_views validation
    # checks against the existing zones on disk so an old config that
    # forgot a roi id is still rejected. We pass the current ids as a
    # kwarg — NEVER as a payload key — so the public top-level-keys
    # check stays strict.
    current_ids = list((target_cam.get("zones") or {}).keys())

    try:
        cleaned = validate_roi_update(payload or {},
                                       current_roi_ids=current_ids)
    except Exception as exc:
        raise HTTPException(status_code=400,
                            detail={"error": str(exc)})

    raw_path = Path(DEFAULT_CONFIG_PATH)
    data = _yaml.safe_load(raw_path.read_text()) or {}
    cameras = data.get("cameras") or []
    target = None
    for c in cameras:
        if c.get("id") == camera_id:
            target = c
            break
    if target is None:
        raise HTTPException(
            status_code=404,
            detail=f"camera {camera_id!r} not in config.yaml")

    before = describe_camera_rois(target)

    if "zones" in cleaned:
        target["zones"] = cleaned["zones"]
    if "model_roi_views" in cleaned:
        existing_views = target.get("model_roi_views") or {}
        merged: dict = dict(existing_views) if isinstance(
            existing_views, dict) else {}
        for model, body in cleaned["model_roi_views"].items():
            merged[model] = {**(merged.get(model) or {}), **body}
        # Drop any keys for unsupported models so a typo doesn't linger.
        merged = {k: v for k, v in merged.items()
                  if k in SUPPORTED_MODELS}
        target["model_roi_views"] = merged

    raw_path.write_text(_yaml.safe_dump(data, sort_keys=False))

    after = describe_camera_rois(target)

    SM = get_sessionmaker()
    with SM() as s:
        audit.record(
            s, action="admin.camera_rois_update",
            entity_type="camera", entity_id=camera_id,
            actor_type="admin_api",
            before={"zones": before["zones"],
                    "model_roi_views": before["model_roi_views"]},
            after={"zones": after["zones"],
                   "model_roi_views": after["model_roi_views"]},
            ip=(request.client.host if request and request.client else None),
            user_agent=(request.headers.get("user-agent")
                        if request else None),
        )
        s.commit()

    return {
        "camera_id": camera_id,
        "updated_keys": sorted(cleaned.keys()),
        "zones": after["zones"],
        "model_roi_views": after["model_roi_views"],
    }
