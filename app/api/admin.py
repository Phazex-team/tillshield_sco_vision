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

from fastapi import APIRouter, HTTPException, Query, Request, Response


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
    from app.config import load_config
    from classifiers import list_classifiers
    cfg = load_config()
    active = {
        str(cam.get("classifier") or "").strip().lower()
        for cam in cfg.cameras
        if cam.get("classifier")
    }
    items = list_classifiers()
    if active:
        items = [it for it in items if it.get("key") in active]
    return {"items": items}


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
# Camera source settings (RTSP URL + POS workstation mapping)
# ---------------------------------------------------------------------------
#
# RTSP URLs and the POS workstation->camera map both live in
# ``config.yaml``. There was previously no way to set them without
# hand-editing the file. These endpoints let an operator (a) point a
# camera at a new RTSP URL and (b) attach the POS workstation id that
# correlates to that camera, persisting both to ``config.yaml`` with an
# audit trail. Changes to the RTSP URL take effect for the next
# recorder reload / case; in-flight analysis keeps its snapshot.


class CameraSettings(BaseModel):
    name: Optional[str] = None
    rtsp_url: Optional[str] = None
    # POS workstation id that maps to this camera (TillShield
    # ``workstation_camera_map``). Pass "" to clear this camera's
    # mapping; omit to leave it unchanged.
    workstation_id: Optional[str] = None


class CameraCreate(BaseModel):
    """Body for ``POST /admin/cameras`` — onboard a brand-new camera.

    ``camera_id`` and ``rtsp_url`` are required; everything else is
    seeded to the shipped per-camera shape so the ROI / prompt / preview
    surfaces work on the new camera immediately.
    """
    camera_id: str
    rtsp_url: str
    name: Optional[str] = None
    classifier: Optional[str] = None
    workstation_id: Optional[str] = None


# Seeded defaults for a newly-created camera, mirroring the shipped
# camera shape in config.yaml so edit/ROI/preview flows work on it with
# no further hand-editing.
_NEW_CAMERA_DEFAULTS = {
    "classifier": "sco_checkout",
    "token_budget": 1120,
    "enable_thinking": "",
    "max_frames": "",
    "cooldown_sec": 30,
}


def _tillshield_block(raw: dict) -> dict:
    return ((raw.get("integrations") or {}).get("tillshield") or {})


def _workstation_for_camera(raw: dict, camera_id: str) -> Optional[str]:
    """Reverse-lookup: the first workstation id mapped to ``camera_id``."""
    ws_map = _tillshield_block(raw).get("workstation_camera_map") or {}
    for ws, cam in ws_map.items():
        if cam == camera_id:
            return str(ws)
    return None


def _workstations_for_camera(raw: dict, camera_id: str) -> list[str]:
    """All POS workstation ids currently mapped to ``camera_id``."""
    ws_map = _tillshield_block(raw).get("workstation_camera_map") or {}
    return sorted(str(ws) for ws, cam in ws_map.items() if cam == camera_id)


def _write_config_atomic(path, data) -> None:
    """Persist ``config.yaml`` as safely as the filesystem allows.

    Preferred path: write a sibling temp file, ``fsync`` it, then
    ``os.replace`` it into place — a truly atomic swap, so a concurrent
    reader (the recorder watches this file's mtime) sees either the old
    or the new bytes, never a truncated middle.

    Fallback: in production ``config.yaml`` is a *single-file bind mount*
    inside the container, and you cannot rename over a bind-mount point
    (``os.replace`` -> EBUSY/EXDEV). There we write in place with a single
    buffered write + fsync. The swap window is sub-millisecond, and the
    recorder tolerates a transient parse error (it simply reconciles again
    on its next poll), so no camera change is lost."""
    import os as _os
    from pathlib import Path

    import yaml as _yaml

    path = Path(path)
    payload = _yaml.safe_dump(data, sort_keys=False)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "w") as f:
            f.write(payload)
            f.flush()
            _os.fsync(f.fileno())
        _os.replace(tmp, path)
    except OSError:
        # Bind-mounted target (or cross-device temp): write in place.
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        with open(path, "w") as f:
            f.write(payload)
            f.flush()
            _os.fsync(f.fileno())


def _read_recorder_runtime() -> dict:
    """Read the recorder's heartbeat (``run/recorder_state.json``, written
    after each reconcile) so the admin API can report — factually, across
    the process boundary — the recorder's live camera set and whether it
    has caught up to the current ``config.yaml``.

    The recorder reconciles asynchronously (it polls the config mtime), so
    right after a write ``caught_up`` is typically false; it flips true
    within the recorder's reconcile interval. Never raises."""
    import json as _json
    import os as _os

    from app.config import DEFAULT_CONFIG_PATH
    from video.recorder_supervisor import default_state_path

    try:
        state = _json.loads(default_state_path().read_text())
    except FileNotFoundError:
        return {"available": False,
                "detail": "recorder heartbeat not found; the recorder may "
                          "be starting or not running"}
    except Exception as exc:  # pragma: no cover - defensive
        return {"available": False, "detail": f"heartbeat unreadable: {exc}"}

    seen = state.get("config_mtime")
    try:
        current = _os.path.getmtime(DEFAULT_CONFIG_PATH)
    except OSError:  # pragma: no cover - defensive
        current = None
    caught_up = (seen is not None and current is not None
                 and seen >= current - 0.001)
    return {
        "available": True,
        "mode": "auto-reconcile (recorder watches config.yaml)",
        "active_cameras": state.get("active_cameras"),
        "config_mtime_applied": seen,
        "config_mtime_current": current,
        "caught_up": caught_up,
        "updated_at": state.get("updated_at"),
    }


def _runtime_apply_report() -> dict:
    """Report whether the config change hot-applied to the live runtime.

    * ``app``: the app reads ``config.yaml`` per request (no cache), so
      camera lists / ops-status reflect the change immediately. We
      re-load here to confirm the just-written file parses; a parse
      failure is surfaced as a runtime-apply failure (config was still
      written — partial success).
    * ``recorder``: reconciles itself; we report its heartbeat state.
    """
    from app.config import load_config

    app_status: dict
    try:
        load_config()
        app_status = {"applied": True,
                      "detail": "app reads config.yaml per request; camera "
                                "lists are live immediately"}
    except Exception as exc:
        app_status = {"applied": False,
                      "detail": f"config reload failed: {exc}"}

    recorder = _read_recorder_runtime()
    # ``ok`` reflects the synchronous, app-side apply (the part we can
    # confirm in-process). The recorder applies asynchronously, so its
    # ``caught_up`` is reported for visibility but does not gate ``ok``.
    return {
        "config_written": True,
        "ok": bool(app_status["applied"]),
        "app": app_status,
        "recorder": recorder,
    }


@router.get("/cameras")
def list_camera_settings() -> dict:
    """Return each camera's source settings (RTSP URL + mapped POS
    workstation). Unlike ``GET /admin/config`` this is NOT redacted —
    it is the editing surface, gated by the same admin token on write.

    ``rtsp_url`` is the *effective* value (``${VAR}`` already expanded).
    ``rtsp_url_is_env_ref`` is true when the raw ``config.yaml`` value is
    an env reference, so the UI can warn that saving replaces it with a
    literal URL.
    """
    from pathlib import Path

    import yaml as _yaml

    from app.config import DEFAULT_CONFIG_PATH, load_config

    cfg = load_config()
    raw_disk = _yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text()) or {}
    raw_by_id = {c.get("id"): c for c in (raw_disk.get("cameras") or [])}

    items = []
    for cam in cfg.cameras:
        cam_id = cam.get("id")
        raw_url = str((raw_by_id.get(cam_id) or {}).get("rtsp_url") or "")
        items.append({
            "camera_id": cam_id,
            "name": cam.get("name") or cam_id,
            "rtsp_url": cam.get("rtsp_url") or "",
            "rtsp_url_is_env_ref": "${" in raw_url,
            "workstation_id": _workstation_for_camera(cfg.raw, cam_id),
        })
    return {"items": items}


@router.patch("/cameras/{camera_id}")
def update_camera_settings(
    camera_id: str,
    settings: CameraSettings,
    request: Request,
    x_phazex_admin_token: Optional[str] = Header(default=None),
) -> dict:
    """Persist a camera's RTSP URL / name and POS workstation mapping
    into ``config.yaml``. Token-gated and audited.

    ``workstation_id``: a non-empty value maps that POS workstation to
    this camera (replacing any prior workstation that pointed here) and
    adds it to ``allowed_workstation_ids``. ``""`` clears this camera's
    mapping. Omitting the field leaves the mapping unchanged.
    """
    _check_admin_token(x_phazex_admin_token)

    from pathlib import Path

    import yaml as _yaml

    from app import audit
    from app.config import DEFAULT_CONFIG_PATH
    from db.session import get_sessionmaker

    raw_path = Path(DEFAULT_CONFIG_PATH)
    data = _yaml.safe_load(raw_path.read_text()) or {}
    cameras = data.get("cameras") or []
    target = next((c for c in cameras if c.get("id") == camera_id), None)
    if target is None:
        raise HTTPException(status_code=404,
                            detail=f"camera {camera_id!r} not in config.yaml")

    before = {
        "name": target.get("name"),
        "rtsp_url": target.get("rtsp_url"),
        "workstation_id": _workstation_for_camera(data, camera_id),
    }
    updated_fields: list[str] = []

    if settings.name is not None:
        target["name"] = settings.name
        updated_fields.append("name")

    if settings.rtsp_url is not None:
        new_url = settings.rtsp_url.strip()
        if not new_url:
            raise HTTPException(status_code=400,
                                detail="rtsp_url must not be empty")
        target["rtsp_url"] = new_url
        updated_fields.append("rtsp_url")

    if settings.workstation_id is not None:
        integrations = data.setdefault("integrations", {})
        ts = integrations.setdefault("tillshield", {})
        ws_map = ts.setdefault("workstation_camera_map", {})
        allowed = ts.setdefault("allowed_workstation_ids", [])
        # Drop any workstation currently pointing at this camera so a
        # camera maps to exactly one workstation id.
        for ws in [w for w, cam in ws_map.items() if cam == camera_id]:
            ws_map.pop(ws, None)
        new_ws = settings.workstation_id.strip()
        if new_ws:
            ws_map[new_ws] = camera_id
            if new_ws not in [str(x) for x in allowed]:
                allowed.append(new_ws)
        updated_fields.append("workstation_id")

    if not updated_fields:
        raise HTTPException(status_code=400, detail="no fields to update")

    _write_config_atomic(raw_path, data)

    after = {
        "name": target.get("name"),
        "rtsp_url": target.get("rtsp_url"),
        "workstation_id": _workstation_for_camera(data, camera_id),
    }
    SM = get_sessionmaker()
    with SM() as s:
        audit.record(
            s, action="admin.camera_settings_update",
            entity_type="camera", entity_id=camera_id,
            actor_type="admin_api",
            before=before, after=after,
            ip=(request.client.host if request.client else None),
            user_agent=request.headers.get("user-agent"),
        )
        s.commit()

    # An rtsp_url change recreates the recorder worker on reconcile; a
    # name/workstation-only change needs no recorder action.
    rtsp_changed = "rtsp_url" in updated_fields
    return {"camera_id": camera_id,
            "updated_fields": updated_fields,
            "settings": after,
            "runtime": _runtime_apply_report(),
            "note": ("config.yaml updated. The app reflects the edit "
                     "immediately; " + ("the segment recorder recreates "
                     "this camera's worker with the new RTSP within its "
                     "reconcile interval — no restart required."
                     if rtsp_changed else "no recorder change is needed "
                     "for a name/workstation edit."))}


@router.post("/cameras")
def create_camera(
    body: CameraCreate,
    request: Request,
    x_phazex_admin_token: Optional[str] = Header(default=None),
) -> dict:
    """Create a brand-new camera in ``config.yaml``. Token-gated and
    audited (``admin.camera_created``).

    ``camera_id`` must be unique (409 on collision); ``rtsp_url`` is
    required (400 if blank). The camera is seeded with the shipped shape
    (classifier / token budget / cooldown / empty zones + prompts) so the
    ROI, prompt and preview surfaces work on it immediately. An optional
    ``workstation_id`` maps a POS workstation to the new camera, taking it
    over from any camera it previously pointed at.

    The new camera is not recorded until the app + segment recorder are
    restarted/reloaded to pick up the RTSP source.
    """
    _check_admin_token(x_phazex_admin_token)

    from pathlib import Path

    import yaml as _yaml

    from app import audit
    from app.config import DEFAULT_CONFIG_PATH
    from db.session import get_sessionmaker

    camera_id = (body.camera_id or "").strip()
    if not camera_id:
        raise HTTPException(status_code=400, detail="camera_id must not be empty")
    rtsp_url = (body.rtsp_url or "").strip()
    if not rtsp_url:
        raise HTTPException(status_code=400, detail="rtsp_url must not be empty")

    raw_path = Path(DEFAULT_CONFIG_PATH)
    data = _yaml.safe_load(raw_path.read_text()) or {}
    cameras = data.setdefault("cameras", [])
    if any(c.get("id") == camera_id for c in cameras):
        raise HTTPException(status_code=409,
                            detail=f"camera {camera_id!r} already exists")

    new_cam = {
        "id": camera_id,
        "name": (body.name or "").strip() or camera_id,
        "rtsp_url": rtsp_url,
        "classifier": (body.classifier or "").strip()
                      or _NEW_CAMERA_DEFAULTS["classifier"],
        "token_budget": _NEW_CAMERA_DEFAULTS["token_budget"],
        "enable_thinking": _NEW_CAMERA_DEFAULTS["enable_thinking"],
        "max_frames": _NEW_CAMERA_DEFAULTS["max_frames"],
        "cooldown_sec": _NEW_CAMERA_DEFAULTS["cooldown_sec"],
        "zones": {},
        "model_roi_views": {},
        "prompts": {},
    }
    cameras.append(new_cam)

    workstation_reassigned_from = None
    if body.workstation_id is not None and body.workstation_id.strip():
        new_ws = body.workstation_id.strip()
        integrations = data.setdefault("integrations", {})
        ts = integrations.setdefault("tillshield", {})
        ws_map = ts.setdefault("workstation_camera_map", {})
        allowed = ts.setdefault("allowed_workstation_ids", [])
        prior = ws_map.get(new_ws)
        if prior and prior != camera_id:
            workstation_reassigned_from = prior
        ws_map[new_ws] = camera_id
        if new_ws not in [str(x) for x in allowed]:
            allowed.append(new_ws)

    _write_config_atomic(raw_path, data)

    after = {
        "camera_id": camera_id,
        "name": new_cam["name"],
        "classifier": new_cam["classifier"],
        "workstation_id": _workstation_for_camera(data, camera_id),
    }
    SM = get_sessionmaker()
    with SM() as s:
        audit.record(
            s, action="admin.camera_created",
            entity_type="camera", entity_id=camera_id,
            actor_type="admin_api",
            before=None, after=after,
            ip=(request.client.host if request.client else None),
            user_agent=request.headers.get("user-agent"),
        )
        s.commit()

    return {"camera_id": camera_id,
            "created": after,
            "workstation_reassigned_from": workstation_reassigned_from,
            "runtime": _runtime_apply_report(),
            "note": ("config.yaml updated. The app reflects the new camera "
                     "immediately; the segment recorder hot-applies it "
                     "(starts recording) within its reconcile interval — "
                     "no restart required.")}


@router.delete("/cameras/{camera_id}")
def delete_camera(
    camera_id: str,
    request: Request,
    clear_workstation_mappings: bool = Query(
        False,
        description="If the camera is still mapped to a POS workstation, "
                    "the delete is refused (409) unless this is true, which "
                    "clears those mappings as part of the delete."),
    x_phazex_admin_token: Optional[str] = Header(default=None),
) -> dict:
    """Remove a camera from ``config.yaml``. Token-gated and audited
    (``admin.camera_deleted``). The recorder stops its worker and the app
    drops it from camera lists on the next reconcile — no restart.

    Safety: a camera still mapped to a POS workstation cannot be deleted
    (409) — that mapping must be explicitly cleared or reassigned first,
    or pass ``clear_workstation_mappings=true`` to clear it here.
    """
    _check_admin_token(x_phazex_admin_token)

    from pathlib import Path

    from app import audit
    from app.config import DEFAULT_CONFIG_PATH
    from db.session import get_sessionmaker

    raw_path = Path(DEFAULT_CONFIG_PATH)
    import yaml as _yaml
    data = _yaml.safe_load(raw_path.read_text()) or {}
    cameras = data.get("cameras") or []
    target = next((c for c in cameras if c.get("id") == camera_id), None)
    if target is None:
        raise HTTPException(status_code=404,
                            detail=f"camera {camera_id!r} not in config.yaml")

    mapped_ws = _workstations_for_camera(data, camera_id)
    if mapped_ws and not clear_workstation_mappings:
        raise HTTPException(
            status_code=409,
            detail={
                "reason": "camera is still mapped to POS workstation(s); "
                          "clear or reassign the mapping first, or pass "
                          "clear_workstation_mappings=true",
                "workstations": mapped_ws,
            })

    # Remove the camera.
    data["cameras"] = [c for c in cameras if c.get("id") != camera_id]

    cleared: list[str] = []
    if mapped_ws and clear_workstation_mappings:
        ws_map = _tillshield_block(data).get("workstation_camera_map") or {}
        for ws in mapped_ws:
            if ws_map.pop(ws, None) is not None:
                cleared.append(ws)

    _write_config_atomic(raw_path, data)

    before = {"camera_id": camera_id,
              "name": target.get("name"),
              "workstations": mapped_ws}
    SM = get_sessionmaker()
    with SM() as s:
        audit.record(
            s, action="admin.camera_deleted",
            entity_type="camera", entity_id=camera_id,
            actor_type="admin_api",
            before=before,
            after={"cleared_workstations": cleared},
            ip=(request.client.host if request.client else None),
            user_agent=request.headers.get("user-agent"),
        )
        s.commit()

    return {"camera_id": camera_id,
            "deleted": True,
            "cleared_workstations": cleared,
            "runtime": _runtime_apply_report(),
            "note": ("config.yaml updated. The app drops the camera from "
                     "its lists immediately; the segment recorder stops "
                     "its worker within the reconcile interval — no "
                     "restart required.")}


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


@router.get("/camera-rois/{camera_id}/snapshot")
def get_camera_roi_snapshot(
        camera_id: str,
        response: Response,
        x_phazex_admin_token: Optional[str] = Header(default=None),
) -> dict:
    """Return a single representative frame from ``camera_id`` so the
    operator can draw ROIs on a real image.

    Hard contract:
      * Admin-token gated via the existing ``_check_admin_token``.
      * The endpoint accepts ONLY ``camera_id`` — never a URL or file
        path from the client.
      * The response NEVER contains ``rtsp_url`` (or any other secret).
      * ``Cache-Control: no-store`` so a browser-cached frame can't
        outlive the segment it came from.

    Source: the newest ``VideoSegment`` row for this camera. We open
    that MP4 with OpenCV, grab a representative middle frame, encode
    it JPEG → data URL, and return JSON ``{image_url, width, height,
    source: "latest_segment", captured_at, segment_id}``.

    Live RTSP fallback is intentionally NOT implemented: OpenCV's
    RTSP open + read has no reliable timeout knob from Python, so a
    misconfigured camera would block the endpoint indefinitely. When
    no local segment exists the endpoint returns 404 with a short
    detail saying the live-RTSP path is unavailable; the operator
    can keep editing numerically with the existing table.
    """
    _check_admin_token(x_phazex_admin_token)
    response.headers["Cache-Control"] = "no-store"

    def _snapshot_error(status_code: int, detail: str) -> None:
        raise HTTPException(
            status_code=status_code,
            detail=detail,
            headers={"Cache-Control": "no-store"},
        )

    from app.config import load_config
    from sqlalchemy import select
    from db.models import VideoSegment
    from db.session import get_sessionmaker

    cfg = load_config()
    if not any(c.get("id") == camera_id for c in cfg.cameras):
        _snapshot_error(404, f"camera {camera_id!r} not configured")

    SM = get_sessionmaker()
    with SM() as s:
        seg = s.execute(
            select(VideoSegment)
            .where(VideoSegment.camera_id == camera_id)
            .order_by(VideoSegment.start_at.desc())
        ).scalars().first()
    if seg is None or not seg.path:
        _snapshot_error(
            404,
            ("no local segment available for camera "
             f"{camera_id!r}; live RTSP snapshot is not "
             "implemented (no reliable timeout)"))

    import os
    if not os.path.exists(seg.path):
        _snapshot_error(
            404,
            f"latest segment for {camera_id!r} no longer on "
            "disk; retention may have cleared it")

    try:
        import cv2  # type: ignore
    except Exception as exc:
        _snapshot_error(503, f"cv2 unavailable: {type(exc).__name__}")

    cap = cv2.VideoCapture(seg.path)
    if not cap.isOpened():
        _snapshot_error(503, "decoder could not open the latest segment")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            _snapshot_error(503, "latest segment reports zero frames")
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            # Fall back to the very first frame if the mid-frame seek
            # missed (some codecs misreport frame count).
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            _snapshot_error(
                503, "decoder returned no frame from the latest segment")
        height, width = frame_bgr.shape[:2]
        ok, buf = cv2.imencode(".jpg", frame_bgr,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            _snapshot_error(503, "jpeg encode failed for snapshot")
    finally:
        cap.release()

    import base64
    b64 = base64.b64encode(bytes(buf)).decode("ascii")
    return {
        "camera_id": camera_id,
        "source": "latest_segment",
        "image_url": f"data:image/jpeg;base64,{b64}",
        "width": int(width),
        "height": int(height),
        # Only stable identifiers — never the on-disk path or RTSP URL.
        "segment_id": seg.id,
        "captured_at": seg.start_at.isoformat() if seg.start_at else None,
    }


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


# ---------------------------------------------------------------------------
# Model controls (per-stage enable/disable, applies to NEXT case/reprocess)
# ---------------------------------------------------------------------------

# Map UI/PATCH key -> config.yaml ``models`` sub-key + UI metadata.
# These are the only stages this control surface manages — anything
# else stays out so the operator cannot accidentally toggle vLLM /
# Gemma server processes from the reviewer UI (those are managed
# externally; see the operations-console contract).
MODEL_CONTROL_SPECS: tuple[dict, ...] = (
    {
        "id": "falcon", "config_key": "falcon",
        "label": "Perception (FL)",
        "role": "perception_detector",
        "dependencies": [],
        "independent": True,
        "caption": ("Independent detector; required for track evidence "
                    "/ VERIFIED path."),
        "default_when_missing": True,
    },
    {
        "id": "sam2", "config_key": "sam2",
        "label": "Segmenter (S2)",
        "role": "perception_segmenter",
        "dependencies": ["falcon"],
        "independent": False,
        "caption": ("Uses Perception (FL) boxes; no useful standalone mode "
                    "in this pipeline."),
        "default_when_missing": True,
    },
    {
        "id": "ocr", "config_key": "falcon_ocr",
        "label": "OCR",
        "role": "perception_ocr",
        "dependencies": ["falcon"],
        "independent": False,
        "caption": "Uses Perception (FL) receipt/document detections.",
        "default_when_missing": False,
    },
    {
        "id": "qwen3_vl", "config_key": "qwen3_vl",
        "label": "Vision Primary (Q)",
        "role": "vlm_primary",
        "dependencies": [],
        "independent": True,
        "caption": "Independent vision verifier.",
        "default_when_missing": True,
    },
    {
        "id": "gemma", "config_key": "gemma",
        "label": "Vision Fallback (G)",
        "role": "vlm_fallback",
        "dependencies": [],
        "independent": True,
        "caption": ("Fallback vision verifier; safe to disable, but a "
                    "Vision Primary (Q) failure then has no fallback."),
        "default_when_missing": True,
    },
    {
        # SAM 3 (concept-prompted video backend). Independent of
        # Falcon — when enabled it runs DIRECTLY on the window with
        # POS-derived text concepts and the case_runner threads its
        # canonical groups into the SCO v2 prompt/policy. Toggling
        # this OFF restores the Falcon-only perception path. Default
        # OFF — this is still an opt-in experimental backend.
        "id": "sam3", "config_key": "sam3",
        "label": "SAM 3 (concept perception)",
        "role": "perception_concept_backend",
        "dependencies": [],
        "independent": True,
        "caption": ("Optional SAM 3 video-concept backend. When ON, "
                    "runs alongside (or instead of) Perception (FL) "
                    "and emits canonical container groups. Safe to "
                    "leave OFF — Falcon is the default perception path."),
        "default_when_missing": False,
    },
)

ALLOWED_MODEL_CONTROL_KEYS: frozenset = frozenset(
    s["id"] for s in MODEL_CONTROL_SPECS)


def _spec_by_id(model_id: str) -> Optional[dict]:
    for s in MODEL_CONTROL_SPECS:
        if s["id"] == model_id:
            return s
    return None


def _current_model_enabled(cfg, model_id: str) -> bool:
    spec = _spec_by_id(model_id)
    if spec is None:
        return False
    m = cfg.models.get(spec["config_key"])
    if m is None:
        return bool(spec.get("default_when_missing", True))
    return bool(m.enabled)


def _model_control_warnings(state: dict) -> list[str]:
    """Operator-facing, non-blocking warnings derived from the current
    flag combination. These mirror the UI tooltips so backend + UI
    cannot drift."""
    warnings: list[str] = []
    if not state.get("falcon") and not state.get("sam3"):
        warnings.append(
            "Both perception backends disabled (Perception (FL) and SAM 3): "
            "no perception evidence will exist, so cases will likely fall "
            "through to REVIEW.")
    elif not state.get("falcon") and state.get("sam3"):
        warnings.append(
            "Perception (FL) disabled, SAM 3 ON: SCO checkout cases will "
            "use SAM 3 only. Legacy callers that still rely on Falcon "
            "track gating will fall through to REVIEW.")
    if not state.get("qwen3_vl") and not state.get("gemma"):
        warnings.append(
            "Both vision providers disabled: the provider chain will return "
            "a structured error and the decision policy will degrade to "
            "REVIEW. No vision narrative will be available.")
    if not state.get("gemma") and state.get("qwen3_vl"):
        warnings.append(
            "Vision Fallback (G) disabled: a Vision Primary (Q) failure will "
            "surface as an error with no fallback.")
    return warnings


@router.get("/model-controls")
def get_model_controls() -> dict:
    """Return current enable/disable state + per-stage metadata."""
    from app.config import load_config

    cfg = load_config()
    state: dict[str, bool] = {}
    items: list[dict] = []
    for spec in MODEL_CONTROL_SPECS:
        enabled = _current_model_enabled(cfg, spec["id"])
        state[spec["id"]] = enabled
        items.append({
            "id": spec["id"],
            "config_key": f"models.{spec['config_key']}.enabled",
            "label": spec["label"],
            "role": spec["role"],
            "dependencies": list(spec["dependencies"]),
            "independent": bool(spec["independent"]),
            "caption": spec["caption"],
            "enabled": enabled,
        })
    return {
        "models": items,
        "state": state,
        "warnings": _model_control_warnings(state),
        "applies_to": ("next case or reprocess; in-flight analyses "
                        "keep the config snapshot they started with"),
    }


def _validate_model_control_update(payload: dict, current: dict) -> dict:
    """Merge ``payload`` with ``current`` state, validate, return the
    new state dict. Raises ``HTTPException(400)`` on any rejection.

    Rejection rules (mirrored in the UI's pre-submit checks):

      * unknown keys are rejected
      * non-boolean values are rejected
      * at least one independent source must remain enabled
        (falcon OR qwen3_vl OR gemma)
      * sam2=true requires falcon=true
      * ocr=true requires falcon=true
    """
    if not isinstance(payload, dict) or not payload:
        raise HTTPException(status_code=400,
                            detail={"error": "payload must be a non-empty "
                                              "object of model_id -> bool"})
    unknown = set(payload) - ALLOWED_MODEL_CONTROL_KEYS
    if unknown:
        raise HTTPException(status_code=400, detail={
            "error": (f"unknown keys: {sorted(unknown)}; allowed: "
                      f"{sorted(ALLOWED_MODEL_CONTROL_KEYS)}")})
    new_state = dict(current)
    for k, v in payload.items():
        if not isinstance(v, bool):
            raise HTTPException(status_code=400, detail={
                "error": f"value for {k!r} must be a boolean (got "
                          f"{type(v).__name__})"})
        new_state[k] = v
    # An independent source must remain enabled. SAM 3 counts as an
    # independent perception backend (it does not require Falcon),
    # so a deployment running SAM 3 + a VLM can disable Falcon
    # safely. Without SAM 3, the legacy rule (FL OR Q OR G) still
    # applies.
    if not (new_state.get("falcon") or new_state.get("qwen3_vl")
            or new_state.get("gemma") or new_state.get("sam3")):
        raise HTTPException(status_code=400, detail={
            "error": "at least one independent source must remain enabled: "
                      "Perception (FL), SAM 3, Vision Primary (Q), or "
                      "Vision Fallback (G)"})
    if new_state.get("sam2") and not new_state.get("falcon"):
        raise HTTPException(status_code=400, detail={
            "error": ("Segmenter (S2) cannot be enabled while Perception (FL) "
                      "is disabled (it consumes Perception (FL) boxes; no "
                      "useful standalone mode)")})
    if new_state.get("ocr") and not new_state.get("falcon"):
        raise HTTPException(status_code=400, detail={
            "error": ("OCR cannot be enabled while Perception (FL) is disabled "
                      "(OCR runs on Perception (FL) receipt/document "
                      "detections)")})
    return new_state


@router.patch("/model-controls")
def update_model_controls(
        payload: dict = Body(...),
        request: Request = None,
        x_phazex_admin_token: Optional[str] = Header(default=None),
) -> dict:
    """Persist enable/disable flags for the five gated stages.

    Changes apply to the NEXT case or reprocess only — an in-flight
    ``analyze_case`` keeps the config snapshot it loaded at start.
    NO external process is started, stopped, or restarted; this
    endpoint is config-level gating only.
    """
    _check_admin_token(x_phazex_admin_token)

    from pathlib import Path
    import yaml as _yaml

    from app import audit
    from app.config import DEFAULT_CONFIG_PATH, load_config
    from db.session import get_sessionmaker

    cfg = load_config()
    current_state = {s["id"]: _current_model_enabled(cfg, s["id"])
                     for s in MODEL_CONTROL_SPECS}
    new_state = _validate_model_control_update(payload, current_state)

    raw_path = Path(DEFAULT_CONFIG_PATH)
    data = _yaml.safe_load(raw_path.read_text()) or {}
    models_block = data.setdefault("models", {})
    for spec in MODEL_CONTROL_SPECS:
        entry = models_block.setdefault(spec["config_key"], {})
        # If the entry has no ``name`` yet (e.g. a fresh config), keep
        # whatever is already there — the operator's PATCH should never
        # delete unrelated keys.
        entry["enabled"] = bool(new_state[spec["id"]])
    raw_path.write_text(_yaml.safe_dump(data, sort_keys=False))

    # Audit ONLY the model enable flags. Secrets and unrelated config
    # never appear in the audit row.
    SM = get_sessionmaker()
    with SM() as s:
        audit.record(
            s, action="admin.model_controls_update",
            entity_type="config", entity_id="model_controls",
            actor_type="admin_api",
            before=dict(current_state),
            after=dict(new_state),
            ip=(request.client.host if request and request.client else None),
            user_agent=(request.headers.get("user-agent")
                         if request else None),
        )
        s.commit()

    # Return the fresh state so the UI can re-render without a second
    # roundtrip. ``applies_to`` reminds the operator this is config-only
    # gating — no process was started or stopped, no memory freed.
    return get_model_controls()
