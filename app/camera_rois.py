"""Camera ROI registry + per-model view assignments.

This module is the *single source of truth* for camera ROIs that the
admin API persists into ``config.yaml`` and the perception/reasoning
runtime consumes at case time.

The canonical ROI registry is the existing ``cameras[].zones`` map.
Each zone keeps the legacy ``{x, y, w, h}`` keys (which
``perception.pipeline._load_zones`` and the decision policy still read
verbatim) and accepts two new sibling keys — ``label`` and
``purpose`` — used by the reviewer UI and the VLM crop captions.

``cameras[].model_roi_views`` is a new optional sibling: it lets an
operator assign one or more ROIs to each model with a per-model
caption explaining what should be visible. The supported model keys
match the actual pipeline consumers:

  * ``falcon`` — detector ROI for ``union_crop`` (Falcon runs on the
    cropped union, detections are translated back to full-frame
    coordinates by the perception pipeline).
  * ``sam2``   — segmenter ROI for ``filter_candidate_crops`` (only
    detections whose centre falls inside the union of assigned ROIs
    are forwarded to SAM 2).
  * ``ocr``    — OCR ROI for ``filter_candidate_crops`` (same filter
    semantics as SAM 2, applied to OCR candidates).
  * ``qwen3_vl`` / ``gemma`` — VLM ROI for ``labeled_crops`` (case
    runner attaches labeled ROI crops as additional manifest frames
    and injects the captions into the user prompt).

If a model entry is missing or has no valid ROI ids, the runtime
preserves its current full-frame behavior. No fake controls.

Decision policy is intentionally *not* exposed here — track-gated
VERIFIED remains the deterministic policy's authority.
"""
from __future__ import annotations

import re
from typing import Any, Optional


# Models the runtime actually consumes (rejected by validation otherwise).
SUPPORTED_MODELS: tuple[str, ...] = (
    "falcon", "sam2", "ocr", "qwen3_vl", "gemma",
)

# How each model uses an ROI assignment. Validation enforces this set.
# Each tuple lists ONLY the modes the active pipeline implements today.
# Adding a mode here is a runtime commitment, not a UI hint.
SUPPORTED_MODES: dict[str, tuple[str, ...]] = {
    "falcon":   ("union_crop",),
    "sam2":     ("filter_candidate_crops",),
    "ocr":      ("filter_candidate_crops",),
    "qwen3_vl": ("labeled_crops",),
    "gemma":    ("labeled_crops",),
}

# Models that consume labeled crops in a VLM manifest. For these the
# safer default is to ALSO keep a full-frame overview so a saved ROI
# view does not make the model blind outside the crops.
VLM_MODELS: frozenset = frozenset({"qwen3_vl", "gemma"})

_ROI_ID_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$")


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

def _camera_dict(cfg, camera_id: str) -> Optional[dict]:
    for cam in (cfg.cameras or []):
        if cam.get("id") == camera_id:
            return cam
    return None


def list_camera_rois(cfg) -> list[dict]:
    """Return the public ROI view for every configured camera.

    Each entry has ``camera_id``, ``name``, ``zones`` (id->descriptor),
    and ``model_roi_views``. The returned shape is what the API exposes
    and what the UI editor reads.
    """
    out: list[dict] = []
    for cam in (cfg.cameras or []):
        out.append(describe_camera_rois(cam))
    return out


def describe_camera_rois(cam: dict) -> dict:
    cid = cam.get("id")
    raw_zones = cam.get("zones") or {}
    zones: dict[str, dict] = {}
    for zid, z in raw_zones.items():
        if not isinstance(z, dict):
            continue
        try:
            zones[str(zid)] = {
                "id": str(zid),
                "label": str(z.get("label") or zid),
                "purpose": str(z.get("purpose") or ""),
                "x": int(z["x"]),
                "y": int(z["y"]),
                "w": int(z["w"]),
                "h": int(z["h"]),
            }
        except (KeyError, TypeError, ValueError):
            # Skip malformed zone — the original storage is left
            # untouched, but we don't expose half-baked data.
            continue
    return {
        "camera_id": cid,
        "name": cam.get("name") or cid,
        "zones": zones,
        "model_roi_views": _normalise_model_views(
            cam.get("model_roi_views") or {}),
    }


def _normalise_model_views(raw: dict) -> dict:
    """Coerce model_roi_views into the canonical shape for the UI/API.
    Unknown model keys are dropped (with no error) so a forward-compat
    config doesn't surface phantom entries here.

    The ``include_full_frame_overview`` default is True for VLM models
    (``qwen3_vl`` / ``gemma``) so a saved ROI view never makes the
    VLM blind outside the labeled crops. Non-VLM models do not use the
    flag at runtime; it defaults to False there.
    """
    out: dict[str, dict] = {}
    for model in SUPPORTED_MODELS:
        entry = raw.get(model) or {}
        if not isinstance(entry, dict):
            continue
        modes = SUPPORTED_MODES.get(model, ())
        default_overview = model in VLM_MODELS
        out[model] = {
            "enabled": bool(entry.get("enabled", False)),
            "roi_ids": [str(r) for r in (entry.get("roi_ids") or [])
                        if isinstance(r, (str, int))],
            "mode": str(entry.get("mode") or (modes[0] if modes else "")),
            "margin_pct": float(entry.get("margin_pct") or 0.0),
            "include_full_frame_overview": bool(
                entry.get("include_full_frame_overview", default_overview)),
            "caption": str(entry.get("caption") or ""),
        }
    return out


# ---------------------------------------------------------------------------
# Runtime helpers (perception + reasoning consume these)
# ---------------------------------------------------------------------------

def model_view(cfg, camera_id: str, model: str) -> Optional[dict]:
    """Return the *active* normalised model view for ``camera_id`` /
    ``model``, or ``None`` when no usable ROI assignment exists.

    A view is *active* iff:
      * the entry exists and ``enabled=True``, AND
      * its ``roi_ids`` list contains at least one id that resolves to
        a zone with valid {x,y,w,h} on this camera.
    """
    if model not in SUPPORTED_MODELS:
        return None
    cam = _camera_dict(cfg, camera_id)
    if cam is None:
        return None
    desc = describe_camera_rois(cam)
    view = desc["model_roi_views"].get(model)
    if not view or not view.get("enabled"):
        return None
    zones = desc["zones"]
    resolved = [zones[r] for r in view["roi_ids"] if r in zones]
    if not resolved:
        return None
    view = dict(view)
    view["resolved_zones"] = resolved
    return view


def union_bbox(zones: list[dict]) -> Optional[tuple[int, int, int, int]]:
    """Bounding box of the union of ``zones`` in ``[x1, y1, x2, y2]``
    integer form. Returns ``None`` when the list is empty."""
    if not zones:
        return None
    x1 = min(int(z["x"]) for z in zones)
    y1 = min(int(z["y"]) for z in zones)
    x2 = max(int(z["x"]) + int(z["w"]) for z in zones)
    y2 = max(int(z["y"]) + int(z["h"]) for z in zones)
    return (x1, y1, x2, y2)


def apply_margin(bbox: tuple[int, int, int, int],
                 margin_pct: float,
                 image_w: int,
                 image_h: int) -> tuple[int, int, int, int]:
    """Expand ``bbox`` by ``margin_pct`` of its own width/height,
    clipped to the image bounds. Margin <= 0 is a no-op."""
    x1, y1, x2, y2 = bbox
    if margin_pct <= 0:
        return (max(0, x1), max(0, y1),
                min(image_w, x2), min(image_h, y2))
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    mx = int(round(w * float(margin_pct)))
    my = int(round(h * float(margin_pct)))
    return (max(0, x1 - mx),
            max(0, y1 - my),
            min(image_w, x2 + mx),
            min(image_h, y2 + my))


def detection_inside_rois(bbox_xyxy: list[float],
                          zones: list[dict]) -> bool:
    """True iff the detection's centre falls inside any of ``zones``.
    Centre semantics mirror ``perception.temporal_memory.Zone.contains``
    so the ROI filter agrees with the existing decision-policy zone
    geometry."""
    if not zones or not bbox_xyxy or len(bbox_xyxy) < 4:
        return False
    cx = 0.5 * (float(bbox_xyxy[0]) + float(bbox_xyxy[2]))
    cy = 0.5 * (float(bbox_xyxy[1]) + float(bbox_xyxy[3]))
    for z in zones:
        zx, zy = int(z["x"]), int(z["y"])
        zw, zh = int(z["w"]), int(z["h"])
        if zx <= cx <= zx + zw and zy <= cy <= zy + zh:
            return True
    return False


# ---------------------------------------------------------------------------
# Validation (PATCH payload)
# ---------------------------------------------------------------------------

class RoiValidationError(ValueError):
    """Raised by ``validate_roi_update`` when the submitted payload is
    structurally invalid. The API layer converts this into HTTP 400 with
    the message body so the operator UI can show the exact rejection."""


def _validate_zone(zid: str, z: dict) -> dict:
    if not _ROI_ID_RE.match(zid or ""):
        raise RoiValidationError(
            f"roi id {zid!r} must match {_ROI_ID_RE.pattern}")
    if not isinstance(z, dict):
        raise RoiValidationError(f"roi {zid!r} body must be an object")
    out: dict = {}
    for key in ("label", "purpose"):
        v = z.get(key)
        if v is None:
            continue
        if not isinstance(v, str):
            raise RoiValidationError(
                f"roi {zid!r}.{key} must be a string")
        out[key] = v
    for key in ("x", "y", "w", "h"):
        if key not in z:
            raise RoiValidationError(f"roi {zid!r} missing field {key!r}")
        try:
            iv = int(z[key])
        except (TypeError, ValueError):
            raise RoiValidationError(
                f"roi {zid!r}.{key} must be an integer (got {z[key]!r})")
        if key in ("x", "y") and iv < 0:
            raise RoiValidationError(
                f"roi {zid!r}.{key} must be >= 0 (got {iv})")
        if key in ("w", "h") and iv <= 0:
            raise RoiValidationError(
                f"roi {zid!r}.{key} must be > 0 (got {iv})")
        out[key] = iv
    return out


def _validate_model_view(model: str, view: dict, valid_roi_ids: set) -> dict:
    if model not in SUPPORTED_MODELS:
        raise RoiValidationError(
            f"unknown model {model!r}; supported: {SUPPORTED_MODELS}")
    if not isinstance(view, dict):
        raise RoiValidationError(f"model_roi_views.{model} must be an object")
    out: dict = {}
    enabled = view.get("enabled")
    if enabled is not None:
        out["enabled"] = bool(enabled)
    roi_ids = view.get("roi_ids")
    if roi_ids is not None:
        if not isinstance(roi_ids, list):
            raise RoiValidationError(
                f"model_roi_views.{model}.roi_ids must be a list")
        norm: list[str] = []
        for r in roi_ids:
            if not isinstance(r, str) or not _ROI_ID_RE.match(r):
                raise RoiValidationError(
                    f"model_roi_views.{model}.roi_ids contains invalid "
                    f"id {r!r}")
            if r not in valid_roi_ids:
                raise RoiValidationError(
                    f"model_roi_views.{model}.roi_ids references "
                    f"unknown roi {r!r}; defined: {sorted(valid_roi_ids)}")
            norm.append(r)
        out["roi_ids"] = norm
    mode = view.get("mode")
    if mode is not None:
        if not isinstance(mode, str):
            raise RoiValidationError(
                f"model_roi_views.{model}.mode must be a string")
        allowed = SUPPORTED_MODES.get(model) or ()
        if mode not in allowed:
            raise RoiValidationError(
                f"model_roi_views.{model}.mode={mode!r} not in {allowed}")
        out["mode"] = mode
    margin = view.get("margin_pct")
    if margin is not None:
        try:
            mf = float(margin)
        except (TypeError, ValueError):
            raise RoiValidationError(
                f"model_roi_views.{model}.margin_pct must be a number")
        if mf < 0 or mf > 0.5:
            raise RoiValidationError(
                f"model_roi_views.{model}.margin_pct must be in [0, 0.5]")
        out["margin_pct"] = mf
    if "include_full_frame_overview" in view:
        out["include_full_frame_overview"] = bool(
            view["include_full_frame_overview"])
    if "caption" in view:
        cap = view["caption"]
        if cap is not None and not isinstance(cap, str):
            raise RoiValidationError(
                f"model_roi_views.{model}.caption must be a string")
        out["caption"] = str(cap or "")
    return out


def validate_roi_update(payload: dict,
                        *,
                        current_roi_ids: Optional[list] = None
                        ) -> dict:
    """Validate a PATCH payload. Returns the cleaned dict ready to write.

    Public payload shape (strict; unknown top-level keys are rejected)::

        {
          "zones": { "<roi_id>": {"label": str, "purpose": str,
                                   "x": int, "y": int, "w": int, "h": int} },
          "model_roi_views": { "<model>": {...} }
        }

    ``current_roi_ids`` is the *server-side* ROI registry for the
    target camera. When the caller updates only ``model_roi_views``,
    this list is used to validate that each assignment references an
    existing ROI. It is a function kwarg — never a payload key — so
    the strict top-level-keys check is unaffected by it.

    Both top-level keys are optional but at least one must be present.
    """
    if not isinstance(payload, dict) or not payload:
        raise RoiValidationError("payload must be a non-empty object")
    allowed_top = {"zones", "model_roi_views"}
    extra = set(payload) - allowed_top
    if extra:
        raise RoiValidationError(
            f"unknown top-level keys: {sorted(extra)}; allowed: "
            f"{sorted(allowed_top)}")
    out: dict = {}
    zones_in = payload.get("zones")
    if zones_in is not None:
        if not isinstance(zones_in, dict) or not zones_in:
            raise RoiValidationError(
                "'zones' must be a non-empty object of roi_id -> body")
        cleaned_zones: dict[str, dict] = {}
        for zid, body in zones_in.items():
            cleaned_zones[str(zid)] = _validate_zone(str(zid), body)
        out["zones"] = cleaned_zones
    # Choose the ROI id set used to validate model assignments. When
    # the PATCH itself includes ``zones``, the new set takes effect;
    # otherwise we fall back to the caller-supplied current registry.
    if "zones" in out:
        valid_ids = set(out["zones"].keys())
    else:
        valid_ids = set(str(r) for r in (current_roi_ids or []))
    mviews_in = payload.get("model_roi_views")
    if mviews_in is not None:
        if not isinstance(mviews_in, dict):
            raise RoiValidationError("'model_roi_views' must be an object")
        cleaned_views: dict[str, dict] = {}
        for model, body in mviews_in.items():
            cleaned_views[str(model)] = _validate_model_view(
                str(model), body, valid_ids)
        out["model_roi_views"] = cleaned_views
    if not out:
        raise RoiValidationError(
            "payload must include at least one of 'zones' or "
            "'model_roi_views'")
    return out
