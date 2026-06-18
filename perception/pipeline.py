"""Perception pipeline orchestration (PRODUCTION_SPEC §10).

The pipeline glues sampling -> falcon detection -> SAM 2 segmentation
-> tracker -> temporal memory -> OCR -> keyframe selection together.

Output shape (consumed by ``app.case_runner.analyze_case``):

    {
      "tracks": [ ... ],
      "keyframes": [ ... ],
      "ocr": [ ... ],
      "detections": [ ... ],
      "limitations": [ ... ],
      "obstructed": bool,
    }

The function is split so tests can hand in a stub FalconClient and
SamplingPolicy + a list of synthetic frames. In production it loads
the real Falcon weights and SAM 2 on demand (lazy).
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .falcon_client import FalconClient
from .keyframes import select_keyframes
from .ocr import OcrEngine, run_ocr
from .sam2_client import Sam2Client
from .sampling import SamplingPolicy, plan_indices
from .schemas import Detection, Keyframe, Mask, OcrResult, Track
from .temporal_memory import Zone, annotate_tracks
from .tracker import Tracker


log = logging.getLogger(__name__)


DEFAULT_FALCON_QUERY = (
    "bag, shopping bag, item, product, box, package, clothing, "
    "receipt, document, paper, hand"
)


def run_perception(session, case, window, *, cfg=None) -> dict:
    """Default production runner. Reads the window MP4 from disk,
    samples frames, and drives the perception pipeline.

    ``cfg`` is the AppConfig snapshot the caller (case_runner) already
    loaded for this case. Passing it through avoids a second
    ``load_config()`` call that could pick up a mid-case ``config.yaml``
    edit. When omitted the snapshot is taken here.
    """
    from app.config import load_config, resolve_model_path

    if cfg is None:
        cfg = load_config()
    falcon_cfg = cfg.models.get("falcon")
    falcon_enabled = bool(falcon_cfg.enabled) if falcon_cfg else True
    falcon_path = None
    if falcon_enabled and falcon_cfg:
        try:
            falcon_path = resolve_model_path(falcon_cfg)
        except Exception:
            falcon_path = None
    sam2_raw_entry = cfg.raw.get("models", {}).get("sam2") or {}
    sam2_model_cfg = cfg.models.get("sam2")
    sam2_enabled = bool(sam2_model_cfg.enabled) if sam2_model_cfg else \
        bool(sam2_raw_entry.get("enabled", True))
    sam2_path = None
    if sam2_enabled and sam2_raw_entry:
        from app.config import ModelConfig
        sam2_path = resolve_model_path(
            ModelConfig(name=sam2_raw_entry.get(
                "name", "facebook/sam2-hiera-large"),
                        enabled=True, extra=sam2_raw_entry),
            production_mode=False,
        )

    ocr_raw_entry = cfg.raw.get("models", {}).get("falcon_ocr") or {}
    ocr_model_cfg = cfg.models.get("falcon_ocr")
    ocr_enabled = bool(ocr_model_cfg.enabled) if ocr_model_cfg else \
        bool(ocr_raw_entry.get("enabled", False))
    ocr_path = None
    if ocr_enabled and ocr_raw_entry:
        from app.config import ModelConfig
        ocr_path = resolve_model_path(
            ModelConfig(name=ocr_raw_entry.get("name", "tiiuae/Falcon-OCR"),
                        enabled=True, extra=ocr_raw_entry),
            production_mode=False,
        )

    window_start_ts = (window.actual_start_at or
                       window.requested_start_at)
    # Resolve per-model ROI views for the active camera. Each view is
    # ``None`` when no usable assignment exists, in which case the
    # corresponding model keeps its full-frame behavior.
    from app.camera_rois import model_view
    # When a model is disabled we DO NOT instantiate its client at all
    # (per the operator contract: no weights load, no GPU touch). The
    # ``*_enabled`` flags below tell ``run_perception_on_window`` to
    # short-circuit and emit ``<stage>_disabled_by_config`` limitations
    # only when there would otherwise have been work to do.
    falcon_client = FalconClient(model_path=falcon_path) \
        if falcon_enabled else None
    sam2_client = Sam2Client(model_path=sam2_path) if sam2_enabled else None
    ocr_engine = OcrEngine(model_path=ocr_path) if ocr_enabled else None
    return run_perception_on_window(
        window_path=window.path,
        window_start_ts=window_start_ts,
        fps=int(cfg.settings.get("gemma_video_fps", 25)),
        zones=_load_zones(cfg, case.camera_id),
        falcon_client=falcon_client,
        sam2_client=sam2_client,
        ocr_engine=ocr_engine,
        sampling=SamplingPolicy(),
        falcon_roi_view=model_view(cfg, case.camera_id, "falcon"),
        sam2_roi_view=model_view(cfg, case.camera_id, "sam2"),
        ocr_roi_view=model_view(cfg, case.camera_id, "ocr"),
        falcon_enabled=falcon_enabled,
        sam2_enabled=sam2_enabled,
        ocr_enabled=ocr_enabled,
    )


def run_perception_on_window(*,
                             window_path: Optional[str],
                             fps: int,
                             zones: list[Zone],
                             falcon_client: Optional[FalconClient],
                             sam2_client: Optional[Sam2Client],
                             sampling: SamplingPolicy,
                             window_start_ts: Optional[datetime] = None,
                             ocr_engine: Optional[OcrEngine] = None,
                             falcon_query: str = DEFAULT_FALCON_QUERY,
                             falcon_roi_view: Optional[dict] = None,
                             sam2_roi_view: Optional[dict] = None,
                             ocr_roi_view: Optional[dict] = None,
                             falcon_enabled: bool = True,
                             sam2_enabled: bool = True,
                             ocr_enabled: bool = True,
                             ) -> dict:
    """Decode + run perception on a single video window. ``window_path``
    may be ``None`` in offline / synthetic test mode — in that case the
    sampler returns no frames and the function returns an empty result
    with the appropriate limitation tag.

    ``window_start_ts`` anchors every detection / track / keyframe
    timestamp to real CCTV wall-clock time. Callers must pass it for
    real-world correctness; tests may omit it (and a limitation is
    emitted noting the fallback)."""
    import time as _time
    limitations: list[str] = []
    detections: list[Detection] = []
    frames_for_ocr: dict[int, object] = {}
    # Per-stage timings (perf_counter, integer ms). Skipped stages are
    # OMITTED rather than reported as 0 so the operator can tell apart
    # "ran in <1ms" from "didn't run at all". ``total_ms`` is always
    # populated. The result block is advisory only — decision policy
    # never reads it.
    t0 = _time.perf_counter()
    timings: dict[str, int] = {}

    def _ms_since(start: float) -> int:
        return int((_time.perf_counter() - start) * 1000)

    t_sample = _time.perf_counter()
    frames = _sample_frames(window_path, fps, sampling, limitations,
                            window_start_ts=window_start_ts)
    timings["sample_frames_ms"] = _ms_since(t_sample)
    if not frames:
        timings["total_ms"] = _ms_since(t0)
        return {
            "detections": [], "tracks": [], "keyframes": [],
            "ocr": [], "masks": [], "limitations": limitations,
            "obstructed": False,
            "timings_ms": timings,
        }

    # Actual decoded frame size for ROI scaling. The recorder writes fixed
    # dimensions per segment, so the first sampled frame is authoritative
    # for this window.
    _frame_size: Optional[tuple[int, int]] = None
    try:
        _frame_size = (int(frames[0][2].size[0]),
                       int(frames[0][2].size[1]))
    except Exception:
        _frame_size = None

    # ---- Falcon stage (independent perception detector) -------------
    # ``falcon_enabled=False`` is a config-level gate: we never touch
    # the detector, the limitation is tagged ``falcon_disabled_by_config``
    # so downstream consumers (decision policy + reviewer UI) can tell
    # this is an operator-chosen state, not a runtime failure.
    # SAM 2 + OCR depend on Falcon detections, so when Falcon is off
    # they are bypassed regardless of their own toggles (the API
    # validation already rejects the combination, but defence in depth
    # belongs here too).
    falcon_crop = _falcon_roi_crop(frames, falcon_roi_view, limitations)
    if not falcon_enabled or falcon_client is None:
        if not falcon_enabled:
            limitations.append("falcon_disabled_by_config")
        else:
            limitations.append("falcon_unavailable")
        timings["total_ms"] = _ms_since(t0)
        return {
            "detections": [], "tracks": [], "keyframes": [],
            "ocr": [], "masks": [], "limitations": limitations,
            "obstructed": False,
            "timings_ms": timings,
        }
    t_falcon = _time.perf_counter()
    try:
        try:
            detections = falcon_client.detect_on_frames(
                [(idx, ts, img) for idx, ts, img in frames],
                query=falcon_query,
                roi_crop=falcon_crop,
            )
            for idx, ts, img in frames:
                frames_for_ocr[idx] = img
        except Exception:
            log.exception("falcon detection failed")
            limitations.append("falcon_unavailable")
    finally:
        timings["falcon_ms"] = _ms_since(t_falcon)

    # SAM 2 ROI filter (filter-only by detection centre — we never crop
    # the image because SAM 2 needs full context for accurate masks).
    # ``frame_size`` lets the filter scale zones from the saved source
    # dimensions onto the actual decoded frame size.
    sam_detections = _filter_by_roi(detections, sam2_roi_view,
                                     limit_tag="sam2_roi_filter",
                                     limitations=limitations,
                                     frame_size=_frame_size)

    masks: list[Mask] = []
    if sam_detections and sam2_enabled \
            and sam2_client is not None and sam2_client.has_capability():
        # Pick a representative frame near the middle of the window
        # for SAM 2; this matches the existing single-frame pipeline.
        mid_idx = frames[len(frames) // 2][0]
        mid_img = next((img for idx, ts, img in frames
                        if idx == mid_idx), frames[0][2])
        t_sam = _time.perf_counter()
        try:
            try:
                masks = sam2_client.segment(mid_img, sam_detections)
            except Exception:
                log.exception("sam2 segment failed")
                limitations.append("sam2_unavailable")
        finally:
            timings["sam2_ms"] = _ms_since(t_sam)
    elif sam_detections and not sam2_enabled:
        # Operator-chosen config gate. NO ``sam2_unavailable`` because
        # that implies a runtime failure; ``sam2_disabled_by_config``
        # is the honest tag.
        limitations.append("sam2_disabled_by_config")
    elif sam_detections:
        limitations.append("sam2_unavailable")
        # Honest omission: sam2_ms is left out when no inference ran.

    t_tracker = _time.perf_counter()
    tracker = Tracker()
    tracker.update(detections)
    tracks = tracker.export()
    tracks = annotate_tracks(
        tracks, detections,
        zones=_scale_temporal_zones(zones, _frame_size, limitations),
    )
    timings["tracker_ms"] = _ms_since(t_tracker)

    ocr_input_detections = _filter_by_roi(detections, ocr_roi_view,
                                           limit_tag="ocr_roi_filter",
                                           limitations=limitations,
                                           frame_size=_frame_size)
    if ocr_enabled:
        t_ocr = _time.perf_counter()
        try:
            ocr, ocr_limitations = run_ocr(ocr_engine, ocr_input_detections,
                                            frames_for_ocr)
        finally:
            timings["ocr_ms"] = _ms_since(t_ocr)
        limitations.extend(ocr_limitations)
    else:
        # Honest gate: only flag when there were OCR candidates that
        # would have been processed. No engine constructed, no work
        # done, no ``ocr_ms`` recorded.
        from .ocr import crops_for_ocr
        ocr = []
        if crops_for_ocr(ocr_input_detections):
            limitations.append("ocr_disabled_by_config")

    t_kf = _time.perf_counter()
    keyframes = select_keyframes(tracks, detections)
    timings["keyframes_ms"] = _ms_since(t_kf)
    obstructed = any("obstruct" in l.lower() for l in limitations)
    timings["total_ms"] = _ms_since(t0)
    return {
        "detections": [_d_dict(d) for d in detections],
        "tracks": [_t_dict(t) for t in tracks],
        "keyframes": [_kf_dict(k) for k in keyframes],
        "ocr": [_ocr_dict(o) for o in ocr],
        "masks": [_m_dict(m) for m in masks],
        "limitations": limitations,
        "obstructed": obstructed,
        "timings_ms": timings,
    }


def _sample_frames(window_path: Optional[str], fps: int,
                   sampling: SamplingPolicy,
                   limitations: list[str],
                   *,
                   window_start_ts: Optional[datetime] = None
                   ) -> list[tuple[int, datetime, object]]:
    if not window_path:
        limitations.append("no_window_path")
        return []
    try:
        from PIL import Image
        import cv2  # type: ignore
    except Exception as exc:
        limitations.append(f"video decoder unavailable: {exc}")
        return []
    cap = cv2.VideoCapture(window_path)
    if not cap.isOpened():
        limitations.append("video_cannot_open")
        return []
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or float(fps)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        cap.release()
        limitations.append("video_zero_frames")
        return []
    # Anchor sampling timestamps to the actual CCTV window start so
    # detections / tracks / keyframes carry real wall-clock values.
    if window_start_ts is None:
        base_ts = datetime.now(timezone.utc).replace(tzinfo=None)
        limitations.append("perception_ts_anchored_to_now")
    elif window_start_ts.tzinfo is not None:
        base_ts = window_start_ts.astimezone(timezone.utc).replace(
            tzinfo=None)
    else:
        base_ts = window_start_ts
    plan = plan_indices(
        fps=actual_fps, frame_count=frame_count,
        base_start_ts=base_ts, policy=sampling,
    )
    out: list[tuple[int, datetime, object]] = []
    for idx, ts in plan:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            continue
        from PIL import Image
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        out.append((idx, ts, Image.fromarray(rgb)))
    cap.release()
    return out


def _load_zones(cfg, camera_id: str) -> list[Zone]:
    zones: list[Zone] = []
    for cam in cfg.cameras:
        if cam.get("id") != camera_id:
            continue
        for name, z in (cam.get("zones") or {}).items():
            if not isinstance(z, dict):
                continue
            try:
                # ``z`` may carry the new ``label``/``purpose`` siblings
                # — only the spatial keys matter for decision-policy
                # zone semantics, so we extract them tolerantly.
                sw = z.get("source_width")
                sh = z.get("source_height")
                src_w = src_h = None
                if sw is not None and sh is not None:
                    src_w = int(sw)
                    src_h = int(sh)
                    if src_w <= 0 or src_h <= 0:
                        src_w = src_h = None
                zones.append(Zone(name=name, x=int(z["x"]), y=int(z["y"]),
                                  w=int(z["w"]), h=int(z["h"]),
                                  source_width=src_w,
                                  source_height=src_h))
            except (KeyError, TypeError, ValueError):
                continue
    return zones


def _scale_temporal_zones(zones: list[Zone],
                          frame_size: Optional[tuple[int, int]],
                          limitations) -> list[Zone]:
    """Scale canonical track/event zones onto the actual decoded frame.

    Falcon/SAM/OCR/VLM model views already scale their assigned ROI
    descriptors. This helper keeps the legacy temporal-memory zone events
    aligned too, so a visually calibrated ``counter_zone`` still produces
    ``entered_counter_zone`` and ``handover_candidate`` on resized
    analysis frames.
    """
    if not zones or frame_size is None:
        return zones
    from app.camera_rois import scale_zone_to_frame

    frame_w, frame_h = int(frame_size[0]), int(frame_size[1])
    out: list[Zone] = []
    for z in zones:
        body = {"x": z.x, "y": z.y, "w": z.w, "h": z.h}
        if z.source_width is not None and z.source_height is not None:
            body["source_width"] = z.source_width
            body["source_height"] = z.source_height
        scaled = scale_zone_to_frame(body, frame_w, frame_h)
        if scaled is None:
            continue
        out.append(Zone(
            name=z.name,
            x=int(scaled["x"]), y=int(scaled["y"]),
            w=int(scaled["w"]), h=int(scaled["h"]),
            source_width=int(scaled.get("source_width") or frame_w),
            source_height=int(scaled.get("source_height") or frame_h),
        ))
    if zones and not out:
        limitations.append("track_zones_collapsed_after_scale")
    return out


def _falcon_roi_crop(frames, view, limitations) -> Optional[tuple[int, int, int, int]]:
    """Build the per-window Falcon crop bbox from the model view, or
    return ``None`` when no usable assignment exists.

    Zones are scaled from their saved ``source_width/source_height``
    onto the actual decoded frame size before union/margin/clipping —
    so an operator who calibrated on a 1920x1080 snapshot still gets
    the right crop when the recorder writes 640x360 segments.
    The crop is expanded by ``margin_pct`` and clipped to the actual
    frame size (frames are assumed to share dimensions across a single
    window, which is true for the recorder's segments)."""
    if not view or view.get("mode") != "union_crop":
        return None
    zones = view.get("resolved_zones") or []
    if not zones:
        return None
    if not frames:
        return None
    _idx, _ts, sample = frames[0]
    try:
        w, h = sample.size
    except Exception:
        limitations.append("falcon_roi_crop_image_size_unavailable")
        return None
    from app.camera_rois import apply_margin, scale_zones_to_frame, union_bbox
    scaled = scale_zones_to_frame(zones, int(w), int(h))
    if not scaled:
        limitations.append("falcon_roi_crop_zones_collapsed_after_scale")
        return None
    bbox = union_bbox(scaled)
    if bbox is None:
        return None
    cropped = apply_margin(bbox, float(view.get("margin_pct") or 0.0),
                            int(w), int(h))
    limitations.append(
        f"falcon_roi_crop={cropped} margin={view.get('margin_pct') or 0.0}")
    return cropped


def _filter_by_roi(detections, view, *, limit_tag: str, limitations,
                    frame_size: Optional[tuple[int, int]] = None) -> list:
    """Return only detections whose centre falls inside any resolved
    ROI for ``view``. When the view is missing/empty, the detections
    are returned unchanged (back-compat). The number of filtered-out
    detections is recorded as a limitation so the reviewer can see
    when ROI gating dropped evidence.

    ``frame_size`` (when supplied as ``(w, h)``) lets the filter
    scale zones from their saved source dimensions onto the actual
    decoded frame size before the centre-in-rect check. When omitted
    the legacy raw-pixel zones are used."""
    if not view or not detections:
        return detections
    zones = view.get("resolved_zones") or []
    if not zones:
        return detections
    from app.camera_rois import detection_inside_rois, scale_zones_to_frame
    if frame_size is not None:
        zones = scale_zones_to_frame(zones, int(frame_size[0]),
                                       int(frame_size[1]))
        if not zones:
            limitations.append(
                f"{limit_tag}: zones collapsed after scale; "
                "keeping all detections")
            return detections
    kept = [d for d in detections
            if detection_inside_rois(list(d.bbox_xyxy), zones)]
    dropped = len(detections) - len(kept)
    if dropped:
        limitations.append(
            f"{limit_tag}: dropped {dropped} detection(s) outside ROI union")
    return kept


def _d_dict(d: Detection) -> dict:
    return {
        "label": d.label, "score": d.score,
        "bbox_xyxy": list(d.bbox_xyxy),
        "frame_id": d.frame_id, "frame_idx": d.frame_idx,
        "ts": d.ts.isoformat() if d.ts else None,
        "query": d.query,
    }


def _t_dict(t: Track) -> dict:
    return {
        "track_id": t.track_id, "label": t.label,
        "first_seen_ts": t.first_seen_ts.isoformat()
            if t.first_seen_ts else None,
        "last_seen_ts": t.last_seen_ts.isoformat()
            if t.last_seen_ts else None,
        "zones": list(t.zones), "events": list(t.events),
        "physical_item_candidate": t.physical_item_candidate,
        "receipt_candidate": t.receipt_candidate,
        "confidence": t.confidence,
    }


def _kf_dict(k: Keyframe) -> dict:
    return {
        "frame_id": k.frame_id, "frame_idx": k.frame_idx,
        "ts": k.ts.isoformat() if k.ts else None,
        "role": k.role, "uri": k.uri, "track_id": k.track_id,
    }


def _ocr_dict(o: OcrResult) -> dict:
    return {"frame_id": o.frame_id, "bbox_xyxy": list(o.bbox_xyxy),
            "text": o.text, "confidence": o.confidence,
            "crop_uri": o.crop_uri}


def _m_dict(m: Mask) -> dict:
    return {"detection_idx": m.detection_idx,
            "mask_uri": m.mask_uri, "score": m.score,
            "bbox_xyxy": list(m.bbox_xyxy) if m.bbox_xyxy else None}
