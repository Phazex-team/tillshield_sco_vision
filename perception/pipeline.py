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


def run_perception(session, case, window) -> dict:
    """Default production runner. Reads the window MP4 from disk,
    samples frames, and drives the perception pipeline."""
    from app.config import load_config, resolve_model_path

    cfg = load_config()
    falcon_cfg = cfg.models.get("falcon")
    falcon_path = None
    if falcon_cfg:
        try:
            falcon_path = resolve_model_path(falcon_cfg)
        except Exception:
            falcon_path = None
    sam2_cfg = cfg.raw.get("models", {}).get("sam2") or {}
    sam2_path = None
    if sam2_cfg:
        from app.config import ModelConfig
        sam2_path = resolve_model_path(
            ModelConfig(name=sam2_cfg.get("name",
                                          "facebook/sam2-hiera-large"),
                        enabled=True, extra=sam2_cfg),
            production_mode=False,
        )

    ocr_cfg = cfg.raw.get("models", {}).get("falcon_ocr") or {}
    ocr_path = None
    if ocr_cfg:
        from app.config import ModelConfig
        ocr_path = resolve_model_path(
            ModelConfig(name=ocr_cfg.get("name", "tiiuae/Falcon-OCR"),
                        enabled=True, extra=ocr_cfg),
            production_mode=False,
        )

    window_start_ts = (window.actual_start_at or
                       window.requested_start_at)
    # Resolve per-model ROI views for the active camera. Each view is
    # ``None`` when no usable assignment exists, in which case the
    # corresponding model keeps its full-frame behavior.
    from app.camera_rois import model_view
    return run_perception_on_window(
        window_path=window.path,
        window_start_ts=window_start_ts,
        fps=int(cfg.settings.get("gemma_video_fps", 25)),
        zones=_load_zones(cfg, case.camera_id),
        falcon_client=FalconClient(model_path=falcon_path),
        sam2_client=Sam2Client(model_path=sam2_path),
        ocr_engine=OcrEngine(model_path=ocr_path),
        sampling=SamplingPolicy(),
        falcon_roi_view=model_view(cfg, case.camera_id, "falcon"),
        sam2_roi_view=model_view(cfg, case.camera_id, "sam2"),
        ocr_roi_view=model_view(cfg, case.camera_id, "ocr"),
    )


def run_perception_on_window(*,
                             window_path: Optional[str],
                             fps: int,
                             zones: list[Zone],
                             falcon_client: FalconClient,
                             sam2_client: Sam2Client,
                             sampling: SamplingPolicy,
                             window_start_ts: Optional[datetime] = None,
                             ocr_engine: Optional[OcrEngine] = None,
                             falcon_query: str = DEFAULT_FALCON_QUERY,
                             falcon_roi_view: Optional[dict] = None,
                             sam2_roi_view: Optional[dict] = None,
                             ocr_roi_view: Optional[dict] = None,
                             ) -> dict:
    """Decode + run perception on a single video window. ``window_path``
    may be ``None`` in offline / synthetic test mode — in that case the
    sampler returns no frames and the function returns an empty result
    with the appropriate limitation tag.

    ``window_start_ts`` anchors every detection / track / keyframe
    timestamp to real CCTV wall-clock time. Callers must pass it for
    real-world correctness; tests may omit it (and a limitation is
    emitted noting the fallback)."""
    limitations: list[str] = []
    detections: list[Detection] = []
    frames_for_ocr: dict[int, object] = {}

    frames = _sample_frames(window_path, fps, sampling, limitations,
                            window_start_ts=window_start_ts)
    if not frames:
        return {
            "detections": [], "tracks": [], "keyframes": [],
            "ocr": [], "masks": [], "limitations": limitations,
            "obstructed": False,
        }

    # Compute Falcon ROI crop (full-frame coords). Detection bboxes
    # come back already offset to full-frame so every downstream
    # consumer keeps the same coordinate space.
    falcon_crop = _falcon_roi_crop(frames, falcon_roi_view, limitations)
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

    # SAM 2 ROI filter (filter-only by detection centre — we never crop
    # the image because SAM 2 needs full context for accurate masks).
    sam_detections = _filter_by_roi(detections, sam2_roi_view,
                                     limit_tag="sam2_roi_filter",
                                     limitations=limitations)

    masks: list[Mask] = []
    if sam_detections and sam2_client.has_capability():
        # Pick a representative frame near the middle of the window
        # for SAM 2; this matches the existing single-frame pipeline.
        mid_idx = frames[len(frames) // 2][0]
        mid_img = next((img for idx, ts, img in frames
                        if idx == mid_idx), frames[0][2])
        try:
            masks = sam2_client.segment(mid_img, sam_detections)
        except Exception:
            log.exception("sam2 segment failed")
            limitations.append("sam2_unavailable")
    elif sam_detections:
        limitations.append("sam2_unavailable")

    tracker = Tracker()
    tracker.update(detections)
    tracks = tracker.export()
    tracks = annotate_tracks(tracks, detections, zones=zones)

    ocr_input_detections = _filter_by_roi(detections, ocr_roi_view,
                                           limit_tag="ocr_roi_filter",
                                           limitations=limitations)
    ocr, ocr_limitations = run_ocr(ocr_engine, ocr_input_detections,
                                    frames_for_ocr)
    limitations.extend(ocr_limitations)

    keyframes = select_keyframes(tracks, detections)
    obstructed = any("obstruct" in l.lower() for l in limitations)
    return {
        "detections": [_d_dict(d) for d in detections],
        "tracks": [_t_dict(t) for t in tracks],
        "keyframes": [_kf_dict(k) for k in keyframes],
        "ocr": [_ocr_dict(o) for o in ocr],
        "masks": [_m_dict(m) for m in masks],
        "limitations": limitations,
        "obstructed": obstructed,
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
                zones.append(Zone(name=name, x=int(z["x"]), y=int(z["y"]),
                                  w=int(z["w"]), h=int(z["h"])))
            except (KeyError, TypeError, ValueError):
                continue
    return zones


def _falcon_roi_crop(frames, view, limitations) -> Optional[tuple[int, int, int, int]]:
    """Build the per-window Falcon crop bbox from the model view, or
    return ``None`` when no usable assignment exists.

    The crop is expanded by ``margin_pct`` and clipped to the actual
    frame size (frames are assumed to share dimensions across a single
    window, which is true for the recorder's segments)."""
    if not view or view.get("mode") != "union_crop":
        return None
    zones = view.get("resolved_zones") or []
    if not zones:
        return None
    from app.camera_rois import apply_margin, union_bbox
    bbox = union_bbox(zones)
    if bbox is None:
        return None
    if not frames:
        return None
    _idx, _ts, sample = frames[0]
    try:
        w, h = sample.size
    except Exception:
        limitations.append("falcon_roi_crop_image_size_unavailable")
        return None
    cropped = apply_margin(bbox, float(view.get("margin_pct") or 0.0),
                            int(w), int(h))
    limitations.append(
        f"falcon_roi_crop={cropped} margin={view.get('margin_pct') or 0.0}")
    return cropped


def _filter_by_roi(detections, view, *, limit_tag: str, limitations) -> list:
    """Return only detections whose centre falls inside any resolved
    ROI for ``view``. When the view is missing/empty, the detections
    are returned unchanged (back-compat). The number of filtered-out
    detections is recorded as a limitation so the reviewer can see
    when ROI gating dropped evidence."""
    if not view or not detections:
        return detections
    zones = view.get("resolved_zones") or []
    if not zones:
        return detections
    from app.camera_rois import detection_inside_rois
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
