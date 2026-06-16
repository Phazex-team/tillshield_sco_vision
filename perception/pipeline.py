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
    return run_perception_on_window(
        window_path=window.path,
        window_start_ts=window_start_ts,
        fps=int(cfg.settings.get("gemma_video_fps", 25)),
        zones=_load_zones(cfg, case.camera_id),
        falcon_client=FalconClient(model_path=falcon_path),
        sam2_client=Sam2Client(model_path=sam2_path),
        ocr_engine=OcrEngine(model_path=ocr_path),
        sampling=SamplingPolicy(),
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

    try:
        detections = falcon_client.detect_on_frames(
            [(idx, ts, img) for idx, ts, img in frames],
            query=falcon_query,
        )
        for idx, ts, img in frames:
            frames_for_ocr[idx] = img
    except Exception:
        log.exception("falcon detection failed")
        limitations.append("falcon_unavailable")

    masks: list[Mask] = []
    if detections and sam2_client.has_capability():
        # Pick a representative frame near the middle of the window
        # for SAM 2; this matches the existing single-frame pipeline.
        mid_idx = frames[len(frames) // 2][0]
        mid_img = next((img for idx, ts, img in frames
                        if idx == mid_idx), frames[0][2])
        try:
            masks = sam2_client.segment(mid_img, detections)
        except Exception:
            log.exception("sam2 segment failed")
            limitations.append("sam2_unavailable")
    elif detections:
        limitations.append("sam2_unavailable")

    tracker = Tracker()
    tracker.update(detections)
    tracks = tracker.export()
    tracks = annotate_tracks(tracks, detections, zones=zones)

    ocr, ocr_limitations = run_ocr(ocr_engine, detections, frames_for_ocr)
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
            try:
                zones.append(Zone(name=name, x=int(z["x"]), y=int(z["y"]),
                                  w=int(z["w"]), h=int(z["h"])))
            except (KeyError, TypeError, ValueError):
                continue
    return zones


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
