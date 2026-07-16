"""Render saved still snapshots of what Falcon detected, per case.

The reviewer UI already draws Falcon detection boxes live over the window
MP4 (a client-side canvas overlay). That needs the video to replay and the
browser to re-derive the geometry. This module produces the *durable*
counterpart: it burns the detection boxes + labels onto a handful of
representative frames and saves them as PNGs under the case's storage dir,
so the case-detail page can show "what Falcon saw" as stills — no video
replay, and the evidence survives even if the window MP4 is later pruned.

Pure/deterministic and cv2-based: no GPU, no model, no network. The box
style deliberately mirrors the live overlay in ``static/review.html``
(green for item boxes, blue for everything else) so the still and the live
overlay read as the same thing.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)


# BGR tuples matching the live overlay's hex colours in review.html:
#   items  -> #3ddc84  (R61 G220 B132) -> BGR (132, 220, 61)
#   other  -> #5b8def  (R91 G141 B239) -> BGR (239, 141, 91)
ITEM_COLOR_BGR = (132, 220, 61)
OTHER_COLOR_BGR = (239, 141, 91)
_LABEL_BG_BGR = (0, 0, 0)

DEFAULT_MAX_SNAPSHOTS = 6

# Per-item crops: how much context to keep around the box (fraction of the
# box's own size, per side). A tight crop of a bare box is often
# unrecognisable, so a little surrounding context is kept — but never so
# much that the crop reads as "the whole frame again".
DEFAULT_CROP_MARGIN = 0.12
# A basket is a handful of lines; this is only a runaway guard.
DEFAULT_MAX_ITEM_CROPS = 40


def _is_item_label(label: Optional[str]) -> bool:
    return "item" in (label or "").lower()


def _valid_bbox(bbox: Any) -> Optional[list[float]]:
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return None
    try:
        return [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None


def annotate_frame(frame_bgr, boxes: list[dict]):
    """Return a copy of ``frame_bgr`` (an HxWx3 BGR ndarray) with each
    detection box + label drawn on it. Boxes with an unusable bbox are
    skipped. Never raises on malformed box dicts."""
    import cv2

    out = frame_bgr.copy()
    for b in boxes:
        if not isinstance(b, dict):
            continue
        bbox = _valid_bbox(b.get("bbox_xyxy") or b.get("bbox"))
        if bbox is None:
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        color = ITEM_COLOR_BGR if _is_item_label(b.get("label")) else \
            OTHER_COLOR_BGR
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = str(b.get("label") or "obj")
        score = b.get("score")
        try:
            tag = f"{label} {float(score):.2f}" if score is not None else label
        except (TypeError, ValueError):
            tag = label
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        ty = y1 - th - 6
        if ty < 0:                       # label would clip off the top edge
            ty = y1 + 2
        cv2.rectangle(out, (x1, ty), (x1 + tw + 4, ty + th + 6),
                      _LABEL_BG_BGR, -1)
        cv2.putText(out, tag, (x1 + 2, ty + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return out


def select_snapshot_frames(detections: list[dict],
                           max_snapshots: int) -> list[tuple[int, list[dict]]]:
    """Group detections by ``frame_idx`` and return up to ``max_snapshots``
    of the busiest frames (most boxes first, earliest frame as tiebreak).
    Detections without a usable bbox are ignored so an empty-box frame is
    never selected."""
    by_frame: dict[int, list[dict]] = {}
    for d in detections or []:
        if not isinstance(d, dict) or _valid_bbox(d.get("bbox_xyxy")) is None:
            continue
        try:
            fi = int(d.get("frame_idx") or 0)
        except (TypeError, ValueError):
            fi = 0
        by_frame.setdefault(fi, []).append(d)
    ordered = sorted(by_frame.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    return ordered[: max(0, int(max_snapshots))]


def _frame_idx_by_frame_id(detections: list[dict]) -> dict[str, int]:
    """Map ``frame_id`` -> ``frame_idx`` from raw detections.

    Canonical groups carry a ``representative_frame_id`` (a string), but
    decoding needs the integer index. Both fields live on every Detection
    (``perception.schemas``), so the raw detection list is the join table —
    this avoids re-deriving the index by parsing the id string, and avoids
    adding a field to the group dicts (which feed the VLM prompt).
    """
    out: dict[str, int] = {}
    for d in detections or []:
        if not isinstance(d, dict):
            continue
        fid = d.get("frame_id")
        idx = d.get("frame_idx")
        if isinstance(fid, str) and fid and isinstance(idx, int):
            out.setdefault(fid, idx)
    return out


def _crop_box(bbox: list[float], w: int, h: int, margin: float
              ) -> Optional[tuple[int, int, int, int]]:
    """Expand ``bbox`` by ``margin`` (fraction of its own size) and clip to
    the frame. Returns None when the result has no area."""
    x1, y1, x2, y2 = bbox
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    mx = (x2 - x1) * margin
    my = (y2 - y1) * margin
    cx1 = max(0, int(round(x1 - mx)))
    cy1 = max(0, int(round(y1 - my)))
    cx2 = min(w, int(round(x2 + mx)))
    cy2 = min(h, int(round(y2 + my)))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return cx1, cy1, cx2, cy2


def render_item_crops(*,
                      window_path: str,
                      groups: list[dict],
                      detections: list[dict],
                      out_dir: str | Path,
                      margin: float = DEFAULT_CROP_MARGIN,
                      max_crops: int = DEFAULT_MAX_ITEM_CROPS,
                      frame_reader: Optional[Callable[[int], Any]] = None,
                      ) -> list[dict]:
    """Write ONE cropped PNG per canonical item group — the item alone, not
    the whole frame.

    ``groups`` are ``perception.item_grouping.group_sco_items`` output: each
    is one distinct physical item (Falcon's fragmented re-detections already
    collapsed), carrying ``representative_bbox`` + ``representative_frame_id``.
    So N groups -> N crops, which is the count the reviewer sees as
    "items detected" (``fl_audit_zone_count``).

    Each crop is the representative box (plus ``margin`` context) cut out of
    its representative frame, with the box outlined in place — green for a
    POS-matched item, blue for an extra (not on the bill), mirroring the
    full-frame stills and the live overlay.

    Same contract as ``render_detection_snapshots``: best-effort, never
    raises, skips anything unusable. Returns descriptors::

        {"filename": "item_00_sco_group_001.png",
         "path": "<abs path>",
         "group_id": "sco_group_001",
         "frame_idx": 42,
         "matched_pos_item": "<basket line>" | None,
         "is_extra": bool,
         "confidence": "high" | "medium" | "low" | None}
    """
    import cv2

    if not groups:
        return []
    idx_by_id = _frame_idx_by_frame_id(detections or [])
    out_dir = Path(out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        log.exception("item crop dir create failed: %s", out_dir)
        return []
    reader = frame_reader or _cv2_frame_reader(window_path)

    results: list[dict] = []
    for order, g in enumerate(groups[: max(0, int(max_crops))]):
        if not isinstance(g, dict):
            continue
        bbox = _valid_bbox(g.get("representative_bbox"))
        if bbox is None:
            continue
        frame_idx = idx_by_id.get(str(g.get("representative_frame_id") or ""))
        if frame_idx is None:
            log.warning("item crop: no frame_idx for group %s (frame_id=%r)",
                        g.get("group_id"), g.get("representative_frame_id"))
            continue
        try:
            frame = reader(frame_idx)
        except Exception:
            log.exception("item crop frame read failed idx=%s", frame_idx)
            frame = None
        if frame is None:
            continue
        try:
            h, w = frame.shape[:2]
            box = _crop_box(bbox, w, h, margin)
            if box is None:
                continue
            cx1, cy1, cx2, cy2 = box
            crop = frame[cy1:cy2, cx1:cx2].copy()
            is_extra = bool(g.get("is_extra_candidate"))
            color = OTHER_COLOR_BGR if is_extra else ITEM_COLOR_BGR
            # Outline the detection inside the crop (box coords are
            # full-frame, so shift them into crop space).
            cv2.rectangle(crop,
                          (int(round(bbox[0])) - cx1, int(round(bbox[1])) - cy1),
                          (int(round(bbox[2])) - cx1, int(round(bbox[3])) - cy1),
                          color, 2)
            gid = str(g.get("group_id") or f"group_{order:03d}")
            name = f"item_{order:02d}_{gid}.png"
            path = out_dir / name
            if not cv2.imwrite(str(path), crop):
                log.warning("item crop write returned false: %s", path)
                continue
        except Exception:
            log.exception("item crop render failed group=%s",
                          g.get("group_id"))
            continue
        results.append({
            "filename": name,
            "path": str(path),
            "group_id": gid,
            "frame_idx": int(frame_idx),
            "frame_ts": g.get("first_seen_ts"),
            "matched_pos_item": g.get("matched_pos_item"),
            "is_extra": is_extra,
            "confidence": g.get("confidence"),
        })
    return results


def _cv2_frame_reader(window_path: str) -> Callable[[int], Any]:
    """Default frame reader: seek + decode a single frame from the window
    MP4. Snapshots are few (<= max_snapshots), so per-frame open/seek is
    fine and keeps each read independent."""
    def read(frame_idx: int):
        import cv2

        cap = cv2.VideoCapture(str(window_path))
        try:
            if not cap.isOpened():
                return None
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
            ok, frame = cap.read()
            return frame if ok else None
        finally:
            cap.release()
    return read


def render_detection_snapshots(*,
                               window_path: str,
                               detections: list[dict],
                               out_dir: str | Path,
                               max_snapshots: int = DEFAULT_MAX_SNAPSHOTS,
                               frame_reader: Optional[
                                   Callable[[int], Any]] = None,
                               ) -> list[dict]:
    """Burn Falcon detection boxes onto representative frames and write one
    PNG per frame under ``out_dir``. Returns a list of descriptors::

        {"filename": "falcon_00_frame_000042.png",
         "path": "<abs path>",
         "frame_idx": 42,
         "frame_ts": "<ISO or None>",
         "box_count": 3}

    Best-effort and side-effect-isolated: a frame that fails to decode or
    write is skipped, never raised, so snapshot generation can never fail a
    case run. ``frame_reader`` is injectable for tests (maps frame_idx ->
    BGR ndarray or None); the default decodes from ``window_path`` via cv2.
    """
    import cv2

    frames = select_snapshot_frames(detections or [], max_snapshots)
    if not frames:
        return []
    reader = frame_reader or _cv2_frame_reader(window_path)
    out_dir = Path(out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        log.exception("snapshot dir create failed: %s", out_dir)
        return []

    results: list[dict] = []
    for order, (frame_idx, boxes) in enumerate(frames):
        try:
            frame = reader(frame_idx)
        except Exception:
            log.exception("snapshot frame read failed idx=%s", frame_idx)
            frame = None
        if frame is None:
            continue
        try:
            annotated = annotate_frame(frame, boxes)
            name = f"falcon_{order:02d}_frame_{int(frame_idx):06d}.png"
            path = out_dir / name
            if not cv2.imwrite(str(path), annotated):
                log.warning("snapshot write returned false: %s", path)
                continue
        except Exception:
            log.exception("snapshot annotate/write failed idx=%s", frame_idx)
            continue
        frame_ts = next(
            (b.get("frame_ts") or b.get("ts")
             for b in boxes if b.get("frame_ts") or b.get("ts")), None)
        results.append({
            "filename": name,
            "path": str(path),
            "frame_idx": int(frame_idx),
            "frame_ts": frame_ts,
            "box_count": len(boxes),
        })
    return results
