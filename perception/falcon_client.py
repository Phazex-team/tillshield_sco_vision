"""Adapter around the existing FalconDetector.

The existing ``falcon_detector.FalconDetector`` is a thin in-process
wrapper around the tiiuae/Falcon-Perception checkpoint. This module
adds repo-local path resolution + a higher-level ``detect_on_frames``
helper that returns ``perception.schemas.Detection`` objects directly.

In production, falcon weights load from
``./models/hf/tiiuae/Falcon-Perception/<snapshot>/`` (resolved through
``app.config.resolve_model_path``). The tests stub this client entirely.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Optional

from PIL import Image

from .schemas import Detection


log = logging.getLogger(__name__)

# Process-wide cache of loaded FalconDetector instances, keyed by the resolved
# model target (local path or hub id). Falcon-Perception weights are ~2.4 GB and
# cost ~15-25 s to load from disk. The perception pipeline builds a fresh
# FalconClient per case, so without this cache every case reloaded the weights.
# Keeping the detector resident across cases removes that per-case reload. The
# transient inference activation (which can spike ~30 GB on a large ROI crop) is
# still freed after each case by the pipeline's torch.cuda.empty_cache(); only
# the small weight tensors persist here (~2.4 GB, well within the memory guard's
# headroom above Qwen's ~70 GB reservation).
_RESIDENT_LOCK = threading.Lock()
_RESIDENT_DETECTORS: dict[str, object] = {}


def warmup_falcon(cfg=None) -> bool:
    """Load the Falcon detector and drive one dummy detection at startup so
    the expensive first-run JIT (Falcon-Perception cold-compiles
    ``flex_attention`` + Triton kernels on the first inference — minutes on
    an sm_121 box) happens HERE, off the critical path, instead of inside
    the first real case while it holds ``falcon_lock`` / the single reprocess
    worker.

    Populates the process-wide resident cache, so the first real case then
    reuses the already-loaded, already-compiled detector. Idempotent and
    fully best-effort: any failure is logged and swallowed — warmup must
    never block or crash startup. Returns True iff a detect completed.

    Pairs with the persistent ``TORCHINDUCTOR_CACHE_DIR`` / ``TRITON_CACHE_DIR``
    (docker-compose): the first ever run still cold-compiles into that cache;
    every subsequent restart is warm and this returns in ~seconds.
    """
    try:
        from app.config import load_config, resolve_model_path
        cfg = cfg or load_config()
        fcfg = cfg.models.get("falcon")
        if fcfg is not None and not fcfg.enabled:
            log.info("falcon warmup skipped (falcon disabled in config)")
            return False
        path = None
        try:
            if fcfg is not None:
                path = resolve_model_path(fcfg)
        except Exception:
            path = None
        keep_resident = bool(
            cfg.raw.get("gpu", {}).get("keep_falcon_resident", True))
        client = FalconClient(model_path=path, keep_resident=keep_resident)
        t0 = time.time()
        log.info("falcon warmup: loading + JIT-compiling detector (first "
                 "cold run compiles flex_attention; can take several "
                 "minutes on a fresh compile cache)")
        client._ensure_loaded()
        # A tiny dummy detect drives the full prefill + decode compile path.
        dummy = Image.new("RGB", (256, 256), (127, 127, 127))
        client._detector.detect(dummy, query="item")
        log.info("falcon warmup complete in %.1fs (real cases now skip the "
                 "JIT)", time.time() - t0)
        return True
    except Exception:
        log.exception("falcon warmup failed (non-fatal; the first real case "
                      "will pay the compile instead)")
        return False


def release_resident_falcon() -> None:
    """Evict all cached resident FalconDetector weights and free GPU memory.

    Call from an emergency memory path (or an ops endpoint) when the resident
    Falcon weights must be reclaimed. Safe to call anytime — the next detect()
    reloads on demand. No-op when nothing is cached."""
    with _RESIDENT_LOCK:
        had = bool(_RESIDENT_DETECTORS)
        _RESIDENT_DETECTORS.clear()
    if not had:
        return
    try:
        import gc

        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        log.debug("cuda empty_cache after falcon release skipped",
                  exc_info=True)

# Frames-x-categories are sent to the detector in batches of this many per
# GPU call. The ROI crops are small, so a modest batch fits comfortably in
# unified memory alongside Qwen while cutting round-trips ~8x.
_FALCON_BATCH = 8


def _iou_px(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms(dets, iou_thresh: float = 0.5):
    """Greedy NMS over one (frame, label)'s boxes — collapse the many
    near-duplicate boxes Falcon emits for a single physical item into one,
    keeping the larger box as representative. Distinct, well-separated items
    (low IoU) are all kept, so this never merges different objects."""
    if len(dets) <= 1:
        return dets
    ordered = sorted(
        dets,
        key=lambda d: (d.bbox_px[2] - d.bbox_px[0]) * (d.bbox_px[3] - d.bbox_px[1]),
        reverse=True)
    kept = []
    for d in ordered:
        if all(_iou_px(d.bbox_px, k.bbox_px) < iou_thresh for k in kept):
            kept.append(d)
    return kept


class FalconClient:
    """Wrap FalconDetector with manifest-friendly outputs."""

    def __init__(self, *, model_path: Optional[str] = None,
                 model_name: str = "tiiuae/Falcon-Perception",
                 keep_resident: bool = True):
        self.model_path = model_path
        self.model_name = model_name
        # When True (default), the loaded detector is cached process-wide so
        # each new per-case FalconClient reuses the already-loaded weights
        # instead of reloading ~2.4 GB from disk. See _RESIDENT_DETECTORS.
        self.keep_resident = keep_resident
        self._detector = None

    def _ensure_loaded(self):
        if self._detector is not None:
            return
        from falcon_detector import FalconDetector
        target = self.model_path or self.model_name
        if not self.keep_resident:
            self._detector = FalconDetector(target)
            return
        # Reuse (or populate) the process-wide resident cache. Guarded so a
        # first-touch load can't race, though case processing is sequential.
        with _RESIDENT_LOCK:
            det = _RESIDENT_DETECTORS.get(target)
            if det is None:
                det = FalconDetector(target)
                _RESIDENT_DETECTORS[target] = det
            self._detector = det

    # Falcon-Perception is a referring detector: it returns boxes that
    # match a query phrase but emits NO per-box class label. To get
    # labels the track-gating can use (``perception.temporal_memory``
    # keys on "item"/"product"/"receipt"/...), we run one query per
    # category and stamp the category name onto every box it returns.
    DEFAULT_CATEGORIES: dict[str, str] = {
        "item": ("item, product, merchandise, bag, shopping bag, box, "
                 "package, clothing, bottle, electronics, phone"),
        "receipt": "receipt, document, paper, invoice",
        "person": "person, hand, arm, cashier, customer",
    }

    # Keys that downstream consumers (decision policy, customer_present gate,
    # track-gating in perception.temporal_memory) rely on. Callers can ADD
    # new categories (e.g. POS-derived SKU queries) but must never overwrite
    # these. A custom categories dict that drops/replaces them would silently
    # break person detection or item gating elsewhere in the pipeline.
    RESERVED_CATEGORY_KEYS: frozenset[str] = frozenset({"item", "person", "receipt"})

    def detect_on_frames(self,
                         frames: list[tuple[int, datetime, Image.Image]],
                         *,
                         query: Optional[str] = None,
                         categories: Optional[dict[str, str]] = None,
                         roi_crop: Optional[tuple[int, int, int, int]] = None
                         ) -> list[Detection]:
        """Run category-aware detection on ``(frame_idx, ts, pil_image)``.

        For each frame and each ``{label: query}`` category, the matched
        boxes are returned as ``Detection``s labelled with the category
        name. ``query`` is accepted for backward compatibility (added to
        the categories under a generic "item" bucket). Exceptions are
        caught per (frame, category) so one decode failure never loses
        the rest of the window.

        ``categories`` ADDS to ``DEFAULT_CATEGORIES`` — it does not
        replace it. Reserved keys (``item``, ``person``, ``receipt``)
        cannot be overwritten via ``categories`` because downstream
        consumers (decision policy, customer-presence gate, track-gating)
        depend on them. Attempts to overwrite are logged and ignored.

        ``roi_crop`` (optional ``(x1, y1, x2, y2)`` in *full-frame*
        pixel coordinates) restricts Falcon to the cropped region.
        Detection ``bbox_xyxy`` values are translated back to full-frame
        coordinates before being returned, so every downstream consumer
        (tracker, SAM 2, OCR, decision policy, evidence package) keeps
        its full-frame coordinate contract.
        """
        self._ensure_loaded()
        # Always start from defaults so person/item/receipt detectors
        # never silently disappear when a scenario adds POS categories.
        cats: dict[str, str] = dict(self.DEFAULT_CATEGORIES)
        if categories:
            for k, v in categories.items():
                if k in self.RESERVED_CATEGORY_KEYS:
                    log.warning(
                        "FalconClient: refusing to overwrite reserved "
                        "category %r via categories= (defaults preserved). "
                        "Use query= to refine the 'item' default.", k)
                    continue
                cats[k] = v
        if query:
            cats.setdefault("item", query)
        ox, oy = 0, 0
        if roi_crop is not None:
            ox, oy = int(roi_crop[0]), int(roi_crop[1])
        # NOTE: we deliberately run detection per (frame, category) rather
        # than batching. The ROI crops are small (~500px); batching pads
        # every image to a common max_dimension canvas + left-pads the
        # heterogeneous category prompts, and that padding overhead exceeds
        # the round-trip savings (measured SLOWER on GB10). Per-image keeps
        # each crop at its native size. Speed comes instead from the NMS +
        # coord-dedup below (fewer boxes) and the sequential frame decode.
        results: list[Detection] = []
        for frame_idx, ts, img in frames:
            target_img = img
            if roi_crop is not None:
                # Clip the crop to the actual image bounds; Falcon would
                # otherwise mis-handle out-of-bounds crops.
                w, h = img.size
                cx1 = max(0, min(int(roi_crop[0]), w))
                cy1 = max(0, min(int(roi_crop[1]), h))
                cx2 = max(cx1, min(int(roi_crop[2]), w))
                cy2 = max(cy1, min(int(roi_crop[3]), h))
                if cx2 > cx1 and cy2 > cy1:
                    target_img = img.crop((cx1, cy1, cx2, cy2))
                    ox, oy = cx1, cy1
                else:
                    target_img = img
                    ox, oy = 0, 0
            for label, cat_query in cats.items():
                try:
                    _, dets = self._detector.detect(target_img,
                                                    query=cat_query)
                except Exception:
                    log.exception("falcon detect failed on frame %d (%s)",
                                  frame_idx, label)
                    continue
                # Collapse the many near-duplicate boxes Falcon emits for the
                # SAME item in the SAME frame (per frame+label NMS). Removes
                # redundant boxes on one object; drops no frames.
                for d in _nms(dets):
                    bx = [float(x) for x in d.bbox_px]
                    if ox or oy:
                        bx = [bx[0] + ox, bx[1] + oy,
                              bx[2] + ox, bx[3] + oy]
                    results.append(Detection(
                        label=label,
                        score=float(getattr(d, "score", 0.0)) or 0.5,
                        bbox_xyxy=bx,
                        frame_id=f"frame_{frame_idx:06d}",
                        frame_idx=frame_idx,
                        ts=ts,
                        query=cat_query,
                    ))
        return results

    def unload(self) -> None:
        # Detach this client's handle only. When keep_resident is set the
        # shared detector stays in the process-wide cache so the next case
        # reuses the already-loaded weights (no disk reload); the pipeline's
        # torch.cuda.empty_cache() still frees the transient inference
        # activation. When not resident this drops the sole reference, so the
        # weights are freed on the next gc/empty_cache. To actually reclaim
        # resident weights under memory pressure, call release_resident_falcon().
        self._detector = None
