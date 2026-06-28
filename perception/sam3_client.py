"""SAM 3 video perception backend (concept-prompted, Falcon-independent).

Unlike ``sam2_client.py``, which segments around Falcon detection
boxes after the fact, this backend runs **directly on the video
window** with **POS-derived natural-language concept prompts**
("biriyani hot food", "hot food container", etc.) and returns
stable per-object identities across frames.

Loads the upstream HuggingFace ``Sam3VideoModel`` + ``Sam3VideoProcessor``
from a repo-local snapshot
(``models/hf/facebook/sam3/<snapshot>/``).

When the ``transformers`` build is missing the ``sam3_video`` model
family OR the weights directory isn't present, ``has_capability()``
is False and callers should record ``sam3_unavailable`` as a
limitation rather than crashing.

Output normalised to the same dict shape the rest of the SCO chain
expects (Detection-style records labelled ``sco_item_NNN`` /
``sco_generic_*`` so the item grouper sees a uniform schema, plus
Track-style records carrying the SAM 3 object ID as the stable
identity).
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Optional


log = logging.getLogger(__name__)


# Reserved SCO label prefixes — the grouper and case_runner already
# split detections by these (sco_item_NNN → POS-derived;
# sco_generic_* → catch-all; bare 'item'/'person'/'receipt' →
# defaults). Keeping them consistent here means the SAM-3 path
# slots into the existing grouper with zero changes.
POS_LABEL_RE = re.compile(r"^sco_item_(\d+)$")
GENERIC_LABEL_PREFIX = "sco_generic_"
DEFAULT_GENERIC_KEY = "sco_generic_products"


@dataclass
class Sam3Concept:
    """One text concept that SAM 3 will search for in the video.

    ``label`` is the schema-friendly key that the grouper consumes
    (``sco_item_000`` or ``sco_generic_*``). ``text`` is the
    natural-language phrase actually sent to SAM 3.
    """
    label: str
    text: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class Sam3Client:
    """Concept-prompted video perception via Meta SAM 3."""

    def __init__(self, *, model_path: Optional[str] = None,
                 device: str = "cuda",
                 dtype: str = "bfloat16",
                 inference_state_device: Optional[str] = "cpu",
                 video_storage_device: Optional[str] = "cpu",
                 score_threshold: float = 0.5):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.inference_state_device = inference_state_device
        self.video_storage_device = video_storage_device
        self.score_threshold = float(score_threshold)
        self._model = None
        self._processor = None
        self._load_err: Optional[str] = None

    # ------------------------------------------------------------------
    # Capability
    # ------------------------------------------------------------------

    def has_capability(self) -> bool:
        if self._model is not None and self._processor is not None:
            return True
        if self._load_err is not None:
            return False
        try:
            import transformers  # noqa: F401
            from transformers.models.sam3_video import (  # noqa: F401
                modeling_sam3_video, processing_sam3_video,
            )
        except Exception as exc:
            self._load_err = f"sam3_video transformers integration missing: {exc}"
            return False
        if not self.model_path or not os.path.isdir(self.model_path):
            self._load_err = f"sam3 weights missing: {self.model_path!r}"
            return False
        return True

    def _resolve_dtype(self):
        import torch
        return {
            "float16": torch.float16, "half": torch.float16,
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float32": torch.float32, "fp32": torch.float32,
        }.get(str(self.dtype).lower(), torch.bfloat16)

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        if not self.has_capability():
            raise RuntimeError(self._load_err or "sam3 unavailable")
        # Hard-gag the HF hub before any import-driven download path.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from transformers import Sam3VideoModel, Sam3VideoProcessor
        self._processor = Sam3VideoProcessor.from_pretrained(
            self.model_path, local_files_only=True)
        self._model = Sam3VideoModel.from_pretrained(
            self.model_path,
            local_files_only=True,
            dtype=self._resolve_dtype(),
        ).to(self.device)
        self._model.eval()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def process_window(self,
                       window_path: str,
                       *,
                       concepts: Iterable[Sam3Concept],
                       window_start_ts: Optional[datetime] = None,
                       fps_hint: int = 25,
                       roi_crop_xyxy: Optional[tuple[int, int, int, int]] = None,
                       max_frames: Optional[int] = None,
                       ) -> dict:
        """Run SAM 3 video propagation on a single window MP4.

        Returns a dict shaped like the rest of the perception pipeline:

            {
              "detections": [<dict>, ...],   # one per (frame, object)
              "tracks":     [<dict>, ...],   # one per stable SAM 3 object id
              "masks":      [<dict>, ...],   # binary mask URIs / shapes
              "limitations": [...],
              "obstructed": False,
              "timings_ms": {...},
              "sam3_meta": {                  # backend audit
                "object_ids": [...],
                "prompt_to_obj_ids": {...},
                "frame_count": int,
              },
            }

        Never raises into the caller; failures populate ``limitations``.
        """
        import time as _time
        from PIL import Image  # noqa: F401  (deferred until call time)

        limitations: list[str] = []
        timings: dict[str, int] = {}
        t0 = _time.perf_counter()

        def _ms_since(start: float) -> int:
            return int((_time.perf_counter() - start) * 1000)

        concepts_list = [c for c in (concepts or []) if c.text]
        if not concepts_list:
            limitations.append("sam3_no_concepts")
            return self._empty_result(limitations, timings, t0, _ms_since)

        if not self.has_capability():
            limitations.append("sam3_unavailable")
            log.warning("sam3 unavailable: %s", self._load_err)
            return self._empty_result(limitations, timings, t0, _ms_since)

        try:
            self._ensure_loaded()
        except Exception as exc:
            limitations.append("sam3_unavailable")
            log.exception("sam3 load failed: %s", exc)
            return self._empty_result(limitations, timings, t0, _ms_since)

        # ---- decode video frames -----------------------------------
        t_dec = _time.perf_counter()
        try:
            video_frames, frame_times, frame_indices = _decode_video(
                window_path, window_start_ts=window_start_ts,
                fps_hint=fps_hint, max_frames=max_frames,
                roi_crop_xyxy=roi_crop_xyxy)
        except Exception as exc:
            limitations.append("sam3_decode_failed")
            log.exception("sam3 decode failed: %s", exc)
            return self._empty_result(limitations, timings, t0, _ms_since)
        timings["sam3_decode_ms"] = _ms_since(t_dec)
        if not video_frames:
            limitations.append("sam3_no_frames")
            return self._empty_result(limitations, timings, t0, _ms_since)

        # ---- run propagation ---------------------------------------
        t_inf = _time.perf_counter()
        try:
            session = self._processor.init_video_session(
                video=video_frames,
                inference_device=self.device,
                inference_state_device=self.inference_state_device,
                video_storage_device=self.video_storage_device,
                dtype=self._resolve_dtype(),
            )
            # Two-step prompt encoding: collect text per label so we
            # can map SAM-3 object IDs back to the schema label after
            # postprocess.
            text_to_label: dict[str, str] = {}
            for c in concepts_list:
                text_to_label[c.text] = c.label
            self._processor.add_text_prompt(
                session, text=[c.text for c in concepts_list])

            per_frame_outputs = []
            for out in self._model.propagate_in_video_iterator(
                    inference_session=session,
                    start_frame_idx=0,
                    show_progress_bar=False):
                per_frame_outputs.append(out)
        except Exception as exc:
            limitations.append("sam3_inference_failed")
            log.exception("sam3 propagation failed: %s", exc)
            return self._empty_result(limitations, timings, t0, _ms_since)
        finally:
            timings["sam3_inference_ms"] = _ms_since(t_inf)

        # ---- normalise -------------------------------------------------
        t_norm = _time.perf_counter()
        try:
            detections, tracks, masks, sam3_meta = self._normalise(
                per_frame_outputs=per_frame_outputs,
                session=session,
                text_to_label=text_to_label,
                frame_times=frame_times,
                frame_indices=frame_indices,
                roi_offset=(roi_crop_xyxy[0] if roi_crop_xyxy else 0,
                            roi_crop_xyxy[1] if roi_crop_xyxy else 0),
            )
        except Exception as exc:
            limitations.append("sam3_postprocess_failed")
            log.exception("sam3 postprocess failed: %s", exc)
            return self._empty_result(limitations, timings, t0, _ms_since)
        finally:
            timings["sam3_postprocess_ms"] = _ms_since(t_norm)

        timings["total_ms"] = _ms_since(t0)
        return {
            "detections": detections,
            "tracks": tracks,
            "masks": masks,
            "keyframes": [],
            "ocr": [],
            "limitations": limitations,
            "obstructed": False,
            "timings_ms": timings,
            "sam3_meta": sam3_meta,
        }

    def unload(self) -> None:
        self._model = None
        self._processor = None
        try:
            import torch
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _empty_result(self, limitations, timings, t0, _ms_since):
        timings["total_ms"] = _ms_since(t0)
        return {
            "detections": [], "tracks": [], "masks": [], "keyframes": [],
            "ocr": [], "limitations": limitations,
            "obstructed": False, "timings_ms": timings,
            "sam3_meta": {},
        }

    def _normalise(self,
                   *,
                   per_frame_outputs,
                   session,
                   text_to_label: dict[str, str],
                   frame_times: list[datetime],
                   frame_indices: list[int],
                   roi_offset: tuple[int, int]) -> tuple[list[dict],
                                                         list[dict],
                                                         list[dict],
                                                         dict]:
        """Convert SAM-3 per-frame outputs + final session state to
        the perception dict shape.

        The grouper consumes ``tracks[]`` with stable identities and
        ``detections[]`` with ``label`` carrying the schema key
        (``sco_item_NNN`` or ``sco_generic_*``). SAM-3's ``object_ids``
        are the stable identities — same ID across frames = same
        physical object, so the grouper's collapse logic gets a clean
        signal.
        """
        import torch

        ox, oy = roi_offset
        detections: list[dict] = []

        # Build per-object accumulators: ID -> (label, list of per-frame
        # records with bbox/score/frame_idx/ts).
        per_obj: dict[int, dict] = {}

        # Postprocess each frame to recover boxes + masks at video res.
        # We don't need masks for the grouper but we attach them for
        # audit / evidence overlay.
        masks_out: list[dict] = []
        for out, frame_idx, ts in zip(per_frame_outputs,
                                       frame_indices, frame_times):
            try:
                post = self._processor.postprocess_outputs(
                    inference_session=session,
                    model_outputs=out,
                )
            except Exception:
                log.exception("sam3 postprocess_outputs failed for "
                              "frame_idx=%s", frame_idx)
                continue
            object_ids = post.get("object_ids")
            if object_ids is None or len(object_ids) == 0:
                continue
            boxes = post["boxes"].detach().cpu().tolist()
            scores = post["scores"].detach().cpu().tolist()
            obj_ids = object_ids.detach().cpu().tolist() \
                if hasattr(object_ids, "detach") else list(object_ids)
            # SAM-3 reports a prompt_id per object via the session; we
            # map prompt_id -> prompt_text -> our schema label.
            prompt_to_text = dict(session.prompts)
            for oid, score, bbox in zip(obj_ids, scores, boxes):
                if float(score) < self.score_threshold:
                    continue
                pid = session.obj_id_to_prompt_id.get(int(oid))
                if pid is None:
                    label = DEFAULT_GENERIC_KEY
                else:
                    prompt_text = prompt_to_text.get(pid, "")
                    label = text_to_label.get(prompt_text,
                                              GENERIC_LABEL_PREFIX + "unknown")
                x1, y1, x2, y2 = (float(v) for v in bbox)
                x1 += ox; x2 += ox; y1 += oy; y2 += oy
                det_idx = len(detections)
                det = {
                    "label": label,
                    "score": float(score),
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "frame_id": f"frame_{int(frame_idx):06d}",
                    "frame_idx": int(frame_idx),
                    "ts": ts.isoformat(),
                    "query": label,
                    "sam3_object_id": int(oid),
                }
                detections.append(det)
                bucket = per_obj.setdefault(int(oid), {
                    "label": label,
                    "first_ts": ts,
                    "last_ts": ts,
                    "det_indices": [],
                    "best_score": float(score),
                })
                bucket["det_indices"].append(det_idx)
                if ts < bucket["first_ts"]:
                    bucket["first_ts"] = ts
                if ts > bucket["last_ts"]:
                    bucket["last_ts"] = ts
                if float(score) > bucket["best_score"]:
                    bucket["best_score"] = float(score)
                # Label stability: prefer the POS-specific label over
                # a generic one if the same object switches prompts.
                if bucket["label"].startswith(GENERIC_LABEL_PREFIX) \
                        and POS_LABEL_RE.match(label):
                    bucket["label"] = label

        # Build track records from accumulators.
        tracks: list[dict] = []
        for oid, b in sorted(per_obj.items()):
            tracks.append({
                "track_id": f"sam3_obj_{int(oid):04d}",
                "label": b["label"],
                "first_seen_ts": b["first_ts"].isoformat(),
                "last_seen_ts": b["last_ts"].isoformat(),
                "detections": list(b["det_indices"]),
                "zones": ["sco_audit_zone"],
                "events": [],
                "physical_item_candidate": b["label"].startswith("sco_"),
                "receipt_candidate": False,
                "confidence": float(b["best_score"]),
                "sam3_object_id": int(oid),
            })

        sam3_meta = {
            "object_ids": sorted(per_obj.keys()),
            "frame_count": len(frame_indices),
            "prompt_to_obj_ids": {
                text_to_label.get(t, t): [
                    int(oid) for oid, p_label in
                    session.obj_id_to_prompt_id.items()
                    if session.prompts.get(p_label) == t
                ]
                for t in text_to_label
            },
        }
        return detections, tracks, masks_out, sam3_meta


# ---------------------------------------------------------------------------
# Concept builder (POS basket + generic catch-alls)
# ---------------------------------------------------------------------------

# Generic catch-alls used in addition to POS-derived concepts. Tuning
# pass v2 — narrowed to physical food-container phrasings. Broad
# "product"/"package" terms over-fire on POS-station hardware in
# real SCO footage and create dozens of false extras, so they are
# OFF by default. Operators that need a true catch-all can opt
# in via ``include_broad_generics=True``.
HOT_FOOD_GENERIC_CONCEPTS: tuple[tuple[str, str], ...] = (
    ("sco_generic_food_container",    "food container"),
    ("sco_generic_takeaway_container","takeaway container"),
    ("sco_generic_plastic_food_box",  "plastic food box"),
)

# Opt-in noisy catch-alls (off by default for SCO hot-food mode).
BROAD_GENERIC_CONCEPTS: tuple[tuple[str, str], ...] = (
    ("sco_generic_products",  "product"),
    ("sco_generic_packaging", "package"),
    ("sco_generic_bag",       "shopping bag"),
)

# Phrasings tuned for SAM-3's "verb + object" recall pattern. The
# SKU translator's cleaned-up output ("biriyani hot food",
# "curry hot food") returned zero hits on real SCO footage, so for
# SCO hot-food mode we wrap each POS line in a container-aware
# phrase. POS line description is matched against these patterns
# via cheap substring lookup; anything that doesn't match falls
# back to the SKU translator output (the v1 behaviour).
HOT_FOOD_POS_PHRASING: tuple[tuple[str, str], ...] = (
    ("biriyani", "takeaway food container with rice"),
    ("rice",     "takeaway food container with rice"),
    ("curry",    "takeaway food container with curry"),
    ("noodle",   "takeaway food container with noodles"),
    ("pasta",    "takeaway food container with pasta"),
    ("soup",     "takeaway soup container"),
)


def _hot_food_phrasing_for(description: str) -> Optional[str]:
    if not description:
        return None
    low = description.lower()
    for needle, phrase in HOT_FOOD_POS_PHRASING:
        if needle in low:
            return phrase
    return None


def build_concepts_from_pos(pos_event,
                            *,
                            generic_concepts: Optional[Iterable[
                                tuple[str, str]]] = None,
                            mode: str = "hot_food",
                            include_broad_generics: bool = False,
                            ) -> list[Sam3Concept]:
    """Build a SAM-3 concept list from a PosEvent's basket items.

    For each POS line: prefer a mode-specific visual phrasing (e.g.
    ``"takeaway food container with rice"`` for hot-food cases). If no
    phrasing matches, fall back to the deterministic SKU-translator
    output. Each line is attached under a ``sco_item_NNN`` label so
    downstream grouping can resolve back to the basket entry.

    Generic catch-alls are appended so SAM 3 can surface
    extra-candidate identities. ``mode="hot_food"`` (default) uses
    only food-container phrasings; broad ``product``/``package``
    terms are OFF by default because they over-fire on POS hardware.
    Pass ``include_broad_generics=True`` to add them back.
    """
    from .sku_translator import build_falcon_categories_from_pos
    cats = build_falcon_categories_from_pos(pos_event)
    out: list[Sam3Concept] = []

    # POS-derived concepts, with hot-food container phrasing applied
    # when the line description matches a known food keyword.
    pos_items = []
    try:
        items = ((pos_event.raw_payload or {}).get("items")
                 if pos_event is not None else None) or []
    except Exception:
        items = []
    for label, fallback_text in cats.items():
        if label == DEFAULT_GENERIC_KEY:
            continue
        # Map sco_item_NNN -> POS item index
        idx = None
        try:
            idx = int(label.split("_")[-1])
        except (ValueError, IndexError):
            idx = None
        desc = ""
        if isinstance(idx, int) and 0 <= idx < len(items):
            it = items[idx]
            if isinstance(it, dict):
                desc = (it.get("description") or it.get("name")
                        or it.get("item_description") or "")
        text = (_hot_food_phrasing_for(desc) if mode == "hot_food"
                else None) or fallback_text
        out.append(Sam3Concept(label=label, text=text))
        pos_items.append((label, text))

    # Generic catch-alls.
    if generic_concepts is None:
        generic_concepts = (HOT_FOOD_GENERIC_CONCEPTS if mode == "hot_food"
                            else (HOT_FOOD_GENERIC_CONCEPTS
                                  + BROAD_GENERIC_CONCEPTS))
    for label, text in generic_concepts:
        out.append(Sam3Concept(label=label, text=text))
    if include_broad_generics:
        # Append broad catch-alls only if explicitly enabled.
        for label, text in BROAD_GENERIC_CONCEPTS:
            if not any(c.label == label for c in out):
                out.append(Sam3Concept(label=label, text=text))
    return out


# Back-compat shim for callers that imported the v1 constant name.
DEFAULT_GENERIC_CONCEPTS = HOT_FOOD_GENERIC_CONCEPTS


# ---------------------------------------------------------------------------
# Frame decoding helper
# ---------------------------------------------------------------------------

def _decode_video(window_path: str,
                  *,
                  window_start_ts: Optional[datetime],
                  fps_hint: int = 25,
                  max_frames: Optional[int] = None,
                  roi_crop_xyxy: Optional[tuple[int, int, int, int]] = None,
                  ) -> tuple[list, list[datetime], list[int]]:
    """Return ([PIL.Image, ...], [datetime, ...], [int frame_idx, ...]).

    SAM-3's video API takes a list of frames; we hand it the same
    sampled frames the rest of the pipeline operates on so timings
    line up with the episode selector and the evidence package.
    """
    from PIL import Image
    import cv2  # type: ignore

    cap = cv2.VideoCapture(window_path)
    if not cap.isOpened():
        raise RuntimeError("cv2 cannot open window")
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or float(fps_hint)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return [], [], []

    # Sample at ~1 fps to keep the SAM-3 propagation memory bounded.
    # The grouper / VLM downstream don't need every video frame; what
    # matters is stable identities across the window.
    stride = max(1, int(round(actual_fps / 1.0)))
    indices = list(range(0, total, stride))
    if max_frames is not None and len(indices) > max_frames:
        # Uniform downsample to the cap.
        keep = max_frames
        step = max(1, len(indices) // keep)
        indices = indices[::step][:keep]

    base_ts = window_start_ts or datetime.utcfromtimestamp(0)

    frames: list = []
    times: list[datetime] = []
    out_indices: list[int] = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            continue
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if roi_crop_xyxy is not None:
            x1, y1, x2, y2 = (int(v) for v in roi_crop_xyxy)
            h, w = rgb.shape[:2]
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(x1, min(x2, w))
            y2 = max(y1, min(y2, h))
            if x2 > x1 and y2 > y1:
                rgb = rgb[y1:y2, x1:x2]
        frames.append(Image.fromarray(rgb))
        times.append(base_ts + timedelta(seconds=i / max(actual_fps, 1.0)))
        out_indices.append(i)
    cap.release()
    return frames, times, out_indices
