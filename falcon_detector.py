"""Falcon Perception (PyTorch/CUDA) wrapper.

Mirrors the upstream demo/perception_single.py engine_type="batch" path:
  load_and_prepare_model(backend='torch') → BatchInferenceEngine
  per-frame: build_prompt_for_task → process_batch_and_generate → engine.generate
Bbox format: normalized center-form {x:cx, y:cy, w, h} in [0,1].
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from falcon_perception import (
    PERCEPTION_300M_MODEL_ID,
    PERCEPTION_MODEL_ID,
    build_prompt_for_task,
    load_and_prepare_model,
    setup_torch_config,
)
from falcon_perception.batch_inference import (
    BatchInferenceEngine,
    process_batch_and_generate,
)

log = logging.getLogger(__name__)

_MODEL_ALIASES = {
    "tiiuae/falcon-perception": PERCEPTION_MODEL_ID,
    "tiiuae/Falcon-Perception": PERCEPTION_MODEL_ID,
    "tiiuae/falcon-perception-300m": PERCEPTION_300M_MODEL_ID,
    "tiiuae/Falcon-Perception-300M": PERCEPTION_300M_MODEL_ID,
}

_DTYPE_ALIASES = {
    "float16": "bfloat16",  # fp16 not in the torch backend's Literal; use bf16 on CUDA
    "fp16": "bfloat16",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "float32": "float32",
    "fp32": "float32",
}


@dataclass
class Detection:
    label: str
    bbox_norm: tuple  # (cx, cy, w, h), each in [0,1]
    bbox_px: tuple    # (x1, y1, x2, y2) in pixels

    def summary(self) -> str:
        x1, y1, x2, y2 = self.bbox_px
        cx, cy, w, h = self.bbox_norm
        return (f"{self.label}[px:{int(x1)},{int(y1)},{int(x2)},{int(y2)}"
                f" norm:cx={cx:.2f},cy={cy:.2f},w={w:.2f},h={h:.2f}]")


class FalconDetector:
    DEFAULT_QUERY = (
        "hand, item, product, bag, box, package, jacket, clothing, "
        "bottle, receipt, phone, merchandise"
    )

    def __init__(self, model_name: str = PERCEPTION_MODEL_ID,
                 dtype: str = "bfloat16",
                 task: str = "detection",
                 query: str | None = None,
                 max_new_tokens: int = 200,
                 min_dim: int = 256,
                 max_dim: int = 1024,
                 device: str | None = None,
                 compile: bool = False):
        setup_torch_config()

        model_id = _MODEL_ALIASES.get(model_name, model_name)
        resolved_dtype = _DTYPE_ALIASES.get(str(dtype).lower(), "bfloat16")
        log.info("loading Falcon Perception (torch/cuda): %s dtype=%s",
                 model_id, resolved_dtype)

        # A repo-local bundle path must go through hf_local_dir, not
        # hf_model_id (the HF downloader rejects a filesystem path as a
        # repo id). Detect a local directory and route accordingly.
        import os as _os
        _is_local_dir = bool(model_id) and _os.path.isdir(model_id)
        model, tokenizer, model_args = load_and_prepare_model(
            hf_model_id=None if _is_local_dir else model_id,
            hf_local_dir=model_id if _is_local_dir else None,
            dtype=resolved_dtype,
            backend="torch",
            device=device,
            compile=compile,
        )
        self.model = model
        self.tokenizer = tokenizer
        self.model_args = model_args
        self.engine = BatchInferenceEngine(model, tokenizer)
        self.device = model.device
        self.task = task if task != "segmentation" or getattr(
            model_args, "do_segmentation", False) else "detection"
        self.query = query or self.DEFAULT_QUERY
        self.max_new_tokens = max_new_tokens
        self.min_dim = min_dim
        self.max_dim = max_dim
        self._stop_token_ids = [tokenizer.eos_token_id]
        eoq = getattr(tokenizer, "end_of_query_token_id", None)
        if eoq is not None:
            self._stop_token_ids.append(eoq)

    @torch.inference_mode()
    def detect(self, image: Image.Image, query: str | None = None
               ) -> tuple[Image.Image, list[Detection]]:
        if image.mode != "RGB":
            image = image.convert("RGB")
        effective_query = (query or "").strip() or self.query
        prompt = build_prompt_for_task(effective_query, self.task)
        batch = process_batch_and_generate(
            self.tokenizer,
            [(image, prompt)],
            max_length=self.model_args.max_seq_len,
            min_dimension=self.min_dim,
            max_dimension=self.max_dim,
        )
        batch = {
            k: (v.to(self.device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        output_tokens, aux_outputs = self.engine.generate(
            tokens=batch["tokens"],
            pos_t=batch["pos_t"],
            pos_hw=batch["pos_hw"],
            pixel_values=batch["pixel_values"],
            pixel_mask=batch["pixel_mask"],
            max_new_tokens=self.max_new_tokens,
            temperature=0.0,
            stop_token_ids=self._stop_token_ids,
            task=self.task,
        )
        aux = aux_outputs[0]
        decoded = self.tokenizer.decode(
            output_tokens[0].detach().cpu().tolist(), skip_special_tokens=False
        )
        labels = _extract_labels(decoded)

        bboxes_norm = _pair_bbox_entries(aux.bboxes_raw)
        dets: list[Detection] = []
        W, H = image.size
        for i, b in enumerate(bboxes_norm):
            cx, cy, w, h = b["x"], b["y"], b["w"], b["h"]
            x1 = max(0.0, (cx - w / 2) * W)
            y1 = max(0.0, (cy - h / 2) * H)
            x2 = min(float(W), (cx + w / 2) * W)
            y2 = min(float(H), (cy + h / 2) * H)
            label = labels[i] if i < len(labels) else "object"
            dets.append(Detection(label=label, bbox_norm=(cx, cy, w, h),
                                  bbox_px=(x1, y1, x2, y2)))
        annotated = _annotate(image, dets)
        return annotated, dets


def _pair_bbox_entries(raw) -> list[dict]:
    """Pair [{x,y}, {h,w}, ...] into [{x,y,h,w}, ...] (upstream convention).

    ``raw`` may contain dicts of scalars or dicts of 0-dim tensors depending
    on backend; coerce tensor values to floats.
    """
    out, cur = [], {}
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        for k, v in entry.items():
            if torch.is_tensor(v):
                v = float(v.detach().cpu().item())
            cur[k] = v
        if all(k in cur for k in ("x", "y", "h", "w")):
            out.append(dict(cur))
            cur = {}
    return out


def _extract_labels(decoded: str) -> list[str]:
    """Best-effort label extraction from the decoded token stream.

    Falcon emits a structured sequence interleaving text labels with bbox
    tokens. We look for short alphabetic runs between bbox coordinate groups.
    If we can't segment cleanly, we return an empty list (caller falls back
    to the generic 'object' label — that's fine since Gemma reasons on the
    image itself, with bbox coords for grounding).
    """
    import re
    # Detection output is a stream of SPECIAL tokens (``<|image_cls|>``,
    # ``<|image_reg_N|>``, ``<|image|>``) with NO real per-box label, so
    # strip special-token markers first; anything left that looks like a
    # token name (contains digits/underscores or image/reg/cls) is noise.
    text = re.sub(r"<\|[^|]*\|>", " ", decoded or "")
    tokens = re.findall(r"[A-Za-z][A-Za-z _\-]{1,40}", text)
    stop = {"image", "detection", "segmentation", "object", "user",
            "assistant", "cls", "reg", "system"}
    cleaned = [t.strip().lower() for t in tokens
               if 2 <= len(t.strip()) <= 30
               and t.strip().lower() not in stop
               and not re.search(r"(image|reg|cls)", t.lower())]
    return cleaned


def _annotate(image: Image.Image, dets: list[Detection]) -> Image.Image:
    out = image.copy()
    drw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for d in dets:
        x1, y1, x2, y2 = d.bbox_px
        drw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
        drw.text((x1 + 2, max(0, y1 - 10)), d.label, fill="lime", font=font)
    return out


def bbox_summary(dets: list[Detection]) -> str:
    if not dets:
        return "no objects detected"
    return "; ".join(d.summary() for d in dets)
