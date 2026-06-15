"""Qwen3-VL provider — real, local-only, no downloads.

Loads ``Qwen3VLMoeForConditionalGeneration`` directly from a repo-local
snapshot resolved via ``app.config.resolve_model_path``. The HuggingFace
Hub is gagged at module import: ``HF_HUB_OFFLINE=1`` and
``TRANSFORMERS_OFFLINE=1`` are set before any transformers symbol is
touched, so a misconfigured snapshot raises locally instead of reaching
out for a download.

This provider is the primary verifier when
``config.yaml -> models.qwen3_vl.enabled: true``. The chain wrapper in
``reasoning.providers.select_active_provider`` falls back to Gemma if
this provider raises during load OR returns an error result during
inference. The decision policy remains the final authority on outcomes.

Implementation notes (verified against the local snapshot at
``models/hf/Qwen/Qwen3-VL-30B-A3B-Instruct/<snapshot>/``):

* ``config.json`` declares ``architectures=[Qwen3VLMoeForConditionalGeneration]``
  and ``model_type=qwen3_vl_moe``. We use the explicit class, not Auto*,
  to avoid relying on registry order in older transformers builds.
* ``AutoProcessor.from_pretrained(local_path, local_files_only=True)``
  returns ``Qwen3VLProcessor`` for this snapshot.
* The processor's ``apply_chat_template`` accepts the OpenAI-style
  multimodal content shape (image + text parts).
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import threading
import time
from typing import Any, Optional

# Hard-gag the hub before any HF import. Module-scope so callers cannot
# accidentally enable network by importing us in the wrong order.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from .base import (
    EvidenceManifest,
    ProviderHealth,
    VLMProvider,
    VLMResult,
    register_provider,
)


log = logging.getLogger(__name__)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*|\s*```\s*$", re.DOTALL)
_THINK_RE = re.compile(r"<\|?think\|?>.*?<\|?/think\|?>", re.DOTALL)

_DATA_URL_RE = re.compile(
    r"^data:image/(?P<mime>jpeg|jpg|png|webp);base64,(?P<b64>.+)$",
    re.DOTALL,
)

_DEFAULT_SYSTEM = (
    "You are an evidence describer reviewing a short retail return / "
    "refund counter clip. You do NOT decide anything. A separate "
    "deterministic policy decides the outcome from the structured "
    "signals you report. Never use the words fraud, fraudulent, "
    "theft, or suspect. If unsure of any field lower confidence to "
    "low. Output ONLY the JSON object the user turn asks for."
)

_DEFAULT_USER = (
    "Describe what the camera shows of the return-counter interaction. "
    "Report exactly these fields:\n"
    "{\n"
    '  "handover_occurred": true|false,\n'
    '  "physical_item_presented": true|false,\n'
    '  "receipt_visible": true|false,\n'
    '  "items_observed": ["brief", "item", "descriptions"],\n'
    '  "narrative": "one short sentence describing what is visible",\n'
    '  "confidence": "high"|"medium"|"low",\n'
    '  "obstructed": true|false,\n'
    '  "camera_view_clear": true|false,\n'
    '  "limitations": ["blind spots, occlusions, ambiguities"]\n'
    "}\n"
    "Output ONLY the JSON object. No preamble, no markdown."
)


class Qwen3VLProvider(VLMProvider):
    name = "qwen3_vl"

    def __init__(self,
                 *,
                 model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
                 enabled: bool = False,
                 local_path: str = "",
                 max_new_tokens: int = 512,
                 temperature: float = 0.1,
                 device: str = "cuda",
                 dtype: str = "bfloat16",
                 **extra: Any):
        super().__init__(model_name=model_name, enabled=enabled, **extra)
        self.local_path = local_path
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.device = device
        self.dtype = dtype
        self._load_lock = threading.Lock()
        self._gen_lock = threading.Lock()
        self._processor: Any = None
        self._model: Any = None
        self._load_err: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has_local_weights(self) -> bool:
        return bool(self.local_path) and os.path.isdir(self.local_path)

    def analyze_evidence(self, manifest: EvidenceManifest) -> VLMResult:
        if not self.enabled:
            return VLMResult(provider=self.name,
                             model_name=self.model_name,
                             error="provider disabled")
        if not self.has_local_weights():
            return VLMResult(
                provider=self.name,
                model_name=self.model_name,
                error=(f"local weights missing at {self.local_path!r}; "
                       "Qwen3-VL provider will not auto-download"),
            )
        if not manifest.frames:
            return VLMResult(provider=self.name,
                             model_name=self.model_name,
                             error="no frames in manifest")

        try:
            self._load()
        except Exception as exc:
            self._load_err = f"{type(exc).__name__}: {exc}"
            log.exception("qwen3_vl load failed")
            return VLMResult(provider=self.name,
                             model_name=self.model_name,
                             error=f"load failed: {self._load_err}")

        t0 = time.time()
        try:
            text, usage = self._generate(manifest)
        except Exception as exc:
            log.exception("qwen3_vl generation failed")
            return VLMResult(
                provider=self.name,
                model_name=self.model_name,
                latency_ms=int((time.time() - t0) * 1000),
                error=f"generate failed: {type(exc).__name__}: {exc}",
            )

        parsed = _parse_json(text)
        return VLMResult(
            provider=self.name,
            model_name=self.model_name,
            raw_text=text,
            parsed=parsed,
            latency_ms=int((time.time() - t0) * 1000),
        )

    def health(self) -> ProviderHealth:
        if not self.enabled:
            return ProviderHealth(self.name, False, "disabled (by config)")
        if not self.has_local_weights():
            return ProviderHealth(self.name, False,
                                  f"local_path missing: {self.local_path!r}")
        if self._load_err:
            return ProviderHealth(self.name, False,
                                  f"load_err: {self._load_err}")
        return ProviderHealth(self.name, True, "local snapshot present")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        with self._load_lock:
            if self._model is not None and self._processor is not None:
                return
            # Belt-and-suspenders before importing transformers.
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

            import torch
            from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

            dtype = getattr(torch, self.dtype, torch.bfloat16)
            log.info("qwen3_vl loading from local snapshot %s "
                     "(dtype=%s device=%s)",
                     self.local_path, self.dtype, self.device)
            self._processor = AutoProcessor.from_pretrained(
                self.local_path,
                local_files_only=True,
            )
            self._model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                self.local_path,
                torch_dtype=dtype,
                device_map=self.device,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
            self._model.eval()
            log.info("qwen3_vl loaded on %s dtype=%s",
                     next(self._model.parameters()).device,
                     next(self._model.parameters()).dtype)

    def _generate(self, manifest: EvidenceManifest) -> tuple[str, dict]:
        import torch
        from PIL import Image

        # Build OpenAI-style content list: each frame is one image part.
        frames = _decode_frames(manifest.frames)
        if not frames:
            raise RuntimeError("no decodable frames in manifest")

        sys_prompt = (manifest.system_prompt or "").strip() or _DEFAULT_SYSTEM
        usr_prompt = (manifest.user_prompt or "").strip() or _DEFAULT_USER

        chat = [
            {"role": "system", "content": [
                {"type": "text", "text": sys_prompt},
            ]},
            {"role": "user", "content": (
                [{"type": "image", "image": img} for img in frames]
                + [{"type": "text", "text": usr_prompt}]
            )},
        ]

        inputs = self._processor.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        # Move tensors; cast floats only.
        dtype = getattr(torch, self.dtype, torch.bfloat16)
        moved = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.dtype.is_floating_point:
                    moved[k] = v.to(self._model.device, dtype=dtype)
                else:
                    moved[k] = v.to(self._model.device)
            else:
                moved[k] = v

        prompt_len = moved["input_ids"].shape[-1]
        do_sample = self.temperature > 0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = self.temperature

        with torch.inference_mode():
            with self._gen_lock:
                out_ids = self._model.generate(**moved, **gen_kwargs)
        new_tokens = out_ids[0][prompt_len:]
        text = self._processor.decode(new_tokens, skip_special_tokens=True)
        usage = {
            "prompt_tokens": int(prompt_len),
            "completion_tokens": int(new_tokens.shape[-1]),
            "total_tokens": int(prompt_len + new_tokens.shape[-1]),
        }
        return text, usage


def _decode_frames(frames: list[dict]):
    from PIL import Image
    out = []
    for f in frames or []:
        url = (f.get("image_url") or "").strip()
        m = _DATA_URL_RE.match(url)
        if not m:
            continue
        raw = base64.b64decode(m.group("b64"))
        img = Image.open(io.BytesIO(raw))
        out.append(img.convert("RGB") if img.mode != "RGB" else img)
    return out


def _parse_json(text: str) -> dict:
    if not text:
        return {"narrative": "(empty response)", "confidence": "low"}
    cleaned = _THINK_RE.sub("", text)
    cleaned = _FENCE_RE.sub("", cleaned).strip()
    m = _JSON_RE.search(cleaned)
    if not m:
        return {"narrative": text[:400], "confidence": "low"}
    blob = m.group(0)
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        try:
            return json.loads(blob.replace("'", '"'))
        except json.JSONDecodeError:
            return {"narrative": text[:400], "confidence": "low"}


def _factory(**config: Any) -> Qwen3VLProvider:
    return Qwen3VLProvider(**config)


register_provider(Qwen3VLProvider.name, _factory)
