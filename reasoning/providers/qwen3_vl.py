"""Qwen3-VL provider with two strictly local-only backends.

``provider=vllm_openai`` (default) ships frames as base64 image blocks
to a vLLM OpenAI-compatible HTTP server running on this same DGX (the
endpoint is locked to ``http://127.0.0.1:8000/v1`` unless the operator
opts into looser hosts/ports). No HF Transformers / bitsandbytes touch
this process under that backend.

``provider=local_transformers`` is the in-process rollback path. It
loads ``Qwen3VLMoeForConditionalGeneration`` + ``AutoProcessor`` from a
repo-local snapshot resolved via ``app.config.resolve_model_path`` and
runs generation directly. ``HF_HUB_OFFLINE`` and ``TRANSFORMERS_OFFLINE``
are set at module import so a misconfigured snapshot raises locally
instead of reaching out for a download.

Both backends honor the same contract:

* Frames are decoded to RGB PIL Images, optionally downscaled, then
  emitted via the backend's native frame channel (HTTP base64 for vLLM
  or processor image part for local_transformers).
* The provider never invents track IDs, frame IDs, or timestamps.
* Transport / OOM / 5xx / envelope errors return a structured
  ``VLMResult(error=...)`` so the ChainProvider falls back to Gemma.
* A *valid* HTTP response whose model text is unparseable / non-dict is
  coerced to a low-confidence parsed dict so the deterministic decision
  policy routes the case to REVIEW (and never crashes the worker).
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
from urllib.parse import urlparse

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

# Public re-export so ``app.case_runner`` (and tests) can compose the
# canonical structured-JSON request with an ROI-guidance preamble
# without forking the prompt body. Drift would silently change the
# review-safe contract for every camera, so we keep one source of truth.
DEFAULT_USER_PROMPT: str = _DEFAULT_USER


_DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8000
_LOCALHOST_ALIASES = {"localhost"}


class Qwen3VLProvider(VLMProvider):
    name = "qwen3_vl"

    def __init__(self,
                 *,
                 model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
                 enabled: bool = False,
                 provider: str = "vllm_openai",
                 base_url: str = _DEFAULT_BASE_URL,
                 served_model_name: str = "qwen3_vl",
                 checkpoint_label: str = "",
                 precision: str = "",
                 timeout_sec: float = 120.0,
                 health_timeout_sec: float = 3.0,
                 request_retries: int = 1,
                 allow_localhost_alias: bool = False,
                 allow_port_override: bool = False,
                 max_frame_long_edge: int = 1280,
                 max_frame_pixels: int = 1_048_576,
                 max_tokens: int = 512,
                 # ---- local_transformers rollback kwargs ----
                 local_path: str = "",
                 max_new_tokens: int = 512,
                 temperature: float = 0.1,
                 device: str = "cuda",
                 dtype: str = "bfloat16",
                 load_in_4bit: bool = False,
                 **extra: Any):
        super().__init__(model_name=model_name, enabled=enabled, **extra)
        self.provider_backend = str(provider).strip() or "vllm_openai"
        # vLLM HTTP backend knobs.
        self.served_model_name = str(served_model_name)
        self.checkpoint_label = str(checkpoint_label)
        self.precision = str(precision)
        self.timeout_sec = float(timeout_sec)
        self.health_timeout_sec = float(health_timeout_sec)
        self.request_retries = max(1, int(request_retries))
        self.allow_localhost_alias = bool(allow_localhost_alias)
        self.allow_port_override = bool(allow_port_override)
        self.max_frame_long_edge = max(64, int(max_frame_long_edge))
        self.max_frame_pixels = max(64 * 64, int(max_frame_pixels))
        self.max_tokens = max(16, int(max_tokens))
        # Validate base_url eagerly so a misconfigured config fails fast.
        self._base_url_err: Optional[str] = None
        try:
            self.base_url = _validate_base_url(
                base_url,
                allow_localhost_alias=self.allow_localhost_alias,
                allow_port_override=self.allow_port_override,
            )
        except ValueError as exc:
            self.base_url = ""
            self._base_url_err = str(exc)
        # local_transformers rollback knobs.
        self.local_path = local_path
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.device = device
        self.dtype = dtype
        self.load_in_4bit = bool(load_in_4bit)
        # Local-transformers runtime state.
        self._load_lock = threading.Lock()
        self._gen_lock = threading.Lock()
        self._processor: Any = None
        self._model: Any = None
        self._load_err: Optional[str] = None
        # vLLM session — lazily constructed so tests can monkeypatch.
        self._session: Any = None

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
        if self.provider_backend == "vllm_openai":
            return self._analyze_vllm(manifest)
        if self.provider_backend == "local_transformers":
            return self._analyze_local_transformers(manifest)
        return VLMResult(
            provider=self.name,
            model_name=self.model_name,
            error=(f"unknown qwen3_vl provider backend "
                   f"{self.provider_backend!r}; expected "
                   f"'vllm_openai' or 'local_transformers'"),
        )

    def health(self) -> ProviderHealth:
        if not self.enabled:
            return ProviderHealth(self.name, False, "disabled (by config)")
        if self.provider_backend == "vllm_openai":
            if self._base_url_err:
                return ProviderHealth(self.name, False,
                                      f"bad base_url: {self._base_url_err}")
            healthy, detail = self._vllm_health()
            return ProviderHealth(self.name, healthy, detail)
        # local_transformers
        if not self.has_local_weights():
            return ProviderHealth(self.name, False,
                                  f"local_path missing: {self.local_path!r}")
        if self._load_err:
            return ProviderHealth(self.name, False,
                                  f"load_err: {self._load_err}")
        return ProviderHealth(self.name, True, "local snapshot present")

    # ------------------------------------------------------------------
    # Internals — vLLM HTTP backend
    # ------------------------------------------------------------------

    def _provider_metadata(self) -> dict:
        return {
            "backend": self.provider_backend,
            "base_url": self.base_url if self.provider_backend == "vllm_openai"
                        else None,
            "served_model_name": self.served_model_name
                if self.provider_backend == "vllm_openai" else None,
            "checkpoint_label": self.checkpoint_label or None,
            "precision": self.precision or None,
            "local_path": (self.local_path or None)
                if self.provider_backend == "local_transformers" else None,
            "max_tokens": self.max_tokens,
            "max_frame_long_edge": self.max_frame_long_edge,
            "max_frame_pixels": self.max_frame_pixels,
        }

    def _ensure_session(self) -> Any:
        if self._session is not None:
            return self._session
        import requests
        # trust_env=False ignores HTTP_PROXY/HTTPS_PROXY/NO_PROXY and any
        # ~/.netrc. The endpoint is local-only by validation, but proxy
        # env from an operator's shell must NOT be allowed to redirect
        # traffic off-box.
        s = requests.Session()
        s.trust_env = False
        self._session = s
        return s

    def _vllm_health(self) -> tuple[bool, str]:
        if self._base_url_err:
            return False, f"bad base_url: {self._base_url_err}"
        try:
            sess = self._ensure_session()
            r = sess.get(f"{self.base_url}/models",
                         timeout=self.health_timeout_sec,
                         allow_redirects=False)
        except Exception as exc:
            return False, f"vllm unreachable: {type(exc).__name__}: {exc}"
        if r.status_code != 200:
            return False, f"vllm /models http {r.status_code}"
        try:
            body = r.json()
        except Exception as exc:
            return False, f"vllm /models non-json: {exc}"
        ids = {str(it.get("id"))
               for it in (body.get("data") or [])
               if isinstance(it, dict)}
        if self.served_model_name not in ids:
            return False, (f"vllm /models missing served model "
                           f"{self.served_model_name!r}; got {sorted(ids)}")
        return True, f"vllm /models contains {self.served_model_name!r}"

    def _analyze_vllm(self, manifest: EvidenceManifest) -> VLMResult:
        if self._base_url_err:
            return VLMResult(provider=self.name,
                             model_name=self.model_name,
                             error=f"bad base_url: {self._base_url_err}")
        frames_pil = _decode_frames(manifest.frames or [])
        if not frames_pil:
            return VLMResult(provider=self.name,
                             model_name=self.model_name,
                             error="no frames in manifest")
        frames_pil = [self._downscale_frame(im) for im in frames_pil]

        sys_prompt = (manifest.system_prompt or "").strip() or _DEFAULT_SYSTEM
        usr_prompt = (manifest.user_prompt or "").strip() or _DEFAULT_USER

        user_content: list[dict] = []
        for img in frames_pil:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        user_content.append({"type": "text", "text": usr_prompt})

        payload = {
            "model": self.served_model_name,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": int(self.max_tokens),
            "temperature": float(self.temperature),
            "stream": False,
        }

        t0 = time.time()
        last_err: Optional[str] = None
        usage: dict = {}
        text: str = ""
        for attempt in range(self.request_retries):
            try:
                sess = self._ensure_session()
                r = sess.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=self.timeout_sec,
                    allow_redirects=False,
                )
            except Exception as exc:
                last_err = (f"vllm transport {type(exc).__name__}: {exc}"
                            f" (attempt {attempt + 1}/{self.request_retries})")
                log.warning("qwen3_vl vllm transport error: %s", last_err)
                continue
            if r.status_code != 200:
                # Read the first chunk of the body for diagnostics, then
                # break — non-200 is a Gemma-fallback signal.
                body_preview = ""
                try:
                    body_preview = r.text[:300]
                except Exception:
                    pass
                last_err = (f"vllm http {r.status_code}: "
                            f"{body_preview!r}")
                break
            try:
                body = r.json()
            except Exception as exc:
                last_err = f"vllm response not JSON: {exc}"
                break
            choices = body.get("choices") if isinstance(body, dict) else None
            if not isinstance(choices, list) or not choices:
                last_err = (f"vllm envelope missing choices: "
                            f"keys={list(body) if isinstance(body, dict) else type(body).__name__}")
                break
            first = choices[0] if isinstance(choices[0], dict) else None
            msg = first.get("message") if isinstance(first, dict) else None
            text = (msg or {}).get("content") if isinstance(msg, dict) else None
            if not isinstance(text, str):
                last_err = "vllm envelope missing message.content string"
                break
            usage = body.get("usage") if isinstance(body, dict) else {}
            if not isinstance(usage, dict):
                usage = {}
            last_err = None
            break

        latency_ms = int((time.time() - t0) * 1000)
        if last_err is not None:
            return VLMResult(
                provider=self.name,
                model_name=self.model_name,
                latency_ms=latency_ms,
                error=last_err,
            )

        parsed = _parse_json(text)
        # Defence in depth: if the parser ever returns a non-dict, coerce
        # to a REVIEW-routing dict instead of letting it propagate.
        if not isinstance(parsed, dict):
            parsed = {"narrative": str(text)[:400], "confidence": "low"}
        parsed["_model_run"] = {
            "provider_metadata": self._provider_metadata(),
            "model_snapshot": self.checkpoint_label or self.served_model_name,
            "usage": usage,
        }
        return VLMResult(
            provider=self.name,
            model_name=self.model_name,
            raw_text=text,
            parsed=parsed,
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    # Frame downscale (shared)
    # ------------------------------------------------------------------

    def _downscale_frame(self, img):
        """Cap long edge AND total pixels before base64. Returns RGB PIL."""
        try:
            from PIL import Image  # noqa: F401  (import to satisfy lint)
        except Exception:
            return img
        w, h = img.size
        if w <= 0 or h <= 0:
            return img
        scale = 1.0
        long_edge = max(w, h)
        if long_edge > self.max_frame_long_edge:
            scale = min(scale, self.max_frame_long_edge / float(long_edge))
        pixels = w * h
        if pixels > self.max_frame_pixels:
            scale = min(scale,
                        (self.max_frame_pixels / float(pixels)) ** 0.5)
        if scale >= 0.999:
            return img if img.mode == "RGB" else img.convert("RGB")
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = img.resize((new_w, new_h))
        return resized if resized.mode == "RGB" else resized.convert("RGB")

    # ------------------------------------------------------------------
    # Internals — local_transformers rollback
    # ------------------------------------------------------------------

    def _analyze_local_transformers(self, manifest: EvidenceManifest) -> VLMResult:
        if not self.has_local_weights():
            return VLMResult(
                provider=self.name,
                model_name=self.model_name,
                error=(f"local weights missing at {self.local_path!r}; "
                       "Qwen3-VL local_transformers will not auto-download"),
            )
        if not manifest.frames:
            return VLMResult(provider=self.name,
                             model_name=self.model_name,
                             error="no frames in manifest")
        try:
            self._load()
        except Exception as exc:
            self._load_err = f"{type(exc).__name__}: {exc}"
            log.exception("qwen3_vl local_transformers load failed")
            return VLMResult(provider=self.name,
                             model_name=self.model_name,
                             error=f"load failed: {self._load_err}")

        t0 = time.time()
        try:
            text, usage = self._generate_via_transformers(manifest)
        except Exception as exc:
            log.exception("qwen3_vl local_transformers generation failed")
            return VLMResult(
                provider=self.name,
                model_name=self.model_name,
                latency_ms=int((time.time() - t0) * 1000),
                error=f"generate failed: {type(exc).__name__}: {exc}",
            )

        parsed = _parse_json(text)
        if not isinstance(parsed, dict):
            parsed = {"narrative": str(text)[:400], "confidence": "low"}
        parsed["_model_run"] = {
            "provider_metadata": self._provider_metadata(),
            "model_snapshot": self.local_path or self.model_name,
            "usage": usage,
        }
        return VLMResult(
            provider=self.name,
            model_name=self.model_name,
            raw_text=text,
            parsed=parsed,
            latency_ms=int((time.time() - t0) * 1000),
        )

    def unload(self) -> None:
        """Drop GPU/CPU references so the next provider in the chain
        can load. Safe to call repeatedly."""
        with self._load_lock:
            self._model = None
            self._processor = None
            try:
                import gc
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

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
            device_map = "auto" if self.device.startswith("cuda") else self.device
            load_kwargs: dict = dict(
                torch_dtype=dtype,
                device_map=device_map,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
            if self.load_in_4bit:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=dtype,
                )
                log.info("qwen3_vl: loading in 4-bit (nf4)")
            self._model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                self.local_path, **load_kwargs)
            self._model.eval()
            log.info("qwen3_vl loaded on %s dtype=%s",
                     next(self._model.parameters()).device,
                     next(self._model.parameters()).dtype)

    def _build_transformers_chat(self,
                                 sys_prompt: str,
                                 usr_prompt: str,
                                 frames) -> list:
        """Build the OpenAI-shape chat for ``apply_chat_template`` using
        concrete lists (not list-comprehensions over a generator).

        Older transformers builds iterate the message ``content`` list
        twice when expanding image placeholders; passing a generator
        works the first time and is empty the second time. Forcing
        concrete lists is the proven-safe shape on this snapshot.
        """
        sys_parts: list[dict] = [{"type": "text", "text": sys_prompt}]
        user_parts: list[dict] = []
        for img in frames:
            user_parts.append({"type": "image", "image": img})
        user_parts.append({"type": "text", "text": usr_prompt})
        return [
            {"role": "system", "content": sys_parts},
            {"role": "user", "content": user_parts},
        ]

    def _generate_via_transformers(self, manifest: EvidenceManifest
                                   ) -> tuple[str, dict]:
        import torch

        frames = [self._downscale_frame(im)
                  for im in _decode_frames(manifest.frames)]
        if not frames:
            raise RuntimeError("no decodable frames in manifest")

        sys_prompt = (manifest.system_prompt or "").strip() or _DEFAULT_SYSTEM
        usr_prompt = (manifest.user_prompt or "").strip() or _DEFAULT_USER

        chat = self._build_transformers_chat(sys_prompt, usr_prompt, frames)

        inputs = self._processor.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
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


def _validate_base_url(url: str, *,
                       allow_localhost_alias: bool,
                       allow_port_override: bool) -> str:
    """Strict offline-only base_url validator.

    Accepts ONLY ``http://127.0.0.1:8000/v1`` by default. Toggles:
      * ``allow_localhost_alias=True`` accepts ``localhost`` too.
      * ``allow_port_override=True`` accepts any port (still 127.0.0.1).
    Rejects: missing scheme, https, query string, fragment, missing
    ``/v1`` path, non-loopback host.
    """
    if not url or not isinstance(url, str):
        raise ValueError("base_url empty")
    if url.endswith("/"):
        url = url.rstrip("/")
    parts = urlparse(url)
    if parts.scheme != "http":
        raise ValueError(f"scheme must be http, got {parts.scheme!r}")
    if parts.query:
        raise ValueError("base_url must not contain a query string")
    if parts.fragment:
        raise ValueError("base_url must not contain a fragment")
    host = (parts.hostname or "").lower()
    if host != _DEFAULT_HOST:
        if not (allow_localhost_alias and host in _LOCALHOST_ALIASES):
            raise ValueError(
                f"host must be {_DEFAULT_HOST!r} "
                f"(allow_localhost_alias={allow_localhost_alias}); got {host!r}"
            )
    port = parts.port
    if port is None:
        raise ValueError("base_url must include an explicit port")
    if port != _DEFAULT_PORT and not allow_port_override:
        raise ValueError(
            f"port must be {_DEFAULT_PORT} "
            f"(allow_port_override=False); got {port}"
        )
    path = parts.path or ""
    if path != "/v1":
        raise ValueError(f"base_url path must be exactly '/v1', got {path!r}")
    return f"{parts.scheme}://{host}:{port}/v1"


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
        loaded = json.loads(blob)
    except json.JSONDecodeError:
        try:
            loaded = json.loads(blob.replace("'", '"'))
        except json.JSONDecodeError:
            return {"narrative": text[:400], "confidence": "low"}
    if not isinstance(loaded, dict):
        return {"narrative": text[:400], "confidence": "low"}
    return loaded


def _factory(**config: Any) -> Qwen3VLProvider:
    return Qwen3VLProvider(**config)


register_provider(Qwen3VLProvider.name, _factory)
