"""Gemma 4 26B-A4B MoE VLM reasoner — vLLM HTTP client (v3).

v3 swaps the in-process ``transformers``/``torch`` Gemma loader for an
HTTP client that talks to a vLLM OpenAI-compatible server (default
``http://localhost:8001``). The model is no longer loaded into the
SCO Vision process; the server holds the weights once and serves
every camera's clips concurrently.

Public interface is unchanged:
    GemmaVideoReasoner(model_name, max_tokens=..., temperature=...,
                       max_video_frames=..., video_fps=...,
                       vllm_url=..., request_timeout_sec=...,
                       request_retries=..., request_retry_backoff_sec=...)
        .reason(frames, *, start_objects, action_objects,
                system_prompt=None, user_prompt=None,
                token_budget=None, classifier=None) -> dict
        .quick_describe(frames, max_frames=3, max_new_tokens=80) -> str

JSON output schema is byte-for-byte the same as v2 so the CSV/dashboard
do not need migrations.

Image encoding: frames are sent as inline ``data:image/jpeg;base64`` URLs
inside an OpenAI-style ``messages`` payload. The vLLM Gemma 4 server
handles the multi-image -> video preproc on its side.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time
from typing import Optional, Sequence

import requests
from PIL import Image

log = logging.getLogger(__name__)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*|\s*```\s*$", re.DOTALL)
_THINK_RE = re.compile(r"<\|?think\|?>.*?<\|?/think\|?>", re.DOTALL)


class GemmaVideoReasoner:
    """vLLM-backed Gemma 4 video reasoner."""

    def __init__(self,
                 model_name: str,
                 max_tokens: int = 768,
                 temperature: float = 0.1,
                 # Safety ceiling on frames sent to the model. Raised from
                 # 60 so the ~1 fps manifest budget (≈150 for a 2.5-min
                 # window) is not silently clipped. The real budget is set
                 # by case_runner's manifest_max_frames.
                 max_video_frames: int = 300,
                 video_fps: int = 1,
                 vllm_url: str = "",
                 request_timeout_sec: float = 120.0,
                 request_retries: int = 3,
                 request_retry_backoff_sec: float = 5.0,
                 # Compat: v2 accepted these and silently ignored most.
                 device_map: str = "",
                 torch_dtype: str = "",
                 **_ignored):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_video_frames = max(1, int(max_video_frames))
        self.video_fps = max(1, int(video_fps))

        url = (vllm_url or os.environ.get("VLLM_URL")
               or f"http://localhost:{os.environ.get('VLLM_PORT', '8001')}")
        self.base_url = url.rstrip("/")
        self.chat_url = f"{self.base_url}/v1/chat/completions"
        self.health_url = f"{self.base_url}/health"
        self.timeout = float(request_timeout_sec)
        self.retries = max(1, int(request_retries))
        self.backoff = max(0.1, float(request_retry_backoff_sec))

        log.info("Gemma reasoner -> vLLM @ %s  model=%s  timeout=%.0fs",
                 self.base_url, model_name, self.timeout)

    # --------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------

    def reason(self,
               frames: Sequence[Image.Image],
               *,
               start_objects: str,
               action_objects: str,
               system_prompt: str | None = None,
               user_prompt: str | None = None,
               token_budget: Optional[int] = None,
               classifier: Optional[str] = None,
               enable_thinking: Optional[bool] = None,
               max_frames: Optional[int] = None,
               camera_id: Optional[str] = None,
               session_id: Optional[str] = None) -> dict:
        if not frames:
            return _fallback("no frames provided")

        cap = int(max_frames) if max_frames else self.max_video_frames
        sampled = self._sample(frames, cap)
        num_frames = len(sampled)

        from classifiers import CLASSIFIERS, coerce_token_budget
        sys_tpl = (system_prompt or "").strip() or _legacy_system_prompt(classifier)
        usr_tpl = (user_prompt or "").strip() or _legacy_user_prompt(classifier)
        fmt = {"start_objects": start_objects, "action_objects": action_objects}
        try:
            system = sys_tpl.format(**fmt)
        except (KeyError, IndexError):
            system = (sys_tpl
                      .replace("{start_objects}", start_objects)
                      .replace("{action_objects}", action_objects))

        # Build OpenAI-style content list: each frame is one image_url part.
        user_content = self._build_image_content(sampled)
        user_content.append({"type": "text", "text": usr_tpl})

        # Thinking mode emits chain-of-thought tokens before the JSON
        # answer. Quadruple the budget so JSON has room left after
        # thinking consumes its share. Leave non-thinking unchanged.
        effective_max_tokens = (int(self.max_tokens) * 4
                                if enable_thinking else int(self.max_tokens))
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user_content},
            ],
            "max_tokens": effective_max_tokens,
            "temperature": float(self.temperature),
            "stream": False,
        }
        # Per-request image-token budget is a vLLM extra arg. The Gemma 4
        # vLLM model accepts it as ``mm_processor_kwargs.image_token_budget``.
        if token_budget is not None:
            tb = coerce_token_budget(token_budget, 560)
            payload["mm_processor_kwargs"] = {"image_token_budget": int(tb)}
        if enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}

        log.info("[gemma.reason] cam=%s sid=%s classifier=%s thinking=%s "
                 "frames=%d budget=%s tokens=%d",
                 camera_id or "-", session_id or "-", classifier or "-",
                 bool(enable_thinking), num_frames, token_budget,
                 effective_max_tokens)
        t0 = time.time()
        text = self._post_chat(payload)
        latency_ms = int((time.time() - t0) * 1000)
        parsed = _parse_json(text)
        parsed["_num_frames"] = num_frames
        parsed["_latency_ms"] = latency_ms

        try:
            from tracer import trace_gemma
            trace_gemma(camera_id=camera_id, session_id=session_id,
                        classifier=classifier, thinking=bool(enable_thinking),
                        prompt_tokens=None, completion_tokens=None,
                        thinking_tokens=None,
                        raw_response=text, parsed_result=parsed,
                        latency_ms=latency_ms, num_frames=num_frames,
                        token_budget=token_budget)
        except Exception:
            log.exception("trace_gemma hook failed (non-fatal)")
        return parsed

    def quick_describe(self,
                       frames: Sequence[Image.Image],
                       max_frames: int = 3,
                       max_new_tokens: int = 80) -> str:
        """Cheap probe used by the SessionDispatcher to decide whether two
        adjacent sessions are the same subject. Returns one short sentence."""
        if not frames:
            return ""
        sampled = self._sample(frames, max_frames)

        prompt_text = (
            "Describe the subject in this short video clip in ONE short "
            "sentence — clothing colours and rough position only (no JSON, "
            "no preamble)."
        )
        user_content = self._build_image_content(sampled)
        user_content.append({"type": "text", "text": prompt_text})

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": user_content}],
            "max_tokens": int(max_new_tokens),
            "temperature": 0.0,
            "stream": False,
        }
        # Quick describe always uses the lowest token budget for speed.
        payload["mm_processor_kwargs"] = {"image_token_budget": 70}
        # Quick describe is a cheap probe — never use thinking.
        payload["chat_template_kwargs"] = {"enable_thinking": False}

        try:
            text = self._post_chat(payload).strip()
        except Exception:
            log.exception("quick_describe failed")
            return ""
        text = _FENCE_RE.sub("", text).strip()
        if text.startswith("{"):
            try:
                data = json.loads(text)
                for key in ("customer_description", "description", "narrative"):
                    v = data.get(key)
                    if isinstance(v, str) and v:
                        return v.strip()
            except Exception:
                pass
        return text

    # --------------------------------------------------------------
    # Internals
    # --------------------------------------------------------------

    def _sample(self, frames: Sequence[Image.Image], cap: int) -> list[Image.Image]:
        n = min(len(frames), cap)
        stride = max(1, len(frames) // n) if n else 1
        return [
            (f if f.mode == "RGB" else f.convert("RGB"))
            for f in list(frames)[::stride][:n]
        ]

    def _build_image_content(self, frames: Sequence[Image.Image]) -> list[dict]:
        out = []
        for img in frames:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            out.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        return out

    def _post_chat(self, payload: dict) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                resp = requests.post(self.chat_url, json=payload,
                                     timeout=self.timeout)
                if not resp.ok:
                    log.error("vLLM 400 body: %s", resp.text[:2000])
                resp.raise_for_status()
                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    raise RuntimeError(f"vLLM returned no choices: {data}")
                msg = choices[0].get("message") or {}
                content = msg.get("content")
                if isinstance(content, list):
                    # Some vLLM builds return a list of parts.
                    content = "".join(p.get("text", "") for p in content
                                      if isinstance(p, dict))
                return content or ""
            except Exception as e:
                last_err = e
                log.warning("vLLM request failed (attempt %d/%d): %s",
                            attempt, self.retries, e)
                if attempt < self.retries:
                    time.sleep(self.backoff)
        raise RuntimeError(f"vLLM request failed after {self.retries} retries: "
                           f"{last_err}")

    def health(self) -> bool:
        try:
            r = requests.get(self.health_url, timeout=5)
            return r.status_code == 200
        except Exception:
            return False


# Backwards-compatible alias used by callers that still import ``GemmaReasoner``.
GemmaReasoner = GemmaVideoReasoner


# --------------------------------------------------------------------
# Legacy prompt fallbacks
# --------------------------------------------------------------------
# If a caller passes ``system_prompt=""`` we still need *something* to send
# to Gemma. Look up the matching classifier default; if no classifier was
# provided, fall back to the fraud one (matches v2 behaviour).

def _legacy_system_prompt(classifier: Optional[str]) -> str:
    from classifiers import get_classifier
    return get_classifier(classifier or "fraud")["gemma_system"]


def _legacy_user_prompt(classifier: Optional[str]) -> str:
    from classifiers import get_classifier
    return get_classifier(classifier or "fraud")["gemma_user"]


# --------------------------------------------------------------------
# JSON parsing (carried verbatim from v2 so output schema is identical)
# --------------------------------------------------------------------

def _parse_json(text: str) -> dict:
    if not text:
        return _fallback(text)
    cleaned = _THINK_RE.sub("", text)
    cleaned = _FENCE_RE.sub("", cleaned).strip()
    match = _JSON_RE.search(cleaned)
    if not match:
        return _fallback(text)
    blob = match.group(0)
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        try:
            data = json.loads(blob.replace("'", '"'))
        except json.JSONDecodeError:
            return _fallback(text)

    handover = bool(data.get("handover_occurred", False))
    items_handed = [str(x) for x in (data.get("items_handed_over") or []) if x]
    try:
        item_count = int(data.get("item_count") or 0)
    except (TypeError, ValueError):
        item_count = 0
    if not handover:
        item_count = 0
        items_handed = []
    customer_desc = str(data.get("customer_description", ""))[:400]
    narrative = str(data.get("narrative", ""))[:2000]
    confidence = str(data.get("confidence", "low")).lower()
    flag = bool(data.get("flag_for_review", not handover and bool(items_handed)))

    people = [{
        "id": 1,
        "description": customer_desc,
        "items_presented": items_handed,
        "presented": handover,
    }] if customer_desc or items_handed else []

    return {
        "handover_occurred": handover,
        "item_count": max(0, item_count),
        "items_handed_over": items_handed,
        "customer_description": customer_desc,
        "narrative": narrative,
        "confidence": confidence,
        "flag_for_review": flag,
        # Back-compat for old call sites
        "people": people,
        "item_presented": handover,
        "objects_detected": items_handed,
    }


def _fallback(text: Optional[str]) -> dict:
    msg = f"unparsable model output: {(text or '')[:180]}"
    return {
        "handover_occurred": False,
        "item_count": 0,
        "items_handed_over": [],
        "customer_description": "",
        "narrative": msg,
        "confidence": "low",
        "flag_for_review": True,
        "people": [],
        "item_presented": False,
        "objects_detected": [],
    }
