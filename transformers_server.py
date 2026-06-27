"""Minimal OpenAI-compatible HTTP server for Gemma 4 BF16 (vision-language).

Stands in for vLLM on :8001 when the NVFP4 path can't be made to work on
this hardware. The SCO Vision app's existing ``gemma_reasoner.py``
HTTP client talks to this without modification.

Single in-process model: ``transformers.Gemma4ForConditionalGeneration``
loaded in bfloat16 on CUDA. Sequential request handling (no batching) —
fine for the ~one-session-per-minute workload of this app.

Endpoints:
  GET  /health
  POST /v1/chat/completions

The chat-completions endpoint accepts the same payload shape vLLM uses,
including multimodal user content (a list with ``{type:"image_url", ...}``
and ``{type:"text", ...}`` parts), and ignores any keys it doesn't
understand (``chat_template_kwargs``, ``mm_processor_kwargs``, etc.).
"""
from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import re
import threading
import time
import uuid
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from PIL import Image
from transformers import AutoProcessor, Gemma4ForConditionalGeneration

log = logging.getLogger("tx_server")

# Defensive server-side ceiling on image frames per request. The app already
# bounds frames (manifest_max_frames), but a single oversized request can pin
# the KV-cache high-water mark for the life of the process, so we hard-cap
# here too. Generous (well above the normal ~64) so legitimate requests are
# never truncated; only runaway inputs are clipped (with a warning).
MAX_IMAGES_PER_REQUEST = int(os.environ.get("TX_MAX_IMAGES", "96"))

MODEL_NAME = os.environ.get(
    "GEMMA_MODEL_NAME", "google/gemma-4-26B-A4B-it"
)
PORT = int(os.environ.get("VLLM_PORT", "8001"))


# ---- model load ----------------------------------------------------------

_LOAD_LOCK = threading.Lock()
_LOAD_DONE = threading.Event()
_GEN_LOCK = threading.Lock()  # serialize generate() calls
_MODEL: Any = None
_PROCESSOR: Any = None


def _load_model() -> None:
    global _MODEL, _PROCESSOR
    with _LOAD_LOCK:
        if _LOAD_DONE.is_set():
            return
        log.info("loading %s in bfloat16 on cuda (local_files_only=True)...",
                 MODEL_NAME)
        # Hard local-only: this server must never reach out to the
        # HuggingFace Hub at runtime. If the snapshot is missing the
        # operator must fetch it explicitly (out of band) before
        # restarting — see scripts/inspect_models.py.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        _PROCESSOR = AutoProcessor.from_pretrained(
            MODEL_NAME, local_files_only=True
        )
        _MODEL = Gemma4ForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        _MODEL.eval()
        log.info(
            "loaded %s on %s (dtype=%s)",
            MODEL_NAME,
            next(_MODEL.parameters()).device,
            next(_MODEL.parameters()).dtype,
        )
        _LOAD_DONE.set()


# ---- request <-> processor adaptation -----------------------------------

_DATA_URL_RE = re.compile(
    r"^data:image/(?P<mime>jpeg|jpg|png|webp);base64,(?P<b64>.+)$",
    re.DOTALL,
)


def _decode_image_url(url: str) -> Image.Image:
    m = _DATA_URL_RE.match((url or "").strip())
    if not m:
        raise ValueError(
            "image_url must be a data:image/{jpeg,png,webp};base64,... URL"
        )
    raw = base64.b64decode(m.group("b64"))
    img = Image.open(io.BytesIO(raw))
    return img.convert("RGB") if img.mode != "RGB" else img


def _build_chat_messages(messages: list[dict]) -> list[dict]:
    """Convert OpenAI-style messages into the structure
    ``Gemma4Processor.apply_chat_template`` expects.

    OpenAI text-only:
        {"role": "user", "content": "hello"}
    OpenAI multimodal:
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:..."}},
            {"type": "text",      "text": "describe"}
        ]}
    Gemma4Processor wants:
        {"role": "user", "content": [
            {"type": "image", "image": <PIL.Image>},
            {"type": "text",  "text": "..."}
        ]}
    """
    out = []
    for m in messages or []:
        role = str(m.get("role", "")).strip() or "user"
        content = m.get("content")
        parts: list[dict] = []
        if isinstance(content, str):
            parts.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for p in content:
                if not isinstance(p, dict):
                    continue
                t = p.get("type")
                if t == "text":
                    parts.append({"type": "text", "text": p.get("text", "")})
                elif t == "image_url":
                    url = (p.get("image_url") or {}).get("url", "")
                    parts.append({"type": "image", "image": _decode_image_url(url)})
                elif t == "image":
                    img = p.get("image")
                    if isinstance(img, Image.Image):
                        parts.append({"type": "image", "image": img})
                # silently ignore other part types (audio, video, ...)
        elif content is not None:
            parts.append({"type": "text", "text": str(content)})
        out.append({"role": role, "content": parts})
    # Defensive cap: clip total image parts across the conversation so one
    # runaway request can't balloon (and permanently pin) the KV cache.
    total_imgs = sum(1 for m in out for p in m["content"]
                     if p.get("type") == "image")
    if total_imgs > MAX_IMAGES_PER_REQUEST:
        log.warning("request had %d images; capping to %d (defensive)",
                    total_imgs, MAX_IMAGES_PER_REQUEST)
        kept = 0
        for m in out:
            new_parts = []
            for p in m["content"]:
                if p.get("type") == "image":
                    if kept >= MAX_IMAGES_PER_REQUEST:
                        continue
                    kept += 1
                new_parts.append(p)
            m["content"] = new_parts
    return out


def _generate(messages: list[dict],
              max_tokens: int,
              temperature: float) -> tuple[str, dict]:
    """Run one generation. Returns (text, usage_dict)."""
    chat = _build_chat_messages(messages)

    # Tokenize + handle images via the processor's chat template path.
    inputs = _PROCESSOR.apply_chat_template(
        chat,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(_MODEL.device, dtype=torch.bfloat16)

    # Image-token shapes are int — keep them as-is. Cast only floating
    # tensors to bfloat16; integer tensors (input_ids, attention_mask)
    # were promoted by the .to() above and need to come back to int64.
    for k in ("input_ids", "attention_mask", "token_type_ids"):
        if k in inputs and inputs[k].dtype != torch.long:
            inputs[k] = inputs[k].long()

    prompt_len = inputs["input_ids"].shape[-1]

    do_sample = temperature is not None and temperature > 0
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_tokens),
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)

    out_ids = None
    try:
        with torch.inference_mode():
            with _GEN_LOCK:
                out_ids = _MODEL.generate(**inputs, **gen_kwargs)

        new_tokens = out_ids[0][prompt_len:]
        text = _PROCESSOR.decode(new_tokens, skip_special_tokens=True)
        completion_len = int(new_tokens.shape[-1])
        return text, {
            "prompt_tokens": int(prompt_len),
            "completion_tokens": completion_len,
            "total_tokens": int(prompt_len + completion_len),
        }
    finally:
        # Release this request's working memory. Multimodal prompts (many
        # image frames) allocate a KV cache sized to the full prompt; without
        # this, PyTorch's caching allocator keeps the high-water mark of the
        # LARGEST request ever served reserved forever, so the server ratchets
        # up to ~87G and never shrinks. Freeing per-request keeps steady-state
        # at weights + current request. Cost is a few ms of cache re-alloc.
        try:
            del inputs
            if out_ids is not None:
                del out_ids
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            log.debug("post-generate cleanup skipped", exc_info=True)


# ---- HTTP app -----------------------------------------------------------

app = FastAPI(title="Transformers BF16 Gemma4 server")


@app.get("/health")
def health():
    if _LOAD_DONE.is_set():
        return {"status": "ok", "model": MODEL_NAME}
    raise HTTPException(503, "model loading")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body: dict = await request.json()
    if not _LOAD_DONE.is_set():
        raise HTTPException(503, "model still loading")
    messages = body.get("messages") or []
    max_tokens = int(body.get("max_tokens") or 512)
    temperature = body.get("temperature")
    if temperature is None:
        temperature = 0.0
    try:
        text, usage = _generate(messages, max_tokens, float(temperature))
    except Exception:
        log.exception("generation failed")
        raise HTTPException(500, "generation failed")
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:24],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }


# ---- entry --------------------------------------------------------------

def main():
    global MODEL_NAME
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--model", default=MODEL_NAME)
    args = ap.parse_args()

    MODEL_NAME = args.model

    # Load synchronously before opening the port so /health=200 means
    # generation is actually possible.
    _load_model()

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info",
                access_log=False)


if __name__ == "__main__":
    main()
