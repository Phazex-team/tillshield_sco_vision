"""Gemma 4 BF16 provider — wraps the existing GemmaVideoReasoner client.

The current MVP uses ``gemma_reasoner.GemmaVideoReasoner`` to talk to
``transformers_server.py`` on :8001. This provider preserves that wire
path so we can ship the new abstraction without changing inference.

Heavy imports are deferred to ``_client()`` so importing the package
(and the test suite) never touches torch/PIL/requests.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from .base import (
    EvidenceManifest,
    ProviderHealth,
    VLMProvider,
    VLMResult,
    register_provider,
)


class GemmaProvider(VLMProvider):
    name = "gemma"

    def __init__(self,
                 *,
                 model_name: str = "google/gemma-4-26B-A4B-it",
                 enabled: bool = True,
                 vllm_url: str = "",
                 max_tokens: int = 512,
                 temperature: float = 0.1,
                 request_timeout_sec: float = 300.0,
                 request_retries: int = 3,
                 request_retry_backoff_sec: float = 5.0,
                 **extra: Any):
        super().__init__(model_name=model_name, enabled=enabled, **extra)
        self.vllm_url = vllm_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.request_timeout_sec = request_timeout_sec
        self.request_retries = request_retries
        self.request_retry_backoff_sec = request_retry_backoff_sec
        self._client_cache: Optional[Any] = None

    def unload(self) -> None:
        """Drop the HTTP client reference. The Gemma server itself
        keeps weights warm; the local provider object stays cheap."""
        self._client_cache = None

    def _client(self):
        if self._client_cache is not None:
            return self._client_cache
        from gemma_reasoner import GemmaVideoReasoner
        self._client_cache = GemmaVideoReasoner(
            model_name=self.model_name,
            vllm_url=self.vllm_url,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            request_timeout_sec=self.request_timeout_sec,
            request_retries=self.request_retries,
            request_retry_backoff_sec=self.request_retry_backoff_sec,
        )
        return self._client_cache

    def analyze_evidence(self, manifest: EvidenceManifest) -> VLMResult:
        if not self.enabled:
            return VLMResult(
                provider=self.name,
                model_name=self.model_name,
                error="provider disabled",
            )
        frames = _decode_frames(manifest.frames)
        if not frames:
            return VLMResult(
                provider=self.name,
                model_name=self.model_name,
                error="no frames in manifest",
            )
        start_objects, action_objects = _summarize_tracks(manifest.tracks)
        # SCO Phase 5 fix: when the active prompt is the SCO basket-match
        # template, Gemma's response JSON has SCO-shaped keys
        # (basket_match / matched / missing / extras / video_usable /
        # confidence / narrative). The legacy refund parser in
        # gemma_reasoner._parse_json would silently project that onto
        # refund fields (handover_occurred / item_count / ...) and drop
        # the SCO keys, so downstream the SCO schema parser sees no
        # basket_match and falls back to uncertain/low — turning a
        # legitimate Gemma answer into REVIEW with sco_low_confidence.
        # Flip to schema-passthrough so the dict is returned verbatim.
        prompt_version = ((manifest.metadata or {}).get("prompt_version")
                          if isinstance(manifest.metadata, dict) else None)
        schema_passthrough = (prompt_version == "sco_basket_match_v1")
        t0 = time.time()
        try:
            parsed = self._client().reason(
                frames,
                start_objects=start_objects,
                action_objects=action_objects,
                system_prompt=manifest.system_prompt or None,
                user_prompt=manifest.user_prompt or None,
                camera_id=manifest.camera_id,
                session_id=manifest.case_id,
                schema_passthrough=schema_passthrough,
            )
        except Exception as exc:
            return VLMResult(
                provider=self.name,
                model_name=self.model_name,
                latency_ms=int((time.time() - t0) * 1000),
                error=f"reason() failed: {exc}",
            )
        return VLMResult(
            provider=self.name,
            model_name=self.model_name,
            raw_text=parsed.get("narrative", ""),
            parsed=parsed,
            latency_ms=int(parsed.get("_latency_ms")
                           or (time.time() - t0) * 1000),
        )

    def health(self) -> ProviderHealth:
        if not self.enabled:
            return ProviderHealth(self.name, False, "disabled")
        try:
            ok = self._client().health()
        except Exception as exc:
            return ProviderHealth(self.name, False, f"health probe failed: {exc}")
        return ProviderHealth(self.name, bool(ok),
                              "transformers_server :/health 200" if ok
                              else "transformers_server unreachable")


def _decode_frames(frames: list[dict]) -> list:
    """Turn EvidenceManifest.frames into PIL images for the legacy reasoner.

    Frame entries look like ``{"frame_id": "...", "ts": "...",
    "image_url": "data:image/jpeg;base64,..."}``.
    """
    import base64
    import io
    import re

    from PIL import Image

    data_url_re = re.compile(
        r"^data:image/(?:jpeg|jpg|png|webp);base64,(?P<b64>.+)$",
        re.DOTALL,
    )
    out = []
    for f in frames or []:
        url = (f.get("image_url") or "").strip()
        m = data_url_re.match(url)
        if not m:
            continue
        raw = base64.b64decode(m.group("b64"))
        img = Image.open(io.BytesIO(raw))
        out.append(img.convert("RGB") if img.mode != "RGB" else img)
    return out


def _summarize_tracks(tracks: list[dict]) -> tuple[str, str]:
    """Map structured tracks to the legacy (start_objects, action_objects).

    The MVP prompt template asks for two comma-separated strings. Pull
    them from tracks tagged ``role=start`` vs ``role=action``. Fall back
    to empty strings so the prompt template still renders.
    """
    start_labels: list[str] = []
    action_labels: list[str] = []
    for t in tracks or []:
        label = (t.get("label") or "").strip()
        if not label:
            continue
        role = (t.get("role") or "").strip().lower()
        if role == "start":
            start_labels.append(label)
        elif role == "action":
            action_labels.append(label)
        else:
            action_labels.append(label)
    return ", ".join(start_labels), ", ".join(action_labels)


def _factory(**config: Any) -> GemmaProvider:
    return GemmaProvider(**config)


register_provider(GemmaProvider.name, _factory)
