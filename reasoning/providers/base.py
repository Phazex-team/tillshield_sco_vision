"""Provider abstraction for the reasoning layer.

A provider receives a fully-built ``EvidenceManifest`` (structured visual
evidence + frame citations) and returns a ``VLMResult`` containing the
model's narrative + a normalized JSON payload. The provider must never
invent or mutate track IDs, frame IDs, or timestamps — those come from
the perception pipeline upstream.

The deterministic policy layer (``reasoning/decision_policy.py``) wraps
this output. Providers themselves should NOT emit fraud accusations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class EvidenceManifest:
    """Structured visual evidence passed to a VLM provider.

    Frames are encoded as data URLs (``data:image/jpeg;base64,...``) so
    provider implementations can ship them over OpenAI-compatible HTTP
    without re-encoding. ``tracks`` and ``ocr`` are advisory context for
    the model; it must cite frame IDs / track IDs but cannot create them.
    """
    case_id: str
    camera_id: str
    window_start_ts: str  # ISO-8601
    window_end_ts: str    # ISO-8601
    frames: list[dict] = field(default_factory=list)
    tracks: list[dict] = field(default_factory=list)
    ocr: list[dict] = field(default_factory=list)
    pos_event: Optional[dict] = None
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class VLMResult:
    provider: str
    model_name: str
    raw_text: str = ""
    parsed: dict = field(default_factory=dict)
    latency_ms: int = 0
    error: Optional[str] = None
    citations: list[dict] = field(default_factory=list)


@dataclass
class ProviderHealth:
    provider: str
    healthy: bool
    detail: str = ""


class VLMProvider:
    """Base interface every provider implements."""

    name: str = "base"

    def __init__(self, *, model_name: str, enabled: bool = True, **config: Any):
        self.model_name = model_name
        self.enabled = enabled
        self.config = config

    def analyze_evidence(self, manifest: EvidenceManifest) -> VLMResult:
        raise NotImplementedError

    def health(self) -> ProviderHealth:
        return ProviderHealth(provider=self.name,
                              healthy=self.enabled,
                              detail="not implemented")


_REGISTRY: dict[str, Callable[..., VLMProvider]] = {}


def register_provider(name: str,
                      factory: Callable[..., VLMProvider]) -> None:
    _REGISTRY[name] = factory


def list_providers() -> list[str]:
    return sorted(_REGISTRY)


def get_provider(name: str, **config: Any) -> VLMProvider:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown provider {name!r}; registered: {list_providers()}"
        )
    return _REGISTRY[name](**config)
