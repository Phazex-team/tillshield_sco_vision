"""Qwen3-VL provider — wired but disabled by default.

We do not download Qwen3-VL: weights must already exist at the local
snapshot path recorded in ``config.yaml``. ``analyze_evidence`` raises
if the provider is enabled without that path being present.

Implementation note: the actual inference path will be filled in during
the MLOps phase (benchmark + calibration). For now the provider only
verifies the local snapshot is reachable; this is enough to ship the
abstraction safely without ever pulling weights at startup.
"""
from __future__ import annotations

import os
from typing import Any

from .base import (
    EvidenceManifest,
    ProviderHealth,
    VLMProvider,
    VLMResult,
    register_provider,
)


class Qwen3VLProvider(VLMProvider):
    name = "qwen3_vl"

    def __init__(self,
                 *,
                 model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
                 enabled: bool = False,
                 local_path: str = "",
                 **extra: Any):
        super().__init__(model_name=model_name, enabled=enabled, **extra)
        self.local_path = local_path

    def has_local_weights(self) -> bool:
        if not self.local_path:
            return False
        return os.path.isdir(self.local_path)

    def analyze_evidence(self, manifest: EvidenceManifest) -> VLMResult:
        if not self.enabled:
            return VLMResult(
                provider=self.name,
                model_name=self.model_name,
                error="provider disabled",
            )
        if not self.has_local_weights():
            return VLMResult(
                provider=self.name,
                model_name=self.model_name,
                error=(
                    f"local weights missing at {self.local_path!r}; "
                    "Qwen3-VL provider will not auto-download"
                ),
            )
        # Real inference path is intentionally deferred to MLOps phase.
        # When activated, this will load the local snapshot via
        # transformers AutoModelForCausalLM (or a dedicated bridge) and
        # mirror the GemmaProvider contract.
        return VLMResult(
            provider=self.name,
            model_name=self.model_name,
            error="qwen3_vl inference not yet implemented; benchmark gate pending",
        )

    def health(self) -> ProviderHealth:
        if not self.enabled:
            return ProviderHealth(self.name, False, "disabled (by config)")
        if not self.has_local_weights():
            return ProviderHealth(self.name, False,
                                  f"local_path missing: {self.local_path!r}")
        return ProviderHealth(self.name, True, "local snapshot present")


def _factory(**config: Any) -> Qwen3VLProvider:
    return Qwen3VLProvider(**config)


register_provider(Qwen3VLProvider.name, _factory)
