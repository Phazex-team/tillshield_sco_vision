"""Chain provider: try Qwen first, fall back to Gemma.

The chain delegates to a list of inner providers in order. The first
provider that returns a result without ``error`` wins. If the primary
provider raises during ``analyze_evidence``, the chain catches and tries
the next.

The decision policy (`reasoning.decision_policy.decide`) is the final
authority; the chain only chooses **which** provider's description goes
into the evidence summary.
"""
from __future__ import annotations

import logging
from typing import Any

from .base import (
    EvidenceManifest,
    ProviderHealth,
    VLMProvider,
    VLMResult,
)


log = logging.getLogger(__name__)


class ChainProvider(VLMProvider):
    name = "chain"

    def __init__(self, *, providers: list[VLMProvider]):
        if not providers:
            raise ValueError("ChainProvider requires at least one provider")
        # The chain itself is always "enabled"; its members may not be.
        super().__init__(model_name="chain", enabled=True)
        self.providers = providers

    def analyze_evidence(self, manifest: EvidenceManifest) -> VLMResult:
        last_result: VLMResult | None = None
        attempts: list[str] = []
        for p in self.providers:
            if not p.enabled:
                attempts.append(f"{p.name}=disabled")
                continue
            try:
                r = p.analyze_evidence(manifest)
            except Exception as exc:
                log.exception("provider %s raised", p.name)
                attempts.append(f"{p.name}=raised:{type(exc).__name__}")
                last_result = VLMResult(
                    provider=p.name, model_name=p.model_name,
                    error=f"raised: {type(exc).__name__}: {exc}",
                )
                continue
            last_result = r
            if r.error is None:
                # Annotate so callers can see chain fallback decisions.
                attempts.append(f"{p.name}=ok")
                r.parsed = dict(r.parsed or {})
                r.parsed.setdefault("_chain_attempts", attempts)
                return r
            attempts.append(f"{p.name}=err:{r.error[:60]}")

        # All members failed. Return the last result with chain notes.
        if last_result is None:
            return VLMResult(
                provider="chain", model_name="chain",
                error="no enabled providers in chain",
            )
        last_result.parsed = dict(last_result.parsed or {})
        last_result.parsed["_chain_attempts"] = attempts
        return last_result

    def health(self) -> ProviderHealth:
        healths = [p.health() for p in self.providers]
        any_healthy = any(h.healthy for h in healths)
        detail = "; ".join(f"{h.provider}={'ok' if h.healthy else h.detail}"
                           for h in healths)
        return ProviderHealth(self.name, any_healthy, detail)


def build_active_provider(cfg: Any) -> VLMProvider:
    """Build the active provider from an ``app.config.AppConfig``.

    Rules:
      * If ``models.qwen3_vl.enabled`` AND its repo-local weights resolve,
        Qwen is the primary. Gemma is the fallback (chain). Decision
        policy stays final authority.
      * Otherwise Gemma is the only provider.

    The function never raises; if Gemma config is also missing the
    returned chain will produce error results and the decision policy
    will degrade to REVIEW upstream.
    """
    from app.config import resolve_model_path
    from . import get_provider

    providers: list[VLMProvider] = []

    qwen_cfg = cfg.models.get("qwen3_vl") if cfg else None
    if qwen_cfg and qwen_cfg.enabled:
        # Resolve repo-local snapshot first; fall through if missing.
        try:
            local_path = resolve_model_path(qwen_cfg)
        except Exception as exc:
            log.warning("qwen3_vl path resolution failed: %s", exc)
            local_path = None
        if local_path:
            providers.append(get_provider(
                "qwen3_vl",
                model_name=qwen_cfg.name,
                enabled=True,
                local_path=local_path,
                **{k: v for k, v in qwen_cfg.extra.items()
                   if k not in ("local_path",)},
            ))
        else:
            log.warning("qwen3_vl enabled but no local snapshot found; "
                        "skipping in chain")

    gemma_cfg = cfg.models.get("gemma") if cfg else None
    if gemma_cfg and gemma_cfg.enabled:
        providers.append(get_provider(
            "gemma",
            model_name=gemma_cfg.name,
            enabled=True,
            **{k: v for k, v in gemma_cfg.extra.items()
               if k not in ("local_path",)},
        ))

    if not providers:
        # Return a disabled Gemma so callers get a structured error.
        return get_provider("gemma", model_name="gemma:none", enabled=False)
    if len(providers) == 1:
        return providers[0]
    return ChainProvider(providers=providers)
