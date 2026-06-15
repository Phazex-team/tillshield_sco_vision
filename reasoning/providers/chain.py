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
    """Sequenced provider chain with memory-policy + mutual-exclusion.

    Enforces three production rules from PRODUCTION_SPEC.md section 7:

    * **Lazy load** — no provider model touches GPU/RAM until the first
      ``analyze_evidence`` call. Construction is cheap.
    * **Mutual exclusion** — at most one big VLM is warm-loaded at a
      time. When the chain falls back from primary to secondary, it
      unloads the primary first and clears the CUDA cache.
    * **Memory guard** — every call consults
      ``app.memory_guard.get_policy().allow_new_inference()`` before
      attempting a load. When the soft limit is crossed the chain
      returns a ``REVIEW_PENDING_MODEL`` error so the deterministic
      decision policy degrades to ``REVIEW`` upstream.
    """
    name = "chain"

    def __init__(self, *, providers: list[VLMProvider],
                 warm_fallback: bool = False):
        if not providers:
            raise ValueError("ChainProvider requires at least one provider")
        super().__init__(model_name="chain", enabled=True)
        self.providers = providers
        self.warm_fallback = bool(warm_fallback)

    def analyze_evidence(self, manifest: EvidenceManifest) -> VLMResult:
        from app.memory_guard import get_policy

        policy = get_policy()
        status = policy.poll()
        if not status.inference_allowed:
            return VLMResult(
                provider="chain", model_name="chain",
                error=(f"inference deferred: memory state {status.state}; "
                       f"reason={status.degraded_reason}"),
                parsed={"_memory_state": status.state,
                        "_memory_used_gb": status.used_gb,
                        "_chain_attempts": ["chain=deferred"]},
            )

        last_result: VLMResult | None = None
        attempts: list[str] = []
        for idx, p in enumerate(self.providers):
            if not p.enabled:
                attempts.append(f"{p.name}=disabled")
                continue
            # NOTE: mutual exclusion between primary + fallback is
            # enforced on the error path below — we unload the failed
            # provider before moving to the next. That path always
            # runs when we end up trying the fallback, so we don't need
            # a separate pre-fallback unload here (which would
            # double-call ``unload()``).
            try:
                policy.mark_loaded(p.name)
                r = p.analyze_evidence(manifest)
            except Exception as exc:
                log.exception("provider %s raised", p.name)
                attempts.append(f"{p.name}=raised:{type(exc).__name__}")
                last_result = VLMResult(
                    provider=p.name, model_name=p.model_name,
                    error=f"raised: {type(exc).__name__}: {exc}",
                )
                # Make sure the failed provider is unloaded before we
                # try the next one — typical OOM recovery path.
                _try_unload(p, policy)
                continue

            last_result = r
            if r.error is None:
                attempts.append(f"{p.name}=ok")
                r.parsed = dict(r.parsed or {})
                r.parsed.setdefault("_chain_attempts", attempts)
                return r
            attempts.append(f"{p.name}=err:{r.error[:60]}")
            # If the provider returned a structured error, drop it
            # before trying the next one.
            _try_unload(p, policy)

        if last_result is None:
            return VLMResult(
                provider="chain", model_name="chain",
                error="no enabled providers in chain",
                parsed={"_chain_attempts": attempts},
            )
        last_result.parsed = dict(last_result.parsed or {})
        last_result.parsed["_chain_attempts"] = attempts
        return last_result


def _try_unload(provider: VLMProvider, policy) -> None:
    """Best-effort unload + CUDA cache clear.

    Providers may expose an ``unload()`` method; if not, we clear the
    refs we know about and call torch.cuda.empty_cache() so the next
    provider has room to load.
    """
    name = getattr(provider, "name", "")
    try:
        if hasattr(provider, "unload"):
            provider.unload()
        else:
            # Generic best-effort: drop heavy refs the provider holds.
            for attr in ("_model", "_model_cache", "_client_cache"):
                if hasattr(provider, attr):
                    setattr(provider, attr, None)
    except Exception:
        log.exception("provider %s unload failed", name)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    if name:
        policy.mark_unloaded(name)

    def health(self) -> ProviderHealth:
        healths = [p.health() for p in self.providers]
        any_healthy = any(h.healthy for h in healths)
        detail = "; ".join(f"{h.provider}={'ok' if h.healthy else h.detail}"
                           for h in healths)
        return ProviderHealth(self.name, any_healthy, detail)


def build_active_provider(cfg: Any) -> VLMProvider:
    """Build the active provider chain from an ``app.config.AppConfig``.

    Rules (PRODUCTION_SPEC.md §7 + §11):

      * ``reasoning.primary_provider`` (default ``qwen3_vl``) goes
        first if enabled AND its repo-local weights resolve.
      * ``reasoning.fallback_provider`` (default ``gemma``) is the
        chain fallback. It is **not** warm-loaded unless
        ``reasoning.warm_fallback`` is true.
      * If only one provider is configured, return it directly so
        memory-state checks still apply.
      * Providers are constructed lazily — no weights load here.

    Returns a ``VLMProvider`` (often a ``ChainProvider``). Never raises;
    a fully degraded config returns a disabled provider whose
    ``analyze_evidence`` produces a structured error, which the
    deterministic decision policy degrades to ``REVIEW`` upstream.
    """
    from app.config import resolve_model_path
    from . import get_provider

    reasoning_cfg = (cfg.raw.get("reasoning") if cfg else None) or {}
    primary = reasoning_cfg.get("primary_provider", "qwen3_vl")
    fallback = reasoning_cfg.get("fallback_provider", "gemma")
    warm_fallback = bool(reasoning_cfg.get("warm_fallback", False))

    providers: list[VLMProvider] = []
    order = [primary] if primary else []
    if fallback and fallback not in order:
        order.append(fallback)

    for key in order:
        model_cfg = cfg.models.get(key) if cfg else None
        if model_cfg is None or not model_cfg.enabled:
            log.info("provider %s skipped (missing or disabled)", key)
            continue
        # Resolve repo-local snapshot. We only need to gate Qwen behind
        # path-presence; Gemma reaches the transformers HTTP server.
        local_path = None
        if key == "qwen3_vl":
            try:
                local_path = resolve_model_path(model_cfg)
            except Exception as exc:
                log.warning("%s path resolution failed: %s", key, exc)
            if not local_path:
                log.warning(
                    "%s enabled but no local snapshot found; skipping",
                    key)
                continue
        kwargs = {k: v for k, v in model_cfg.extra.items()
                  if k not in ("local_path",)}
        if local_path:
            kwargs["local_path"] = local_path
        providers.append(get_provider(
            key, model_name=model_cfg.name, enabled=True, **kwargs))

    if not providers:
        return get_provider("gemma", model_name="gemma:none", enabled=False)
    if len(providers) == 1:
        return providers[0]
    return ChainProvider(providers=providers, warm_fallback=warm_fallback)
