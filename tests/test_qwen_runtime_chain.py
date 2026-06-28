"""Council tests for Qwen runtime + memory-guard exemption.

Four acceptance lines:

  1. Qwen vLLM backend stays FIRST in the provider chain.
  2. An external HTTP Qwen provider can be called when /v1/models is
     healthy.
  3. memory_guard EMERGENCY state does NOT block a healthy external
     HTTP provider purely because the app process's memory is high.
  4. Gemma fallback still works when the Qwen HTTP call fails (raises
     OR returns a structured error).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _force_emergency_policy(monkeypatch):
    """Install a MemoryPolicy that classifies as EMERGENCY no matter
    what (probe returns total_gb=121, used_gb=120). Use this to test
    that external-HTTP providers are still attempted under the gate."""
    from app.memory_guard import (
        MemoryPolicy, MemoryPolicyConfig, set_policy_for_test,
    )
    cfg = MemoryPolicyConfig(soft_gb=90.0, hard_gb=100.0, emergency_gb=110.0)
    policy = MemoryPolicy(cfg, probe=lambda: (121.0, 120.0))
    set_policy_for_test(policy)
    return policy


def _force_normal_policy(monkeypatch):
    from app.memory_guard import (
        MemoryPolicy, MemoryPolicyConfig, set_policy_for_test,
    )
    cfg = MemoryPolicyConfig(soft_gb=90.0, hard_gb=100.0, emergency_gb=110.0)
    policy = MemoryPolicy(cfg, probe=lambda: (121.0, 10.0))
    set_policy_for_test(policy)
    return policy


# ---------------------------------------------------------------------------
# 1. Qwen stays first in the provider chain
# ---------------------------------------------------------------------------

def test_qwen_vllm_backend_stays_first_in_chain():
    """build_active_provider must build a ChainProvider with qwen3_vl
    first when reasoning.primary_provider == qwen3_vl. The vllm_openai
    backend must NOT be filtered out by the repo-local snapshot gate
    (the active gate is the /v1/models startup probe)."""
    from types import SimpleNamespace
    from reasoning.providers.chain import build_active_provider, ChainProvider

    qwen_cfg = SimpleNamespace(
        name="Qwen/Qwen3-VL-30B-A3B-Instruct",
        enabled=True,
        extra={
            "provider": "vllm_openai",
            "base_url": "http://127.0.0.1:8000/v1",
            "served_model_name": "qwen3_vl",
        },
    )
    gemma_cfg = SimpleNamespace(
        name="google/gemma-4-26B-A4B-it",
        enabled=True,
        extra={"vllm_url": "http://localhost:8001"},
    )
    cfg = SimpleNamespace(
        raw={"reasoning": {"primary_provider": "qwen3_vl",
                             "fallback_provider": "gemma",
                             "warm_fallback": False}},
        models={"qwen3_vl": qwen_cfg, "gemma": gemma_cfg},
    )
    chain = build_active_provider(cfg)
    assert isinstance(chain, ChainProvider), type(chain)
    assert chain.providers[0].name == "qwen3_vl", \
        [p.name for p in chain.providers]
    assert chain.providers[1].name == "gemma", \
        [p.name for p in chain.providers]


# ---------------------------------------------------------------------------
# 2. External HTTP provider is callable when /v1/models is healthy
# ---------------------------------------------------------------------------

def test_external_qwen_provider_callable_when_vllm_models_healthy(monkeypatch):
    """When the vLLM server responds 200 on /v1/models with the served
    model id, Qwen3VLProvider's vLLM health probe returns healthy and
    the chain calls it."""
    _force_normal_policy(monkeypatch)

    from reasoning.providers.qwen3_vl import Qwen3VLProvider
    from reasoning.providers.gemma import GemmaProvider
    from reasoning.providers.chain import ChainProvider
    from reasoning.providers.base import EvidenceManifest, VLMResult

    qwen = Qwen3VLProvider(
        model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
        enabled=True, provider="vllm_openai",
        base_url="http://127.0.0.1:8000/v1",
        served_model_name="qwen3_vl",
    )
    # Stub the HTTP path: analyze_evidence returns a successful result.
    monkeypatch.setattr(qwen, "_analyze_vllm",
                        lambda m: VLMResult(provider="qwen3_vl",
                                              model_name="qwen3_vl",
                                              parsed={"ok": True}))
    gemma = GemmaProvider(model_name="google/gemma-4-26B-A4B-it",
                           enabled=True, vllm_url="http://localhost:8001")
    chain = ChainProvider(providers=[qwen, gemma])
    manifest = EvidenceManifest(case_id="c", camera_id="cam_01",
                                 window_start_ts="x", window_end_ts="y",
                                 frames=[{"image_url": "data:image/jpeg;base64,"}])
    r = chain.analyze_evidence(manifest)
    assert r.error is None, r.error
    assert r.provider == "qwen3_vl"
    assert "qwen3_vl=ok" in (r.parsed.get("_chain_attempts") or [])


# ---------------------------------------------------------------------------
# 3. memory_guard EMERGENCY does NOT block external HTTP providers
# ---------------------------------------------------------------------------

def test_memory_emergency_does_not_block_external_http_chain(monkeypatch):
    """All-external-HTTP chain must still attempt providers even when
    the app process's memory has crossed the emergency threshold.
    The two providers (Qwen vLLM, Gemma HTTP bridge) keep their weights
    in DIFFERENT processes — the app's RAM ceiling does not bound them."""
    _force_emergency_policy(monkeypatch)

    from reasoning.providers.qwen3_vl import Qwen3VLProvider
    from reasoning.providers.gemma import GemmaProvider
    from reasoning.providers.chain import ChainProvider
    from reasoning.providers.base import EvidenceManifest, VLMResult

    qwen = Qwen3VLProvider(model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
                            enabled=True, provider="vllm_openai",
                            base_url="http://127.0.0.1:8000/v1",
                            served_model_name="qwen3_vl")
    monkeypatch.setattr(qwen, "_analyze_vllm",
                        lambda m: VLMResult(provider="qwen3_vl",
                                              model_name="qwen3_vl",
                                              parsed={"ok": True}))
    gemma = GemmaProvider(model_name="google/gemma-4-26B-A4B-it",
                           enabled=True, vllm_url="http://localhost:8001")
    chain = ChainProvider(providers=[qwen, gemma])
    manifest = EvidenceManifest(case_id="c", camera_id="cam_01",
                                 window_start_ts="x", window_end_ts="y",
                                 frames=[{"image_url": "data:image/jpeg;base64,"}])
    r = chain.analyze_evidence(manifest)
    # CRITICAL: NOT a "memory state emergency_limit" defer.
    assert r.error is None, (
        f"chain short-circuited on memory state for ALL-external "
        f"providers: {r.error}")
    assert "inference deferred" not in (r.error or "")
    assert r.provider == "qwen3_vl"


def test_in_process_provider_still_gated_when_memory_emergency(monkeypatch):
    """Regression: an in-process provider IS still gated by the
    emergency threshold — this commit only exempts external HTTP."""
    _force_emergency_policy(monkeypatch)

    from reasoning.providers.qwen3_vl import Qwen3VLProvider
    from reasoning.providers.chain import ChainProvider
    from reasoning.providers.base import EvidenceManifest

    qwen_local = Qwen3VLProvider(
        model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
        enabled=True, provider="local_transformers",
        local_path="/tmp/does/not/exist",
    )
    assert qwen_local.is_external_http() is False
    chain = ChainProvider(providers=[qwen_local])
    manifest = EvidenceManifest(case_id="c", camera_id="cam_01",
                                 window_start_ts="x", window_end_ts="y",
                                 frames=[])
    r = chain.analyze_evidence(manifest)
    # Chain has an in-process provider AND gate is closed → defer.
    assert r.error is not None
    assert "inference deferred" in r.error
    assert "emergency_limit" in r.error


def test_mixed_chain_skips_in_process_but_tries_external_under_pressure(monkeypatch):
    """If a chain has BOTH an in-process and an external-HTTP provider,
    the gate skips the in-process one but still tries the external one
    when memory state blocks new inference."""
    _force_emergency_policy(monkeypatch)

    from reasoning.providers.qwen3_vl import Qwen3VLProvider
    from reasoning.providers.gemma import GemmaProvider
    from reasoning.providers.chain import ChainProvider
    from reasoning.providers.base import EvidenceManifest, VLMResult

    qwen_local = Qwen3VLProvider(
        model_name="qwen-local", enabled=True,
        provider="local_transformers",
        local_path="/tmp/does/not/exist",
    )
    gemma = GemmaProvider(model_name="google/gemma-4-26B-A4B-it",
                           enabled=True, vllm_url="http://localhost:8001")
    # Stub Gemma's HTTP call so it returns success without a real server.
    monkeypatch.setattr(gemma, "analyze_evidence",
                        lambda m: VLMResult(provider="gemma",
                                              model_name="gemma",
                                              parsed={"ok": True}))
    chain = ChainProvider(providers=[qwen_local, gemma])
    manifest = EvidenceManifest(case_id="c", camera_id="cam_01",
                                 window_start_ts="x", window_end_ts="y",
                                 frames=[])
    r = chain.analyze_evidence(manifest)
    # Primary (local Qwen) skipped under the gate; Gemma (external HTTP)
    # tried and succeeded.
    assert r.error is None, r.error
    assert r.provider == "gemma"
    attempts = (r.parsed.get("_chain_attempts") or [])
    assert any("qwen3_vl=deferred" in a for a in attempts), attempts
    assert "gemma=ok" in attempts


# ---------------------------------------------------------------------------
# 4. Gemma fallback fires when Qwen HTTP call fails
# ---------------------------------------------------------------------------

def test_gemma_fallback_fires_when_qwen_raises(monkeypatch):
    _force_normal_policy(monkeypatch)

    from reasoning.providers.qwen3_vl import Qwen3VLProvider
    from reasoning.providers.gemma import GemmaProvider
    from reasoning.providers.chain import ChainProvider
    from reasoning.providers.base import EvidenceManifest, VLMResult

    qwen = Qwen3VLProvider(model_name="qwen", enabled=True,
                            provider="vllm_openai",
                            base_url="http://127.0.0.1:8000/v1",
                            served_model_name="qwen3_vl")

    def _qwen_raises(m):
        raise ConnectionError("vllm unreachable")
    monkeypatch.setattr(qwen, "_analyze_vllm", _qwen_raises)

    gemma = GemmaProvider(model_name="gemma", enabled=True,
                           vllm_url="http://localhost:8001")
    monkeypatch.setattr(gemma, "analyze_evidence",
                        lambda m: VLMResult(provider="gemma",
                                              model_name="gemma",
                                              parsed={"fallback": True}))
    chain = ChainProvider(providers=[qwen, gemma])
    manifest = EvidenceManifest(case_id="c", camera_id="cam_01",
                                 window_start_ts="x", window_end_ts="y",
                                 frames=[])
    r = chain.analyze_evidence(manifest)
    assert r.error is None, r.error
    assert r.provider == "gemma"
    assert r.parsed.get("fallback") is True
    attempts = r.parsed.get("_chain_attempts") or []
    assert any("qwen3_vl=raised" in a for a in attempts), attempts
    assert "gemma=ok" in attempts


def test_gemma_fallback_fires_when_qwen_returns_error_result(monkeypatch):
    _force_normal_policy(monkeypatch)

    from reasoning.providers.qwen3_vl import Qwen3VLProvider
    from reasoning.providers.gemma import GemmaProvider
    from reasoning.providers.chain import ChainProvider
    from reasoning.providers.base import EvidenceManifest, VLMResult

    qwen = Qwen3VLProvider(model_name="qwen", enabled=True,
                            provider="vllm_openai",
                            base_url="http://127.0.0.1:8000/v1",
                            served_model_name="qwen3_vl")
    monkeypatch.setattr(qwen, "_analyze_vllm",
                        lambda m: VLMResult(provider="qwen3_vl",
                                              model_name="qwen3_vl",
                                              error="vllm 500 internal"))
    gemma = GemmaProvider(model_name="gemma", enabled=True,
                           vllm_url="http://localhost:8001")
    monkeypatch.setattr(gemma, "analyze_evidence",
                        lambda m: VLMResult(provider="gemma",
                                              model_name="gemma",
                                              parsed={"fallback": True}))
    chain = ChainProvider(providers=[qwen, gemma])
    manifest = EvidenceManifest(case_id="c", camera_id="cam_01",
                                 window_start_ts="x", window_end_ts="y",
                                 frames=[])
    r = chain.analyze_evidence(manifest)
    assert r.error is None, r.error
    assert r.provider == "gemma"


# ---------------------------------------------------------------------------
# Provider-classifier unit tests
# ---------------------------------------------------------------------------

def test_is_external_http_classifier_per_backend():
    from reasoning.providers.qwen3_vl import Qwen3VLProvider
    from reasoning.providers.gemma import GemmaProvider

    q_http = Qwen3VLProvider(model_name="x", enabled=True,
                              provider="vllm_openai",
                              base_url="http://127.0.0.1:8000/v1",
                              served_model_name="qwen3_vl")
    q_local = Qwen3VLProvider(model_name="x", enabled=True,
                                provider="local_transformers",
                                local_path="/tmp/none")
    g = GemmaProvider(model_name="g", enabled=True,
                       vllm_url="http://localhost:8001")
    assert q_http.is_external_http() is True
    assert q_local.is_external_http() is False
    assert g.is_external_http() is True


def test_base_provider_defaults_external_http_to_false():
    from reasoning.providers.base import VLMProvider
    p = VLMProvider(model_name="x", enabled=True)
    assert p.is_external_http() is False
