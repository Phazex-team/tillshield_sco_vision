"""Memory guard + lazy provider loading tests.

Pins the contracts:
  * Constructing the provider chain does NOT load any model weights.
  * Qwen + Gemma are never warm-loaded simultaneously by default.
  * Soft memory limit blocks new inference work.
  * Hard memory limit triggers unload callbacks.
  * Emergency memory limit triggers stop callbacks for inference workers.
  * Recorder/API code never calls ``allow_new_inference`` so it stays
    alive even under emergency state.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.memory_guard import (  # noqa: E402
    STATE_EMERGENCY,
    STATE_HARD,
    STATE_NORMAL,
    STATE_SOFT,
    MemoryPolicy,
    MemoryPolicyConfig,
    set_policy_for_test,
)


def _fake_probe(total_gb: float, used_gb: float):
    return lambda: (total_gb, used_gb)


@pytest.fixture
def policy():
    p = MemoryPolicy(MemoryPolicyConfig(
        soft_gb=90, hard_gb=100, emergency_gb=110, poll_interval_sec=999,
    ), probe=_fake_probe(120.0, 30.0))
    set_policy_for_test(p)
    yield p
    p.reset_for_test()


def test_normal_state_allows_inference(policy):
    s = policy.poll()
    assert s.state == STATE_NORMAL
    assert s.inference_allowed is True
    assert policy.allow_new_inference() is True


def test_soft_limit_blocks_new_inference(policy):
    policy._probe = _fake_probe(120.0, 92.0)
    s = policy.poll()
    assert s.state == STATE_SOFT
    assert s.inference_allowed is False
    assert "soft memory limit" in s.degraded_reason


def test_hard_limit_triggers_unload_callbacks(policy):
    # Legacy behaviour, now opt-in: on_hard_limit="unload_models" still
    # drops non-active weights when explicitly configured.
    policy.cfg.on_hard_limit = "unload_models"
    unloaded: list[str] = []
    policy.register_unload_callback("qwen3_vl",
                                    lambda: unloaded.append("qwen3_vl"))
    policy.register_unload_callback("gemma",
                                    lambda: unloaded.append("gemma"))
    policy.mark_loaded("qwen3_vl")

    policy._probe = _fake_probe(120.0, 101.0)
    s = policy.poll()
    assert s.state == STATE_HARD
    assert s.inference_allowed is False
    assert "qwen3_vl" in unloaded
    # Loaded set is cleared once unload fires.
    assert "qwen3_vl" not in policy.loaded_providers()


def test_hard_limit_default_defers_jobs_without_unloading(policy):
    """Default on_hard_limit='defer_jobs': at the hard limit we pause
    admission (queued jobs wait) but never fire unload callbacks — so
    loaded weights stay put and accuracy is preserved."""
    assert policy.cfg.on_hard_limit == "defer_jobs"
    unloaded: list[str] = []
    policy.register_unload_callback("qwen3_vl",
                                    lambda: unloaded.append("qwen3_vl"))
    policy.mark_loaded("qwen3_vl")

    policy._probe = _fake_probe(120.0, 101.0)
    s = policy.poll()
    assert s.state == STATE_HARD
    assert s.admission_paused is True
    # No weights dropped -> accuracy untouched.
    assert unloaded == []
    assert "qwen3_vl" in policy.loaded_providers()


def test_admission_gate_hysteresis(policy):
    """Once paused at the hard limit, admission stays closed until used
    RAM falls back below resume_gb (default = soft), not merely below the
    hard limit — so the gate doesn't flap at the 100 GB boundary."""
    # Under the hard limit -> open.
    policy._probe = _fake_probe(120.0, 95.0)
    assert policy.poll().admission_paused is False

    # Cross the hard limit -> paused.
    policy._probe = _fake_probe(120.0, 101.0)
    assert policy.poll().admission_paused is True

    # Drop into the hysteresis band (below hard, above resume) -> still paused.
    policy._probe = _fake_probe(120.0, 95.0)
    assert policy.poll().admission_paused is True

    # Drop below resume -> admits again.
    policy._probe = _fake_probe(120.0, 89.0)
    assert policy.poll().admission_paused is False


def test_wait_for_headroom_returns_immediately_when_open(policy):
    policy._probe = _fake_probe(120.0, 30.0)
    assert policy.wait_for_headroom() is True


def test_wait_for_headroom_aborts_without_deadlock(policy):
    """When memory stays pinned above the hard limit, wait_for_headroom
    honours should_abort and returns False rather than blocking forever."""
    policy.cfg.poll_interval_sec = 0.01
    policy._probe = _fake_probe(120.0, 105.0)
    assert policy.wait_for_headroom(should_abort=lambda: True) is False


def test_emergency_limit_stops_inference_workers(policy):
    stopped: list[str] = []
    policy.register_stop_callback("inference",
                                  lambda: stopped.append("inference"))
    policy._probe = _fake_probe(120.0, 115.0)
    s = policy.poll()
    assert s.state == STATE_EMERGENCY
    assert s.inference_allowed is False
    assert stopped == ["inference"]


def test_recorder_path_does_not_gate_on_memory(policy):
    """The recorder never calls allow_new_inference. Even at emergency
    state the policy object simply reports — it does not stop the
    recorder. This test pins the contract by exercising the policy in
    emergency state and confirming no exception is raised when callers
    that *don't* check inference run."""
    policy._probe = _fake_probe(120.0, 115.0)
    s = policy.poll()
    assert s.state == STATE_EMERGENCY
    # Recorder code path: just consult status for logging.
    assert s.total_gb == 120.0
    assert s.degraded_reason


# ----------------------------------------------------------------------
# Provider lazy-load contracts
# ----------------------------------------------------------------------

def test_chain_provider_construction_does_not_load_weights(monkeypatch,
                                                           tmp_path):
    """build_active_provider must not load Qwen or Gemma weights at
    construction time. (Test forces the chain shape by opting in to
    the Gemma fallback — production default is Qwen-only.)"""
    import app.config as ac
    fake_root = tmp_path / "models" / "hf"
    qwen_snap = fake_root / "Qwen/Qwen3-VL-30B-A3B-Instruct" / "snap"
    qwen_snap.mkdir(parents=True)
    monkeypatch.setattr(ac, "BUNDLE_ROOT", fake_root)

    from reasoning.providers import build_active_provider, ChainProvider
    cfg = ac.load_config()
    # Production default is reasoning.fallback_provider: null (Qwen-only);
    # for THIS test we want to assert the multi-provider construction
    # path is also lazy. Force the chain shape.
    cfg.raw.setdefault("reasoning", {})["fallback_provider"] = "gemma"
    p = build_active_provider(cfg)
    assert isinstance(p, ChainProvider)
    # Qwen private fields must remain unset until analyze_evidence.
    qwen = p.providers[0]
    assert qwen.name == "qwen3_vl"
    assert qwen._model is None
    assert qwen._processor is None
    # Gemma provider's lazy HTTP client also unset.
    gemma = p.providers[1]
    assert gemma.name == "gemma"
    assert gemma._client_cache is None


def test_qwen_and_gemma_not_loaded_simultaneously_by_default():
    """When the operator opts in to the Gemma fallback, the chain
    constructor must not pre-warm it (``warm_fallback`` defaults to
    false). Production-default Qwen-only mode skips the ChainProvider
    entirely; this test pins the chain semantics when the operator
    re-enables fallback."""
    from app.config import load_config
    from reasoning.providers import build_active_provider, ChainProvider
    cfg = load_config()
    cfg.raw.setdefault("reasoning", {})["fallback_provider"] = "gemma"
    p = build_active_provider(cfg)
    assert isinstance(p, ChainProvider)
    assert p.warm_fallback is False


def test_in_process_chain_deferred_when_memory_above_soft_limit(
        monkeypatch, tmp_path):
    """When the memory guard reports a degraded state AND every provider
    in the chain is IN-PROCESS, the chain returns a structured error
    so the decision policy can degrade the case to REVIEW upstream —
    without loading a model.

    Updated for the per-provider memory gate: external HTTP providers
    are NOT blocked even under memory pressure (their weights live in
    a different process and the app's RAM ceiling does not bound
    them). To pin the gate's intended behaviour we therefore force an
    all-in-process chain by configuring Qwen with local_transformers
    backend and disabling Gemma (which is always external HTTP)."""
    from reasoning.providers.qwen3_vl import Qwen3VLProvider
    from reasoning.providers.chain import ChainProvider
    from reasoning.providers.base import EvidenceManifest

    p = MemoryPolicy(MemoryPolicyConfig(soft_gb=90, hard_gb=100,
                                        emergency_gb=110,
                                        poll_interval_sec=999),
                     probe=_fake_probe(120.0, 95.0))
    set_policy_for_test(p)
    try:
        qwen_local = Qwen3VLProvider(
            model_name="qwen-local", enabled=True,
            provider="local_transformers",
            local_path=str(tmp_path / "does/not/exist"),
        )
        assert qwen_local.is_external_http() is False
        chain = ChainProvider(providers=[qwen_local])
        manifest = EvidenceManifest(
            case_id="c0", camera_id="cam_01",
            window_start_ts="2026-06-15T14:00:00Z",
            window_end_ts="2026-06-15T14:01:00Z",
            frames=[],
        )
        result = chain.analyze_evidence(manifest)
        assert result.error is not None
        assert "inference deferred" in result.error
        # Crucially, the provider was not loaded.
        assert qwen_local._model is None
        # And the per-provider deferral surfaced in the audit trail.
        attempts = (result.parsed or {}).get("_chain_attempts") or []
        assert any("qwen3_vl=deferred" in a for a in attempts), attempts
    finally:
        p.reset_for_test()


def test_external_http_chain_NOT_deferred_when_memory_above_soft_limit(
        monkeypatch, tmp_path):
    """The complement of the in-process test: when every provider in
    the chain is EXTERNAL HTTP, memory pressure does not defer the
    chain. The external server's weights live in another process and
    the app's RAM ceiling has no bearing on its ability to serve."""
    from reasoning.providers.qwen3_vl import Qwen3VLProvider
    from reasoning.providers.gemma import GemmaProvider
    from reasoning.providers.chain import ChainProvider
    from reasoning.providers.base import EvidenceManifest, VLMResult

    p = MemoryPolicy(MemoryPolicyConfig(soft_gb=90, hard_gb=100,
                                        emergency_gb=110,
                                        poll_interval_sec=999),
                     probe=_fake_probe(120.0, 95.0))
    set_policy_for_test(p)
    try:
        qwen = Qwen3VLProvider(
            model_name="qwen", enabled=True,
            provider="vllm_openai",
            base_url="http://127.0.0.1:8000/v1",
            served_model_name="qwen3_vl",
        )
        # Stub the HTTP path so this test doesn't need a real server.
        monkeypatch.setattr(qwen, "_analyze_vllm",
                            lambda m: VLMResult(provider="qwen3_vl",
                                                  model_name="qwen3_vl",
                                                  parsed={"ok": True}))
        gemma = GemmaProvider(model_name="gemma", enabled=True,
                               vllm_url="http://localhost:8001")
        chain = ChainProvider(providers=[qwen, gemma])
        manifest = EvidenceManifest(
            case_id="c0", camera_id="cam_01",
            window_start_ts="x", window_end_ts="y", frames=[])
        result = chain.analyze_evidence(manifest)
        assert result.error is None, result.error
        assert result.provider == "qwen3_vl"
    finally:
        p.reset_for_test()


def test_chain_unloads_primary_before_fallback(monkeypatch, tmp_path):
    """When primary returns an error, the chain must call its unload()
    before trying the secondary."""
    from reasoning.providers.base import VLMProvider, VLMResult
    from reasoning.providers.chain import ChainProvider

    unload_calls: list[str] = []

    class FailingPrimary(VLMProvider):
        name = "qwen3_vl"

        def __init__(self):
            super().__init__(model_name="primary", enabled=True)
            self._model = "fake"

        def analyze_evidence(self, m):
            return VLMResult(provider=self.name, model_name=self.model_name,
                             error="simulated load failure")

        def unload(self):
            unload_calls.append("primary")
            self._model = None

    class OKFallback(VLMProvider):
        name = "gemma"

        def __init__(self):
            super().__init__(model_name="fallback", enabled=True)

        def analyze_evidence(self, m):
            return VLMResult(provider=self.name, model_name=self.model_name,
                             parsed={"narrative": "fallback wins"})

    p = MemoryPolicy(MemoryPolicyConfig(soft_gb=90, hard_gb=100,
                                        emergency_gb=110,
                                        poll_interval_sec=999),
                     probe=_fake_probe(120.0, 30.0))
    set_policy_for_test(p)
    try:
        chain = ChainProvider(providers=[FailingPrimary(), OKFallback()])
        from reasoning.providers.base import EvidenceManifest
        m = EvidenceManifest(
            case_id="c", camera_id="cam_01",
            window_start_ts="2026-06-15T14:00:00Z",
            window_end_ts="2026-06-15T14:01:00Z",
            frames=[],
        )
        result = chain.analyze_evidence(m)
        assert result.error is None
        assert result.provider == "gemma"
        assert unload_calls == ["primary"]
    finally:
        p.reset_for_test()
