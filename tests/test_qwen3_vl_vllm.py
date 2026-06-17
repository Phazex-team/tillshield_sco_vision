"""Tests for the Qwen3-VL vLLM backend + backend-conditional gates.

These tests must NOT touch the network, NOT load any model weights, and
NOT import HF Transformers / bitsandbytes when exercising the
``vllm_openai`` path. They cover:

  * Strict base_url validation (scheme, host, port, path, no query/frag)
  * Session hygiene (trust_env=False, allow_redirects=False)
  * /v1/models gate (served_model_name must be present)
  * Wired ``max_tokens`` ends up in the POST payload
  * vllm mode never imports transformers/bitsandbytes
  * build_active_provider bypasses Qwen HF bundle for vllm_openai but
    still requires it for local_transformers
  * Failure routing: transport / non-200 / malformed envelope → Gemma
  * Valid response + unparseable model text → REVIEW path (not crash)
  * Downscale happens BEFORE base64
  * local_transformers chat builder uses concrete lists, not generators
  * VLM metadata threading (provider_metadata / model_snapshot)
  * Offline verifier reclassifies Qwen as rollback-only under vllm_openai
  * Decision policy still gates VERIFIED on perception tracks
"""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasoning.providers.base import EvidenceManifest  # noqa: E402
from reasoning.providers.qwen3_vl import (  # noqa: E402
    Qwen3VLProvider,
    _parse_json,
    _validate_base_url,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _png_frame(width: int = 320, height: int = 240) -> dict:
    """Return a frame dict whose ``image_url`` is a 320x240 PNG data URL.
    Small enough to avoid downscale unless the test sets a tiny limit."""
    from PIL import Image
    img = Image.new("RGB", (width, height), color=(40, 80, 160))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {
        "frame_id": "f0",
        "frame_idx": 0,
        "ts": "2026-06-15T14:00:00",
        "image_url": f"data:image/png;base64,{b64}",
    }


def _manifest(frames: int = 2) -> EvidenceManifest:
    return EvidenceManifest(
        case_id="case-test",
        camera_id="cam_01",
        window_start_ts="2026-06-15T14:00:00",
        window_end_ts="2026-06-15T14:00:30",
        frames=[_png_frame() for _ in range(frames)],
    )


class _Resp:
    """Minimal response stub honoring ``.status_code``, ``.json()``,
    ``.text``. Behaves like requests.Response enough for our code path."""

    def __init__(self, *, status_code: int = 200, body: Any = None,
                 text: str = "", raise_on_json: bool = False):
        self.status_code = status_code
        self._body = body
        self.text = text or ""
        self._raise_on_json = raise_on_json

    def json(self):
        if self._raise_on_json:
            raise ValueError("not json")
        return self._body


class _FakeSession:
    """A requests-like session whose GET/POST return canned responses
    and records every call so tests can assert allow_redirects=False
    and timeouts."""

    def __init__(self, *, get_resp: _Resp | None = None,
                 post_resp: _Resp | None = None,
                 raise_on_post: Exception | None = None,
                 raise_on_get: Exception | None = None):
        self.trust_env = True  # default; provider will set False
        self.calls: list[dict] = []
        self._get = get_resp
        self._post = post_resp
        self._raise_get = raise_on_get
        self._raise_post = raise_on_post

    def get(self, url, *, timeout, allow_redirects):
        self.calls.append({"method": "GET", "url": url, "timeout": timeout,
                           "allow_redirects": allow_redirects})
        if self._raise_get is not None:
            raise self._raise_get
        return self._get or _Resp(status_code=200,
                                  body={"data": [{"id": "qwen3_vl"}]})

    def post(self, url, *, json, timeout, allow_redirects):
        self.calls.append({"method": "POST", "url": url, "timeout": timeout,
                           "allow_redirects": allow_redirects, "json": json})
        if self._raise_post is not None:
            raise self._raise_post
        return self._post or _Resp(status_code=200, body={
            "choices": [{"message": {"content": '{"narrative":"ok",'
                                                '"confidence":"medium"}'}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        })


# ---------------------------------------------------------------------
# Strict base_url validation
# ---------------------------------------------------------------------

@pytest.mark.parametrize("good", [
    "http://127.0.0.1:8000/v1",
    "http://127.0.0.1:8000/v1/",  # trailing slash trimmed
])
def test_validate_base_url_accepts_defaults(good):
    assert _validate_base_url(good, allow_localhost_alias=False,
                              allow_port_override=False) \
        == "http://127.0.0.1:8000/v1"


@pytest.mark.parametrize("bad", [
    "",
    "127.0.0.1:8000/v1",                          # no scheme
    "https://127.0.0.1:8000/v1",                  # https
    "http://example.com:8000/v1",                 # non-loopback host
    "http://127.0.0.1:9000/v1",                   # wrong port
    "http://127.0.0.1:8000",                      # missing /v1
    "http://127.0.0.1:8000/v2",                   # wrong path
    "http://127.0.0.1:8000/v1?foo=1",             # query
    "http://127.0.0.1:8000/v1#frag",              # fragment
    "http://127.0.0.1/v1",                        # missing port
])
def test_validate_base_url_rejects(bad):
    with pytest.raises(ValueError):
        _validate_base_url(bad, allow_localhost_alias=False,
                           allow_port_override=False)


def test_validate_base_url_localhost_alias_opt_in():
    with pytest.raises(ValueError):
        _validate_base_url("http://localhost:8000/v1",
                           allow_localhost_alias=False,
                           allow_port_override=False)
    assert _validate_base_url("http://localhost:8000/v1",
                              allow_localhost_alias=True,
                              allow_port_override=False) \
        == "http://localhost:8000/v1"


def test_validate_base_url_port_override_opt_in():
    with pytest.raises(ValueError):
        _validate_base_url("http://127.0.0.1:9001/v1",
                           allow_localhost_alias=False,
                           allow_port_override=False)
    assert _validate_base_url("http://127.0.0.1:9001/v1",
                              allow_localhost_alias=False,
                              allow_port_override=True) \
        == "http://127.0.0.1:9001/v1"


def test_provider_with_bad_base_url_returns_error_not_raise():
    p = Qwen3VLProvider(model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
                        enabled=True,
                        provider="vllm_openai",
                        base_url="https://evil.example.com/v1")
    result = p.analyze_evidence(_manifest())
    assert result.error is not None
    assert "bad base_url" in result.error
    h = p.health()
    assert h.healthy is False
    assert "bad base_url" in h.detail


# ---------------------------------------------------------------------
# Session hygiene
# ---------------------------------------------------------------------

def test_session_trust_env_false_and_redirects_disabled(monkeypatch):
    fake = _FakeSession()
    p = Qwen3VLProvider(model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
                        enabled=True, provider="vllm_openai")
    p._session = fake  # inject before any call
    result = p.analyze_evidence(_manifest())
    assert result.error is None, result.error
    # The provider sets trust_env=False on the session it creates; when
    # the session is injected, we still assert allow_redirects=False on
    # every call (the wire-level guarantee tests actually care about).
    assert all(c["allow_redirects"] is False for c in fake.calls)


def test_session_created_by_provider_disables_trust_env(monkeypatch):
    # Force the lazy path: when _session is None, _ensure_session uses
    # requests.Session and sets trust_env=False.
    class FakeRequestsSession:
        def __init__(self):
            self.trust_env = True

    import reasoning.providers.qwen3_vl as mod
    # Build a fake ``requests`` module so the lazy import inside
    # _ensure_session resolves to our stub.
    fake_requests = SimpleNamespace(Session=FakeRequestsSession)
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    p = Qwen3VLProvider(model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
                        enabled=True, provider="vllm_openai")
    sess = p._ensure_session()
    assert sess.trust_env is False


# ---------------------------------------------------------------------
# /v1/models health gate
# ---------------------------------------------------------------------

def test_health_requires_served_model_name_in_v1_models():
    p = Qwen3VLProvider(model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
                        enabled=True, provider="vllm_openai",
                        served_model_name="qwen3_vl")
    p._session = _FakeSession(get_resp=_Resp(
        status_code=200, body={"data": [{"id": "other_model"}]}))
    h = p.health()
    assert h.healthy is False
    assert "missing served model" in h.detail


def test_health_ok_when_served_model_listed():
    p = Qwen3VLProvider(model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
                        enabled=True, provider="vllm_openai",
                        served_model_name="qwen3_vl")
    p._session = _FakeSession(get_resp=_Resp(
        status_code=200, body={"data": [{"id": "qwen3_vl"}]}))
    h = p.health()
    assert h.healthy is True
    assert "qwen3_vl" in h.detail


# ---------------------------------------------------------------------
# max_tokens wired into payload
# ---------------------------------------------------------------------

def test_max_tokens_payload_follows_config():
    p = Qwen3VLProvider(model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
                        enabled=True, provider="vllm_openai",
                        max_tokens=137)
    fake = _FakeSession()
    p._session = fake
    p.analyze_evidence(_manifest())
    posts = [c for c in fake.calls if c["method"] == "POST"]
    assert posts, "expected at least one POST"
    assert posts[0]["json"]["max_tokens"] == 137
    assert posts[0]["json"]["stream"] is False
    assert posts[0]["json"]["model"] == "qwen3_vl"


# ---------------------------------------------------------------------
# No HF imports in vllm mode
# ---------------------------------------------------------------------

def test_vllm_mode_imports_no_transformers_or_bitsandbytes(monkeypatch):
    # Pre-poison the modules so any import attempt would raise loudly.
    sentinel = ImportError("HF must not be imported in vllm mode")

    class _Forbidden:
        def __getattr__(self, item):
            raise sentinel

    for mod_name in ("transformers", "bitsandbytes"):
        monkeypatch.setitem(sys.modules, mod_name, _Forbidden())

    p = Qwen3VLProvider(model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
                        enabled=True, provider="vllm_openai")
    p._session = _FakeSession()
    result = p.analyze_evidence(_manifest())
    assert result.error is None, result.error


# ---------------------------------------------------------------------
# Chain backend-conditional gate
# ---------------------------------------------------------------------

def _stub_cfg(provider: str) -> Any:
    """Construct the minimal cfg shape ``build_active_provider`` needs."""
    from app.config import ModelConfig
    qwen_extra = {
        "provider": provider,
        "base_url": "http://127.0.0.1:8000/v1",
        "served_model_name": "qwen3_vl",
        "max_tokens": 32,
    }
    if provider == "local_transformers":
        qwen_extra["local_path"] = "/nonexistent/snapshot"
    cfg = SimpleNamespace(
        raw={"reasoning": {"primary_provider": "qwen3_vl",
                            "fallback_provider": "gemma",
                            "warm_fallback": False}},
        models={
            "qwen3_vl": ModelConfig(name="Qwen/Qwen3-VL-30B-A3B-Instruct",
                                    enabled=True, extra=qwen_extra),
            "gemma": ModelConfig(name="google/gemma-4-26B-A4B-it",
                                 enabled=True,
                                 extra={"vllm_url": "http://127.0.0.1:1"}),
        },
    )
    return cfg


def test_chain_does_not_require_qwen_bundle_for_vllm_openai(monkeypatch):
    import reasoning.providers.chain as chain
    calls = []

    def fake_resolve(model_cfg, *, production_mode=None):
        calls.append(model_cfg.name)
        return None  # would normally cause Qwen to be skipped

    monkeypatch.setattr(chain, "resolve_model_path", fake_resolve)
    cfg = _stub_cfg(provider="vllm_openai")
    p = chain.build_active_provider(cfg)
    # Active path is Qwen via vLLM, Gemma fallback.
    assert p.name == "chain"
    member_names = [m.name for m in p.providers]
    assert "qwen3_vl" in member_names
    assert "gemma" in member_names
    # resolve_model_path must NOT be called for Qwen under vllm_openai.
    assert "Qwen/Qwen3-VL-30B-A3B-Instruct" not in calls


def test_chain_requires_qwen_bundle_for_local_transformers(monkeypatch):
    import reasoning.providers.chain as chain
    seen = []

    def fake_resolve(model_cfg, *, production_mode=None):
        seen.append(model_cfg.name)
        return None  # simulate missing repo-local bundle

    monkeypatch.setattr(chain, "resolve_model_path", fake_resolve)
    cfg = _stub_cfg(provider="local_transformers")
    p = chain.build_active_provider(cfg)
    # Qwen is dropped because the rollback gate failed; only Gemma remains.
    if p.name == "chain":
        names = [m.name for m in p.providers]
    else:
        names = [p.name]
    assert "qwen3_vl" not in names
    assert "Qwen/Qwen3-VL-30B-A3B-Instruct" in seen


# ---------------------------------------------------------------------
# Failure routing: transport / non-200 / malformed envelope
# ---------------------------------------------------------------------

@pytest.mark.parametrize("fake", [
    _FakeSession(raise_on_post=TimeoutError("read timeout")),
    _FakeSession(raise_on_post=ConnectionError("refused")),
    _FakeSession(post_resp=_Resp(status_code=503, text="service unavailable")),
    _FakeSession(post_resp=_Resp(status_code=200, body={"choices": []})),
    _FakeSession(post_resp=_Resp(status_code=200,
                                  body={"choices": [{"message": {}}]})),
])
def test_transport_or_envelope_failures_return_error(fake):
    p = Qwen3VLProvider(model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
                        enabled=True, provider="vllm_openai")
    p._session = fake
    result = p.analyze_evidence(_manifest())
    assert result.error is not None, "transport/envelope failure must error"
    # The chain wrapper would route to Gemma; here we only assert the
    # provider returned a structured error rather than raising.


def test_chain_falls_back_to_gemma_on_qwen_transport_error(monkeypatch):
    """Wire Qwen + a fake Gemma into a ChainProvider; force Qwen to error
    via injected session; expect the chain to use Gemma's result."""
    from reasoning.providers.chain import ChainProvider
    qwen = Qwen3VLProvider(model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
                           enabled=True, provider="vllm_openai")
    qwen._session = _FakeSession(
        raise_on_post=TimeoutError("read timeout"),
        get_resp=_Resp(status_code=200, body={"data": [{"id": "qwen3_vl"}]}),
    )

    from reasoning.providers.base import VLMProvider, VLMResult
    from reasoning.providers.base import ProviderHealth

    class FakeGemma(VLMProvider):
        name = "gemma"

        def __init__(self):
            super().__init__(model_name="gemma:test", enabled=True)
            self.calls = 0

        def analyze_evidence(self, manifest):
            self.calls += 1
            return VLMResult(provider="gemma", model_name="gemma:test",
                             raw_text='{"narrative":"fallback",'
                                      '"confidence":"low"}',
                             parsed={"narrative": "fallback",
                                     "confidence": "low"})

        def health(self):
            return ProviderHealth("gemma", True, "ok")

    gemma = FakeGemma()
    chain = ChainProvider(providers=[qwen, gemma])
    r = chain.analyze_evidence(_manifest())
    assert r.error is None
    assert r.provider == "gemma"
    assert gemma.calls == 1


# ---------------------------------------------------------------------
# Valid response + unparseable text → REVIEW route, no crash
# ---------------------------------------------------------------------

@pytest.mark.parametrize("content", [
    "not json at all",
    "[]",          # JSON but not a dict
    "[1, 2, 3]",   # ditto
    "42",          # ditto
])
def test_unparseable_or_non_dict_text_does_not_crash(content):
    p = Qwen3VLProvider(model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
                        enabled=True, provider="vllm_openai")
    p._session = _FakeSession(post_resp=_Resp(
        status_code=200,
        body={"choices": [{"message": {"content": content}}],
              "usage": {}}))
    r = p.analyze_evidence(_manifest())
    assert r.error is None, "valid envelope must not error on unparseable text"
    assert isinstance(r.parsed, dict)
    assert r.parsed.get("confidence") == "low"


def test_parse_json_non_dict_returns_low_confidence_dict():
    assert _parse_json("[]") == {"narrative": "[]", "confidence": "low"}
    assert _parse_json("42") == {"narrative": "42", "confidence": "low"}
    assert _parse_json("totally not json") == \
        {"narrative": "totally not json", "confidence": "low"}


# ---------------------------------------------------------------------
# Downscale happens before base64
# ---------------------------------------------------------------------

def test_downscale_applied_before_base64():
    """Big frame, small downscale cap → POST body's image_url is smaller
    than the raw input image. We use PIL to decode the base64 back and
    compare dimensions to the cap."""
    from PIL import Image
    big = Image.new("RGB", (2560, 1440), color=(10, 20, 30))
    buf = io.BytesIO()
    big.save(buf, format="PNG")
    big_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    manifest = EvidenceManifest(
        case_id="case-d",
        camera_id="cam_01",
        window_start_ts="2026-06-15T14:00:00",
        window_end_ts="2026-06-15T14:00:01",
        frames=[{
            "frame_id": "f0",
            "frame_idx": 0,
            "ts": "2026-06-15T14:00:00",
            "image_url": f"data:image/png;base64,{big_b64}",
        }],
    )
    p = Qwen3VLProvider(model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
                        enabled=True, provider="vllm_openai",
                        max_frame_long_edge=640,
                        max_frame_pixels=640 * 360)
    fake = _FakeSession()
    p._session = fake
    p.analyze_evidence(manifest)
    posts = [c for c in fake.calls if c["method"] == "POST"]
    assert posts
    user_block = posts[0]["json"]["messages"][1]["content"]
    image_url_block = next(b for b in user_block if b.get("type") == "image_url")
    url = image_url_block["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    payload_b64 = url.split(",", 1)[1]
    decoded = base64.b64decode(payload_b64)
    sent_img = Image.open(io.BytesIO(decoded))
    assert max(sent_img.size) <= 640, sent_img.size
    assert sent_img.size[0] * sent_img.size[1] <= 640 * 360 + 4


# ---------------------------------------------------------------------
# Local-transformers chat builder uses concrete lists
# ---------------------------------------------------------------------

def test_local_transformers_chat_builder_uses_concrete_lists():
    p = Qwen3VLProvider(model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
                        enabled=True, provider="local_transformers",
                        local_path="/tmp")
    frames = ["frame_a", "frame_b"]  # opaque placeholders are fine
    chat = p._build_transformers_chat("sys", "usr", frames)
    assert isinstance(chat, list)
    assert isinstance(chat[0]["content"], list)
    assert isinstance(chat[1]["content"], list)
    # Iterating user content twice must yield the same images both times
    # (catches generator-exhausted shape).
    seen_a = [b for b in chat[1]["content"] if b.get("type") == "image"]
    seen_b = [b for b in chat[1]["content"] if b.get("type") == "image"]
    assert seen_a == seen_b == [
        {"type": "image", "image": "frame_a"},
        {"type": "image", "image": "frame_b"},
    ]
    assert chat[1]["content"][-1] == {"type": "text", "text": "usr"}


# ---------------------------------------------------------------------
# Model metadata is stamped
# ---------------------------------------------------------------------

def test_model_run_metadata_present_in_parsed():
    p = Qwen3VLProvider(model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
                        enabled=True, provider="vllm_openai",
                        served_model_name="qwen3_vl",
                        checkpoint_label="Qwen3-VL-30B-A3B-Instruct-FP8",
                        precision="fp8")
    p._session = _FakeSession()
    r = p.analyze_evidence(_manifest())
    assert r.error is None
    mr = r.parsed.get("_model_run")
    assert isinstance(mr, dict)
    meta = mr["provider_metadata"]
    assert meta["backend"] == "vllm_openai"
    assert meta["base_url"] == "http://127.0.0.1:8000/v1"
    assert meta["served_model_name"] == "qwen3_vl"
    assert meta["precision"] == "fp8"
    assert mr["model_snapshot"] == "Qwen3-VL-30B-A3B-Instruct-FP8"


def test_case_runner_adapter_threads_metadata():
    """``_adapt_vlm_result`` must lift ``_model_run`` out of parsed into
    top-level provider_metadata / model_snapshot / usage."""
    from app.case_runner import _adapt_vlm_result
    from reasoning.providers.base import VLMResult

    parsed = {
        "narrative": "ok",
        "_model_run": {
            "provider_metadata": {"backend": "vllm_openai",
                                  "base_url": "http://127.0.0.1:8000/v1"},
            "model_snapshot": "Qwen3-VL-30B-A3B-Instruct-FP8",
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        },
    }
    res = VLMResult(provider="qwen3_vl", model_name="Qwen3",
                    raw_text="", parsed=parsed)
    out = _adapt_vlm_result(res, prompt_version="v1")
    assert out["provider_metadata"]["backend"] == "vllm_openai"
    assert out["model_snapshot"] == "Qwen3-VL-30B-A3B-Instruct-FP8"
    assert out["usage"]["completion_tokens"] == 2
    # _model_run must be stripped from parsed so output_json stays clean.
    assert "_model_run" not in out["parsed"]


# ---------------------------------------------------------------------
# Offline verifier conditional behavior
# ---------------------------------------------------------------------

def test_offline_verifier_classifies_qwen_rollback_only(tmp_path, monkeypatch):
    from app.config import ModelConfig
    import scripts.verify_offline_bundle as v

    cfg = SimpleNamespace(
        models={
            "qwen3_vl": ModelConfig(
                name="Qwen/Qwen3-VL-30B-A3B-Instruct",
                enabled=True,
                extra={"provider": "vllm_openai"}),
        },
    )
    assert v._qwen_backend(cfg) == "vllm_openai"
    note = v._qwen_runtime_note(cfg)
    assert "vLLM" in note

    registry = {"required": [{
        "name": "Qwen3-VL",
        "repo": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "official_source": "...",
    }]}
    manifest = {"models": []}  # Qwen NOT present anywhere
    summary = v.verify_required_assets(registry, manifest, full_hash=False,
                                        qwen_backend="vllm_openai")
    assert summary["required_missing"] == []
    assert summary["rollback_only_missing"]
    assert summary["rollback_only_missing"][0]["name"] == "Qwen3-VL"


def test_offline_verifier_keeps_qwen_required_for_local_transformers():
    import scripts.verify_offline_bundle as v
    registry = {"required": [{
        "name": "Qwen3-VL",
        "repo": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "official_source": "...",
    }]}
    manifest = {"models": []}
    summary = v.verify_required_assets(registry, manifest, full_hash=False,
                                        qwen_backend="local_transformers")
    assert summary["rollback_only_missing"] == []
    assert summary["required_missing"]
    assert summary["required_missing"][0]["name"] == "Qwen3-VL"


# ---------------------------------------------------------------------
# Decision policy still gates VERIFIED on perception tracks
# ---------------------------------------------------------------------

def test_decision_policy_still_gates_verified_on_perception_tracks():
    """Even if the VLM says ``physical_item_presented: true``, without a
    matching perception track the deterministic policy must NOT return
    VERIFIED. This guards the wiring after the vLLM migration."""
    from reasoning.decision_policy import decide, summary_from_vlm

    vlm_parsed = {
        "handover_occurred": True,
        "physical_item_presented": True,
        "receipt_visible": True,
        "narrative": "customer hands item over and receipt visible",
        "confidence": "high",
        "obstructed": False,
        "camera_view_clear": True,
    }
    # No perception tracks at all → policy must NOT verify on VLM alone.
    summary = summary_from_vlm(vlm_parsed, footage_valid=True,
                                obstructed=False, camera_gap=False,
                                perception_result={"tracks": []})
    decision = decide(summary)
    assert decision.outcome != "VERIFIED"
