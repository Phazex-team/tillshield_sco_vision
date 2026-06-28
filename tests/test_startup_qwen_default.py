"""Production-startup defaults + Qwen-as-primary chain.

Council scope: chain is Qwen-only by default when fallback is null;
Gemma comes back when fallback is explicitly enabled; startup/status
reports Qwen-unavailable clearly; qwen_vllm_start.sh idempotency.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# 1. Provider chain: Qwen-only when fallback null
# ---------------------------------------------------------------------------

def _qwen_cfg(enabled=True, backend="vllm_openai"):
    return SimpleNamespace(
        name="Qwen/Qwen3-VL-30B-A3B-Instruct",
        enabled=enabled,
        extra={"provider": backend,
                "base_url": "http://127.0.0.1:8000/v1",
                "served_model_name": "qwen3_vl"},
    )


def _gemma_cfg(enabled=True):
    return SimpleNamespace(
        name="google/gemma-4-26B-A4B-it", enabled=enabled,
        extra={"vllm_url": "http://localhost:8001"},
    )


def test_chain_is_qwen_only_when_fallback_null():
    """Production default: reasoning.fallback_provider: null →
    build_active_provider returns the bare Qwen3VLProvider, not a
    ChainProvider with Gemma. Gemma is preserved in tree but the
    chain does not silently fall through."""
    from reasoning.providers.chain import (
        build_active_provider, ChainProvider,
    )
    from reasoning.providers.qwen3_vl import Qwen3VLProvider

    cfg = SimpleNamespace(
        raw={"reasoning": {"primary_provider": "qwen3_vl",
                             "fallback_provider": None}},
        models={"qwen3_vl": _qwen_cfg(), "gemma": _gemma_cfg()},
    )
    provider = build_active_provider(cfg)
    # Single-provider path returns the provider directly (not a chain)
    assert isinstance(provider, Qwen3VLProvider)
    assert provider.name == "qwen3_vl"
    assert not isinstance(provider, ChainProvider)


def test_chain_is_qwen_only_when_fallback_key_missing():
    """If the operator removes fallback_provider entirely, the default
    is null (production behaviour) — same outcome as explicitly null."""
    from reasoning.providers.chain import build_active_provider
    from reasoning.providers.qwen3_vl import Qwen3VLProvider

    cfg = SimpleNamespace(
        raw={"reasoning": {"primary_provider": "qwen3_vl"}},
        models={"qwen3_vl": _qwen_cfg(), "gemma": _gemma_cfg()},
    )
    provider = build_active_provider(cfg)
    assert isinstance(provider, Qwen3VLProvider)


def test_chain_re_enables_gemma_when_operator_opts_in():
    """Operator opt-in restores the v1 behaviour: chain with Qwen
    primary + Gemma fallback."""
    from reasoning.providers.chain import (
        build_active_provider, ChainProvider,
    )
    cfg = SimpleNamespace(
        raw={"reasoning": {"primary_provider": "qwen3_vl",
                             "fallback_provider": "gemma"}},
        models={"qwen3_vl": _qwen_cfg(), "gemma": _gemma_cfg()},
    )
    provider = build_active_provider(cfg)
    assert isinstance(provider, ChainProvider)
    assert [p.name for p in provider.providers] == ["qwen3_vl", "gemma"]


def test_real_config_has_qwen_primary_and_null_fallback_by_default():
    """The committed config.yaml ships with Qwen-only by default."""
    from app.config import load_config
    cfg = load_config()
    reasoning = (cfg.raw.get("reasoning") or {})
    assert reasoning.get("primary_provider") == "qwen3_vl"
    assert reasoning.get("fallback_provider") in (None, "null"), \
        ("fallback_provider must be null/None in production config; "
         f"got {reasoning.get('fallback_provider')!r}")


# ---------------------------------------------------------------------------
# 2. Qwen-unavailable surfaces a clear error (no silent fallback)
# ---------------------------------------------------------------------------

def test_qwen_only_unavailable_returns_structured_error(monkeypatch):
    """When the chain is Qwen-only and Qwen errors, the upstream
    case_runner sees a structured error — not a silent Gemma success."""
    from reasoning.providers.chain import build_active_provider
    from reasoning.providers.qwen3_vl import Qwen3VLProvider
    from reasoning.providers.base import EvidenceManifest, VLMResult
    from app.memory_guard import (
        MemoryPolicy, MemoryPolicyConfig, set_policy_for_test,
    )
    set_policy_for_test(MemoryPolicy(
        MemoryPolicyConfig(soft_gb=90, hard_gb=100, emergency_gb=110),
        probe=lambda: (121.0, 10.0)))

    cfg = SimpleNamespace(
        raw={"reasoning": {"primary_provider": "qwen3_vl",
                             "fallback_provider": None}},
        models={"qwen3_vl": _qwen_cfg()},
    )
    provider = build_active_provider(cfg)
    assert isinstance(provider, Qwen3VLProvider)
    monkeypatch.setattr(provider, "_analyze_vllm",
                        lambda m: VLMResult(provider="qwen3_vl",
                                              model_name="qwen3_vl",
                                              error="vllm unreachable"))
    manifest = EvidenceManifest(case_id="c", camera_id="cam_01",
                                 window_start_ts="x", window_end_ts="y",
                                 frames=[])
    r = provider.analyze_evidence(manifest)
    assert r.error is not None
    assert "vllm unreachable" in r.error


def test_qwen_status_reports_vllm_health(monkeypatch):
    """app.startup.qwen_vllm_status surfaces a structured health dict
    for the ops/status aggregator — used by the reviewer UI to render
    'Qwen unavailable' instead of guessing from a silent failure."""
    from app.startup import qwen_vllm_status
    cfg = SimpleNamespace(
        raw={"reasoning": {"primary_provider": "qwen3_vl"}},
        models={"qwen3_vl": _qwen_cfg()},
    )
    # Don't actually hit a real Qwen server in CI — assert the
    # function returns the structured shape regardless of outcome.
    out = qwen_vllm_status(cfg)
    assert "enabled" in out
    assert "backend" in out
    assert "healthy" in out
    assert "detail" in out
    assert out["enabled"] is True
    assert out["backend"] == "vllm_openai"


# ---------------------------------------------------------------------------
# 3. Qwen launcher present + has the working flags
# ---------------------------------------------------------------------------

QWEN_LAUNCHER = ROOT / "scripts" / "qwen_vllm_start.sh"


def test_qwen_launcher_exists_and_is_executable():
    assert QWEN_LAUNCHER.is_file()
    assert os.access(QWEN_LAUNCHER, os.X_OK), \
        f"{QWEN_LAUNCHER} should be chmod +x"


def test_qwen_launcher_uses_the_known_good_flags():
    """The flag set this DGX (GB10 / sm_121) needs. The launcher MUST
    pass these by default; operator env vars exist to override only
    after re-verifying a different flag set works."""
    src = QWEN_LAUNCHER.read_text()
    # The flags appear as defaults the launcher embeds when building
    # the vllm serve argv.
    assert '--moe-backend' in src
    assert 'QWEN_MOE_BACKEND:-triton' in src
    assert '--enforce-eager' in src
    assert 'QWEN_ENFORCE_EAGER:-1' in src
    assert '--no-enable-flashinfer-autotune' in src
    assert 'QWEN_DISABLE_FLASHINFER_AUTOTUNE:-1' in src


def test_qwen_launcher_defaults_max_model_len_to_65536():
    """vLLM's Qwen3-VL default 262144 needs ~24 GiB of KV cache, which
    won't fit after weights load on this DGX Spark. 65536 is the
    largest value that fits in the ~22 GiB free; it's the operator
    default. Operators can override with QWEN_MAX_MODEL_LEN=N."""
    src = QWEN_LAUNCHER.read_text()
    # Default literal must be embedded in the launcher
    assert 'QWEN_MAX_MODEL_LEN:-65536' in src, (
        "qwen_vllm_start.sh must default QWEN_MAX_MODEL_LEN to 65536")
    # The flag still has to actually be appended to the vllm serve argv
    assert '--max-model-len' in src
    # Override path is preserved: the default is wrapped in the
    # ${VAR:-DEFAULT} idiom so an explicit export still wins.
    assert 'QWEN_MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-65536}"' in src


def test_qwen_launcher_emits_max_model_len_in_default_invocation(tmp_path,
                                                                  monkeypatch):
    """Smoke-run the launcher with a stub ``vllm`` on PATH and verify
    the resolved argv contains ``--max-model-len 65536``. Catches
    regressions where the default literal is updated but the conditional
    that appends the flag is broken (or vice-versa)."""
    import shutil
    import subprocess
    import textwrap

    # Stub `vllm`: print the argv to a file and exit 0 immediately so
    # the launcher's health gate sees the process exit and we skip the
    # long wait loop.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    argv_log = tmp_path / "vllm_argv.txt"
    vllm_stub = bin_dir / "vllm"
    vllm_stub.write_text(textwrap.dedent(f"""\
        #!/usr/bin/env bash
        printf '%s\\n' "$@" > {argv_log}
        # exit 1 so the launcher does NOT linger in the health loop.
        exit 1
    """))
    vllm_stub.chmod(0o755)

    # Stub `curl` to always fail health (so launcher doesn't think the
    # already-running probe passed).
    curl_stub = bin_dir / "curl"
    curl_stub.write_text("#!/usr/bin/env bash\nexit 22\n")
    curl_stub.chmod(0o755)

    # Run launcher with PATH pointing at our stubs first.
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env.get('PATH','')}"
    env["QWEN_HEALTH_TIMEOUT_SEC"] = "2"
    env["QWEN_LOG"] = str(tmp_path / "qwen.log")
    env["QWEN_PID_FILE"] = str(tmp_path / "qwen.pid")
    # No venv in tmp_path → launcher will exit 1. Create a stub one.
    (ROOT / "venv").mkdir(exist_ok=True)
    # Run from a temp checkout to avoid mutating the real run/logs.
    res = subprocess.run(["bash", str(QWEN_LAUNCHER)],
                          env=env, capture_output=True, text=True,
                          timeout=30)
    # We don't care about the exit code (stub vllm exits non-zero so
    # launcher reports failure) — only that argv was captured before
    # exit and contains the max-model-len flag.
    assert argv_log.exists(), (
        f"vllm stub never invoked; launcher stderr:\n{res.stderr}")
    captured = argv_log.read_text().splitlines()
    assert "--max-model-len" in captured, (
        f"launcher did not pass --max-model-len; argv={captured}")
    i = captured.index("--max-model-len")
    assert captured[i + 1] == "65536", (
        f"default --max-model-len value should be 65536; got "
        f"{captured[i+1]!r} (full argv={captured})")


def test_qwen_launcher_max_model_len_override_wins(tmp_path):
    """An operator export of QWEN_MAX_MODEL_LEN replaces the 65536
    default."""
    import subprocess
    import textwrap

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    argv_log = tmp_path / "vllm_argv.txt"
    vllm_stub = bin_dir / "vllm"
    vllm_stub.write_text(textwrap.dedent(f"""\
        #!/usr/bin/env bash
        printf '%s\\n' "$@" > {argv_log}
        exit 1
    """))
    vllm_stub.chmod(0o755)
    curl_stub = bin_dir / "curl"
    curl_stub.write_text("#!/usr/bin/env bash\nexit 22\n")
    curl_stub.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env.get('PATH','')}"
    env["QWEN_HEALTH_TIMEOUT_SEC"] = "2"
    env["QWEN_LOG"] = str(tmp_path / "qwen.log")
    env["QWEN_PID_FILE"] = str(tmp_path / "qwen.pid")
    env["QWEN_MAX_MODEL_LEN"] = "32768"
    (ROOT / "venv").mkdir(exist_ok=True)
    subprocess.run(["bash", str(QWEN_LAUNCHER)],
                    env=env, capture_output=True, text=True, timeout=30)
    captured = argv_log.read_text().splitlines()
    i = captured.index("--max-model-len")
    assert captured[i + 1] == "32768"


def test_qwen_launcher_two_stage_health_gate():
    """/health is liveness; /v1/models is readiness. Both required."""
    src = QWEN_LAUNCHER.read_text()
    assert "/health" in src
    assert "/v1/models" in src
    # Health gate uses BOTH probes with && (single curl chain).
    assert "is_healthy()" in src


def test_qwen_launcher_refuses_to_double_launch():
    """Stale PID file with a running process → exit 2, don't start a
    second server. (Council instruction: do not double-launch.)"""
    src = QWEN_LAUNCHER.read_text()
    assert "NOT starting a duplicate" in src
    assert "exit 2" in src


# ---------------------------------------------------------------------------
# 4. start.sh defaults — Qwen ON, Gemma OFF
# ---------------------------------------------------------------------------

START_SH = ROOT / "start.sh"


def test_start_sh_defaults_qwen_on_gemma_off():
    src = START_SH.read_text()
    # Explicit defaults
    assert 'START_QWEN="${START_QWEN:-1}"' in src
    assert 'START_GEMMA="${START_GEMMA:-0}"' in src
    assert 'START_PHOENIX="${START_PHOENIX:-0}"' in src
    # Calls the Qwen launcher when START_QWEN=1
    assert "scripts/qwen_vllm_start.sh" in src
    # Calls the Gemma launcher ONLY when START_GEMMA=1
    assert 'if [[ "$START_GEMMA" == "1" ]]; then' in src


def test_start_sh_prints_reviewer_ui_url_on_success():
    src = START_SH.read_text()
    assert "/static/review.html" in src
    assert "/api/v1/docs" in src


def test_stop_sh_stops_qwen_and_gemma():
    src = (ROOT / "stop.sh").read_text()
    assert 'stop_pid "qwen"' in src
    assert 'stop_pid "gemma"' in src


# ---------------------------------------------------------------------------
# 5. STARTUP.md exists and pins the working flag set
# ---------------------------------------------------------------------------

STARTUP_MD = ROOT / "docs" / "STARTUP.md"


def test_startup_doc_exists_and_documents_known_good_flags():
    assert STARTUP_MD.is_file()
    text = STARTUP_MD.read_text()
    for flag in ("--moe-backend triton",
                  "--enforce-eager",
                  "--no-enable-flashinfer-autotune"):
        assert flag in text, f"STARTUP.md missing flag {flag!r}"
    # Health gates documented
    assert "/v1/models" in text
    assert "/api/v1/health" in text
    assert "/static/review.html" in text
    # Default policy documented
    assert "Qwen-only" in text or "fallback_provider: null" in text
    # QWEN_MAX_MODEL_LEN default documented (including the KV-cache
    # explanation that motivates the 65536 value).
    assert "QWEN_MAX_MODEL_LEN" in text
    assert "65536" in text
    assert "KV cache" in text
