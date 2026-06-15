"""Smoke tests for the reasoning provider abstraction.

These tests must run WITHOUT loading any model weights, opening any
HTTP connection, or touching torch. They verify:

* The registry exposes both built-in providers.
* Qwen3-VL provider correctly detects presence/absence of the local
  snapshot and refuses to auto-enable.
* Both providers return a structured ``VLMResult`` (with a meaningful
  ``error`` field) when called offline, instead of raising.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasoning import providers  # noqa: E402
from reasoning.providers.base import EvidenceManifest  # noqa: E402


def _empty_manifest() -> EvidenceManifest:
    return EvidenceManifest(
        case_id="case-0",
        camera_id="cam_01",
        window_start_ts="2026-06-15T14:00:00Z",
        window_end_ts="2026-06-15T14:01:00Z",
    )


def test_registry_lists_both_builtins():
    names = providers.list_providers()
    assert "gemma" in names
    assert "qwen3_vl" in names


def test_gemma_provider_no_frames_returns_error_not_raise():
    p = providers.get_provider(
        "gemma",
        model_name="google/gemma-4-26B-A4B-it",
        enabled=True,
        vllm_url="http://127.0.0.1:1",  # closed port; only reached if frames present
    )
    result = p.analyze_evidence(_empty_manifest())
    assert result.provider == "gemma"
    assert result.error is not None
    assert "no frames" in result.error.lower()


def test_qwen3_disabled_by_default():
    p = providers.get_provider(
        "qwen3_vl",
        model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
        local_path="/nonexistent/path",
        enabled=False,
    )
    assert p.enabled is False
    result = p.analyze_evidence(_empty_manifest())
    assert result.error == "provider disabled"
    health = p.health()
    assert health.healthy is False


def test_qwen3_detects_missing_local_path_when_enabled():
    p = providers.get_provider(
        "qwen3_vl",
        model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
        local_path="/definitely/not/here",
        enabled=True,
    )
    assert p.has_local_weights() is False
    result = p.analyze_evidence(_empty_manifest())
    assert result.error is not None
    assert "local weights missing" in result.error


def test_qwen3_detects_real_local_path(tmp_path):
    # Create a directory that looks like a snapshot folder; we only
    # check existence, not contents.
    snapshot = tmp_path / "snapshots" / "fake_rev"
    snapshot.mkdir(parents=True)
    p = providers.get_provider(
        "qwen3_vl",
        model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
        local_path=str(snapshot),
        enabled=True,
    )
    assert p.has_local_weights() is True
    health = p.health()
    assert health.healthy is True


def test_get_provider_unknown_name_raises():
    import pytest
    with pytest.raises(KeyError):
        providers.get_provider("does_not_exist")
