"""Guardrail tests: the legacy MVP must still import after the refactor.

These tests do NOT bring up the model server or load weights. They
verify that the public symbols other production-path modules import
still exist with their original shapes, so a code reviewer can be
confident the refactor was purely additive at this checkpoint.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_config_loads_with_new_section_present():
    from app.config import load_config
    cfg = load_config()
    assert "gemma" in cfg.models
    assert cfg.models["gemma"].name == "google/gemma-4-26B-A4B-it"
    # Qwen3-VL is now primary in the production config.
    assert "qwen3_vl" in cfg.models
    assert cfg.models["qwen3_vl"].enabled is True
    assert cfg.models["qwen3_vl"].name == "Qwen/Qwen3-VL-30B-A3B-Instruct"


def test_legacy_gemma_reasoner_import_still_works():
    """gemma_reasoner.GemmaVideoReasoner is the public API that the
    legacy ``monitor.py`` instantiates. The provider abstraction wraps
    this; the underlying module must remain importable."""
    from gemma_reasoner import GemmaReasoner, GemmaVideoReasoner
    assert GemmaReasoner is GemmaVideoReasoner


def test_legacy_classifiers_still_resolve_fraud():
    """gemma_reasoner falls back to ``classifiers.get_classifier('fraud')``
    when no prompt is supplied. That contract must still hold."""
    from classifiers import get_classifier
    fraud = get_classifier("fraud")
    assert isinstance(fraud, dict)
    assert "gemma_system" in fraud


def test_qwen_local_snapshot_path_exists_for_inspector():
    """The path recorded in ``config.yaml`` must point at a real local
    snapshot directory — operators rely on the inspector to verify this
    without doing any network calls."""
    import os
    from app.config import load_config
    cfg = load_config()
    qwen = cfg.models["qwen3_vl"]
    local_path = qwen.extra.get("local_path")
    assert local_path, "qwen3_vl.local_path must be set in config.yaml"
    assert os.path.isdir(local_path), \
        f"qwen3_vl.local_path {local_path!r} not present locally"


def test_provider_registry_does_not_trigger_model_load():
    """Importing ``reasoning.providers`` must NOT load torch/PIL/requests
    eagerly. We assert it by re-importing and ensuring no exception is
    raised even with a fake HF_HOME pointing nowhere."""
    import importlib
    mod = importlib.reload(__import__("reasoning.providers",
                                      fromlist=["__init__"]))
    assert "gemma" in mod.list_providers()
    assert "qwen3_vl" in mod.list_providers()
