"""Startup integrity checks must fail fast in production when required
assets are missing, and must succeed in dev mode even without them."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_startup_dev_mode_tolerates_missing_bundle(monkeypatch, tmp_path):
    monkeypatch.delenv("FRAUD_OFFLINE_MODE", raising=False)
    monkeypatch.delenv("OFFLINE_MODE", raising=False)
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    import app.config as ac
    monkeypatch.setattr(ac, "BUNDLE_ROOT", tmp_path / "nope" / "hf")

    from app.startup import run_startup_checks
    summary = run_startup_checks(strict=False)
    assert summary["production"] is False


def test_startup_production_fails_when_required_asset_missing(monkeypatch,
                                                              tmp_path):
    monkeypatch.setenv("FRAUD_OFFLINE_MODE", "1")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    # Point BUNDLE_ROOT at an empty dir so every required asset is
    # missing.
    import app.config as ac
    monkeypatch.setattr(ac, "BUNDLE_ROOT", tmp_path / "empty" / "hf")

    from app.startup import StartupCheckError, run_startup_checks
    with pytest.raises(StartupCheckError) as exc:
        run_startup_checks()
    msg = str(exc.value).lower()
    assert "qwen3-vl" in msg.lower() or "qwen" in msg.lower()
    assert "sam2-hiera-large" in msg.lower() or "sam2" in msg.lower()


def test_startup_production_passes_against_real_bundle(monkeypatch, tmp_path):
    """If the actual bundle on this DGX is present (post-prep) AND
    the sam2 runtime is installed, startup in production mode
    succeeds. Skip when either is missing."""
    import importlib.util
    monkeypatch.setenv("FRAUD_OFFLINE_MODE", "1")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    bundle = ROOT / "models" / "hf"
    required = ("Qwen/Qwen3-VL-30B-A3B-Instruct",
                "google/gemma-4-26B-A4B-it",
                "tiiuae/Falcon-Perception",
                "facebook/sam2-hiera-large")
    for rel in required:
        if not (bundle.joinpath(*rel.split("/")).is_dir()):
            pytest.skip(f"bundle asset missing: {rel}; run prepare first")
    if importlib.util.find_spec("sam2") is None:
        pytest.skip("sam2 runtime not installed; install via wheelhouse")
    from app.startup import run_startup_checks
    summary = run_startup_checks()
    assert summary["production"] is True
    assert summary["issues"] == []
    assert "qwen3_vl" in summary["provider_chain"]["members"]
    assert "gemma" in summary["provider_chain"]["members"]
