"""SAM2 runtime checks.

Pins:

* startup in production mode fails fast when ``sam2`` cannot be
  imported, even when the weights bundle is present;
* SAM2 client correctly reports unavailability when the package is
  missing;
* verify_offline_python_env reports the package as missing-runtime;
* config.yaml carries a models.sam2 entry resolving to the bundled
  snapshot.
"""
from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_config_has_sam2_entry_resolving_to_bundle():
    from app.config import ModelConfig, load_config, resolve_model_path
    cfg = load_config()
    assert "sam2" in cfg.models
    sam2_cfg = cfg.models["sam2"]
    bundle = ROOT / "models" / "hf" / "facebook" / "sam2-hiera-large"
    if not bundle.is_dir():
        pytest.skip("sam2 weights bundle missing; run prepare script")
    path = resolve_model_path(sam2_cfg)
    assert path and Path(path).is_dir()


def test_sam2_client_marks_unavailable_when_package_missing():
    """When the ``sam2`` package isn't installed, the client must not
    pretend it can segment. ``has_capability`` returns False AND the
    load_err string explains the reason."""
    if importlib.util.find_spec("sam2") is not None:
        pytest.skip("sam2 package installed; cannot exercise missing path")
    from perception.sam2_client import Sam2Client
    client = Sam2Client(model_path=str(ROOT / "models" / "hf"
                                        / "facebook" / "sam2-hiera-large"))
    assert client.has_capability() is False
    assert "sam2 package not installed" in (client._load_err or "")


def test_production_startup_fails_when_sam2_missing(monkeypatch, tmp_path):
    """Even with weights bundled, if the sam2 runtime can't import,
    production startup raises StartupCheckError."""
    if importlib.util.find_spec("sam2") is not None:
        pytest.skip("sam2 already installed; missing-runtime branch covered "
                    "in CI when env lacks the wheelhouse install")
    monkeypatch.setenv("FRAUD_OFFLINE_MODE", "1")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    # Provider chain build doesn't depend on sam2 — exercise only the
    # startup check.
    from app.startup import StartupCheckError, run_startup_checks
    with pytest.raises(StartupCheckError) as exc:
        run_startup_checks()
    msg = str(exc.value).lower()
    assert "sam2" in msg


def test_verify_python_env_flags_required_runtime_missing(monkeypatch,
                                                          tmp_path):
    """The verifier script must call out missing required-runtime
    packages by name when ``sam2`` isn't installed."""
    if importlib.util.find_spec("sam2") is not None:
        pytest.skip("sam2 already installed")
    proc = subprocess.run(
        [sys.executable, "scripts/verify_offline_python_env.py"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    assert proc.returncode == 1
    assert "sam2" in (proc.stderr + proc.stdout).lower()
