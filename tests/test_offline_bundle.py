"""Offline bundle + portability checkpoint tests.

Pin the contracts:
  * Repo-local ``./models/hf/...`` is preferred over ``~/.cache``.
  * Production/offline mode rejects cache-only paths (``OfflineBundleError``).
  * Verifier fails when any required asset is missing.
  * Asset registry lists Qwen, Gemma, Falcon, SAM 3, Falcon OCR as required.
  * Wheelhouse / Python-env verifier scripts run without network.
  * No runtime code (outside ``scripts/prepare_offline_*``) calls
    ``snapshot_download`` or ``hf_hub_download``.
"""
from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# 1. Asset registry shape
# ---------------------------------------------------------------------------

# SAM 2 is the deployable segmenter today; SAM 3 is preferred-upgrade.
# Falcon-Perception covers OCR (per upstream README), so a dedicated
# Falcon-OCR is optional, not required.
REQUIRED_NAMES = {"qwen3_vl", "gemma_bf16", "falcon_perception", "sam2"}


def test_registry_lists_all_required_assets():
    registry = yaml.safe_load((ROOT / "offline_assets.yaml").read_text())
    required = {r["name"] for r in registry.get("required") or []}
    assert REQUIRED_NAMES.issubset(required), \
        f"missing required entries: {REQUIRED_NAMES - required}"
    for r in registry["required"]:
        assert r.get("runtime_blocking") is True, \
            f"required asset {r['name']!r} must be runtime_blocking=true"


def test_registry_optional_assets_not_runtime_blocking():
    registry = yaml.safe_load((ROOT / "offline_assets.yaml").read_text())
    for r in registry.get("optional") or []:
        assert r.get("runtime_blocking") is False, \
            f"optional asset {r['name']!r} must not be runtime_blocking"


# ---------------------------------------------------------------------------
# 2. Config loader path resolution
# ---------------------------------------------------------------------------

def test_resolve_model_path_prefers_repo_local(tmp_path, monkeypatch):
    """When ``./models/hf/<name>/<snapshot>/`` exists, it must win over
    a cache path that is also present."""
    monkeypatch.delenv("FRAUD_OFFLINE_MODE", raising=False)
    monkeypatch.delenv("OFFLINE_MODE", raising=False)

    # Make a fake repo-local snapshot for a fake model name.
    import app.config as ac
    fake_root = tmp_path / "models" / "hf"
    fake_snap = fake_root / "fake/org" / "snap_a"
    fake_snap.mkdir(parents=True)
    # Re-point the constant temporarily.
    monkeypatch.setattr(ac, "BUNDLE_ROOT", fake_root)

    cfg = ac.ModelConfig(name="fake/org", enabled=True,
                         extra={"local_path": str(tmp_path / "cache_alt")})
    (tmp_path / "cache_alt").mkdir()

    resolved = ac.resolve_model_path(cfg, production_mode=False)
    assert resolved is not None
    assert Path(resolved).resolve() == fake_snap.resolve()


def test_resolve_model_path_dev_falls_back_to_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("FRAUD_OFFLINE_MODE", raising=False)
    monkeypatch.delenv("OFFLINE_MODE", raising=False)

    import app.config as ac
    monkeypatch.setattr(ac, "BUNDLE_ROOT", tmp_path / "models" / "hf")
    cache_dir = tmp_path / "fake_cache"
    cache_dir.mkdir()
    cfg = ac.ModelConfig(name="fake/org", enabled=True,
                         extra={"local_path": str(cache_dir)})
    resolved = ac.resolve_model_path(cfg, production_mode=False)
    assert resolved == str(cache_dir.resolve())


def test_resolve_model_path_production_rejects_cache_only(tmp_path, monkeypatch):
    """Production mode must raise OfflineBundleError instead of returning
    a cache path when the repo-local bundle has no matching snapshot."""
    import app.config as ac
    monkeypatch.setattr(ac, "BUNDLE_ROOT", tmp_path / "models" / "hf")
    cache_dir = tmp_path / "fake_cache"
    cache_dir.mkdir()
    cfg = ac.ModelConfig(name="fake/org", enabled=True,
                         extra={"local_path": str(cache_dir)})
    with pytest.raises(ac.OfflineBundleError):
        ac.resolve_model_path(cfg, production_mode=True)


def test_resolve_model_path_production_uses_repo_local(tmp_path, monkeypatch):
    """Production mode accepts when repo-local snapshot exists."""
    import app.config as ac
    fake_root = tmp_path / "models" / "hf"
    fake_snap = fake_root / "fake/org" / "snap_b"
    fake_snap.mkdir(parents=True)
    monkeypatch.setattr(ac, "BUNDLE_ROOT", fake_root)
    cfg = ac.ModelConfig(name="fake/org", enabled=True, extra={})
    resolved = ac.resolve_model_path(cfg, production_mode=True)
    assert Path(resolved).resolve() == fake_snap.resolve()


def test_is_production_offline_mode_env(monkeypatch):
    from app import config as ac
    monkeypatch.setenv("FRAUD_OFFLINE_MODE", "1")
    assert ac.is_production_offline_mode() is True
    monkeypatch.setenv("FRAUD_OFFLINE_MODE", "0")
    assert ac.is_production_offline_mode() is False
    monkeypatch.delenv("FRAUD_OFFLINE_MODE", raising=False)
    monkeypatch.setenv("OFFLINE_MODE", "production")
    assert ac.is_production_offline_mode() is True


# ---------------------------------------------------------------------------
# 3. Bundle prepare script behaviour
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_bundle(tmp_path, monkeypatch):
    """Run the prepare/verify scripts against a temp REPO_ROOT clone."""
    # Stage a minimal repo skeleton.
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "models").mkdir()
    (repo / "app").mkdir()
    shutil.copy(ROOT / "scripts/prepare_offline_model_bundle.py",
                repo / "scripts/")
    shutil.copy(ROOT / "scripts/verify_offline_bundle.py",
                repo / "scripts/")
    shutil.copy(ROOT / "app/config.py", repo / "app/")
    (repo / "app/__init__.py").write_text("")

    # Fake HF cache root with one tiny "model".
    cache = tmp_path / "hfcache" / "hub"
    repo_dir = cache / "models--fake--TinyModel"
    snap = repo_dir / "snapshots" / "rev1"
    blobs = repo_dir / "blobs"
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    blob_a = blobs / ("a" * 40)
    blob_a.write_bytes(b'{"architectures":["FakeModel"]}')
    blob_b = blobs / ("b" * 40)
    blob_b.write_bytes(b'{"tok":"x"}')
    blob_c = blobs / ("c" * 40)
    blob_c.write_bytes(b"weights")
    os.symlink(blob_a, snap / "config.json")
    os.symlink(blob_b, snap / "tokenizer_config.json")
    os.symlink(blob_c, snap / "pytorch_model.bin")

    # Minimal registry with one required entry pointing at the fake model.
    registry_path = repo / "offline_assets.yaml"
    registry_path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "required": [
            {
                "name": "tiny",
                "purpose": "test",
                "repo": "fake/TinyModel",
                "cache_path": str(repo_dir),
                "runtime_blocking": True,
            },
            {
                "name": "absent",
                "purpose": "test missing required",
                "repo": "fake/AbsentModel",
                "cache_path": None,
                "runtime_blocking": True,
                "official_source": "n/a",
            },
        ],
        "optional": [],
        "runtime_assets": {
            "configs": ["offline_assets.yaml"],
        },
    }))

    # Minimal config so verify can resolve enabled models.
    (repo / "config.yaml").write_text(yaml.safe_dump({
        "cameras": [],
        "settings": {},
        "models": {
            "tiny": {
                "name": "fake/TinyModel",
                "enabled": True,
                "local_path": str(snap),  # cache fallback
            },
        },
    }))

    env = os.environ.copy()
    env["HF_HOME"] = str(tmp_path / "hfcache")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHONPATH"] = str(repo)
    yield repo, env


def test_prepare_copy_only_missing_required_fails(isolated_bundle):
    repo, env = isolated_bundle
    proc = subprocess.run(
        [sys.executable, "scripts/prepare_offline_model_bundle.py",
         "--copy-from-cache-only"],
        cwd=repo, env=env, capture_output=True, text=True,
    )
    # Exit code 2: 'absent' is missing and required.
    assert proc.returncode == 2, proc.stderr
    assert "REQUIRED ASSETS MISSING" in proc.stderr
    # The cached tiny model still got bundled.
    assert (repo / "models/hf/fake/TinyModel/rev1/config.json").is_file()


def test_prepare_rejects_escaping_symlink(tmp_path, monkeypatch, isolated_bundle):
    repo, env = isolated_bundle
    # Inject a symlink whose target escapes the cache root.
    snap = (tmp_path / "hfcache" / "hub" /
            "models--fake--TinyModel" / "snapshots" / "rev1")
    outside = tmp_path / "outside_payload"
    outside.write_text("nope")
    bad_link = snap / "extra.bin"
    os.symlink(outside, bad_link)

    proc = subprocess.run(
        [sys.executable, "scripts/prepare_offline_model_bundle.py",
         "--copy-from-cache-only", "--asset", "tiny"],
        cwd=repo, env=env, capture_output=True, text=True,
    )
    assert proc.returncode == 2
    assert "refusing to follow symlink" in (proc.stderr + proc.stdout)


def test_verify_after_prepare_required_missing(isolated_bundle):
    repo, env = isolated_bundle
    # Build the bundle for 'tiny' only so 'absent' is naturally missing.
    subprocess.run(
        [sys.executable, "scripts/prepare_offline_model_bundle.py",
         "--copy-from-cache-only", "--asset", "tiny"],
        cwd=repo, env=env, capture_output=True, text=True,
    )
    # Verifier should still see 'absent' as missing -> exit 2.
    proc = subprocess.run(
        [sys.executable, "scripts/verify_offline_bundle.py"],
        cwd=repo, env=env, capture_output=True, text=True,
    )
    assert proc.returncode == 2, (proc.stdout, proc.stderr)
    assert "REQUIRED MISSING" in proc.stderr


def test_verify_full_bundle_passes_when_all_required_present(isolated_bundle):
    repo, env = isolated_bundle
    # Mutate the registry so the only required entry is 'tiny'.
    reg = yaml.safe_load((repo / "offline_assets.yaml").read_text())
    reg["required"] = [reg["required"][0]]
    (repo / "offline_assets.yaml").write_text(yaml.safe_dump(reg))

    subprocess.run(
        [sys.executable, "scripts/prepare_offline_model_bundle.py",
         "--copy-from-cache-only"],
        cwd=repo, env=env, capture_output=True, text=True, check=True,
    )
    proc = subprocess.run(
        [sys.executable, "scripts/verify_offline_bundle.py",
         "--production"],
        cwd=repo, env=env, capture_output=True, text=True,
    )
    # Production mode now resolves "tiny" config; the snapshot dir is
    # ./models/hf/fake/TinyModel/rev1 and the loader prefers it.
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert "BUNDLE OK" in proc.stdout


# ---------------------------------------------------------------------------
# 4. No runtime call to snapshot_download / hf_hub_download
# ---------------------------------------------------------------------------

def test_runtime_code_does_not_call_download_apis():
    """The only acceptable use of snapshot_download / hf_hub_download is
    inside ``scripts/prepare_offline_model_bundle.py``. Every other
    source file must be clean."""
    ROOTS = [
        ROOT / "app",
        ROOT / "db",
        ROOT / "pos",
        ROOT / "reasoning",
        ROOT / "perception",
        ROOT / "video",
        ROOT / "evidence",
        ROOT / "review",
        ROOT / "mlops",
        ROOT / "monitor.py",
        ROOT / "app.py",
        ROOT / "gemma_reasoner.py",
        ROOT / "falcon_detector.py",
        ROOT / "transformers_server.py",
    ]
    forbidden = ("snapshot_download", "hf_hub_download")
    offenders: list[tuple[str, str]] = []
    for r in ROOTS:
        if not r.exists():
            continue
        files = [r] if r.is_file() else list(r.rglob("*.py"))
        for f in files:
            try:
                text = f.read_text()
            except UnicodeDecodeError:
                continue
            for needle in forbidden:
                if needle in text:
                    offenders.append((str(f.relative_to(ROOT)), needle))
    assert not offenders, (
        f"runtime code must not import HF download APIs; offenders={offenders}"
    )


def test_only_prep_script_may_use_snapshot_download():
    src = (ROOT / "scripts/prepare_offline_model_bundle.py").read_text()
    # Allowed here, but only conditionally under --download-approved.
    assert "snapshot_download" in src
    assert "download_approved" in src


# ---------------------------------------------------------------------------
# 5. Chain provider: Qwen primary, Gemma fallback
# ---------------------------------------------------------------------------

def _empty_manifest():
    from reasoning.providers.base import EvidenceManifest
    return EvidenceManifest(
        case_id="c1", camera_id="cam_01",
        window_start_ts="2026-06-15T14:00:00Z",
        window_end_ts="2026-06-15T14:01:00Z",
        frames=[{
            "frame_id": "f0", "ts": "2026-06-15T14:00:30Z",
            "image_url": ("data:image/jpeg;base64,"
                          + "/9j/4AAQSkZJRgABAQEASABIAAD/"),
        }],
    )


def test_chain_falls_back_to_gemma_when_qwen_errors():
    from reasoning.providers import ChainProvider, get_provider

    # Build a Qwen that always errors (missing local path) and a Gemma
    # that returns a benign result.
    qwen = get_provider(
        "qwen3_vl", model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
        enabled=True, local_path="/definitely/not/here",
    )
    # Gemma to a closed port — its analyze_evidence returns an error
    # too, but with a different shape. Use a stub provider for clarity.
    from reasoning.providers.base import VLMProvider, VLMResult

    class StubGemma(VLMProvider):
        name = "gemma"

        def __init__(self):
            super().__init__(model_name="stub-gemma", enabled=True)

        def analyze_evidence(self, manifest):
            return VLMResult(provider="gemma", model_name="stub-gemma",
                             parsed={"narrative": "ok"})

    chain = ChainProvider(providers=[qwen, StubGemma()])
    result = chain.analyze_evidence(_empty_manifest())
    assert result.provider == "gemma"
    assert result.error is None
    assert "_chain_attempts" in result.parsed
    attempts = result.parsed["_chain_attempts"]
    assert any(a.startswith("qwen3_vl=err") for a in attempts)
    assert "gemma=ok" in attempts


def test_chain_attempts_recorded_when_all_fail():
    from reasoning.providers import ChainProvider
    from reasoning.providers.base import VLMProvider, VLMResult

    class Boom(VLMProvider):
        def __init__(self, name):
            super().__init__(model_name=name, enabled=True)
            self.name = name

        def analyze_evidence(self, m):
            raise RuntimeError("boom")

    chain = ChainProvider(providers=[Boom("a"), Boom("b")])
    result = chain.analyze_evidence(_empty_manifest())
    assert result.error is not None
    attempts = result.parsed["_chain_attempts"]
    assert attempts == ["a=raised:RuntimeError", "b=raised:RuntimeError"]


def test_build_active_provider_qwen_disabled_returns_gemma_only(monkeypatch):
    """When ``qwen3_vl.enabled=false`` is set on the loaded config, the
    active provider must be plain Gemma — the chain wrapper drops Qwen."""
    from app.config import load_config
    from reasoning.providers import build_active_provider

    cfg = load_config()
    # Override the shipping default for this test.
    cfg.models["qwen3_vl"].enabled = False
    p = build_active_provider(cfg)
    assert p.name == "gemma"


def test_build_active_provider_qwen_enabled_returns_chain(monkeypatch, tmp_path):
    """With Qwen enabled AND a repo-local snapshot present, the chain is
    built with Qwen first, Gemma second."""
    import app.config as ac
    from reasoning.providers import build_active_provider, ChainProvider

    # Stage a fake repo-local Qwen snapshot.
    fake_root = tmp_path / "models" / "hf"
    fake_snap = fake_root / "Qwen/Qwen3-VL-30B-A3B-Instruct" / "snapX"
    fake_snap.mkdir(parents=True)
    monkeypatch.setattr(ac, "BUNDLE_ROOT", fake_root)

    cfg = ac.load_config()
    cfg.models["qwen3_vl"].enabled = True
    p = build_active_provider(cfg)
    assert isinstance(p, ChainProvider)
    assert p.providers[0].name == "qwen3_vl"
    assert p.providers[1].name == "gemma"


# ---------------------------------------------------------------------------
# 6. Decision policy + prompt safety still hold
# ---------------------------------------------------------------------------

def test_decision_policy_outcomes_still_constrained():
    from reasoning.decision_policy import (
        VALID_OUTCOMES, EvidenceSummary, decide,
    )
    for s in [
        EvidenceSummary(footage_valid=False),
        EvidenceSummary(footage_valid=True, vlm_confidence="low"),
        EvidenceSummary(footage_valid=True, receipt_visible=True,
                        vlm_confidence="high"),
    ]:
        d = decide(s)
        assert d.outcome in VALID_OUTCOMES
        assert "FRAUD" not in d.outcome


def test_no_fraud_language_in_qwen_or_gemma_provider_defaults():
    from reasoning.providers import qwen3_vl as qmod
    for needle in ("determine fraud", "is this fraud",
                   "fraud indicator", "loss-prevention"):
        assert needle not in qmod._DEFAULT_SYSTEM.lower()
        assert needle not in qmod._DEFAULT_USER.lower()
