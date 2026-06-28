"""Regression tests for app.config._repo_local_snapshot.

The earlier implementation mis-resolved facebook/sam3 to
``models/hf/facebook/sam3/.cache`` because it picked the largest
subdir and ``.cache`` was the only one (model files were directly
under sam3/ thanks to snapshot_download(local_dir=...)).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _layout_a(tmp_path: Path) -> Path:
    """HF cache layout: <org>/<name>/<snapshot_hash>/<files>."""
    base = tmp_path / "models" / "hf" / "facebook" / "sam2-hiera-large"
    snap = base / "deadbeef1234567890"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "model.safetensors").write_bytes(b"\x00" * 1024)
    return base


def _layout_b_with_cache_sibling(tmp_path: Path) -> Path:
    """snapshot_download(local_dir=...) layout: <org>/<name>/<files>,
    with a sneaky .cache sibling dir.  The pre-fix resolver picked
    .cache here."""
    base = tmp_path / "models" / "hf" / "facebook" / "sam3"
    base.mkdir(parents=True)
    (base / "config.json").write_text("{}")
    (base / "model.safetensors").write_bytes(b"\x00" * 1024)
    cache = base / ".cache"
    cache.mkdir()
    (cache / "huggingface_metadata.txt").write_text("garbage")
    return base


def _layout_b_no_safetensors(tmp_path: Path) -> Path:
    """Layout B with only config.json present (still valid HF dir)."""
    base = tmp_path / "models" / "hf" / "facebook" / "model_no_st"
    base.mkdir(parents=True)
    (base / "config.json").write_text("{}")
    (base / "model-00001-of-00002.bin").write_bytes(b"\x00")  # sharded
    return base


def _patch_bundle_root(monkeypatch, tmp_path: Path) -> None:
    """Repoint app.config.BUNDLE_ROOT at our temp tree without
    relying on import-time side effects."""
    import app.config as cfg_mod
    new_root = tmp_path / "models" / "hf"
    monkeypatch.setattr(cfg_mod, "BUNDLE_ROOT", new_root)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def test_layout_b_returns_base_dir_not_cache_subdir(tmp_path, monkeypatch):
    """The bug. snapshot_download(local_dir=...) produced
    sam3/{config.json, model.safetensors, .cache/}. The resolver
    must return sam3, NOT sam3/.cache."""
    _layout_b_with_cache_sibling(tmp_path)
    _patch_bundle_root(monkeypatch, tmp_path)

    from app.config import _repo_local_snapshot
    out = _repo_local_snapshot("facebook/sam3")
    assert out is not None
    p = Path(out)
    assert p.name == "sam3", f"resolver returned {p}, expected sam3"
    assert (p / "config.json").exists(), \
        "resolved path must point at the dir holding config.json"
    assert ".cache" not in p.parts


def test_layout_b_with_only_config_json_works(tmp_path, monkeypatch):
    _layout_b_no_safetensors(tmp_path)
    _patch_bundle_root(monkeypatch, tmp_path)

    from app.config import _repo_local_snapshot
    out = _repo_local_snapshot("facebook/model_no_st")
    assert out is not None and Path(out).name == "model_no_st"


# ---------------------------------------------------------------------------
# Back-compat (layout A still works)
# ---------------------------------------------------------------------------

def test_layout_a_returns_snapshot_subdir(tmp_path, monkeypatch):
    _layout_a(tmp_path)
    _patch_bundle_root(monkeypatch, tmp_path)

    from app.config import _repo_local_snapshot
    out = _repo_local_snapshot("facebook/sam2-hiera-large")
    assert out is not None
    p = Path(out)
    assert p.name == "deadbeef1234567890"


def test_layout_a_skips_hidden_subdirs(tmp_path, monkeypatch):
    """Even when the snapshot subdir layout is in play, hidden dirs
    (e.g. .cache) at the org/name level must never be selected."""
    base = tmp_path / "models" / "hf" / "facebook" / "mixedmodel"
    snap = base / "snap_a"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").write_bytes(b"\x00" * 64)
    (base / ".cache").mkdir()
    (base / ".cache" / "noise.txt").write_text("noise")
    _patch_bundle_root(monkeypatch, tmp_path)

    from app.config import _repo_local_snapshot
    out = _repo_local_snapshot("facebook/mixedmodel")
    assert Path(out).name == "snap_a"


# ---------------------------------------------------------------------------
# Real config end-to-end (no mocks; checks the committed repo state)
# ---------------------------------------------------------------------------

def test_real_repo_facebook_sam3_resolves_to_dir_with_config_json():
    """The actual repo-local SAM 3 snapshot must resolve to a
    directory containing config.json. Pins the bug that put SAM 3
    on .cache in the field."""
    from app.config import _repo_local_snapshot
    out = _repo_local_snapshot("facebook/sam3")
    if out is None:
        pytest.skip("facebook/sam3 weights not present locally")
    p = Path(out)
    assert (p / "config.json").exists(), \
        f"resolver returned {p} but config.json is missing there"
    assert p.name != ".cache"
    assert "model.safetensors" in [f.name for f in p.iterdir()]
