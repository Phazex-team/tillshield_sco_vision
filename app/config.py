"""Typed access to ``config.yaml`` plus environment overrides.

Single source of truth for module dirs that need to read configuration
without importing the legacy ``app.py``. Keeps backwards-compatible
field names so the existing MVP keeps working.

This module also owns ``resolve_model_path`` — the only place runtime
code is allowed to map a model name onto a local directory. It enforces
the offline-portable rule: in production/offline mode it returns ONLY
repo-local ``./models/hf/...`` paths and never silently falls back to
``~/.cache/huggingface``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
BUNDLE_ROOT = PROJECT_ROOT / "models" / "hf"
CACHE_PREFIX = str(Path.home() / ".cache")


class OfflineBundleError(RuntimeError):
    """Raised when offline mode is in effect and a required model is
    not available under ``./models/hf/...``."""


def is_production_offline_mode() -> bool:
    """True iff the runtime must reject cache-only model paths.

    Toggle with ``FRAUD_OFFLINE_MODE=1`` in the environment (or `.env`).
    """
    v = (os.environ.get("FRAUD_OFFLINE_MODE")
         or os.environ.get("OFFLINE_MODE", "")).strip().lower()
    return v in ("1", "true", "yes", "production", "on")


@dataclass
class ModelConfig:
    name: str
    enabled: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AppConfig:
    raw: dict
    cameras: list[dict]
    settings: dict
    models: dict[str, ModelConfig]
    observability: dict

    @property
    def storage_root(self) -> Path:
        return Path(os.environ.get("STORAGE_ROOT",
                                   str(PROJECT_ROOT / "storage")))

    @property
    def database_url(self) -> str:
        return os.environ.get(
            "DATABASE_URL",
            f"sqlite:///{PROJECT_ROOT / 'fraud_detection_v3.sqlite'}",
        )


_DOTENV_LOADED = False


def _load_dotenv_once() -> None:
    """Load ``.env`` (gitignored) so secrets referenced as ``${VAR}`` in
    config.yaml resolve on a direct ``run_app.py`` launch too. No-op if
    python-dotenv is absent."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except Exception:
        pass


def _expand_env(obj):
    """Recursively expand ``${VAR}`` references in string values from the
    environment. Unknown vars are left literal so a missing secret fails
    visibly rather than silently."""
    if isinstance(obj, str):
        return os.path.expandvars(obj) if "${" in obj else obj
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    return obj


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> AppConfig:
    _load_dotenv_once()
    p = Path(path)
    with p.open() as f:
        raw = yaml.safe_load(f) or {}
    raw = _expand_env(raw)

    models_raw = raw.get("models") or {}
    models: dict[str, ModelConfig] = {}
    for key, m in models_raw.items():
        if not isinstance(m, dict):
            continue
        name = m.get("name") or ""
        enabled = bool(m.get("enabled", True))
        extra = {k: v for k, v in m.items() if k not in ("name", "enabled")}
        models[key] = ModelConfig(name=name, enabled=enabled, extra=extra)

    return AppConfig(
        raw=raw,
        cameras=list(raw.get("cameras") or []),
        settings=dict(raw.get("settings") or {}),
        models=models,
        observability=dict(raw.get("observability") or {}),
    )


def _repo_local_snapshot(model_name: str) -> Optional[str]:
    """Return the repo-local snapshot dir for ``model_name`` if one
    exists under ``./models/hf/...``. The HF cache layout is
    ``<repo>/<org>/<name>/<snapshot>/``, mirrored here."""
    if not model_name:
        return None
    parts = model_name.split("/")
    base = BUNDLE_ROOT.joinpath(*parts)
    if not base.is_dir():
        return None
    snaps = [p for p in base.iterdir() if p.is_dir()]
    if not snaps:
        return None
    # Single snapshot is the common case (the bundler picks one); if
    # there are multiple, prefer the largest by content size.
    if len(snaps) == 1:
        return str(snaps[0])
    def _size(p: Path) -> int:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return str(max(snaps, key=_size))


def resolve_model_path(model_cfg: "ModelConfig",
                       *,
                       production_mode: Optional[bool] = None
                       ) -> Optional[str]:
    """Return an absolute local directory for ``model_cfg``.

    Resolution order:
      1. Repo-local ``./models/hf/<name>/<snapshot>/`` (preferred).
      2. ``model_cfg.extra['local_path']`` if it exists on disk
         (development mode only).

    In production mode (``FRAUD_OFFLINE_MODE=1``):
      * step 2 is skipped — cache paths are NEVER returned.
      * if step 1 yields nothing, ``OfflineBundleError`` is raised.

    Returns ``None`` if no path is usable AND production mode is off, so
    callers in dev can still degrade gracefully.
    """
    if production_mode is None:
        production_mode = is_production_offline_mode()

    repo_local = _repo_local_snapshot(model_cfg.name)
    if repo_local is not None:
        return repo_local

    if production_mode:
        raise OfflineBundleError(
            f"model {model_cfg.name!r} not present under {BUNDLE_ROOT}; "
            "production/offline mode forbids the ~/.cache fallback. "
            "Run scripts/prepare_offline_model_bundle.py before starting."
        )

    cache_path = model_cfg.extra.get("local_path")
    if cache_path and Path(cache_path).is_dir():
        return str(Path(cache_path).resolve())
    return None
