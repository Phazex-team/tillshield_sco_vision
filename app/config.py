"""Typed access to ``config.yaml`` plus environment overrides.

Single source of truth for module dirs that need to read configuration
without importing the legacy ``app.py``. Keeps backwards-compatible
field names so the existing MVP keeps working.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


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


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> AppConfig:
    p = Path(path)
    with p.open() as f:
        raw = yaml.safe_load(f) or {}

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
