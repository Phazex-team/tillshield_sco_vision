"""Startup integrity checks (PRODUCTION_SPEC §17).

The app must fail fast in production mode if any required asset is
missing. ``run_startup_checks`` is invoked by ``scripts/run_app.py``
before opening the API port; if it raises, the operator sees a clear
message and the API never serves a request with a half-loaded stack.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional


log = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[1]


class StartupCheckError(RuntimeError):
    pass


def run_startup_checks(*, strict: Optional[bool] = None) -> dict:
    """Verify production readiness. Returns a summary dict; raises
    ``StartupCheckError`` on any blocking failure when in strict mode.

    Strict mode follows ``FRAUD_OFFLINE_MODE`` by default.
    """
    from app.config import is_production_offline_mode, load_config

    cfg = load_config()
    production = (strict if strict is not None
                  else is_production_offline_mode())

    issues: list[str] = []
    warnings: list[str] = []

    # 1. Required runtime assets
    issues.extend(_check_required_assets(production))

    # 2. Provider chain construction (lazy — no model load)
    chain_summary = _check_provider_chain(cfg, production)
    if chain_summary.get("error"):
        issues.append(chain_summary["error"])

    # 3. Storage roots
    storage = cfg.storage_root
    storage.mkdir(parents=True, exist_ok=True)
    if not os.access(storage, os.W_OK):
        issues.append(f"storage root not writable: {storage}")

    # 4. Memory guard initialises
    try:
        from app.memory_guard import get_policy
        get_policy().poll()
    except Exception as exc:
        issues.append(f"memory guard failed to initialise: {exc}")

    if production and issues:
        joined = "\n  - ".join(issues)
        raise StartupCheckError(
            f"startup checks failed in production mode:\n  - {joined}"
        )

    if not production and issues:
        warnings.extend(issues)
        issues = []

    return {
        "production": production,
        "issues": issues,
        "warnings": warnings,
        "provider_chain": chain_summary,
        "storage_root": str(storage),
    }


def _check_required_assets(production: bool) -> list[str]:
    """Mirror scripts/verify_offline_bundle.py for the required set."""
    if not production:
        return []
    issues: list[str] = []
    # Read BUNDLE_ROOT dynamically from app.config so tests that
    # monkeypatch the constant get the patched value.
    from app import config as ac
    bundle_root = ac.BUNDLE_ROOT
    try:
        import yaml
        registry = yaml.safe_load(
            (REPO_ROOT / "offline_assets.yaml").read_text()) or {}
        for entry in (registry.get("required") or []):
            repo = entry.get("repo") or ""
            base = bundle_root.joinpath(*repo.split("/"))
            if not base.is_dir():
                issues.append(
                    f"required asset {entry['name']!r} missing under "
                    f"{base} — bundle the model before starting"
                )
    except FileNotFoundError as exc:
        issues.append(f"offline_assets.yaml unreadable: {exc}")
    return issues


def _check_provider_chain(cfg, production: bool) -> dict:
    try:
        from reasoning.providers import build_active_provider
        provider = build_active_provider(cfg)
        if provider.name == "chain":
            chain_members = [p.name for p in provider.providers]
        else:
            chain_members = [provider.name]
    except Exception as exc:
        return {"members": [], "error": f"provider chain build failed: {exc}"}
    return {"members": chain_members, "error": None}
