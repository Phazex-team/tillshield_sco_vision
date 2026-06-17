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

    # 1. Required model bundles under ./models/hf/
    issues.extend(_check_required_assets(production))

    # 2. Required Python runtime packages (sam2 etc.) declared in
    # offline_assets.yaml.
    issues.extend(_check_required_python_packages(production))

    # 2b. TillShield poller config (only when polling is enabled).
    issues.extend(_check_tillshield_poll_config(cfg))

    # 3. Provider chain construction (lazy — no model load)
    chain_summary = _check_provider_chain(cfg, production)
    if chain_summary.get("error"):
        issues.append(chain_summary["error"])

    # 4. Storage roots
    storage = cfg.storage_root
    storage.mkdir(parents=True, exist_ok=True)
    if not os.access(storage, os.W_OK):
        issues.append(f"storage root not writable: {storage}")

    # 5. Memory guard initialises
    try:
        from app.memory_guard import get_policy
        get_policy().poll()
    except Exception as exc:
        issues.append(f"memory guard failed to initialise: {exc}")

    # 6. SAM2 runtime capability (production-only)
    if production:
        sam2_issue = _check_sam2_runtime(cfg)
        if sam2_issue:
            issues.append(sam2_issue)

    # 7. vLLM readiness for the Qwen3-VL active runtime. WARNING-ONLY:
    # the API / recorder / reviewer UI must come up even if vLLM is
    # not running yet, because the chain falls back to Gemma and the
    # decision policy degrades to REVIEW.
    if production:
        vllm_warning = _check_qwen_vllm_backend(cfg)
        if vllm_warning:
            warnings.append(vllm_warning)

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
    """Mirror the offline bundle verifier for the required set."""
    if not production:
        return []
    import json

    issues: list[str] = []
    from app import config as ac
    bundle_root = ac.BUNDLE_ROOT
    try:
        import yaml
        registry = yaml.safe_load(
            (REPO_ROOT / "offline_assets.yaml").read_text()) or {}
        manifest = json.loads(
            (REPO_ROOT / "models" / "manifest.json").read_text()
        )
        by_name = {
            m.get("name") or m.get("model_id"): m
            for m in (manifest.get("models") or [])
        }
        for entry in (registry.get("required") or []):
            repo = entry.get("repo") or ""
            base = bundle_root.joinpath(*repo.split("/"))
            model = by_name.get(entry.get("name"))
            if model is None or model.get("status") != "present":
                issues.append(
                    f"required asset {entry['name']!r} missing from "
                    "models/manifest.json"
                )
                continue
            snapshot = model.get("snapshot")
            snap_dir = base / str(snapshot)
            if not base.is_dir() or not snap_dir.is_dir():
                issues.append(
                    f"required asset {entry['name']!r} missing under "
                    f"{snap_dir} — bundle the model before starting"
                )
                continue
            for tracked in (model.get("files") or []):
                rel_path = tracked.get("rel_path")
                if not rel_path:
                    continue
                fp = snap_dir / rel_path
                if not fp.is_file():
                    issues.append(
                        f"required asset {entry['name']!r} incomplete: "
                        f"missing {fp}"
                    )
                    break
                expected_bytes = tracked.get("bytes")
                if isinstance(expected_bytes, int) and \
                        fp.stat().st_size != expected_bytes:
                    issues.append(
                        f"required asset {entry['name']!r} incomplete: "
                        f"size mismatch on {fp.name}"
                    )
                    break
    except FileNotFoundError as exc:
        issues.append(f"offline bundle metadata unreadable: {exc}")
    except json.JSONDecodeError as exc:
        issues.append(f"models/manifest.json invalid: {exc}")
    return issues


def _check_required_python_packages(production: bool) -> list[str]:
    if not production:
        return []
    import importlib.util
    issues: list[str] = []
    try:
        import yaml
        registry = yaml.safe_load(
            (REPO_ROOT / "offline_assets.yaml").read_text()) or {}
    except FileNotFoundError:
        return issues
    for pkg in (registry.get("required_python_packages") or []):
        if importlib.util.find_spec(str(pkg)) is None:
            issues.append(
                f"required python package {pkg!r} not importable. "
                f"Install via wheelhouse: "
                f"pip install --no-index --find-links wheelhouse {pkg}"
            )
    return issues


def _check_tillshield_poll_config(cfg) -> list[str]:
    """Validate the TillShield poller config. Returns issue strings when
    polling is enabled but misconfigured (workstation without a camera
    mapping, mapped camera absent from cameras, missing endpoint, bad
    interval). No-op when polling is disabled."""
    try:
        from pos.tillshield_poll import validate_poll_config
        return [f"tillshield poller: {m}" for m in validate_poll_config(cfg)]
    except Exception as exc:  # never let validation import crash startup
        log.warning("tillshield poll config validation skipped: %s", exc)
        return []


def _check_sam2_runtime(cfg) -> Optional[str]:
    """SAM2 runtime is required for production perception. We confirm
    the package imports AND the configured local snapshot exists."""
    import importlib.util
    if importlib.util.find_spec("sam2") is None:
        return ("sam2 python package not importable; perception cannot "
                "produce masks. Install via wheelhouse on this machine.")
    sam2_cfg = cfg.models.get("sam2")
    if sam2_cfg is None:
        return "config.yaml missing models.sam2 entry"
    try:
        from app.config import resolve_model_path
        path = resolve_model_path(sam2_cfg)
    except Exception as exc:
        return f"sam2 weights path could not be resolved: {exc}"
    if not path:
        return "sam2 weights not bundled under ./models/hf/"
    return None


def _check_qwen_vllm_backend(cfg) -> Optional[str]:
    """Warning-only check that Qwen3-VL's vLLM endpoint is reachable
    AND advertises the configured ``served_model_name``.

    Returns ``None`` when:
      * Qwen3-VL is disabled OR the configured backend is not
        ``vllm_openai`` (rollback path), OR
      * /v1/models contains the served model name.
    Returns a short warning string in all other situations. NEVER
    raises; this function must never block startup."""
    try:
        qwen_cfg = cfg.models.get("qwen3_vl") if cfg else None
        if qwen_cfg is None or not qwen_cfg.enabled:
            return None
        backend = (qwen_cfg.extra.get("provider") or "vllm_openai")
        if backend != "vllm_openai":
            return None
        from reasoning.providers.qwen3_vl import Qwen3VLProvider
        # Construct a probe with only the kwargs the validator + health
        # check need. Constructor is cheap (no model load, no socket).
        kwargs = {k: v for k, v in qwen_cfg.extra.items()
                  if k not in ("local_path",)}
        probe = Qwen3VLProvider(
            model_name=qwen_cfg.name,
            enabled=True,
            **kwargs,
        )
        healthy, detail = probe._vllm_health()
        if healthy:
            return None
        return (f"qwen3_vl vllm backend not ready (warning, not blocking): "
                f"{detail}")
    except Exception as exc:  # never block startup on a check failure
        return (f"qwen3_vl vllm readiness check raised "
                f"(warning, not blocking): {type(exc).__name__}: {exc}")


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
