"""Verify the offline bundle without using the network.

Reads ``offline_assets.yaml`` for required/optional asset definitions
and ``models/manifest.json`` for the current bundle state, then:

1. Confirms every REQUIRED asset is present under ``./models/hf/``.
   Production verification fails if any of:
        Qwen3-VL, Gemma BF16, Falcon Perception, SAM 3, Falcon OCR
   is missing.

2. For each "runtime_assets" path declared in the registry (prompts,
   config, db migrations, static frontend, launcher scripts, python
   env files), confirms presence.

3. Resolves every enabled model in ``config.yaml`` via
   ``app.config.resolve_model_path``. In production mode the path
   MUST be repo-local; cache-only paths fail loudly.

Exit codes:
    0  bundle + runtime assets + config all OK for current mode
    1  bundle present but config/runtime assets have problems
    2  required model missing or manifest corrupted
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import yaml

# Network OFF before any HF import.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = REPO_ROOT / "models" / "hf"
MANIFEST_PATH = REPO_ROOT / "models" / "manifest.json"
REGISTRY_PATH = REPO_ROOT / "offline_assets.yaml"
CACHE_PREFIX = str(Path.home() / ".cache")


class VerificationError(RuntimeError):
    pass


def _sha256(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _load_registry() -> dict:
    if not REGISTRY_PATH.is_file():
        raise VerificationError(f"registry missing: {REGISTRY_PATH}")
    return yaml.safe_load(REGISTRY_PATH.read_text()) or {}


def _load_manifest() -> dict:
    if not MANIFEST_PATH.is_file():
        raise VerificationError(
            f"manifest missing at {MANIFEST_PATH}. "
            "Run scripts/prepare_offline_model_bundle.py first."
        )
    return json.loads(MANIFEST_PATH.read_text())


def verify_required_assets(registry: dict, manifest: dict,
                           *, full_hash: bool) -> dict:
    by_name = {m.get("name") or m.get("model_id"): m
               for m in manifest.get("models", [])}

    summary = {"required_present": [], "required_missing": [],
               "optional_present": [], "optional_missing": []}

    for entry in registry.get("required", []) or []:
        name = entry.get("name")
        m = by_name.get(name)
        if m is None or m.get("status") != "present":
            summary["required_missing"].append({
                "name": name, "repo": entry.get("repo"),
                "official_source": entry.get("official_source") or "",
                "fallback": entry.get("fallback"),
            })
            continue
        # Confirm files actually present on disk and (optionally) the
        # sha256 of every tracked file.
        snap_dir = BUNDLE_ROOT / m["model_id"] / m["snapshot"]
        if not snap_dir.is_dir():
            summary["required_missing"].append({
                "name": name, "repo": entry.get("repo"),
                "official_source": "manifest claims present but "
                                   f"{snap_dir} not on disk",
            })
            continue
        for f in m.get("files", []):
            fp = snap_dir / f["rel_path"]
            if not fp.is_file():
                summary["required_missing"].append({
                    "name": name, "repo": entry.get("repo"),
                    "official_source": f"file missing: {fp}",
                })
                break
            if fp.stat().st_size != f["bytes"]:
                summary["required_missing"].append({
                    "name": name, "repo": entry.get("repo"),
                    "official_source": (
                        f"size mismatch on {fp.name}: "
                        f"manifest={f['bytes']} disk={fp.stat().st_size}"
                    ),
                })
                break
            if full_hash and _sha256(fp) != f["sha256"]:
                summary["required_missing"].append({
                    "name": name, "repo": entry.get("repo"),
                    "official_source": f"sha256 drift on {fp.name}",
                })
                break
        else:
            summary["required_present"].append({
                "name": name, "snapshot": m["snapshot"],
                "files": m["file_count"], "bytes": m["total_bytes"],
            })

    for entry in registry.get("optional", []) or []:
        name = entry.get("name")
        m = by_name.get(name)
        if m is None or m.get("status") != "present":
            summary["optional_missing"].append({
                "name": name, "repo": entry.get("repo"),
                "official_source": entry.get("official_source") or "",
            })
        else:
            summary["optional_present"].append({
                "name": name, "snapshot": m["snapshot"],
            })
    return summary


def verify_runtime_assets(registry: dict) -> list[str]:
    issues: list[str] = []
    for group, items in (registry.get("runtime_assets") or {}).items():
        for rel in items or []:
            p = REPO_ROOT / rel
            if not p.exists():
                issues.append(f"[{group}] missing: {rel}")
            elif p.is_dir() and not any(p.iterdir()):
                issues.append(f"[{group}] empty directory: {rel}")
    return issues


def verify_config_paths(*, production_mode: bool) -> list[str]:
    from app.config import load_config, resolve_model_path

    cfg = load_config()
    issues: list[str] = []
    for key, model_cfg in cfg.models.items():
        if not model_cfg.enabled:
            continue
        try:
            resolved = resolve_model_path(model_cfg,
                                          production_mode=production_mode)
        except Exception as exc:
            issues.append(f"[{key}] resolve_model_path raised: {exc}")
            continue
        if resolved is None:
            issues.append(
                f"[{key}] no usable local path for model "
                f"{model_cfg.name!r}"
            )
            continue
        if production_mode and resolved.startswith(CACHE_PREFIX):
            issues.append(
                f"[{key}] production mode requires repo-local path, "
                f"got cache path: {resolved}"
            )
        if not Path(resolved).is_dir():
            issues.append(
                f"[{key}] resolved path does not exist: {resolved}"
            )
    return issues


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full-hash", action="store_true",
                    help="Re-hash every file (slow).")
    ap.add_argument("--production", action="store_true",
                    help="Enforce production-mode rules: no cache "
                         "fallback for any enabled model.")
    args = ap.parse_args()

    print(f"repo_root  = {REPO_ROOT}")
    print(f"bundle     = {BUNDLE_ROOT}")
    print(f"registry   = {REGISTRY_PATH}")
    print(f"manifest   = {MANIFEST_PATH}")
    print(f"mode       = {'production' if args.production else 'dev'}")
    print(f"offline env: HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')}  "
          f"TRANSFORMERS_OFFLINE={os.environ.get('TRANSFORMERS_OFFLINE')}")
    print()

    try:
        registry = _load_registry()
        manifest = _load_manifest()
    except VerificationError as exc:
        print(f"BUNDLE FAIL: {exc}", file=sys.stderr)
        return 2

    summary = verify_required_assets(registry, manifest,
                                     full_hash=args.full_hash)

    for m in summary["required_present"]:
        print(f"OK  REQUIRED  {m['name']}  snapshot={m['snapshot']}  "
              f"files={m['files']}  bytes={m['bytes']}")
    for m in summary["optional_present"]:
        print(f"OK  optional  {m['name']}  snapshot={m['snapshot']}")

    if summary["required_missing"]:
        print()
        for m in summary["required_missing"]:
            print(f"REQUIRED MISSING: {m['name']}  ({m['repo']})",
                  file=sys.stderr)
            print(f"   official_source: {m['official_source']}",
                  file=sys.stderr)
            if m.get("fallback"):
                print(f"   fallback        : {m['fallback']}",
                      file=sys.stderr)
        # In production mode this is fatal; in dev we still exit 2 so
        # CI can pick it up.
        return 2

    runtime_issues = verify_runtime_assets(registry)
    for issue in runtime_issues:
        print(f"RUNTIME ASSET FAIL: {issue}", file=sys.stderr)

    config_issues = verify_config_paths(production_mode=args.production)
    for issue in config_issues:
        print(f"CONFIG FAIL: {issue}", file=sys.stderr)

    for m in summary["optional_missing"]:
        print(f"warn  optional missing: {m['name']}  "
              f"-> {m['official_source']}")

    if runtime_issues or config_issues:
        return 1

    print()
    print("BUNDLE OK — required assets, runtime assets, and config all "
          "resolve to local paths for the current mode.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
