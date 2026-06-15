"""Inspect locally cached HuggingFace model snapshots.

Read-only. No network calls, no downloads.

Lists each model expected by config.yaml + the local snapshot path,
its existence, snapshot revisions, and total on-disk size. Used to
verify operators see the same artifacts the running stack uses.

    python scripts/inspect_models.py
    python scripts/inspect_models.py --json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None


HF_HUB = Path(os.environ.get("HF_HOME",
                             str(Path.home() / ".cache" / "huggingface"))) / "hub"


def _dir_size(p: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(p, followlinks=False):
        for f in files:
            fp = Path(root) / f
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total


def _hf_repo_dir(name: str) -> Path:
    return HF_HUB / ("models--" + name.replace("/", "--"))


def _human(n: int) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f}PB"


def inspect_repo(name: str, expected_local_path: str | None = None) -> dict:
    repo_dir = _hf_repo_dir(name)
    snapshots_dir = repo_dir / "snapshots"
    snapshots = []
    if snapshots_dir.is_dir():
        for snap in sorted(snapshots_dir.iterdir()):
            if snap.is_dir():
                snapshots.append({
                    "revision": snap.name,
                    "path": str(snap.resolve()),
                    "size_bytes": _dir_size(snap),
                })
    expected_match = None
    if expected_local_path:
        ep = Path(expected_local_path).resolve()
        expected_match = any(Path(s["path"]) == ep for s in snapshots)
    return {
        "name": name,
        "repo_dir": str(repo_dir),
        "repo_dir_exists": repo_dir.is_dir(),
        "snapshots": snapshots,
        "expected_local_path": expected_local_path,
        "expected_local_path_matches": expected_match,
    }


def load_config_models(path: Path) -> dict:
    if yaml is None:
        return {}
    try:
        with path.open() as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
    return cfg.get("models") or {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--json", action="store_true", help="Emit JSON.")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    models_cfg = load_config_models(cfg_path)

    targets: list[tuple[str, str | None]] = []
    for key in ("falcon", "gemma", "qwen3_vl"):
        m = models_cfg.get(key) or {}
        name = m.get("name")
        if name:
            targets.append((name, m.get("local_path")))

    # Fallback so the script does *something* useful if config is missing.
    if not targets:
        targets = [
            ("tiiuae/Falcon-Perception", None),
            ("google/gemma-4-26B-A4B-it", None),
            ("Qwen/Qwen3-VL-30B-A3B-Instruct", None),
        ]

    results = [inspect_repo(name, lp) for name, lp in targets]

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    print(f"HF hub: {HF_HUB}")
    print()
    for r in results:
        ok = "OK" if r["repo_dir_exists"] else "MISSING"
        print(f"[{ok}] {r['name']}")
        print(f"  repo_dir: {r['repo_dir']}")
        if r["expected_local_path"]:
            tag = "match" if r["expected_local_path_matches"] else "MISMATCH"
            print(f"  expected_local_path ({tag}): {r['expected_local_path']}")
        if not r["snapshots"]:
            print("  snapshots: (none)")
        for s in r["snapshots"]:
            print(f"  snapshot {s['revision']}  "
                  f"size={_human(s['size_bytes'])}  path={s['path']}")
        print()

    missing = [r for r in results if not r["repo_dir_exists"]]
    if missing:
        print(f"WARNING: {len(missing)} model(s) not present locally", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
