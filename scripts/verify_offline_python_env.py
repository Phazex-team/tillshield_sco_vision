"""Verify the active Python environment matches ``requirements.lock``.

Used during offline deploy: confirms every locked package is already
installed at the locked version. Does not install anything; does not
touch the network.

Exit codes:
    0  every locked package version satisfied
    1  one or more mismatches or missing packages
    2  no requirements.lock found
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Belt-and-suspenders offline; importlib.metadata never touches network
# anyway but external resolvers might.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_FILE = REPO_ROOT / "requirements.lock"


def _parse_lock(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            continue
        # PEP 440 markers, comments, extras stripped.
        line = line.split(";", 1)[0].strip()
        line = line.split(" ", 1)[0]
        name, _, version = line.partition("==")
        if name and version:
            out[name.lower()] = version
    return out


def _installed_versions() -> dict[str, str]:
    from importlib.metadata import distributions
    return {dist.metadata["Name"].lower(): dist.version
            for dist in distributions()
            if dist.metadata and dist.metadata["Name"]}


def _required_packages_from_registry() -> list[str]:
    import yaml
    p = REPO_ROOT / "offline_assets.yaml"
    if not p.is_file():
        return []
    data = yaml.safe_load(p.read_text()) or {}
    return [str(x).lower() for x in
            (data.get("required_python_packages") or [])]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-extras", action="store_true",
                    help="Don't flag packages installed but not in "
                         "lock file (default: warn only).")
    args = ap.parse_args()

    if not LOCK_FILE.is_file():
        print(f"no requirements.lock at {LOCK_FILE}", file=sys.stderr)
        return 2

    locked = _parse_lock(LOCK_FILE)
    installed = _installed_versions()

    missing: list[str] = []
    mismatched: list[str] = []
    for name, version in sorted(locked.items()):
        got = installed.get(name)
        if got is None:
            missing.append(f"{name}=={version}")
        elif got != version:
            mismatched.append(f"{name}: locked={version} installed={got}")

    extras = sorted(set(installed) - set(locked))

    # Required-python-packages from offline_assets.yaml MUST also be
    # installed, even if their pinned version is not yet in the lock
    # file. These are deployment-blocking.
    registry_required = _required_packages_from_registry()
    runtime_missing: list[str] = []
    for pkg in registry_required:
        if pkg not in installed:
            runtime_missing.append(pkg)

    print(f"lock_file: {LOCK_FILE}")
    print(f"locked:    {len(locked)} packages")
    print(f"installed: {len(installed)} packages")
    print(f"required runtime packages: {sorted(registry_required) or '(none)'}")
    print()

    for m in missing:
        print(f"MISSING:    {m}", file=sys.stderr)
    for m in mismatched:
        print(f"MISMATCHED: {m}", file=sys.stderr)
    for m in runtime_missing:
        print(f"REQUIRED RUNTIME MISSING: {m} "
              f"(install via wheelhouse on the destination DGX)",
              file=sys.stderr)
    if extras and not args.allow_extras:
        for e in extras[:10]:
            print(f"extra (not in lock): {e}", file=sys.stderr)
        if len(extras) > 10:
            print(f"... and {len(extras) - 10} more extras",
                  file=sys.stderr)

    if missing or mismatched or runtime_missing:
        return 1
    print("OK — every locked + required runtime package is installed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
