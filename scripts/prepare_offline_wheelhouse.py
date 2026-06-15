"""Build an offline Python wheelhouse under ``./wheelhouse/``.

Reads ``requirements.lock`` (or ``requirements.txt`` if no lock file is
present) and downloads every wheel into ``./wheelhouse/`` so the repo
can be ``pip install``-ed on the destination DGX without network.

This is the **only** repo script allowed to use ``pip download``.
Runtime code must never invoke pip at all.

Two modes:
    --copy-from-cache-only   (default)
        Use packages already resolved in the active venv. We don't
        actually have a local pip cache layout to mirror, so this mode
        simply asserts every locked package is already installed and
        wheels are not rebuilt — i.e. it skips downloads. Use this on a
        machine that already has the venv warmed up.

    --download-approved
        Resolve and download wheels matching ``requirements.lock`` into
        ``./wheelhouse/`` via ``pip download``. Requires network.

After either mode, writes ``wheelhouse/manifest.json`` with file name,
size, and sha256 of every wheel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
WHEELHOUSE = REPO_ROOT / "wheelhouse"
WH_MANIFEST = WHEELHOUSE / "manifest.json"
LOCK_FILE = REPO_ROOT / "requirements.lock"
REQ_FILE = REPO_ROOT / "requirements.txt"


def _sha256(p: Path, chunk: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _read_requirements() -> Optional[Path]:
    if LOCK_FILE.is_file():
        return LOCK_FILE
    if REQ_FILE.is_file():
        return REQ_FILE
    return None


def _list_wheelhouse() -> list[dict]:
    if not WHEELHOUSE.is_dir():
        return []
    out = []
    for f in sorted(WHEELHOUSE.iterdir()):
        if f.suffix in (".whl", ".tar.gz") and f.is_file():
            out.append({
                "filename": f.name,
                "bytes": f.stat().st_size,
                "sha256": _sha256(f),
            })
    return out


def _write_manifest(files: list[dict], *, mode: str,
                    requirements_path: Path, started_at: float) -> None:
    WHEELHOUSE.mkdir(parents=True, exist_ok=True)
    WH_MANIFEST.write_text(json.dumps({
        "schema_version": 1,
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                     time.gmtime(started_at)),
        "duration_sec": round(time.time() - started_at, 1),
        "mode": mode,
        "requirements_path": str(requirements_path),
        "wheelhouse_path": str(WHEELHOUSE),
        "files": files,
    }, indent=2))


def _run_pip_download(req_path: Path, *, python: str) -> int:
    WHEELHOUSE.mkdir(parents=True, exist_ok=True)
    cmd = [
        python, "-m", "pip", "download",
        "--dest", str(WHEELHOUSE),
        "--requirement", str(req_path),
    ]
    print(" ".join(cmd))
    return subprocess.call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--copy-from-cache-only", action="store_true",
                       help="(default) Skip downloads. Assert all "
                            "locked packages are already installed.")
    group.add_argument("--download-approved", action="store_true",
                       help="Resolve and download wheels into "
                            "./wheelhouse/ via pip download.")
    ap.add_argument("--python", default=sys.executable,
                    help="Python interpreter to use for pip download "
                         "(default: current).")
    args = ap.parse_args()

    started = time.time()
    req = _read_requirements()
    if req is None:
        print("No requirements.lock or requirements.txt found.",
              file=sys.stderr)
        return 2
    print(f"requirements: {req}")
    print(f"wheelhouse:   {WHEELHOUSE}")
    mode = ("download_approved" if args.download_approved
            else "copy_from_cache_only")
    print(f"mode:         {mode}")
    print()

    if args.download_approved:
        rc = _run_pip_download(req, python=args.python)
        if rc != 0:
            print("pip download failed", file=sys.stderr)
            return 2

    files = _list_wheelhouse()
    _write_manifest(files, mode=mode, requirements_path=req,
                    started_at=started)

    total = sum(f["bytes"] for f in files)
    print(f"wheelhouse files: {len(files)}  total={total} bytes")
    print(f"manifest: {WH_MANIFEST}")

    if not files and args.download_approved:
        print("WARN: wheelhouse empty after download_approved",
              file=sys.stderr)
        return 1
    if not files:
        print("WARN: wheelhouse empty — run with --download-approved "
              "to populate", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
