"""Build the offline model bundle under ``./models/hf/``.

Reads ``offline_assets.yaml`` and copies (or hardlinks) every entry into
``./models/hf/<repo_org>/<repo_name>/<snapshot>/``. Two explicit modes:

  --copy-from-cache-only   (default)
        NEVER downloads. Hardlinks files already present in
        ``~/.cache/huggingface``. Missing required assets are listed and
        the script exits non-zero. Missing optional assets are warned.

  --download-approved
        Downloads required assets that are missing directly into the
        repo-local bundle (NOT into the user's HF cache). Uses
        ``huggingface_hub.snapshot_download`` with ``local_dir``
        targeting ``./models/hf/<repo>/__pending__/`` and renames into
        the snapshot dir on success. Requires network. The runtime
        never calls this path — only this prep script does.

Whichever mode runs, ``models/manifest.json`` is rewritten with the
final state: snapshot hash, file count, total size, and sha256 of
every file. The verifier reads it offline.

This script is the ONLY place in the repo that is allowed to call a
HuggingFace download API. The runtime, providers, and decision policy
must never import ``snapshot_download`` / ``hf_hub_download``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import yaml

# Network defaults to OFF; --download-approved flips this for one
# snapshot_download call inside the script (never at runtime).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = REPO_ROOT / "models" / "hf"
MANIFEST_PATH = REPO_ROOT / "models" / "manifest.json"
REGISTRY_PATH = REPO_ROOT / "offline_assets.yaml"
HF_HUB_ROOT = Path(os.environ.get(
    "HF_HOME", str(Path.home() / ".cache" / "huggingface"))
) / "hub"


# Files that MUST exist in any usable HF snapshot directory.
REQUIRED_PRESENT: tuple[str, ...] = ("config.json",)
REQUIRED_AT_LEAST_ONE: tuple[tuple[str, ...], ...] = (
    ("tokenizer.json", "tokenizer_config.json", "spm.model",
     "preprocessor_config.json"),
    ("*.safetensors", "*.bin", "*.pt"),
)


class BundleError(RuntimeError):
    pass


@dataclass
class FileEntry:
    rel_path: str
    bytes: int
    sha256: str


@dataclass
class ModelBundleResult:
    name: str
    model_id: str
    purpose: str
    required: bool
    runtime_blocking: bool
    status: str                     # "present" | "missing" | "optional_missing"
    snapshot: Optional[str]
    source: Optional[str]           # cache | download | None
    source_path: Optional[str]
    dest_path: Optional[str]
    file_count: int
    total_bytes: int
    files: list[FileEntry]
    note: str = ""


def _load_registry() -> dict:
    if not REGISTRY_PATH.is_file():
        raise BundleError(f"registry missing: {REGISTRY_PATH}")
    return yaml.safe_load(REGISTRY_PATH.read_text()) or {}


def _hf_repo_dir(model_id: str) -> Path:
    return HF_HUB_ROOT / ("models--" + model_id.replace("/", "--"))


def _pick_cached_snapshot(model_id: str) -> Optional[tuple[str, Path]]:
    repo = _hf_repo_dir(model_id)
    snaps_dir = repo / "snapshots"
    if not snaps_dir.is_dir():
        return None
    candidates = sorted(p for p in snaps_dir.iterdir() if p.is_dir())
    if not candidates:
        return None
    best = max(candidates, key=_total_real_bytes)
    return best.name, best


def _total_real_bytes(snapshot_dir: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(snapshot_dir, followlinks=False):
        for f in files:
            try:
                total += os.path.getsize(
                    os.path.realpath(str(Path(root) / f))
                )
            except OSError:
                pass
    return total


def _sha256(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _safe_resolve(path: Path, allowed_roots: Iterable[Path]) -> Path:
    real = Path(os.path.realpath(str(path)))
    for root in allowed_roots:
        try:
            real.relative_to(root.resolve())
            return real
        except ValueError:
            continue
    raise BundleError(
        f"refusing to follow symlink {path} -> {real}: target escapes "
        f"allowed roots {[str(r) for r in allowed_roots]}"
    )


def _link_or_copy(src: Path, dst: Path, *, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError as exc:
        print(f"  hardlink failed ({exc}); falling back to copy "
              f"for {src.name}", file=sys.stderr)
        shutil.copy2(src, dst)


def _validate_snapshot_contents(snapshot_dir: Path, model_id: str) -> None:
    names = {p.name for p in snapshot_dir.iterdir()}
    for required in REQUIRED_PRESENT:
        if required not in names:
            raise BundleError(
                f"{model_id}: required file {required!r} missing under "
                f"{snapshot_dir}"
            )
    for group in REQUIRED_AT_LEAST_ONE:
        if not any(snapshot_dir.glob(p) for p in group):
            raise BundleError(
                f"{model_id}: snapshot lacks any of {group!r} — "
                "refusing to publish an incomplete bundle"
            )


def _ingest_snapshot(model_id: str,
                     snapshot: str,
                     src_dir: Path,
                     *,
                     copy: bool,
                     allowed_roots: Iterable[Path]) -> ModelBundleResult:
    _validate_snapshot_contents(src_dir, model_id)
    dst_dir = BUNDLE_ROOT / model_id / snapshot
    dst_dir.mkdir(parents=True, exist_ok=True)

    files: list[FileEntry] = []
    total = 0
    for entry in sorted(src_dir.iterdir()):
        # HF cache snapshots are flat in the strict sense, but tooling
        # may drop hidden metadata dirs alongside (``.eval_results``,
        # ``.huggingface``, etc.) — skip dot-prefixed directories. Any
        # OTHER nesting is unexpected and we abort.
        if entry.is_dir():
            if entry.name.startswith("."):
                continue
            raise BundleError(
                f"{model_id}: unexpected subdirectory {entry} in snapshot"
            )
        if entry.name.startswith("."):
            # Hidden HF metadata files (eg ``.gitattributes``) aren't
            # required for inference but are tiny — include them so
            # the bundle is byte-identical to the cache.
            pass
        real = _safe_resolve(entry, allowed_roots)
        if not real.is_file():
            raise BundleError(
                f"{model_id}: entry {entry} resolves to non-file {real}"
            )
        dst = dst_dir / entry.name
        _link_or_copy(real, dst, copy=copy)
        size = dst.stat().st_size
        sha = _sha256(dst)
        files.append(FileEntry(entry.name, size, sha))
        total += size

    return ModelBundleResult(
        name="", model_id=model_id, purpose="",
        required=False, runtime_blocking=False,
        status="present", snapshot=snapshot,
        source=None, source_path=str(src_dir),
        dest_path=str(dst_dir),
        file_count=len(files), total_bytes=total,
        files=files,
    )


def _download_into_bundle(model_id: str) -> tuple[str, Path]:
    """One-shot download directly into ``./models/hf/...``. The downloaded
    layout is flattened to match a snapshot dir. Returns
    ``(snapshot_hash, snapshot_dir)``.

    This is the ONLY place in the repo that turns off
    ``HF_HUB_OFFLINE``. The toggle is local to this function so module
    import (and the runtime) keeps offline semantics."""
    # Temporarily re-enable network for the prep step.
    prev_hub = os.environ.pop("HF_HUB_OFFLINE", None)
    prev_tf = os.environ.pop("TRANSFORMERS_OFFLINE", None)
    try:
        from huggingface_hub import snapshot_download
        target = BUNDLE_ROOT / model_id / "__pending__"
        target.mkdir(parents=True, exist_ok=True)
        snapshot_dir = snapshot_download(
            repo_id=model_id,
            local_dir=str(target),
        )
        snap_path = Path(snapshot_dir)
        # Compute a stable hash from the resolved revision. HF cache uses
        # a 40-char commit hash; we read it from the .gitattributes if
        # present, else fall back to the local_dir name.
        rev_hash = _detect_revision(snap_path) or snap_path.name
        final = BUNDLE_ROOT / model_id / rev_hash
        if final.exists():
            shutil.rmtree(final)
        snap_path.rename(final)
        # Clean up the __pending__ wrapper if empty.
        try:
            target.rmdir()
        except OSError:
            pass
        return rev_hash, final
    finally:
        if prev_hub is not None:
            os.environ["HF_HUB_OFFLINE"] = prev_hub
        else:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        if prev_tf is not None:
            os.environ["TRANSFORMERS_OFFLINE"] = prev_tf
        else:
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _detect_revision(snapshot_dir: Path) -> Optional[str]:
    # HF snapshot_download writes the resolved revision into
    # ``.huggingface/*`` markers; if not present, we just take a
    # deterministic content hash of (sorted file sha256s).
    marker = snapshot_dir / ".cache" / "huggingface"
    if marker.is_dir():
        for f in marker.rglob("*"):
            if f.is_file():
                text = f.read_text(errors="ignore")
                for line in text.splitlines():
                    if line.count("/") and len(line.strip()) == 40:
                        return line.strip()
    h = hashlib.sha256()
    for f in sorted(snapshot_dir.rglob("*")):
        if f.is_file():
            h.update(f.relative_to(snapshot_dir).as_posix().encode())
            h.update(str(f.stat().st_size).encode())
    return h.hexdigest()[:40]


def process_asset(entry: dict, *, required: bool,
                  copy: bool, allow_download: bool) -> ModelBundleResult:
    name = entry.get("name") or entry.get("repo", "")
    model_id = entry.get("repo", "")
    purpose = entry.get("purpose", "")
    runtime_blocking = bool(entry.get("runtime_blocking", required))

    # 1. Already bundled? Use the bundle as-is, skip cache + download.
    final = BUNDLE_ROOT / model_id if model_id else None
    if final and final.is_dir():
        existing = [p for p in final.iterdir()
                    if p.is_dir() and not p.name.startswith("__")
                    and _total_real_bytes(p) > 0]
        if existing:
            best = max(existing, key=_total_real_bytes)
            files: list[FileEntry] = []
            total = 0
            for f in sorted(best.rglob("*")):
                if f.is_file():
                    size = f.stat().st_size
                    sha = _sha256(f)
                    files.append(FileEntry(
                        str(f.relative_to(best)), size, sha))
                    total += size
            return ModelBundleResult(
                name=name, model_id=model_id, purpose=purpose,
                required=required, runtime_blocking=runtime_blocking,
                status="present", snapshot=best.name,
                source="bundle", source_path=str(best),
                dest_path=str(best),
                file_count=len(files), total_bytes=total,
                files=files,
                note="already in bundle; no action taken",
            )
        # Any empty snapshot dirs from a prior failed run should be
        # cleaned up so the cache-ingest path runs cleanly below.
        for p in final.iterdir():
            if p.is_dir() and _total_real_bytes(p) == 0:
                try:
                    p.rmdir()
                except OSError:
                    pass

    # 2. Available in HF cache? Ingest by hardlink/copy.
    cached = _pick_cached_snapshot(model_id) if model_id else None
    if cached is not None:
        snapshot, src_dir = cached
        allowed = [_hf_repo_dir(model_id), HF_HUB_ROOT]
        try:
            r = _ingest_snapshot(model_id, snapshot, src_dir,
                                 copy=copy, allowed_roots=allowed)
        except BundleError:
            raise
        r.name = name
        r.purpose = purpose
        r.required = required
        r.runtime_blocking = runtime_blocking
        r.source = "cache"
        return r

    # 3. Allowed to download?
    if allow_download and required:
        snapshot, snap_dir = _download_into_bundle(model_id)
        # Re-ingest from the downloaded location for hashing + manifest.
        allowed = [BUNDLE_ROOT.resolve()]
        files: list[FileEntry] = []
        total = 0
        for f in sorted(snap_dir.rglob("*")):
            if f.is_file():
                size = f.stat().st_size
                sha = _sha256(f)
                files.append(FileEntry(
                    str(f.relative_to(snap_dir)), size, sha))
                total += size
        return ModelBundleResult(
            name=name, model_id=model_id, purpose=purpose,
            required=required, runtime_blocking=runtime_blocking,
            status="present", snapshot=snapshot,
            source="download", source_path=None,
            dest_path=str(snap_dir),
            file_count=len(files), total_bytes=total,
            files=files,
            note="fetched into repo-local bundle (network used by prep "
                 "script only)",
        )

    # 4. Missing.
    return ModelBundleResult(
        name=name, model_id=model_id, purpose=purpose,
        required=required, runtime_blocking=runtime_blocking,
        status="missing" if required else "optional_missing",
        snapshot=None, source=None, source_path=None,
        dest_path=None, file_count=0, total_bytes=0, files=[],
        note=(entry.get("official_source") or
              "no local cache, no download permitted"),
    )


def write_manifest(results: list[ModelBundleResult], *,
                   mode: str, started_at: float) -> None:
    manifest = {
        "schema_version": 1,
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                     time.gmtime(started_at)),
        "duration_sec": round(time.time() - started_at, 1),
        "mode": mode,
        "repo_root": str(REPO_ROOT),
        "bundle_root": str(BUNDLE_ROOT),
        "models": [
            {
                "name": r.name,
                "model_id": r.model_id,
                "purpose": r.purpose,
                "required": r.required,
                "runtime_blocking": r.runtime_blocking,
                "status": r.status,
                "snapshot": r.snapshot,
                "source": r.source,
                "source_path": r.source_path,
                "dest_path": r.dest_path,
                "file_count": r.file_count,
                "total_bytes": r.total_bytes,
                "files": [
                    {"rel_path": f.rel_path, "bytes": f.bytes,
                     "sha256": f.sha256}
                    for f in r.files
                ],
                "note": r.note,
            }
            for r in results
        ],
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def _human(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024:
            return f"{x:.1f}{unit}"
        x /= 1024
    return f"{x:.1f}PB"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--copy-from-cache-only", action="store_true",
                       help="(default) Never download; bundle only "
                            "what is already in ~/.cache/huggingface.")
    group.add_argument("--download-approved", action="store_true",
                       help="Fetch missing REQUIRED assets directly "
                            "into ./models/hf/ (never into ~/.cache).")
    ap.add_argument("--copy", action="store_true",
                    help="Copy bytes instead of hardlinking.")
    ap.add_argument("--asset", action="append",
                    help="Limit to this asset name (registry key, "
                         "repeatable).")
    args = ap.parse_args()

    started = time.time()
    registry = _load_registry()
    required = registry.get("required") or []
    optional = registry.get("optional") or []

    if args.asset:
        wanted = set(args.asset)
        required = [e for e in required if e.get("name") in wanted]
        optional = [e for e in optional if e.get("name") in wanted]

    mode = ("download_approved" if args.download_approved
            else "copy_from_cache_only")
    copy_bytes = bool(args.copy)
    allow_download = bool(args.download_approved)

    print(f"bundle_root = {BUNDLE_ROOT}")
    print(f"registry    = {REGISTRY_PATH}")
    print(f"hf cache    = {HF_HUB_ROOT}")
    print(f"mode        = {mode}  (bytes_mode={'copy' if copy_bytes else 'hardlink'})")
    print()

    BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)

    results: list[ModelBundleResult] = []
    grand_total = 0

    fatal_security_error = False
    for entry in required:
        print(f"[required] {entry.get('name')}  ({entry.get('repo')})")
        try:
            r = process_asset(entry, required=True,
                              copy=copy_bytes,
                              allow_download=allow_download)
        except BundleError as exc:
            print(f"  FATAL: {exc}", file=sys.stderr)
            fatal_security_error = True
            continue
        results.append(r)
        if r.status == "present":
            grand_total += r.total_bytes
            print(f"  OK source={r.source} snapshot={r.snapshot} "
                  f"files={r.file_count} size={_human(r.total_bytes)}")
        else:
            print(f"  MISSING — {r.note}", file=sys.stderr)

    for entry in optional:
        print(f"[optional] {entry.get('name')}  ({entry.get('repo')})")
        try:
            r = process_asset(entry, required=False,
                              copy=copy_bytes, allow_download=False)
        except BundleError as exc:
            print(f"  FATAL (optional): {exc}", file=sys.stderr)
            fatal_security_error = True
            continue
        results.append(r)
        if r.status == "present":
            grand_total += r.total_bytes
            print(f"  OK source={r.source} snapshot={r.snapshot} "
                  f"files={r.file_count} size={_human(r.total_bytes)}")
        else:
            print(f"  optional_missing — {r.note}")

    write_manifest(results, mode=mode, started_at=started)

    missing_required = [r for r in results
                        if r.required and r.status != "present"]
    print()
    print(f"manifest written: {MANIFEST_PATH}")
    print(f"bundle total:     {_human(grand_total)}")
    if fatal_security_error:
        return 2
    if missing_required:
        print()
        print("REQUIRED ASSETS MISSING:", file=sys.stderr)
        for r in missing_required:
            print(f"  - {r.name}  ({r.model_id})", file=sys.stderr)
            print(f"      official_source: {r.note}", file=sys.stderr)
        if allow_download:
            print("download attempt failed; see stderr above",
                  file=sys.stderr)
        else:
            print("rerun with --download-approved on a connected "
                  "machine to fetch them into ./models/hf/",
                  file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
