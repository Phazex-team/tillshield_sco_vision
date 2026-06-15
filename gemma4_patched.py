"""Source-level patches for vLLM's Gemma 4 implementation
(``vllm/model_executor/models/gemma4.py``) so the
``bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4`` checkpoint loads
correctly on the working DGX stack:

    vLLM nightly + torch 2.11+cu130 + transformers 5.6.2

Two fixes are applied to vLLM's in-tree gemma4.py:

1. **MoE expert key prefix** — vLLM's ``expert_params_mapping`` looks for
   ``f"experts.{expert_id}.{proj_name}"`` but the NVFP4 checkpoint
   exposes them under ``moe.experts.{expert_id}.{proj_name}``. Without
   this prefix, every expert weight load misses, the loader silently
   continues, and the model serves zeroed expert weights. The fix is a
   one-string change inside ``Gemma4Model.load_weights``.

2. **Remove ``reduce_results=True``** from the FusedMoE constructor
   inside ``Gemma4MoE.__init__``. vLLM nightly's FusedMoE no longer
   accepts ``reduce_results`` as a kwarg; the call raises
   ``TypeError: FusedMoE.__init__() got an unexpected keyword
   argument 'reduce_results'`` at engine init.

These were originally discovered by hand-editing vLLM's source. This
module makes the same edits programmatically and is invoked by
``install.sh`` on a fresh venv (so v3 is self-contained) and re-checked
by ``vllm_start.sh`` on every boot.

Invocation:
    # from python
    import gemma4_patched
    gemma4_patched.apply_manual_fixes()      # idempotent
    gemma4_patched.verify_manual_fixes_applied()  # bool

    # from shell
    python gemma4_patched.py apply           # explicit
    python gemma4_patched.py verify          # exit 0 if OK
    python gemma4_patched.py restore         # revert from backup

A `.preupstreampatch.bak` is written next to the target file the first
time a real edit is made; further runs are no-ops.
"""
from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

log = logging.getLogger("gemma4_patched")

BACKUP_SUFFIX = ".preupstreampatch.bak"

# Fix A: prefix the per-expert weight name with ``moe.``.
_FIX_A_BEFORE = 'f"experts.{expert_id}.{proj_name}"'
_FIX_A_AFTER  = 'f"moe.experts.{expert_id}.{proj_name}"'

# Fix B: a regex that matches the ``reduce_results=True,`` line (with any
# leading indentation, optional spaces, optional trailing comment) inside
# the FusedMoE call in Gemma4MoE.__init__. Multiline-mode so we can drop
# the entire line including its trailing newline.
_FIX_B_RE = re.compile(
    r"^[ \t]*reduce_results\s*=\s*True\s*,\s*(?:#[^\n]*)?\n",
    re.MULTILINE,
)


def _locate_vllm_gemma4(explicit: Optional[str | Path] = None) -> Path:
    """Return the absolute path to the active vLLM gemma4 module."""
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    try:
        import vllm.model_executor.models.gemma4 as g4
    except ImportError as e:
        raise RuntimeError(
            "vllm.model_executor.models.gemma4 is not importable; "
            "is vLLM installed in this environment?"
        ) from e
    return Path(g4.__file__).resolve()


def _ensure_backup(target: Path) -> None:
    backup = target.with_suffix(target.suffix + BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(target, backup)
        log.info("[gemma4_patched] wrote backup -> %s", backup)


def apply_manual_fixes(vllm_gemma4_path: Optional[str | Path] = None) -> bool:
    """Apply both manual fixes to the vLLM gemma4 module in place.

    Idempotent: returns ``True`` if either edit was made, ``False`` if
    the file was already fully patched. Raises if the source doesn't
    contain the expected un-patched anchor for either fix (so we don't
    silently corrupt a future vLLM release).
    """
    target = _locate_vllm_gemma4(vllm_gemma4_path)
    src = target.read_text(encoding="utf-8")
    original = src
    fix_a_done = _FIX_A_BEFORE not in src and _FIX_A_AFTER in src
    fix_b_done = _FIX_B_RE.search(src) is None

    # Fix A: replace the bare experts.{expert_id}... with moe.experts...
    if not fix_a_done:
        if _FIX_A_BEFORE not in src:
            raise RuntimeError(
                f"[gemma4_patched] Fix-A anchor not found in {target}. "
                "Either vLLM upstream changed the expert_params_mapping "
                "or the file is already patched in an unexpected way."
            )
        src = src.replace(_FIX_A_BEFORE, _FIX_A_AFTER, 1)
        log.info("[gemma4_patched] Fix A applied: prefixed expert key with 'moe.'")
    else:
        log.info("[gemma4_patched] Fix A already in place")

    # Fix B: drop the ``reduce_results=True,`` line.
    if not fix_b_done:
        new_src, n = _FIX_B_RE.subn("", src, count=1)
        if n == 0:
            raise RuntimeError(
                f"[gemma4_patched] Fix-B anchor (`reduce_results=True,`) "
                f"not found in {target}. Either vLLM upstream changed the "
                "FusedMoE init or the file is already patched."
            )
        src = new_src
        log.info("[gemma4_patched] Fix B applied: removed reduce_results=True from FusedMoE init")
    else:
        log.info("[gemma4_patched] Fix B already in place")

    if src == original:
        return False

    _ensure_backup(target)
    target.write_text(src, encoding="utf-8")
    log.info("[gemma4_patched] wrote patched %s", target)

    # Drop any cached import so a fresh import picks up the new file.
    sys.modules.pop("vllm.model_executor.models.gemma4", None)
    return True


def verify_manual_fixes_applied(vllm_gemma4_path: Optional[str | Path] = None) -> bool:
    """Return True if both fixes are present in the active vLLM gemma4
    module. Does NOT modify anything.
    """
    target = _locate_vllm_gemma4(vllm_gemma4_path)
    src = target.read_text(encoding="utf-8")
    fix_a_ok = _FIX_A_AFTER in src and _FIX_A_BEFORE not in src
    fix_b_ok = _FIX_B_RE.search(src) is None
    return fix_a_ok and fix_b_ok


def restore_manual_fixes(vllm_gemma4_path: Optional[str | Path] = None) -> bool:
    """Restore the pre-patch backup if one exists. Returns True on
    success, False if no backup was found."""
    target = _locate_vllm_gemma4(vllm_gemma4_path)
    backup = target.with_suffix(target.suffix + BACKUP_SUFFIX)
    if not backup.exists():
        log.warning("[gemma4_patched] no backup at %s; nothing to restore",
                    backup)
        return False
    shutil.copy2(backup, target)
    log.info("[gemma4_patched] restored %s from %s", target, backup)
    sys.modules.pop("vllm.model_executor.models.gemma4", None)
    return True


def _main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Apply / verify / restore the two source-level "
                    "patches to vLLM's gemma4.py needed by the NVFP4 model.")
    ap.add_argument("action", choices=["apply", "verify", "restore"])
    ap.add_argument("--path", default=None,
                    help="Override the vllm gemma4.py path (default: "
                         "auto-detect from the active venv).")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.action == "apply":
        try:
            changed = apply_manual_fixes(args.path)
        except Exception as e:
            log.error("apply failed: %s", e)
            return 2
        return 0 if (changed or verify_manual_fixes_applied(args.path)) else 3
    if args.action == "verify":
        ok = verify_manual_fixes_applied(args.path)
        print("verify_manual_fixes_applied =", ok)
        return 0 if ok else 1
    if args.action == "restore":
        return 0 if restore_manual_fixes(args.path) else 1
    return 64  # EX_USAGE


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
