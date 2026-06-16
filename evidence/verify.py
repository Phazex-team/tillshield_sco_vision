"""Verify a written evidence package matches its embedded hash.

Hash scheme ``sha256_with_zeroed_field``:

  1. Read the file.
  2. Replace the ``"package_sha256": "<64 hex>"`` field value with 64
     ``"0"`` characters.
  3. Compute sha256 of the resulting bytes.
  4. Compare to the original embedded ``package_sha256``.

This lets the file contain its own integrity hash without the
chicken-and-egg problem.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path


SCHEMA_FIELD_RE = re.compile(
    r'"package_sha256":\s*"([0-9a-f]{64})"'
)


def verify_package_file(path: str | Path) -> dict:
    p = Path(path)
    blob = p.read_bytes()
    m = SCHEMA_FIELD_RE.search(blob.decode("utf-8", errors="replace"))
    if not m:
        return {"ok": False, "reason": "no package_sha256 field"}
    embedded = m.group(1)
    zeroed = SCHEMA_FIELD_RE.sub(
        '"package_sha256": "' + ("0" * 64) + '"',
        blob.decode("utf-8"))
    recomputed = hashlib.sha256(zeroed.encode("utf-8")).hexdigest()
    return {
        "ok": embedded == recomputed,
        "embedded": embedded,
        "recomputed": recomputed,
        "path": str(p),
    }
