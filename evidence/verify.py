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
    """Verify a written evidence package.

    Returns:

      * ``ok``: True iff the self-verifying ``audit.package_sha256``
        inside the file matches the recomputed zeroed-field hash.
      * ``embedded``: the value of ``audit.package_sha256`` inside the
        file.
      * ``recomputed``: sha256 of the file with the ``package_sha256``
        field zeroed (must equal ``embedded`` when ``ok`` is True).
      * ``literal_file_sha256``: sha256 of the actual bytes on disk.
        This is what ``sha256sum pkg_*.json`` reports and what
        ``Artifact.sha256`` stores. It is NOT embedded inside the file
        — a file cannot truthfully contain its own literal sha.
    """
    p = Path(path)
    blob = p.read_bytes()
    literal_file_sha = hashlib.sha256(blob).hexdigest()
    text = blob.decode("utf-8", errors="replace")
    m = SCHEMA_FIELD_RE.search(text)
    if not m:
        return {"ok": False, "reason": "no package_sha256 field",
                "literal_file_sha256": literal_file_sha,
                "path": str(p)}
    embedded = m.group(1)
    zeroed = SCHEMA_FIELD_RE.sub(
        '"package_sha256": "' + ("0" * 64) + '"', text)
    recomputed = hashlib.sha256(zeroed.encode("utf-8")).hexdigest()
    return {
        "ok": embedded == recomputed,
        "embedded": embedded,
        "recomputed": recomputed,
        "literal_file_sha256": literal_file_sha,
        "path": str(p),
    }
