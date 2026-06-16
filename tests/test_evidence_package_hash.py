"""Evidence package hash semantics — both names land correctly.

Three distinct hashes are recorded for every package:

  * ``audit.content_sha256`` — payload hash before the package
    self-hash field is added.
  * ``audit.package_sha256`` — self-verifying zeroed-field hash;
    ``evidence.verify_package_file`` can reproduce it offline.
  * ``audit.file_sha256`` / ``Artifact.sha256`` — literal sha256 of
    the bytes on disk. Reviewers can verify with
    ``sha256sum pkg_*.json``.

These tests pin each.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _fresh_session(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'t.sqlite'}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    import db.session as s
    s._ENGINE = None
    s._SESSION_FACTORY = None
    s.init_schema()
    return s.get_sessionmaker()


def _new_pkg(SM):
    from db.models import Case
    from evidence.package import write_package
    with SM() as s:
        case = Case(camera_id="cam_01", status="OPEN",
                    opened_at=datetime(2026, 6, 15, 14))
        s.add(case)
        s.commit()
        pkg = write_package(s, case.id)
        s.commit()
        return pkg, case.id


def test_artifact_sha_equals_literal_file_sha(tmp_path, monkeypatch):
    """``Artifact.sha256`` is the literal sha256 of the bytes on disk —
    what ``sha256sum`` reports."""
    SM = _fresh_session(tmp_path, monkeypatch)
    pkg, case_id = _new_pkg(SM)
    literal = hashlib.sha256(Path(pkg["uri"]).read_bytes()).hexdigest()
    assert pkg["sha256"] == literal
    assert pkg["literal_file_sha256"] == literal

    from db.models import Artifact
    with SM() as s:
        art = s.query(Artifact).filter(
            Artifact.case_id == case_id,
            Artifact.artifact_type == "PACKAGE").first()
    assert art.sha256 == literal


def test_self_verifying_hash_reproducible(tmp_path, monkeypatch):
    """The self-verifying hash inside the file matches what
    ``evidence.verify_package_file`` recomputes under the zeroed-field
    scheme. ``Artifact.metadata.package_self_sha256`` keeps the same
    value for downstream tooling."""
    SM = _fresh_session(tmp_path, monkeypatch)
    pkg, case_id = _new_pkg(SM)

    from evidence.verify import verify_package_file
    r = verify_package_file(pkg["uri"])
    assert r["ok"] is True, r
    assert r["embedded"] == pkg["package_self_sha256"]
    # Literal file sha is exposed on the verify result too.
    assert r["literal_file_sha256"] == pkg["sha256"]

    from db.models import Artifact
    with SM() as s:
        art = s.query(Artifact).filter(
            Artifact.case_id == case_id,
            Artifact.artifact_type == "PACKAGE").first()
    md = art.artifact_metadata
    assert md["package_self_sha256"] == r["embedded"]
    assert md["hash_scheme"] == "sha256_with_zeroed_field"


def test_sha256sum_reproduces_artifact_sha(tmp_path, monkeypatch):
    """A reviewer can verify integrity with the standard ``sha256sum``
    cli — no special knowledge of the zeroed-field scheme required."""
    SM = _fresh_session(tmp_path, monkeypatch)
    pkg, _ = _new_pkg(SM)
    proc = subprocess.run(["sha256sum", pkg["uri"]],
                          capture_output=True, text=True)
    cli_sha = proc.stdout.strip().split()[0]
    assert cli_sha == pkg["sha256"]


def test_content_sha_distinct_from_file_sha(tmp_path, monkeypatch):
    SM = _fresh_session(tmp_path, monkeypatch)
    pkg, _ = _new_pkg(SM)
    assert pkg["content_sha256"] and len(pkg["content_sha256"]) == 64
    assert pkg["sha256"] != pkg["content_sha256"]
    assert pkg["package_self_sha256"] != pkg["sha256"]


def test_tamper_breaks_self_verifying_hash(tmp_path, monkeypatch):
    SM = _fresh_session(tmp_path, monkeypatch)
    pkg, _ = _new_pkg(SM)
    from evidence.verify import verify_package_file
    p = Path(pkg["uri"])
    p.write_text(p.read_text().replace("cam_01", "cam_02"))
    assert verify_package_file(p)["ok"] is False
