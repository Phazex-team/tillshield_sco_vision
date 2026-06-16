"""The evidence package file's recorded sha must match the file."""
from __future__ import annotations

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


def test_written_package_file_sha_matches_embedded_value(tmp_path,
                                                         monkeypatch):
    SM = _fresh_session(tmp_path, monkeypatch)
    from db.models import Case
    from evidence.package import write_package
    from evidence.verify import verify_package_file

    with SM() as s:
        case = Case(camera_id="cam_01", status="OPEN",
                    opened_at=datetime(2026, 6, 15, 14))
        s.add(case)
        s.commit()
        case_id = case.id
        pkg = write_package(s, case_id)
        s.commit()

    # The file's self-described sha (audit.package_sha256) matches the
    # actual file hash computed under the zeroed-field scheme.
    result = verify_package_file(pkg["uri"])
    assert result["ok"] is True, result
    assert result["embedded"] == pkg["sha256"]

    # The artifact row's stored sha also equals the file's recorded sha.
    from db.models import Artifact
    with SM() as s:
        art = s.query(Artifact).filter(
            Artifact.case_id == case_id,
            Artifact.artifact_type == "PACKAGE").first()
    assert art.sha256 == pkg["sha256"]
    assert art.artifact_metadata.get("hash_scheme") == \
        "sha256_with_zeroed_field"


def test_content_sha_is_separately_recorded(tmp_path, monkeypatch):
    """The audit.content_sha256 captures the package contents BEFORE
    the package_sha256 field is added. file sha != content sha by
    design — content sha tracks payload changes; file sha is a tamper
    seal."""
    SM = _fresh_session(tmp_path, monkeypatch)
    from db.models import Case
    from evidence.package import write_package
    from evidence.verify import verify_package_file

    with SM() as s:
        case = Case(camera_id="cam_01", status="OPEN",
                    opened_at=datetime(2026, 6, 15, 14))
        s.add(case)
        s.commit()
        pkg = write_package(s, case.id)
        s.commit()
    assert verify_package_file(pkg["uri"])["ok"]
    assert pkg["content_sha256"] and len(pkg["content_sha256"]) == 64
    assert pkg["sha256"] != pkg["content_sha256"]


def test_tamper_detection(tmp_path, monkeypatch):
    """Any modification to the written file must break the
    self-verifying hash."""
    SM = _fresh_session(tmp_path, monkeypatch)
    from db.models import Case
    from evidence.package import write_package
    from evidence.verify import verify_package_file

    with SM() as s:
        case = Case(camera_id="cam_01", status="OPEN",
                    opened_at=datetime(2026, 6, 15, 14))
        s.add(case)
        s.commit()
        pkg = write_package(s, case.id)
        s.commit()
    p = Path(pkg["uri"])
    text = p.read_text()
    # Flip the camera_id silently.
    p.write_text(text.replace("cam_01", "cam_02"))
    result = verify_package_file(p)
    assert result["ok"] is False
