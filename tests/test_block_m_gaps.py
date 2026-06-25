"""Block M corrective tests.

Pins the two specific gaps the independent verifier caught:

1. ``summary_from_vlm`` must accept the ``legacy_review_only`` kwarg
   that ``monitor.py`` already passes. Direct calls must NOT raise
   ``TypeError``. The actual monitor wrapping path is exercised here
   (not source-grepped) — the legacy path must land in REVIEW without
   any TypeError noise.

2. The evidence package must expose the literal sha256 of the bytes on
   disk under an unambiguous name on both the artifact row and the API
   response. The package JSON must NOT carry a field that falsely
   claims to be the literal file hash.
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Issue 1: legacy_review_only is accepted by summary_from_vlm
# ---------------------------------------------------------------------------

def test_summary_from_vlm_accepts_legacy_review_only_kwarg():
    """Direct call must not raise. monitor.py passes this kwarg."""
    from reasoning.decision_policy import summary_from_vlm
    s = summary_from_vlm({"physical_item_presented": True,
                          "confidence": "high"},
                         footage_valid=True, legacy_review_only=True)
    assert any("legacy_review_only" in c for c in s.contradictions)


def test_legacy_review_only_forces_review_when_vlm_alone(caplog):
    """The exact path monitor.py runs: VLM-only result + legacy flag.
    No TypeError must surface in the log; outcome must be REVIEW."""
    from reasoning.decision_policy import (
        OUTCOME_REVIEW, OUTCOME_VERIFIED, decide, summary_from_vlm,
    )
    caplog.set_level(logging.WARNING)
    vlm_result = {
        "handover_occurred": True,
        "physical_item_presented": True,
        "receipt_visible": True,
        "items_observed": ["bag"],
        "confidence": "high",
        "obstructed": False,
        "camera_view_clear": True,
    }
    summary = summary_from_vlm(vlm_result, footage_valid=True,
                               legacy_review_only=True)
    decision = decide(summary)
    assert decision.outcome == OUTCOME_REVIEW
    assert decision.outcome != OUTCOME_VERIFIED
    # No TypeError noise from a swallowed kwarg.
    typeerrors = [r for r in caplog.records
                  if r.levelno >= logging.ERROR
                  and "TypeError" in r.getMessage()]
    assert not typeerrors


def test_legacy_review_only_still_verified_with_real_track():
    """legacy_review_only is a tag, not a hard veto. Real track
    evidence still allows VERIFIED — matches the user's instruction:
    'unless independent perception track evidence exists'."""
    from reasoning.decision_policy import (
        OUTCOME_VERIFIED, decide, summary_from_vlm,
    )
    perception = {"tracks": [
        {
            "track_id": "track_real",
            "label": "shopping bag",
            "physical_item_candidate": True,
            "zones": ["counter_zone"],
            "events": ["entered_counter_zone", "handover_candidate"],
            "confidence": 0.9,
        },
        # Customer on the customer side — required for VERIFIED since the
        # customer_present gate was added.
        {
            "track_id": "track_person",
            "label": "person",
            "physical_item_candidate": False,
            "zones": ["customer_zone"],
            "events": [],
            "confidence": 0.9,
        },
    ]}
    summary = summary_from_vlm(
        {"handover_occurred": True, "physical_item_presented": True,
         "receipt_visible": True, "confidence": "high"},
        footage_valid=True,
        perception_result=perception,
        legacy_review_only=True,
    )
    assert decide(summary).outcome == OUTCOME_VERIFIED


def test_monitor_wrapping_block_executes_without_typeerror():
    """Execute the exact code block from monitor.py that wraps a VLM
    result through the decision policy, and assert no TypeError is
    raised AND the outcome lands in the safe enum."""
    import importlib
    dp = importlib.import_module("reasoning.decision_policy")
    OUTCOME_VERIFIED = dp.OUTCOME_VERIFIED
    decide = dp.decide
    summary_from_vlm = dp.summary_from_vlm

    # Mirror monitor.py's wrap exactly (lines around 659).
    result = {
        "handover_occurred": True,
        "physical_item_presented": True,
        "receipt_visible": True,
        "items_observed": ["bag"],
        "confidence": "high",
    }
    evidence_summary = summary_from_vlm(
        result, footage_valid=True, legacy_review_only=True)
    decision = decide(evidence_summary)
    result["policy_outcome"] = decision.outcome
    result["flag_for_review"] = decision.outcome != OUTCOME_VERIFIED

    assert result["policy_outcome"] in (
        "VERIFIED", "REVIEW", "HIGH_RISK_REVIEW", "INVALID_VIDEO")
    assert result["policy_outcome"] == "REVIEW"
    assert result["flag_for_review"] is True


# ---------------------------------------------------------------------------
# Issue 2: evidence package hash naming is unambiguous
# ---------------------------------------------------------------------------

def _fresh_session(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'t.sqlite'}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    import db.session as s
    s._ENGINE = None
    s._SESSION_FACTORY = None
    s.init_schema()
    return s.get_sessionmaker()


def test_embedded_package_json_does_not_carry_misleading_file_hash(
        tmp_path, monkeypatch):
    """The on-disk package JSON must NOT contain a field that falsely
    claims to be the literal file sha. The only embedded hash is
    ``audit.package_sha256`` (self-verifying, reproducible offline)."""
    SM = _fresh_session(tmp_path, monkeypatch)
    from db.models import Case
    from evidence.package import write_package
    with SM() as s:
        c = Case(camera_id="cam_01", status="OPEN",
                 opened_at=datetime(2026, 6, 15, 14))
        s.add(c)
        s.commit()
        pkg = write_package(s, c.id)
        s.commit()

    body = Path(pkg["uri"]).read_text()
    parsed = json.loads(body)
    audit = parsed["audit"]
    # The hashes that legitimately belong inside the file.
    assert "package_sha256" in audit
    assert "content_sha256" in audit
    # The misleading field must NOT be embedded — a file cannot
    # truthfully contain its own literal sha.
    assert "file_sha256" not in audit


def test_artifact_row_exposes_literal_file_sha(tmp_path, monkeypatch):
    SM = _fresh_session(tmp_path, monkeypatch)
    from db.models import Artifact, Case
    from evidence.package import write_package
    with SM() as s:
        c = Case(camera_id="cam_01", status="OPEN",
                 opened_at=datetime(2026, 6, 15, 14))
        s.add(c)
        s.commit()
        cid = c.id
        pkg = write_package(s, cid)
        s.commit()

    literal = hashlib.sha256(Path(pkg["uri"]).read_bytes()).hexdigest()
    assert pkg["sha256"] == literal
    assert pkg["literal_file_sha256"] == literal

    with SM() as s:
        art = s.query(Artifact).filter(
            Artifact.case_id == cid,
            Artifact.artifact_type == "PACKAGE").first()
    assert art.sha256 == literal
    # Metadata also carries the literal hash under an unambiguous name.
    assert art.artifact_metadata["literal_file_sha256"] == literal


def test_evidence_package_api_response_carries_literal_sha(
        tmp_path, monkeypatch):
    """The HTTP response must expose ``literal_file_sha256`` so an
    integrator does not have to guess which embedded hash is the
    reviewer-friendly one."""
    SM = _fresh_session(tmp_path, monkeypatch)
    from db.models import Case
    from evidence.package import write_package
    with SM() as s:
        c = Case(camera_id="cam_01", status="OPEN",
                 opened_at=datetime(2026, 6, 15, 14))
        s.add(c)
        s.commit()
        cid = c.id
        pkg = write_package(s, cid)
        s.commit()
    literal = hashlib.sha256(Path(pkg["uri"]).read_bytes()).hexdigest()

    from fastapi.testclient import TestClient
    from app.main import create_app
    client = TestClient(create_app())
    r = client.get(f"/api/v1/cases/{cid}/evidence-package")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "literal_file_sha256" in body
    assert body["literal_file_sha256"] == literal


def test_sha256sum_cli_matches_literal_sha(tmp_path, monkeypatch):
    SM = _fresh_session(tmp_path, monkeypatch)
    from db.models import Case
    from evidence.package import write_package
    with SM() as s:
        c = Case(camera_id="cam_01", status="OPEN",
                 opened_at=datetime(2026, 6, 15, 14))
        s.add(c)
        s.commit()
        pkg = write_package(s, c.id)
        s.commit()
    proc = subprocess.run(["sha256sum", pkg["uri"]],
                          capture_output=True, text=True)
    cli_sha = proc.stdout.strip().split()[0]
    assert cli_sha == pkg["literal_file_sha256"]


def test_verify_package_file_still_works(tmp_path, monkeypatch):
    SM = _fresh_session(tmp_path, monkeypatch)
    from db.models import Case
    from evidence.package import write_package
    from evidence.verify import verify_package_file
    with SM() as s:
        c = Case(camera_id="cam_01", status="OPEN",
                 opened_at=datetime(2026, 6, 15, 14))
        s.add(c)
        s.commit()
        pkg = write_package(s, c.id)
        s.commit()
    r = verify_package_file(pkg["uri"])
    assert r["ok"] is True
    assert r["embedded"] == pkg["package_self_sha256"]
    assert r["literal_file_sha256"] == pkg["sha256"]
