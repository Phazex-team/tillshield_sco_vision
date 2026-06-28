"""End-to-end: SAM3 identities + VLM semantic reasoning, prompt v2 + policy v2.

Pins the council's acceptance scenario:
  * SAM3 detects 2 generic food containers in the audit zone
    (no Falcon person tracks — item-occupancy episode fallback).
  * VLM says physical_count_match=yes, semantic_identity_match=uncertain
    ("items inside closed takeaway containers").
  * Final outcome: REVIEW with sco_identity_uncertain — NOT a
    basket_mismatch and NOT a missing_items false-flag.
"""
from __future__ import annotations

import io
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'e2e_sam3.sqlite'}")
    import db.session as s
    s._ENGINE = None
    s._SESSION_FACTORY = None
    s.init_schema()
    monkeypatch.chdir(tmp_path)
    from video.window_builder import WindowBuildResult

    def _ok_build(*, segments, requested_start, requested_end, out_path):
        out_path = str(out_path)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(b"\x00" * 1024)
        return WindowBuildResult(
            ok=True, out_path=out_path, sha256="0" * 64,
            actual_start_at=requested_start, actual_end_at=requested_end,
            segment_ids=[s.id for s in segments],
        )
    monkeypatch.setattr("video.window_builder.build_window", _ok_build)
    monkeypatch.setattr("app.case_runner._extract_keyframe_data_urls",
                        lambda **kw: [])
    yield s.get_sessionmaker()


def _seed_segment(SM, storage_root, *, start_at, duration_sec=300):
    from db.models import VideoSegment
    storage_root.mkdir(parents=True, exist_ok=True)
    seg = storage_root / "cctv" / "cam_01" / "seg_a.mp4"
    seg.parent.mkdir(parents=True, exist_ok=True)
    seg.write_bytes(b"\x00" * 4096)
    with SM() as s:
        s.add(VideoSegment(camera_id="cam_01",
                           start_at=start_at,
                           end_at=start_at + timedelta(seconds=duration_sec),
                           path=str(seg)))
        s.commit()


def _post_event(pos_time):
    from app.api.pos import ingest_event, PosEventBody
    body = PosEventBody(
        store_id="store_sco_1", terminal_id="t1",
        transaction_id="txn-SAM3-VLM-COMBO",
        line_id="transaction", event_type="SALE",
        pos_event_at=pos_time,
        raw_payload={"items": [
            {"description": "Biriyani Hot Food", "quantity": 1},
            {"description": "Curry Hot Food", "quantity": 1},
        ]},
    )
    fake = type("R", (), {"client": None,
                          "headers": {"x-forwarded-for": "127.0.0.1"}})()
    ingest_event(body, fake)


def _sam3_perception_two_closed_containers(base):
    """Synthetic SAM3 perception output: 2 stable identities, both
    extra-candidates (no POS-specific concept fired), in the
    sco_audit_zone, spanning the POS time."""
    det0 = {
        "label": "sco_generic_food_container", "score": 0.82,
        "bbox_xyxy": [746, 512, 824, 575],
        "frame_id": "frame_000350", "frame_idx": 350,
        "ts": base.isoformat(), "sam3_object_id": 3,
        "query": "sco_generic_food_container",
    }
    det1 = {
        "label": "sco_generic_plastic_food_box", "score": 0.71,
        "bbox_xyxy": [893, 733, 953, 792],
        "frame_id": "frame_001000", "frame_idx": 1000,
        "ts": (base + timedelta(seconds=2)).isoformat(),
        "sam3_object_id": 4,
        "query": "sco_generic_plastic_food_box",
    }
    return {
        "detections": [det0, det1],
        "tracks": [
            {"track_id": "sam3_obj_0003",
             "label": "sco_generic_food_container",
             "first_seen_ts": (base - timedelta(seconds=10)).isoformat(),
             "last_seen_ts": (base + timedelta(seconds=10)).isoformat(),
             "detections": [0], "zones": ["sco_audit_zone"],
             "events": [], "physical_item_candidate": True,
             "receipt_candidate": False, "confidence": 0.82,
             "sam3_object_id": 3},
            {"track_id": "sam3_obj_0004",
             "label": "sco_generic_plastic_food_box",
             "first_seen_ts": (base - timedelta(seconds=8)).isoformat(),
             "last_seen_ts": (base + timedelta(seconds=12)).isoformat(),
             "detections": [1], "zones": ["sco_audit_zone"],
             "events": [], "physical_item_candidate": True,
             "receipt_candidate": False, "confidence": 0.71,
             "sam3_object_id": 4},
        ],
        "masks": [], "keyframes": [], "ocr": [],
        "limitations": ["falcon_disabled_by_config"],
        "obstructed": False,
        "timings_ms": {"total_ms": 1, "sam3_inference_ms": 15000},
        "sam3_meta": {"object_ids": [3, 4], "frame_count": 24,
                      "prompt_to_obj_ids": {
                          "sco_item_000": [],
                          "sco_item_001": [],
                          "sco_generic_food_container": [3],
                          "sco_generic_plastic_food_box": [4],
                      }},
    }


def _vlm_says_count_match_identity_uncertain(captured: dict):
    """Council-prescribed VLM behaviour for the closed-container case."""

    def _stub(session, case, window, manifest=None):
        captured["manifest_user_prompt"] = manifest.user_prompt \
            if manifest else None
        captured["prompt_version"] = (manifest.metadata or {}).get(
            "prompt_version") if manifest else None
        captured["canonical_group_count"] = len(
            (manifest.metadata or {}).get("sco_canonical_groups") or []
        ) if manifest else 0
        return {
            "provider": "qwen3_vl", "model_name": "stub",
            "parsed": {
                "physical_count_match": "yes",
                "semantic_identity_match": "uncertain",
                "matched_items": [
                    {"pos_item": "Biriyani Hot Food",
                     "group_id": "sco_group_001",
                     "visible_count_class": "one"},
                    {"pos_item": "Curry Hot Food",
                     "group_id": "sco_group_002",
                     "visible_count_class": "one"},
                ],
                "missing_visible_items": [],
                "extra_visible_items": [],
                "uncertainty_reason":
                    "items inside closed takeaway containers",
                "video_usable": True,
                "confidence": "high",
                "narrative": "Two takeaway containers visible; contents "
                             "not legible from this angle.",
            },
            "latency_ms": 1, "error": None,
        }
    return _stub


# ---------------------------------------------------------------------------
# Council acceptance: closed-container case → REVIEW + sco_identity_uncertain
# ---------------------------------------------------------------------------

def test_sam3_two_containers_closed_yields_review_with_identity_uncertain(
        fresh_db, tmp_path, monkeypatch):
    SM = fresh_db
    pos_time = datetime(2026, 6, 28, 14, 0, 30)
    _seed_segment(SM, tmp_path / "storage",
                  start_at=pos_time - timedelta(seconds=120))
    _post_event(pos_time)

    from db.models import Case
    with SM() as s:
        case_id = s.query(Case).first().id

    def _perception(session, case, window):
        return _sam3_perception_two_closed_containers(pos_time)

    captured: dict = {}
    vlm_stub = _vlm_says_count_match_identity_uncertain(captured)

    from app.case_runner import analyze_case
    with SM() as s:
        result = analyze_case(s, case_id,
                               perception_runner=_perception,
                               vlm_runner=vlm_stub,
                               prompt_version="sco_basket_match_v2")

    # The case lands in REVIEW (NOT verified, NOT invalid_video)
    assert result["outcome"] == "REVIEW"
    reasons = result.get("reasons") or []
    # The substantive signal: identity uncertain, NOT mismatch
    assert "sco_identity_uncertain" in reasons, reasons
    assert "sco_basket_mismatch" not in reasons, (
        "closed-container case must not be tagged as basket_mismatch")
    assert "sco_missing_items" not in reasons, (
        "policy must suppress missing-items false flag when identity "
        "is uncertain")
    # And the v2 policy version surfaced
    from db.models import Case as _C
    with SM() as s:
        case = s.get(_C, case_id)
        assert case.decision_policy_version == "sco_v2"

    # Wiring evidence: the v2 prompt was used, and exactly 2 canonical
    # groups (the SAM3 identities) reached the VLM.
    assert captured["prompt_version"] == "sco_basket_match_v2"
    assert captured["canonical_group_count"] == 2
    # And the v2 user prompt forbids re-collapsing identities
    assert "physical_count_match" in (captured["manifest_user_prompt"] or "")
    assert "semantic_identity_match" in (captured["manifest_user_prompt"] or "")


# ---------------------------------------------------------------------------
# Genuine semantic contradiction is still a mismatch
# ---------------------------------------------------------------------------

def test_sam3_with_semantic_contradiction_still_flags_mismatch(
        fresh_db, tmp_path, monkeypatch):
    """If POS says food but the VLM clearly sees electronics, that is
    a real mismatch — v2 must still flag it."""
    SM = fresh_db
    pos_time = datetime(2026, 6, 28, 14, 0, 30)
    _seed_segment(SM, tmp_path / "storage",
                  start_at=pos_time - timedelta(seconds=120))
    _post_event(pos_time)
    from db.models import Case
    with SM() as s:
        case_id = s.query(Case).first().id

    def _perception(s, c, w):
        return _sam3_perception_two_closed_containers(pos_time)

    def _stub(session, case, window, manifest=None):
        return {
            "provider": "qwen3_vl", "model_name": "stub",
            "parsed": {
                "physical_count_match": "yes",
                "semantic_identity_match": "no",
                "matched_items": [],
                "missing_visible_items": [],
                "extra_visible_items": [
                    {"group_id": "sco_group_001",
                     "description": "Laptop"}],
                "uncertainty_reason": "",
                "video_usable": True, "confidence": "high",
                "narrative": "POS says food, visible items are electronics.",
            },
            "latency_ms": 1, "error": None,
        }

    from app.case_runner import analyze_case
    with SM() as s:
        result = analyze_case(s, case_id,
                               perception_runner=_perception,
                               vlm_runner=_stub,
                               prompt_version="sco_basket_match_v2")
    assert result["outcome"] == "REVIEW"
    reasons = result.get("reasons") or []
    assert "sco_basket_mismatch" in reasons
    assert "sco_extra_candidates" in reasons


# ---------------------------------------------------------------------------
# Episode-fallback regression: SAM3-only mode (no person tracks) still
# produces a valid episode via item-occupancy
# ---------------------------------------------------------------------------

def test_sam3_only_mode_has_item_occupancy_episode(
        fresh_db, tmp_path, monkeypatch):
    SM = fresh_db
    pos_time = datetime(2026, 6, 28, 14, 0, 30)
    _seed_segment(SM, tmp_path / "storage",
                  start_at=pos_time - timedelta(seconds=120))
    _post_event(pos_time)
    from db.models import Case
    with SM() as s:
        case_id = s.query(Case).first().id

    def _perception(s, c, w):
        # No person tracks at all — just two SAM3 item identities
        return _sam3_perception_two_closed_containers(pos_time)

    def _stub(session, case, window, manifest=None):
        ep = (manifest.metadata or {}).get("sco_episode") if manifest else None
        # Verify episode falls back to item_occupancy (not no_activity)
        # in SAM3-only mode.
        assert ep is not None
        assert ep.get("reason") == "item_occupancy"
        assert ep.get("ambiguous") is False
        assert ep.get("coverage_ratio", 0) > 0.0
        return {
            "provider": "qwen3_vl", "model_name": "stub",
            "parsed": {
                "physical_count_match": "yes",
                "semantic_identity_match": "uncertain",
                "matched_items": [], "missing_visible_items": [],
                "extra_visible_items": [],
                "uncertainty_reason": "closed containers",
                "video_usable": True, "confidence": "high",
                "narrative": "ok",
            },
            "latency_ms": 1, "error": None,
        }

    from app.case_runner import analyze_case
    with SM() as s:
        result = analyze_case(s, case_id,
                               perception_runner=_perception,
                               vlm_runner=_stub,
                               prompt_version="sco_basket_match_v2")
    # No sco_episode_short tag if item-occupancy fallback produced
    # a usable coverage ratio
    reasons = result.get("reasons") or []
    assert "sco_episode_short" not in reasons, reasons
