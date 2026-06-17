"""Review UI safety + structural checks.

The dashboard is the only user-facing surface for reviewers. These
tests pin the contract that no accusation language sneaks back in.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Absolute bans: these words must never appear in any static UI, even
# as CSS class names or JS identifiers.
FORBIDDEN_EVERYWHERE = (
    "fraud", "fraudulent", "theft", "suspect", "accusation",
    "loss-prevention", "loss_prevention",
)

# The new reviewer UI also bans "flagged" — there is no legacy data
# field to keep compatibility with. The legacy index.html still ships
# `flagged` as a CSS class name and a JSON field reference (s.flagged);
# those are internal identifiers, not user-visible labels.
FORBIDDEN_IN_REVIEW_UI = (
    "flagged",
)


def _texts():
    return [
        (ROOT / "static" / "index.html").read_text(),
        (ROOT / "static" / "review.html").read_text(),
    ]


def test_no_accusation_language_in_any_static_ui():
    for src in _texts():
        low = src.lower()
        for phrase in FORBIDDEN_EVERYWHERE:
            assert phrase not in low, \
                f"forbidden phrase {phrase!r} present in static UI"


def test_review_ui_strict_word_ban():
    src = (ROOT / "static" / "review.html").read_text().lower()
    for phrase in FORBIDDEN_IN_REVIEW_UI:
        assert phrase not in src, \
            f"{phrase!r} present in the production reviewer UI"


def test_review_ui_exposes_review_safe_actions():
    src = (ROOT / "static" / "review.html").read_text()
    for a in ("verified_physical_return", "needs_review",
              "high_risk_review", "invalid_video",
              "camera_blind_spot", "pos_camera_mismatch"):
        assert a in src, f"missing reviewer action {a!r}"
    for outcome in ("VERIFIED", "REVIEW", "HIGH_RISK_REVIEW",
                    "INVALID_VIDEO"):
        assert outcome in src


def test_review_ui_calls_v1_api():
    """The UI uses a JS template literal (``${API}/cases``) with
    ``const API = "/api/v1"``. Check for the constant + the
    sub-paths."""
    src = (ROOT / "static" / "review.html").read_text()
    assert 'const API = "/api/v1"' in src
    assert "/cases" in src
    assert "/memory" in src
    assert "/review-actions" in src
    assert "/reprocess" in src


def test_review_ui_wires_video_stream():
    src = (ROOT / "static" / "review.html").read_text()
    assert "/video/windows/" in src and "/stream" in src
    # The UI must actually set video.src to the stream endpoint.
    assert "video.src" in src
    assert "latest_window" in src


def test_review_ui_renders_real_perception_payload():
    src = (ROOT / "static" / "review.html").read_text()
    assert "perception.keyframes" in src or "perception.tracks" in src
    assert "tracker_id" in src
    assert "frame_id" in src


# ---------------------------------------------------------------------------
# Operations console — Pipeline / Prompts / Storage / Config tabs
# ---------------------------------------------------------------------------

def test_review_ui_has_five_tabs():
    src = (ROOT / "static" / "review.html").read_text()
    for name in ("Cases", "Pipeline", "Prompts", "Storage", "Config"):
        assert name in src, f"tab label {name!r} missing"
    for ident in ("tab-cases", "tab-pipeline", "tab-prompts",
                  "tab-storage", "tab-config"):
        assert ident in src, f"tab pane id {ident!r} missing"
    # All four status pills must be styled so the UI can render them.
    for pill in (".pill.OK", ".pill.WARNING", ".pill.ERROR", ".pill.UNKNOWN"):
        assert pill in src, f"missing pill class {pill!r}"


def test_review_ui_pipeline_calls_ops_status():
    src = (ROOT / "static" / "review.html").read_text()
    assert "/ops/status" in src
    assert "refreshOps" in src


def test_review_ui_prompts_call_admin_prompts():
    src = (ROOT / "static" / "review.html").read_text()
    assert "/admin/prompts" in src
    # Prompt editor must send the admin token header and the three fields.
    assert "X-PhazeX-Admin-Token" in src
    for fid in ("prompt-falcon", "prompt-gemma-system",
                "prompt-gemma-user", "prompt-safety",
                "prompt-save-status"):
        assert fid in src


def test_review_ui_storage_calls_disk_and_cleanup():
    src = (ROOT / "static" / "review.html").read_text()
    assert "/storage/disk" in src
    assert "/storage/cleanup/dry-run" in src
    assert "/storage/cleanup/execute" in src
    # Execute button is gated by the admin-token input.
    assert "storage-admin-token" in src


def test_review_ui_config_calls_admin_config_readonly():
    src = (ROOT / "static" / "review.html").read_text()
    assert "/admin/config" in src
    # Config tab is read-only — no YAML/POST/PATCH writers.
    assert "config-cameras" in src
    assert "config-models" in src


def test_review_ui_no_start_stop_external_processes():
    """The console must NOT pretend to start/stop vLLM, Gemma, the
    recorder, or the POS poller — those are managed externally."""
    src = (ROOT / "static" / "review.html").read_text().lower()
    for forbidden in (
        "start vllm", "stop vllm",
        "start gemma", "stop gemma",
        "start recorder", "stop recorder",
        "start poller", "stop poller",
    ):
        assert forbidden not in src, \
            f"UI must not pretend to control external process: {forbidden!r}"
