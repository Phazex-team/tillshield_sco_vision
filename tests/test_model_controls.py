"""Model-control surface tests — config-level enable/disable for the
five gated stages (Falcon / SAM 2 / OCR / Qwen3-VL / Gemma).

Covers:
  * GET /api/v1/admin/model-controls returns all five with metadata.
  * PATCH persists into config.yaml, audits, is admin-token gated.
  * Validation rules: unknown keys, non-bool values, dependency rules,
    "at least one independent source" rule.
  * Runtime gating: disabled Falcon short-circuits perception with
    ``falcon_disabled_by_config``; disabled SAM 2 / OCR skip honestly
    without claiming "unavailable".
  * Provider chain shapes: Qwen-only, Gemma-only, both-disabled
    degraded provider.
  * Config snapshot: a mid-case config.yaml edit does NOT swap the
    provider for the in-flight run.
"""
from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("ADMIN_EDIT_TOKEN", raising=False)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'t.sqlite'}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))

    cfg_path = ROOT / "config.yaml"
    backup = tmp_path / "config_backup.yaml"
    shutil.copy(cfg_path, backup)

    import db.session as ds
    ds._ENGINE = None
    ds._SESSION_FACTORY = None
    ds.init_schema()
    try:
        from app.memory_guard import get_policy
        get_policy().reset_for_test()
    except Exception:
        pass

    from fastapi.testclient import TestClient
    from app.main import create_app
    c = TestClient(create_app())
    yield c
    shutil.copy(backup, cfg_path)


# ---------------------------------------------------------------------
# GET /admin/model-controls
# ---------------------------------------------------------------------

def test_get_model_controls_returns_all_six_with_metadata(client):
    r = client.get("/api/v1/admin/model-controls")
    assert r.status_code == 200, r.text
    body = r.json()
    # Six toggles after the SAM 3 rebrand: falcon, sam2, ocr, qwen3_vl,
    # gemma, sam3. SAM 3 is independent (does NOT require Falcon) and
    # defaults OFF in active config.
    assert set(body["state"].keys()) == \
        {"falcon", "sam2", "ocr", "qwen3_vl", "gemma", "sam3"}
    ids = {m["id"] for m in body["models"]}
    assert ids == {"falcon", "sam2", "ocr", "qwen3_vl", "gemma", "sam3"}
    by_id = {m["id"]: m for m in body["models"]}
    assert by_id["sam2"]["dependencies"] == ["falcon"]
    assert by_id["ocr"]["dependencies"] == ["falcon"]
    assert by_id["sam3"]["dependencies"] == []
    assert by_id["falcon"]["independent"] is True
    assert by_id["sam2"]["independent"] is False
    assert by_id["sam3"]["independent"] is True
    assert by_id["qwen3_vl"]["independent"] is True
    # Operator-facing copy is the same wording as the UI captions. (Model
    # names are deliberately hidden behind the client-facing scheme:
    # Perception (FL) / Segmenter (S2) / Vision Primary (Q) / Fallback (G).)
    assert "Independent detector" in by_id["falcon"]["caption"]
    assert "Perception (FL)" in by_id["sam2"]["caption"]
    assert by_id["sam2"]["label"] == "Segmenter (S2)"
    assert by_id["falcon"]["label"] == "Perception (FL)"
    # config_key points at the YAML field this surface manages.
    assert by_id["falcon"]["config_key"] == "models.falcon.enabled"
    assert by_id["ocr"]["config_key"] == "models.falcon_ocr.enabled"
    # The endpoint explicitly states changes apply to the next case.
    assert "next case" in body["applies_to"].lower()


# ---------------------------------------------------------------------
# PATCH /admin/model-controls — happy path + persistence + audit
# ---------------------------------------------------------------------

def test_patch_persists_into_config_yaml(client):
    r = client.patch("/api/v1/admin/model-controls", json={
        "falcon": True, "sam2": False, "ocr": False,
        "qwen3_vl": True, "gemma": False, "sam3": True,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["state"] == {"falcon": True, "sam2": False, "ocr": False,
                              "qwen3_vl": True, "gemma": False,
                              "sam3": True}
    # Round-trip via GET (re-reads config.yaml).
    r2 = client.get("/api/v1/admin/model-controls")
    assert r2.json()["state"] == body["state"]
    # config.yaml actually has the flags persisted with the right keys.
    import yaml as _yaml
    data = _yaml.safe_load((ROOT / "config.yaml").read_text())
    assert data["models"]["falcon"]["enabled"] is True
    assert data["models"]["sam2"]["enabled"] is False
    assert data["models"]["falcon_ocr"]["enabled"] is False
    assert data["models"]["qwen3_vl"]["enabled"] is True
    assert data["models"]["gemma"]["enabled"] is False
    assert data["models"]["sam3"]["enabled"] is True


def test_patch_writes_audit_log(client):
    client.patch("/api/v1/admin/model-controls",
                  json={"gemma": False})
    from db.models import AuditLog
    import db.session as ds
    SM = ds.get_sessionmaker()
    with SM() as s:
        rows = s.query(AuditLog).filter(
            AuditLog.action == "admin.model_controls_update").all()
    assert rows
    row = rows[-1]
    # Audit row contains BEFORE + AFTER for the six flags only —
    # no secrets, no unrelated config keys. (Now six: SAM 3 added.)
    after = row.after_json or {}
    assert set(after.keys()) == \
        {"falcon", "sam2", "ocr", "qwen3_vl", "gemma", "sam3"}
    assert after["gemma"] is False


# ---------------------------------------------------------------------
# PATCH validation
# ---------------------------------------------------------------------

def test_patch_rejects_unknown_key(client):
    r = client.patch("/api/v1/admin/model-controls",
                      json={"some_random_model": False})
    assert r.status_code == 400
    assert "unknown" in r.json()["detail"]["error"].lower()


def test_patch_rejects_non_boolean_value(client):
    r = client.patch("/api/v1/admin/model-controls",
                      json={"sam2": "off"})
    assert r.status_code == 400
    assert "boolean" in r.json()["detail"]["error"]


def test_patch_rejects_only_sam2_ocr_enabled(client):
    """Falcon off + Qwen off + Gemma off leaves no independent source
    even if SAM2/OCR are enabled — must be rejected."""
    r = client.patch("/api/v1/admin/model-controls", json={
        "falcon": False, "sam2": True, "ocr": True,
        "qwen3_vl": False, "gemma": False,
    })
    assert r.status_code == 400
    err = r.json()["detail"]["error"]
    # Either the at-least-one rule OR the SAM2/OCR-needs-Falcon rule
    # may fire first. Both are acceptable rejections.
    assert ("independent source" in err) or ("Perception (FL) is disabled" in err)


def test_patch_rejects_sam2_when_falcon_disabled(client):
    r = client.patch("/api/v1/admin/model-controls", json={
        "falcon": False, "sam2": True, "ocr": False,
        "qwen3_vl": True, "gemma": True,
    })
    assert r.status_code == 400
    assert "segmenter (s2)" in r.json()["detail"]["error"].lower()


def test_patch_rejects_ocr_when_falcon_disabled(client):
    r = client.patch("/api/v1/admin/model-controls", json={
        "falcon": False, "sam2": False, "ocr": True,
        "qwen3_vl": True, "gemma": True,
    })
    assert r.status_code == 400
    assert "ocr" in r.json()["detail"]["error"].lower()


def test_patch_requires_token_when_configured(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_EDIT_TOKEN", "phzx_admin")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'t.sqlite'}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    cfg_path = ROOT / "config.yaml"
    backup = tmp_path / "config_backup.yaml"
    shutil.copy(cfg_path, backup)
    import db.session as ds
    ds._ENGINE = None
    ds._SESSION_FACTORY = None
    ds.init_schema()
    from fastapi.testclient import TestClient
    from app.main import create_app
    c = TestClient(create_app())
    try:
        r = c.patch("/api/v1/admin/model-controls", json={"gemma": False})
        assert r.status_code == 401
        r2 = c.patch("/api/v1/admin/model-controls",
                     json={"gemma": False},
                     headers={"X-PhazeX-Admin-Token": "phzx_admin"})
        assert r2.status_code == 200
    finally:
        shutil.copy(backup, cfg_path)


# ---------------------------------------------------------------------
# Runtime gating: perception pipeline respects the flags
# ---------------------------------------------------------------------

def test_perception_skips_falcon_when_disabled(monkeypatch):
    """Falcon disabled → no detect call, empty result, limitation tag."""
    import perception.pipeline as pl
    from perception.sampling import SamplingPolicy
    from perception.temporal_memory import Zone

    fake_frames = [(0, datetime(2026, 6, 17, 14, 0, 0), object())]
    monkeypatch.setattr(pl, "_sample_frames",
                        lambda *a, **k: fake_frames)
    # Falcon client would explode if called.
    sentinel = RuntimeError("falcon must not run when disabled")

    class _Boom:
        def detect_on_frames(self, *a, **k): raise sentinel
        def _ensure_loaded(self): pass

    class _Sam2:
        def has_capability(self): return False
        def segment(self, *a, **k): return []

    result = pl.run_perception_on_window(
        window_path="/tmp/fake.mp4", fps=25,
        zones=[Zone(name="counter_zone", x=0, y=0, w=1000, h=1000)],
        falcon_client=_Boom(),
        sam2_client=_Sam2(),
        sampling=SamplingPolicy(),
        falcon_enabled=False,
    )
    assert result["detections"] == []
    assert result["tracks"] == []
    assert "falcon_disabled_by_config" in result["limitations"]
    # ``falcon_unavailable`` must NOT appear — disabled is operator
    # choice, not a runtime failure.
    assert "falcon_unavailable" not in result["limitations"]


def test_perception_skips_sam2_without_unavailable_tag(monkeypatch):
    """SAM 2 disabled → no segment call, no ``sam2_unavailable`` tag.
    The ``sam2_disabled_by_config`` tag appears only when there were
    detections that would otherwise have been segmented."""
    import perception.pipeline as pl
    from perception.sampling import SamplingPolicy
    from perception.schemas import Detection
    from perception.temporal_memory import Zone

    base = datetime(2026, 6, 17, 14, 0, 0)
    detections = [Detection(label="bag", score=0.9,
                             bbox_xyxy=[10, 10, 50, 50],
                             frame_id="f0", frame_idx=0, ts=base)]
    fake_frames = [(0, base, object())]

    class _Falcon:
        def detect_on_frames(self, frames, *, query, **kwargs):
            return detections
        def _ensure_loaded(self): pass

    sentinel = RuntimeError("sam2 must not run when disabled")

    class _Boom:
        def has_capability(self): return True
        def segment(self, *a, **k): raise sentinel

    monkeypatch.setattr(pl, "_sample_frames",
                        lambda *a, **k: fake_frames)
    monkeypatch.setattr(pl, "run_ocr", lambda *a, **k: ([], []))

    result = pl.run_perception_on_window(
        window_path="/tmp/fake.mp4", fps=25,
        zones=[Zone(name="counter_zone", x=0, y=0, w=1000, h=1000)],
        falcon_client=_Falcon(),
        sam2_client=_Boom(),
        sampling=SamplingPolicy(),
        sam2_enabled=False,
    )
    assert result["tracks"], "falcon still ran; tracks expected"
    assert "sam2_disabled_by_config" in result["limitations"]
    assert "sam2_unavailable" not in result["limitations"]
    assert "sam2_ms" not in result["timings_ms"]


def test_perception_skips_ocr_without_unavailable_tag(monkeypatch):
    import perception.pipeline as pl
    from perception.sampling import SamplingPolicy
    from perception.schemas import Detection
    from perception.temporal_memory import Zone

    base = datetime(2026, 6, 17, 14, 0, 0)
    # A receipt-labelled detection IS an OCR candidate, so the
    # disabled_by_config tag should appear.
    detections = [Detection(label="receipt", score=0.9,
                             bbox_xyxy=[10, 10, 80, 80],
                             frame_id="f0", frame_idx=0, ts=base)]
    fake_frames = [(0, base, object())]

    class _Falcon:
        def detect_on_frames(self, frames, *, query, **kwargs):
            return detections
        def _ensure_loaded(self): pass

    class _NoSam2:
        def has_capability(self): return False
        def segment(self, *a, **k): return []

    monkeypatch.setattr(pl, "_sample_frames",
                        lambda *a, **k: fake_frames)
    # run_ocr would crash if called.
    monkeypatch.setattr(pl, "run_ocr",
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("ocr must not run when disabled")))

    result = pl.run_perception_on_window(
        window_path="/tmp/fake.mp4", fps=25,
        zones=[Zone(name="counter_zone", x=0, y=0, w=1000, h=1000)],
        falcon_client=_Falcon(),
        sam2_client=_NoSam2(),
        sampling=SamplingPolicy(),
        ocr_enabled=False,
    )
    assert "ocr_disabled_by_config" in result["limitations"]
    assert "ocr_unavailable" not in result["limitations"]
    assert "ocr_ms" not in result["timings_ms"]
    assert result["ocr"] == []


# ---------------------------------------------------------------------
# Provider chain: Qwen-only, Gemma-only, both disabled
# ---------------------------------------------------------------------

def _make_cfg(qwen_enabled: bool, gemma_enabled: bool):
    from app.config import ModelConfig
    return SimpleNamespace(
        raw={"reasoning": {"primary_provider": "qwen3_vl",
                            "fallback_provider": "gemma",
                            "warm_fallback": False}},
        models={
            "qwen3_vl": ModelConfig(
                name="Qwen/Qwen3-VL-30B-A3B-Instruct",
                enabled=qwen_enabled,
                extra={"provider": "vllm_openai",
                        "base_url": "http://127.0.0.1:8000/v1"}),
            "gemma": ModelConfig(
                name="google/gemma-4-26B-A4B-it",
                enabled=gemma_enabled,
                extra={"vllm_url": "http://127.0.0.1:1"}),
        },
    )


def test_provider_chain_qwen_only_when_gemma_disabled():
    from reasoning.providers import build_active_provider
    p = build_active_provider(_make_cfg(qwen_enabled=True,
                                          gemma_enabled=False))
    if p.name == "chain":
        members = [m.name for m in p.providers]
    else:
        members = [p.name]
    assert "qwen3_vl" in members
    assert "gemma" not in members


def test_provider_chain_gemma_only_when_qwen_disabled():
    from reasoning.providers import build_active_provider
    p = build_active_provider(_make_cfg(qwen_enabled=False,
                                          gemma_enabled=True))
    if p.name == "chain":
        members = [m.name for m in p.providers]
    else:
        members = [p.name]
    assert "gemma" in members
    assert "qwen3_vl" not in members


def test_provider_chain_degrades_when_both_disabled():
    """Both off → a disabled provider is returned and analyze_evidence
    yields a structured error rather than crashing."""
    from reasoning.providers import build_active_provider
    from reasoning.providers.base import EvidenceManifest
    p = build_active_provider(_make_cfg(qwen_enabled=False,
                                          gemma_enabled=False))
    manifest = EvidenceManifest(
        case_id="c", camera_id="cam_01",
        window_start_ts="2026-06-17T14:00:00",
        window_end_ts="2026-06-17T14:00:30",
    )
    result = p.analyze_evidence(manifest)
    assert result.error is not None


# ---------------------------------------------------------------------
# Config snapshot — analyze_case must not be retargeted mid-flight
# ---------------------------------------------------------------------

def test_storage_root_uses_passed_snapshot(monkeypatch):
    """``_storage_root(cfg)`` must read the passed snapshot rather than
    re-invoke ``load_config()``. A mid-case ``config.yaml`` edit must
    NOT retarget the storage root for an in-flight case."""
    from app.case_runner import _storage_root
    snapshot = SimpleNamespace(storage_root=Path("/snapshot/storage"))
    other = SimpleNamespace(storage_root=Path("/changed-after-start"))
    monkeypatch.setattr("app.config.load_config", lambda: other)
    assert _storage_root(snapshot) == Path("/snapshot/storage")


def test_camera_cfg_uses_passed_snapshot(monkeypatch):
    """``_camera_cfg(camera_id, cfg=cfg)`` honours the snapshot. A
    mid-case ROI/camera edit must not change the camera dict the case
    sees for ``_try_nvr_window`` etc."""
    from app.case_runner import _camera_cfg
    snapshot = SimpleNamespace(
        cameras=[{"id": "cam_01", "name": "snapshot-name"}])
    other = SimpleNamespace(
        cameras=[{"id": "cam_01", "name": "edited-after-start"}])
    monkeypatch.setattr("app.config.load_config", lambda: other)
    cam = _camera_cfg("cam_01", cfg=snapshot)
    assert cam == {"id": "cam_01", "name": "snapshot-name"}


def test_build_active_provider_honors_explicit_cfg(monkeypatch):
    """The provider chain that case_runner picks is built from the
    explicit cfg snapshot. Even if ``load_config`` would now return a
    different config (because config.yaml changed mid-case), the
    in-flight chain stays bound to the snapshot."""
    from reasoning.providers import build_active_provider
    snapshot = _make_cfg(qwen_enabled=True, gemma_enabled=False)
    edited = _make_cfg(qwen_enabled=False, gemma_enabled=True)
    monkeypatch.setattr("app.config.load_config", lambda: edited)
    p = build_active_provider(snapshot)
    if p.name == "chain":
        members = [m.name for m in p.providers]
    else:
        members = [p.name]
    assert "qwen3_vl" in members
    assert "gemma" not in members, (
        "build_active_provider must use the explicitly-passed snapshot, "
        "not a fresh load_config() that reflects the mid-case edit")


# ---------------------------------------------------------------------
# UI surface
# ---------------------------------------------------------------------

def test_review_ui_has_model_controls_panel():
    src = (ROOT / "static" / "review.html").read_text()
    assert "Model controls" in src
    assert "model-controls-list" in src
    assert "model-controls-save" in src
    assert "model-controls-admin-token" in src
    assert "/admin/model-controls" in src
    # Each model id appears so the toggle group renders.
    for mid in ("falcon", "sam2", "ocr", "qwen3_vl", "gemma"):
        assert mid in src
    # Captions / dependency warnings — backend's exact UI wording
    # must appear somewhere so a local validation prompt is visible.
    assert "Segmenter (S2) cannot run while Perception (FL) is disabled" in src
    assert "OCR cannot run while Perception (FL) is disabled" in src
    assert "vision providers disabled" in src
    # Hard contract: changes apply to the NEXT case, no process is
    # started/stopped, no memory freed.
    assert "next case or reprocess" in src
    assert "No memory is freed" in src


def test_review_ui_does_not_add_process_controls_for_models():
    """The Model Controls panel must NOT contain start/stop/restart
    buttons for any external process."""
    src = (ROOT / "static" / "review.html").read_text().lower()
    for forbidden in (
        "start falcon", "stop falcon", "restart falcon",
        "start sam", "stop sam", "restart sam",
        "start ocr", "stop ocr", "restart ocr",
        "start qwen", "stop qwen", "restart qwen",
        "start gemma", "stop gemma", "restart gemma",
        "unload falcon", "unload sam", "unload qwen", "unload gemma",
    ):
        assert forbidden not in src, \
            f"Model controls must not pretend to control processes: {forbidden!r}"
