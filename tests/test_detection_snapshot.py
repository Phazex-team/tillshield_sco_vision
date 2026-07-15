"""Saved Falcon detection-snapshot feature.

Covers:
  * annotate_frame burns coloured boxes onto a frame (item vs other colour).
  * select_snapshot_frames groups by frame, busiest-first, capped, and
    ignores boxless detections.
  * render_detection_snapshots writes one PNG per selected frame, returns
    descriptors, and is resilient to unreadable frames.
  * build_package surfaces DETECTION_SNAPSHOT artifacts as
    perception.detection_snapshots with a browser-fetchable URL.
  * the /video/cases/{id}/detection-snapshots/{file} route serves the PNG
    and rejects path traversal.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidence.detection_snapshot import (  # noqa: E402
    ITEM_COLOR_BGR,
    OTHER_COLOR_BGR,
    annotate_frame,
    render_detection_snapshots,
    select_snapshot_frames,
)


def _frame(h=120, w=160):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _det(label, bbox, *, frame_idx=0, score=0.8, ts=None):
    d = {"label": label, "bbox_xyxy": list(bbox), "frame_idx": frame_idx,
         "score": score}
    if ts is not None:
        d["ts"] = ts
    return d


def _has_color(img, bgr) -> bool:
    return bool((img == np.array(bgr, dtype=np.uint8)).all(axis=2).any())


# ---------------------------------------------------------------------------
# annotate_frame
# ---------------------------------------------------------------------------

def test_annotate_draws_item_and_other_colors():
    frame = _frame()
    out = annotate_frame(frame, [
        _det("sco_item_000", [10, 10, 60, 60]),
        _det("sco_generic_products", [80, 40, 130, 100]),
    ])
    # Original frame is untouched (annotate returns a copy).
    assert frame.sum() == 0
    # Item box in green, generic box in blue.
    assert _has_color(out, ITEM_COLOR_BGR)
    assert _has_color(out, OTHER_COLOR_BGR)


def test_annotate_skips_boxes_with_bad_bbox_and_never_raises():
    frame = _frame()
    out = annotate_frame(frame, [
        {"label": "x", "bbox_xyxy": [1, 2, 3]},        # too short
        {"label": "y", "bbox_xyxy": "nope"},           # wrong type
        {"label": "z"},                                # missing bbox
        _det("sco_item_000", [5, 5, 40, 40]),          # the one valid box
    ])
    assert _has_color(out, ITEM_COLOR_BGR)


def test_annotate_tolerates_missing_score():
    frame = _frame()
    out = annotate_frame(frame, [{"label": "item", "bbox_xyxy": [5, 5, 40, 40]}])
    assert _has_color(out, ITEM_COLOR_BGR)


# ---------------------------------------------------------------------------
# select_snapshot_frames
# ---------------------------------------------------------------------------

def test_select_orders_by_box_count_then_frame_and_caps():
    dets = (
        [_det("item", [0, 0, 5, 5], frame_idx=7)] +               # 1 box
        [_det("item", [0, 0, 5, 5], frame_idx=3) for _ in range(3)] +  # 3
        [_det("item", [0, 0, 5, 5], frame_idx=5) for _ in range(2)]    # 2
    )
    picked = select_snapshot_frames(dets, max_snapshots=2)
    assert [fi for fi, _ in picked] == [3, 5]          # busiest two, in order
    assert len(picked[0][1]) == 3


def test_select_ignores_detections_without_bbox():
    dets = [
        _det("item", [0, 0, 5, 5], frame_idx=1),
        {"label": "item", "frame_idx": 1},             # no bbox → ignored
    ]
    picked = select_snapshot_frames(dets, max_snapshots=5)
    assert len(picked) == 1 and len(picked[0][1]) == 1


def test_select_empty_returns_empty():
    assert select_snapshot_frames([], 5) == []
    assert select_snapshot_frames(None, 5) == []  # type: ignore


# ---------------------------------------------------------------------------
# render_detection_snapshots
# ---------------------------------------------------------------------------

def test_render_writes_one_png_per_selected_frame(tmp_path):
    import cv2

    dets = [
        _det("sco_item_000", [10, 10, 60, 60], frame_idx=2, ts="2026-06-27T14:02:32"),
        _det("sco_generic_products", [80, 40, 130, 100], frame_idx=2),
        _det("sco_item_001", [10, 10, 40, 40], frame_idx=9),
    ]
    reader = lambda idx: _frame()  # noqa: E731 — every frame decodes
    out = render_detection_snapshots(
        window_path="unused.mp4", detections=dets,
        out_dir=tmp_path / "snapshots", frame_reader=reader,
        max_snapshots=6,
    )
    assert len(out) == 2
    busiest = out[0]
    assert busiest["frame_idx"] == 2 and busiest["box_count"] == 2
    assert busiest["frame_ts"] == "2026-06-27T14:02:32"
    for s in out:
        p = Path(s["path"])
        assert p.is_file() and p.name == s["filename"]
        img = cv2.imread(str(p))
        assert img is not None and img.shape[2] == 3
    # The busiest snapshot really has the burned-in green item box.
    assert _has_color(cv2.imread(busiest["path"]), ITEM_COLOR_BGR)


def test_render_skips_unreadable_frames(tmp_path):
    dets = [_det("item", [1, 1, 5, 5], frame_idx=n) for n in range(3)]
    out = render_detection_snapshots(
        window_path="x.mp4", detections=dets, out_dir=tmp_path / "s",
        frame_reader=lambda idx: None,          # nothing decodes
    )
    assert out == []


def test_render_no_detections_returns_empty(tmp_path):
    out = render_detection_snapshots(
        window_path="x.mp4", detections=[], out_dir=tmp_path / "s",
        frame_reader=lambda idx: _frame())
    assert out == []


# ---------------------------------------------------------------------------
# build_package exposure
# ---------------------------------------------------------------------------

def _fresh_session(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'t.sqlite'}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    import db.session as s
    s._ENGINE = None
    s._SESSION_FACTORY = None
    s.init_schema()
    return s.get_sessionmaker()


def test_build_package_exposes_detection_snapshots(tmp_path, monkeypatch):
    SM = _fresh_session(tmp_path, monkeypatch)
    from db.models import Case
    from evidence.artifacts import register_artifact
    from evidence.package import build_package

    with SM() as s:
        case = Case(camera_id="cam_01", status="OPEN",
                    opened_at=datetime(2026, 6, 15, 14))
        s.add(case)
        s.commit()
        register_artifact(
            s, case_id=case.id, artifact_type="DETECTION_SNAPSHOT",
            uri=f"/data/cases/case_id={case.id}/snapshots/falcon_00_frame_000002.png",
            mime_type="image/png", frame_idx=2,
            metadata={"filename": "falcon_00_frame_000002.png",
                      "box_count": 3, "frame_ts": "2026-06-27T14:02:32"})
        s.commit()
        pkg = build_package(s, case.id)

    snaps = pkg["perception"]["detection_snapshots"]
    assert len(snaps) == 1
    snap = snaps[0]
    assert snap["url"] == (
        f"/api/v1/video/cases/{case.id}"
        f"/detection-snapshots/falcon_00_frame_000002.png")
    assert snap["frame_idx"] == 2
    assert snap["box_count"] == 3


# ---------------------------------------------------------------------------
# serving route
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'t.sqlite'}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
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
    return TestClient(create_app()), tmp_path


def _write_png(path: Path):
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), _frame())


def test_route_serves_snapshot_png(client):
    c, tmp_path = client
    case_id = "case-abc"
    name = "falcon_00_frame_000002.png"
    _write_png(tmp_path / "storage" / "cases" / f"case_id={case_id}"
               / "snapshots" / name)
    r = c.get(f"/api/v1/video/cases/{case_id}/detection-snapshots/{name}")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_route_404_when_missing(client):
    c, _ = client
    r = c.get("/api/v1/video/cases/case-x/detection-snapshots/nope.png")
    assert r.status_code == 404


def test_route_rejects_path_traversal(client):
    c, tmp_path = client
    # Plant a secret one level above the snapshots dir.
    secret = tmp_path / "storage" / "cases" / "case_id=case-y" / "secret.txt"
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_text("top secret")
    # Encoded traversal that decodes to a name containing separators.
    r = c.get("/api/v1/video/cases/case-y/detection-snapshots/..%2Fsecret.txt")
    assert r.status_code in (400, 404)
    assert "secret" not in r.text
