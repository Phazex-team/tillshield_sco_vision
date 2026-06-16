"""Tests for the Dahua NVR on-demand retrieval path. All mocked — no
real NVR. Verifies: NVR-preferred selection, explicit fallback states,
channel/tz handling, the bounded tx window, observability metadata, and
that cameras without NVR config are unchanged."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from video import nvr_dahua as N  # noqa: E402


CAM = {
    "id": "cam_return_01",
    "nvr": {
        "enabled": True, "base_url": "http://192.168.1.13",
        "username": "u", "password": "p%1",
        "live_channel": 15, "playback_channel": 15, "subtype": 0,
        "timezone": "Asia/Dubai", "pre_roll_sec": 120, "post_roll_sec": 180,
        "max_window_sec": 900, "prefer_on_demand_for_pos": True,
        "export_enabled": False,
    },
}
POS_UTC = datetime(2026, 6, 16, 10, 36, 0)  # naive UTC


def _match():
    return N.RecordingMatch(
        file_path="/mnt/dvr/2026-06-16/14/dav/14/.../14.00.06-15.00.06.dav",
        start_at=datetime(2026, 6, 16, 14, 0, 6),
        end_at=datetime(2026, 6, 16, 15, 0, 6),
        result_channel=14, raw={})


# ---------------------------------------------------------------------------
# Config + window bounding
# ---------------------------------------------------------------------------

def test_load_nvr_config_present():
    cfg = N.load_nvr_config(CAM)
    assert cfg and cfg.enabled
    assert cfg.playback_channel == 15 and cfg.live_channel == 15
    assert cfg.host == "192.168.1.13"


def test_load_nvr_config_absent_is_backward_compatible():
    assert N.load_nvr_config({"id": "cam_01"}) is None


def test_env_credentials_override(monkeypatch):
    monkeypatch.setenv("NVR_USERNAME", "envuser")
    monkeypatch.setenv("NVR_PASSWORD", "envpass")
    cfg = N.load_nvr_config(CAM)
    assert cfg.username == "envuser" and cfg.password == "envpass"


def test_bounded_window_tz_and_cap():
    cfg = N.load_nvr_config(CAM)
    s, e = N.bounded_window(POS_UTC, cfg)
    # 10:36 UTC -> 14:36 Asia/Dubai (+4); 120s pre + 180s post.
    assert s == datetime(2026, 6, 16, 14, 34, 0)
    assert e == datetime(2026, 6, 16, 14, 39, 0)
    assert (e - s).total_seconds() == 300


def test_window_never_exceeds_15min_ceiling():
    cam = {**CAM, "nvr": {**CAM["nvr"], "pre_roll_sec": 100000,
                          "post_roll_sec": 100000, "max_window_sec": 99999}}
    cfg = N.load_nvr_config(cam)
    assert cfg.max_window_sec == N.MAX_WINDOW_SEC_CEILING  # capped at 900
    s, e = N.bounded_window(POS_UTC, cfg)
    assert (e - s).total_seconds() <= N.MAX_WINDOW_SEC_CEILING


# ---------------------------------------------------------------------------
# acquire_window state machine + observability
# ---------------------------------------------------------------------------

def test_recording_found_no_export_when_export_off():
    cfg = N.load_nvr_config(CAM)
    acq = N.acquire_window(cfg, POS_UTC, camera_id="cam_return_01",
                           search_fn=lambda c, a, b: [_match()])
    assert acq.state == N.STATE_FOUND_NO_EXPORT
    assert acq.metadata["recordings_found"] == 1
    assert acq.metadata["first_match"]["result_channel"] == 14
    assert acq.metadata["fallback_used"] is True  # local still used


def test_no_recording_falls_back():
    cfg = N.load_nvr_config(CAM)
    acq = N.acquire_window(cfg, POS_UTC, search_fn=lambda c, a, b: [])
    assert acq.state == N.STATE_NO_RECORDING
    assert acq.metadata["fallback_used"] is True


def test_query_failure_falls_back():
    cfg = N.load_nvr_config(CAM)

    def boom(c, a, b):
        raise ConnectionError("nvr unreachable")

    acq = N.acquire_window(cfg, POS_UTC, search_fn=boom)
    assert acq.state == N.STATE_QUERY_FAILED
    assert "nvr unreachable" in acq.metadata["error"]
    assert acq.metadata["fallback_used"] is True


def test_disabled_when_no_nvr_config():
    acq = N.acquire_window(None, POS_UTC)
    assert acq.state == N.STATE_DISABLED
    assert acq.attempted is False


def test_clip_retrieved_when_export_succeeds(tmp_path):
    cam = {**CAM, "nvr": {**CAM["nvr"], "export_enabled": True}}
    cfg = N.load_nvr_config(cam)
    out = str(tmp_path / "clip.mp4")
    acq = N.acquire_window(cfg, POS_UTC, camera_id="cam_return_01",
                           out_path=out,
                           search_fn=lambda c, a, b: [_match()],
                           export_fn=lambda c, a, b, o: True)
    assert acq.state == N.STATE_CLIP_RETRIEVED
    assert acq.clip_path == out
    assert acq.metadata["export_ok"] is True
    assert acq.metadata["fallback_used"] is False


def test_export_failure_falls_back_to_found_no_export(tmp_path):
    cam = {**CAM, "nvr": {**CAM["nvr"], "export_enabled": True}}
    cfg = N.load_nvr_config(cam)
    acq = N.acquire_window(cfg, POS_UTC, out_path=str(tmp_path / "c.mp4"),
                           search_fn=lambda c, a, b: [_match()],
                           export_fn=lambda c, a, b, o: False)
    assert acq.state == N.STATE_FOUND_NO_EXPORT
    assert acq.metadata["export_attempted"] is True
    assert acq.metadata["export_ok"] is False


# ---------------------------------------------------------------------------
# Search uses playback_channel (not live RTSP channel) + parsing
# ---------------------------------------------------------------------------

def test_search_sends_playback_channel(monkeypatch):
    cfg = N.load_nvr_config(CAM)
    sent = {}
    find_text = (
        "items[0].Channel=14\nitems[0].Type=dav\n"
        "items[0].StartTime=2026-06-16 14:00:06\n"
        "items[0].EndTime=2026-06-16 15:00:06\n"
        "items[0].FilePath=/mnt/dvr/2026-06-16/14/dav/14/x.dav\nfound=1\n")

    class _Resp:
        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            pass

    state = {"next_calls": 0}

    def fake_get(url, auth=None, timeout=None):
        if "factory.create" in url:
            return _Resp("result=123")
        if "findFile" in url:
            sent["findFile_url"] = url
            return _Resp("OK")
        if "findNextFile" in url:
            state["next_calls"] += 1
            # First call returns the batch; then Dahua reports exhaustion.
            return _Resp(find_text if state["next_calls"] == 1 else "found=0\n")
        return _Resp("OK")

    import requests
    monkeypatch.setattr(requests, "get", fake_get)
    matches = N.search_recordings(cfg, datetime(2026, 6, 16, 14, 34),
                                  datetime(2026, 6, 16, 14, 39))
    # playback_channel (15) is the condition.Channel, NOT live RTSP-only.
    assert "condition.Channel=15" in sent["findFile_url"]
    # Datetime space must be %20 (Dahua rejects '+').
    assert "%20" in sent["findFile_url"] and "+" not in sent["findFile_url"]
    assert len(matches) == 1 and matches[0].result_channel == 14


def test_parse_find_response_overlap_filter():
    # findNextFile returns two files; only the overlapping one is kept by
    # acquire via search; parser itself returns both rows.
    text = ("items[0].Channel=14\nitems[0].StartTime=2026-06-16 13:00:06\n"
            "items[0].EndTime=2026-06-16 14:00:06\n"
            "items[0].FilePath=/a.dav\n"
            "items[1].Channel=14\nitems[1].StartTime=2026-06-16 14:00:06\n"
            "items[1].EndTime=2026-06-16 15:00:06\nitems[1].FilePath=/b.dav\n")
    rows = N._parse_find_response(text)
    assert len(rows) == 2
    assert rows[1].file_path == "/b.dav"


def test_playback_rtsp_url_encodes_password_and_uses_live_channel():
    cfg = N.load_nvr_config(CAM)
    url = N.build_playback_rtsp_url(cfg, datetime(2026, 6, 16, 14, 34),
                                    datetime(2026, 6, 16, 14, 39))
    assert "channel=15" in url           # live channel for RTSP
    assert "p%251" in url                # password '%' -> %25
    assert "starttime=2026_06_16_14_34_00" in url
    # subtype MUST precede starttime on this NVR (subtype-last -> RTSP 404).
    assert url.index("subtype=") < url.index("starttime=")


# ---------------------------------------------------------------------------
# Integration with analyze_case: fallback states recorded
# ---------------------------------------------------------------------------

def _fresh_session(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'t.sqlite'}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    import db.session as s
    s._ENGINE = None
    s._SESSION_FACTORY = None
    s.init_schema()
    return s.get_sessionmaker()


def _make_case(session, camera_id):
    from db.models import Case, PosEvent
    ev = PosEvent(store_id="2270", terminal_id="52", transaction_id="T1",
                  line_id="transaction", event_type="RETURN",
                  pos_event_at=POS_UTC, amount=-100.0)
    session.add(ev)
    session.flush()
    c = Case(pos_event_id=ev.id, camera_id=camera_id, status="OPEN")
    session.add(c)
    session.flush()
    return c.id


def test_analyze_case_records_nvr_found_state(tmp_path, monkeypatch):
    """NVR finds a recording but export is off and there are NO local
    segments -> case is INVALID_VIDEO but the window records that footage
    is available on the NVR (operator visibility)."""
    SM = _fresh_session(tmp_path, monkeypatch)
    import video.nvr_dahua as nv
    monkeypatch.setattr(nv, "search_recordings", lambda c, a, b: [_match()])
    # Point case_runner at a camera that HAS nvr config.
    from app import case_runner
    monkeypatch.setattr(case_runner, "_camera_cfg", lambda cid: CAM)
    from db.models import VideoWindow
    from app.case_runner import analyze_case
    with SM() as s:
        cid = _make_case(s, "cam_return_01")
        s.commit()
    with SM() as s:
        res = analyze_case(s, cid, vlm_runner=lambda *a: {}, perception_runner=lambda *a: None)
        s.commit()
    assert res["outcome"] == "INVALID_VIDEO"  # no local segments
    with SM() as s:
        w = s.query(VideoWindow).filter_by(case_id=cid).first()
    assert w.acquisition_source == "nvr_recording_found_no_export"
    assert w.nvr_metadata["recordings_found"] == 1
    assert w.nvr_metadata["first_match"]["result_channel"] == 14


def test_analyze_case_without_nvr_config_unchanged(tmp_path, monkeypatch):
    """A camera with no nvr block behaves exactly as before."""
    SM = _fresh_session(tmp_path, monkeypatch)
    from app import case_runner
    monkeypatch.setattr(case_runner, "_camera_cfg", lambda cid: {"id": cid})
    from db.models import VideoWindow
    from app.case_runner import analyze_case
    with SM() as s:
        cid = _make_case(s, "cam_01")
        s.commit()
    with SM() as s:
        res = analyze_case(s, cid, vlm_runner=lambda *a: {}, perception_runner=lambda *a: None)
        s.commit()
    assert res["outcome"] == "INVALID_VIDEO"
    with SM() as s:
        w = s.query(VideoWindow).filter_by(case_id=cid).first()
    # No NVR attempted; falls straight to local (no segments) path.
    assert w.acquisition_source in ("nvr_disabled", "local_no_segments")
    assert (w.nvr_metadata or {}).get("attempted") in (False, None)
