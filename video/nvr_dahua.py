"""Dahua-family NVR on-demand retrieval client (read-only).

Stage A (proven): historical recording SEARCH via the Dahua mediaFileFind
CGI — for a camera + time window, return the ``.dav`` recordings that
cover it.

Stage B (best-effort): clip EXPORT via the Dahua RTSP playback-by-time
URL, re-muxed to a local MP4 with ffmpeg. Export is treated as
preferred-but-nonblocking: if it fails or is disabled, the caller falls
back to the local recorder/segment path.

Channel numbering caveat (per deployment):
  * The live RTSP URL uses a 1-based ``channel=`` (e.g. 15).
  * mediaFileFind ``condition.Channel`` is also 1-based on this NVR, and
    the RESULT ``Channel`` field + on-disk path are 0-based (channel-1).
  So searching ``condition.Channel=15`` returns results labelled
  ``Channel=14`` under ``/mnt/dvr/.../14/dav/`` — the SAME camera.
``playback_channel`` is therefore explicitly configurable and is sent
verbatim as ``condition.Channel``; the returned channel + file path are
logged so an operator can confirm the mapping. Nothing is hardcoded.

Nothing here ever blocks the pipeline: every network call is wrapped and
surfaced as a structured result the caller can branch on.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Optional
from urllib.parse import quote


log = logging.getLogger(__name__)


# Retrieval window is always bounded to the transaction period — never
# long-hour pulls. This is the hard ceiling regardless of config.
MAX_WINDOW_SEC_CEILING = 900  # 15 minutes


# Acquisition states surfaced to operators (case/window metadata).
STATE_CLIP_RETRIEVED = "nvr_clip_retrieved"
STATE_FOUND_NO_EXPORT = "nvr_recording_found_no_export"
STATE_NO_RECORDING = "nvr_no_recording_found"
STATE_QUERY_FAILED = "nvr_query_failed"
STATE_DISABLED = "nvr_disabled"
STATE_LOCAL_USED = "local_segments_used"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class NvrConfig:
    enabled: bool = False
    base_url: str = ""
    username: str = ""
    password: str = ""
    host: str = ""                 # derived from base_url for RTSP
    rtsp_port: int = 554
    live_channel: int = 1
    playback_channel: int = 1      # sent verbatim as condition.Channel
    subtype: int = 0
    timezone: str = "UTC"
    pre_roll_sec: int = 120
    post_roll_sec: int = 180
    max_window_sec: int = MAX_WINDOW_SEC_CEILING
    request_timeout_sec: float = 20.0
    prefer_on_demand_for_pos: bool = False
    export_enabled: bool = False


def load_nvr_config(camera_cfg: dict) -> Optional[NvrConfig]:
    """Build an ``NvrConfig`` from a camera's ``nvr`` block, or ``None``
    when the camera has no NVR config (backward compatible).

    Credentials prefer environment variables (``NVR_USERNAME`` /
    ``NVR_PASSWORD``) when set, matching the repo's secrets pattern.
    """
    nvr = (camera_cfg or {}).get("nvr")
    if not isinstance(nvr, dict):
        return None
    base_url = str(nvr.get("base_url") or "").rstrip("/")
    host = re.sub(r"^https?://", "", base_url).split(":")[0].split("/")[0]
    cap = int(nvr.get("max_window_sec", MAX_WINDOW_SEC_CEILING) or
              MAX_WINDOW_SEC_CEILING)
    return NvrConfig(
        enabled=bool(nvr.get("enabled", False)),
        base_url=base_url,
        username=os.environ.get("NVR_USERNAME") or str(nvr.get("username") or ""),
        password=os.environ.get("NVR_PASSWORD") or str(nvr.get("password") or ""),
        host=host,
        rtsp_port=int(nvr.get("rtsp_port", 554) or 554),
        live_channel=int(nvr.get("live_channel", 1) or 1),
        playback_channel=int(nvr.get("playback_channel",
                                     nvr.get("live_channel", 1)) or 1),
        subtype=int(nvr.get("subtype", 0) or 0),
        timezone=str(nvr.get("timezone") or "UTC"),
        pre_roll_sec=int(nvr.get("pre_roll_sec", 120) or 120),
        post_roll_sec=int(nvr.get("post_roll_sec", 180) or 180),
        max_window_sec=min(cap, MAX_WINDOW_SEC_CEILING),
        request_timeout_sec=float(nvr.get("request_timeout_sec", 20) or 20),
        prefer_on_demand_for_pos=bool(nvr.get("prefer_on_demand_for_pos", False)),
        export_enabled=bool(nvr.get("export_enabled", False)),
    )


@dataclass
class RecordingMatch:
    file_path: str
    start_at: datetime           # NVR-local naive datetime
    end_at: datetime
    result_channel: Optional[int]  # 0-based channel the NVR reports
    raw: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def _tz(name: str) -> tzinfo:
    if not name or name.upper() == "UTC":
        return timezone.utc
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(name)
    except Exception:
        log.warning("nvr: unknown timezone %r; using UTC", name)
        return timezone.utc


def utc_to_nvr_local(dt_utc_naive: datetime, tz_name: str) -> datetime:
    """Convert a naive-UTC ``pos_event_at`` to the NVR's local wall clock
    (naive), since Dahua search/playback use local device time."""
    aware = dt_utc_naive.replace(tzinfo=timezone.utc)
    return aware.astimezone(_tz(tz_name)).replace(tzinfo=None)


def bounded_window(pos_event_at_utc: datetime,
                   cfg: NvrConfig) -> tuple[datetime, datetime]:
    """Return (start_local, end_local) for the tx window, bounded so it
    never exceeds ``max_window_sec`` (<= 15 min ceiling)."""
    center = utc_to_nvr_local(pos_event_at_utc, cfg.timezone)
    pre = max(0, cfg.pre_roll_sec)
    post = max(0, cfg.post_roll_sec)
    total = pre + post
    if total > cfg.max_window_sec:
        # Shrink symmetrically around the event to respect the cap.
        scale = cfg.max_window_sec / total
        pre = int(pre * scale)
        post = int(post * scale)
    return (center - timedelta(seconds=pre), center + timedelta(seconds=post))


# ---------------------------------------------------------------------------
# Stage A: search
# ---------------------------------------------------------------------------

def _parse_find_response(text: str) -> list[RecordingMatch]:
    """Parse the ``items[N].Field=value`` block from findNextFile."""
    rows: dict[int, dict] = {}
    for line in text.splitlines():
        m = re.match(r"items\[(\d+)\]\.([\w.]+)=(.*)$", line.strip())
        if not m:
            continue
        idx, key, val = int(m.group(1)), m.group(2), m.group(3)
        rows.setdefault(idx, {})[key] = val
    out: list[RecordingMatch] = []
    for _, r in sorted(rows.items()):
        fp = r.get("FilePath")
        st, et = r.get("StartTime"), r.get("EndTime")
        if not (fp and st and et):
            continue
        try:
            start = datetime.strptime(st, "%Y-%m-%d %H:%M:%S")
            end = datetime.strptime(et, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        ch = r.get("Channel")
        out.append(RecordingMatch(
            file_path=fp, start_at=start, end_at=end,
            result_channel=int(ch) if ch and ch.isdigit() else None, raw=r))
    return out


# Dahua findFile returns files whose START falls inside the condition
# window, so a short tx window landing mid-file matches nothing. Pad the
# condition start backward by just over the max Dahua segment length
# (~1h) and rely on the app-side overlap filter to narrow to the tx.
_CONDITION_PAD_SEC = 3700


def search_recordings(cfg: NvrConfig,
                      start_local: datetime,
                      end_local: datetime,
                      *,
                      max_results: int = 50,
                      condition_pad_sec: int = _CONDITION_PAD_SEC
                      ) -> list[RecordingMatch]:
    """Search the NVR for recordings overlapping [start_local, end_local].

    Raises on transport/auth failure so the caller can record
    ``nvr_query_failed`` and fall back. Returns [] when no recording
    covers the window.
    """
    import requests
    from requests.auth import HTTPDigestAuth

    auth = HTTPDigestAuth(cfg.username, cfg.password)
    base = f"{cfg.base_url}/cgi-bin/mediaFileFind.cgi"
    to = cfg.request_timeout_sec
    cond_start = start_local - timedelta(seconds=max(0, condition_pad_sec))

    from urllib.parse import urlencode

    def _get(params: dict) -> str:
        # Dahua needs the datetime space as %20 with literal ':'. requests'
        # default encoder uses '+' for spaces, which the CGI rejects — so
        # build the query with quote_via=quote and ':' kept safe.
        qs = urlencode(params, quote_via=quote, safe=":")
        r = requests.get(f"{base}?{qs}", auth=auth, timeout=to)
        r.raise_for_status()
        return r.text

    created = _get({"action": "factory.create"})
    m = re.search(r"result=(\d+)", created) or re.search(r"(\d+)", created)
    if not m:
        raise RuntimeError(f"nvr: could not create finder ({created[:80]!r})")
    obj = m.group(1)
    try:
        ff = _get({
            "action": "findFile", "object": obj,
            "condition.Channel": cfg.playback_channel,
            "condition.StartTime": cond_start.strftime("%Y-%m-%d %H:%M:%S"),
            "condition.EndTime": end_local.strftime("%Y-%m-%d %H:%M:%S"),
        })
        if "OK" not in ff:
            raise RuntimeError(f"nvr: findFile rejected ({ff[:80]!r})")
        matches: list[RecordingMatch] = []
        while len(matches) < max_results:
            text = _get({"action": "findNextFile", "object": obj,
                         "count": min(100, max_results - len(matches))})
            batch = _parse_find_response(text)
            matches.extend(batch)
            found = re.search(r"found=(\d+)", text)
            if not batch or (found and int(found.group(1)) == 0):
                break
        # Keep only recordings that actually overlap the requested window.
        overlap = [r for r in matches
                   if r.end_at >= start_local and r.start_at <= end_local]
        log.info("nvr: search ch=%s [%s..%s] -> %d match(es)"
                 "%s", cfg.playback_channel,
                 start_local, end_local, len(overlap),
                 f" (first result_channel={overlap[0].result_channel}, "
                 f"{overlap[0].file_path})" if overlap else "")
        return overlap
    finally:
        try:
            _get({"action": "close", "object": obj})
            _get({"action": "destroy", "object": obj})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Stage B: export (best-effort, bounded)
# ---------------------------------------------------------------------------

def build_playback_rtsp_url(cfg: NvrConfig,
                            start_local: datetime,
                            end_local: datetime) -> str:
    """Dahua playback-by-time RTSP URL for the bounded window. Password
    is URL-encoded; the live (1-based) ``channel`` is used for RTSP."""
    user = quote(cfg.username, safe="")
    pw = quote(cfg.password, safe="")
    st = start_local.strftime("%Y_%m_%d_%H_%M_%S")
    et = end_local.strftime("%Y_%m_%d_%H_%M_%S")
    # NOTE: on this NVR ``subtype`` MUST precede starttime/endtime — the
    # subtype-last ordering returns RTSP 404.
    return (f"rtsp://{user}:{pw}@{cfg.host}:{cfg.rtsp_port}/cam/playback"
            f"?channel={cfg.live_channel}&subtype={cfg.subtype}"
            f"&starttime={st}&endtime={et}")


def export_clip(cfg: NvrConfig,
                start_local: datetime,
                end_local: datetime,
                out_path: str,
                *,
                ffmpeg: Optional[str] = None) -> bool:
    """Pull the bounded historical window via RTSP playback to ``out_path``
    (MP4). Returns True on success. Never raises — export is optional."""
    duration = int((end_local - start_local).total_seconds())
    if duration <= 0 or duration > MAX_WINDOW_SEC_CEILING:
        log.warning("nvr export: refusing window of %ss (cap %ss)",
                    duration, MAX_WINDOW_SEC_CEILING)
        return False
    url = build_playback_rtsp_url(cfg, start_local, end_local)
    if ffmpeg is None:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg = "ffmpeg"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = [ffmpeg, "-y", "-rtsp_transport", "tcp",
           "-i", url, "-t", str(duration + 5),
           "-c", "copy", "-movflags", "+faststart", out_path]
    # Dahua playback can stream BELOW real-time, so allow up to ~3x the
    # window length, hard-capped at 15 min wall so a case never hangs.
    timeout = min(max(duration * 3, duration + 120), 900)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        _, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Gracefully tell ffmpeg to quit so the MP4 is finalized, then
        # KEEP the partial clip — it still covers the tx moment and is
        # valid evidence (better than discarding a slow-but-good pull).
        try:
            _, err = proc.communicate(input=b"q", timeout=20)
        except Exception:
            proc.kill()
            _, err = proc.communicate()
        log.warning("nvr export: playback slower than real-time; finalized "
                    "partial clip after %ss timeout", timeout)
    except Exception as exc:
        log.warning("nvr export: ffmpeg failed: %s", exc)
        return False
    # Accept any finalized clip with meaningful content (>1 MB ~ a few s).
    ok = os.path.exists(out_path) and os.path.getsize(out_path) > (1 << 20)
    if not ok:
        log.warning("nvr export: no usable clip (rc=%s, stderr=%s)",
                    proc.returncode,
                    (err or b"").decode("utf-8", "ignore")[-300:])
    return ok


# ---------------------------------------------------------------------------
# Orchestration: try NVR first, return a structured result to branch on
# ---------------------------------------------------------------------------

@dataclass
class NvrAcquisition:
    attempted: bool
    state: str
    clip_path: Optional[str]
    metadata: dict


def acquire_window(cfg: Optional[NvrConfig],
                   pos_event_at_utc: datetime,
                   *,
                   camera_id: str = "",
                   out_path: Optional[str] = None,
                   search_fn=None,
                   export_fn=None) -> NvrAcquisition:
    """Try NVR on-demand retrieval for one POS-triggered window.

    Returns an ``NvrAcquisition`` whose ``state`` is one of the
    ``STATE_*`` constants. ``search_fn``/``export_fn`` are injectable for
    tests. NEVER raises — a failure becomes ``nvr_query_failed`` and the
    caller falls back to local segments.
    """
    meta: dict = {
        "attempted": False, "camera_id": camera_id,
        "playback_channel": cfg.playback_channel if cfg else None,
        "window_start": None, "window_end": None,
        "recordings_found": 0, "first_match": None,
        "export_attempted": False, "export_ok": False,
        "fallback_used": True, "error": None,
    }
    if cfg is None or not cfg.enabled or not cfg.prefer_on_demand_for_pos:
        return NvrAcquisition(False, STATE_DISABLED, None, meta)

    # Resolve at call time so tests can monkeypatch module functions.
    search_fn = search_fn or search_recordings
    export_fn = export_fn or export_clip
    start_local, end_local = bounded_window(pos_event_at_utc, cfg)
    meta.update(attempted=True,
                window_start=start_local.isoformat(),
                window_end=end_local.isoformat())
    try:
        matches = search_fn(cfg, start_local, end_local)
    except Exception as exc:
        meta["error"] = f"{type(exc).__name__}: {exc}"
        log.warning("nvr: query failed for %s: %s", camera_id, meta["error"])
        return NvrAcquisition(True, STATE_QUERY_FAILED, None, meta)

    meta["recordings_found"] = len(matches)
    if not matches:
        return NvrAcquisition(True, STATE_NO_RECORDING, None, meta)
    first = matches[0]
    meta["first_match"] = {
        "file_path": first.file_path,
        "start_at": first.start_at.isoformat(),
        "end_at": first.end_at.isoformat(),
        "result_channel": first.result_channel,
    }

    if cfg.export_enabled and out_path:
        meta["export_attempted"] = True
        try:
            ok = export_fn(cfg, start_local, end_local, out_path)
        except Exception as exc:
            ok = False
            meta["error"] = f"export: {type(exc).__name__}: {exc}"
        meta["export_ok"] = bool(ok)
        if ok:
            meta["fallback_used"] = False
            return NvrAcquisition(True, STATE_CLIP_RETRIEVED, out_path, meta)

    # Recording exists on the NVR but we did not export it: Stage A.
    return NvrAcquisition(True, STATE_FOUND_NO_EXPORT, None, meta)
