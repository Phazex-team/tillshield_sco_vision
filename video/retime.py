"""On-demand re-timing of slow-motion CCTV segments.

Segments recorded before the recorder fps fix hold ~25 fps of real frames
but were written declaring 5 fps, so they play ~5x slow — the window
builder then trims by playback seconds and produces a clip that does not
cover the transaction. The frames themselves are intact; only the timing
metadata is wrong.

``retime_segment`` re-encodes one segment so it plays back in its true
real-time duration (``end_at - start_at``), keeping every frame, and
updates the DB row (fps / duration / sha256). ``retime_segments_for_case``
does this for exactly the segments a case's window needs, then the caller
reprocesses. Already-real-time segments are skipped (idempotent).
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
from typing import Optional


log = logging.getLogger(__name__)

_DUR_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)")


def _ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _probe_playback_seconds(path: str, ffmpeg: str) -> Optional[float]:
    """Return the file's playback duration in seconds, or None."""
    try:
        out = subprocess.run([ffmpeg, "-i", path], capture_output=True,
                             text=True).stderr
    except Exception:
        return None
    m = _DUR_RE.search(out or "")
    if not m:
        return None
    h, mn, s = m.groups()
    return int(h) * 3600 + int(mn) * 60 + float(s)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def retime_segment(seg, ffmpeg: Optional[str] = None) -> dict:
    """Re-time one VideoSegment ORM row's file to real time, in place.

    Returns a status dict. Mutates ``seg`` (fps/duration_sec/sha256) when
    a retime happens — the caller must commit the session.
    """
    ffmpeg = ffmpeg or _ffmpeg()
    path = seg.path
    if not path or not os.path.exists(path):
        return {"id": seg.id, "status": "missing_file"}
    try:
        real_dur = (seg.end_at - seg.start_at).total_seconds()
    except Exception:
        real_dur = 0.0
    if real_dur <= 0:
        return {"id": seg.id, "status": "bad_real_duration"}

    play = _probe_playback_seconds(path, ffmpeg)
    if play is None or play <= 0:
        return {"id": seg.id, "status": "probe_failed"}

    # Already real time (within 10% / 2s)? Nothing to do — idempotent.
    if abs(play - real_dur) <= max(2.0, 0.1 * real_dur):
        return {"id": seg.id, "status": "already_realtime",
                "playback_s": round(play, 1)}

    # Speed the clip up so it plays in ``real_dur`` seconds, keeping frames.
    factor = real_dur / play  # < 1 → faster
    tmp = path + ".retime.mp4"
    cmd = [ffmpeg, "-y", "-i", path,
           "-vf", f"setpts=PTS*{factor:.6f}",
           "-an", "-c:v", "libx264", "-preset", "veryfast",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not (os.path.exists(tmp)
                                    and os.path.getsize(tmp) > 0):
        if os.path.exists(tmp):
            os.unlink(tmp)
        return {"id": seg.id, "status": "ffmpeg_failed",
                "error": (proc.stderr or "")[-200:]}

    os.replace(tmp, path)
    new_play = _probe_playback_seconds(path, ffmpeg) or real_dur
    if seg.frame_count and real_dur:
        seg.fps = round(seg.frame_count / real_dur, 3)
    seg.duration_sec = round(real_dur, 3)
    seg.sha256 = _sha256(path)
    return {"id": seg.id, "status": "retimed",
            "old_playback_s": round(play, 1),
            "new_playback_s": round(new_play, 1),
            "new_fps": seg.fps}


def retime_segments_for_case(case_id: str, *,
                             pre_roll_sec=None,
                             post_roll_sec=None) -> dict:
    """Re-time the slow-mo segments a case's window needs. Returns a
    summary; commits the DB.

    ``pre_roll_sec`` / ``post_roll_sec`` widen the window the same way as
    the reprocess, so a larger requested window pulls in (and re-times)
    the extra segments it now spans."""
    from db.models import Case, PosEvent, VideoSegment
    from db.session import get_sessionmaker
    from pos.correlation import plan_window

    ffmpeg = _ffmpeg()
    SM = get_sessionmaker()
    results: list[dict] = []
    with SM() as s:
        case = s.get(Case, case_id)
        if case is None:
            return {"error": "case not found", "case_id": case_id}
        pos = s.get(PosEvent, case.pos_event_id) if case.pos_event_id else None
        if pos is None:
            return {"error": "case has no POS event", "case_id": case_id}
        plan = plan_window(s, case.camera_id, pos.pos_event_at,
                           pos_event_end_at=pos.pos_event_end_at,
                           pre_roll_sec=pre_roll_sec,
                           post_roll_sec=post_roll_sec)
        ids = list(plan.matched_segment_ids or [])
        segs = (s.query(VideoSegment)
                .filter(VideoSegment.id.in_(ids)).all()) if ids else []
        for seg in segs:
            results.append(retime_segment(seg, ffmpeg))
        s.commit()
    retimed = sum(1 for r in results if r.get("status") == "retimed")
    return {"case_id": case_id, "segments_considered": len(results),
            "retimed": retimed, "details": results}
