"""MP4 encoder for the evidence (audit) path only.

Deliberately separate from the inference path: the frames that go into
the MP4 are already downscaled (caller's job) and are NEVER passed to
Falcon or Gemma. The MP4 exists purely so a human reviewer can scrub
through a past session; disk footprint matters more than pixel quality.

Uses ffmpeg via subprocess, piping raw BGR frames over stdin. Prefers
the system ffmpeg (on Jetson/Linux it's present and hardware-accelerated
if available); falls back to the bundled imageio-ffmpeg binary.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

log = logging.getLogger(__name__)


def _ffmpeg_exe() -> str:
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def encode_evidence_mp4(
    frames_bgr: Sequence[np.ndarray] | Iterable[np.ndarray],
    out_path: str | Path,
    *,
    width: int,
    height: int,
    fps: int,
    crf: int = 28,
    pix_fmt_in: str = "bgr24",
) -> Path:
    """Encode an iterable of BGR frames into an H.264 MP4 at the given
    resolution/fps/CRF. Frames must already be resized to ``width x height``.

    Returns the written path. Raises ``RuntimeError`` if ffmpeg fails.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    exe = _ffmpeg_exe()
    cmd = [
        exe, "-y",
        "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", pix_fmt_in,
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-tune", "zerolatency",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_path),
    ]
    log.info("encoding MP4: %s (%dx%d @ %dfps, crf=%d)",
             out_path, width, height, fps, crf)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)
    n = 0
    try:
        assert proc.stdin is not None
        for frame in frames_bgr:
            if frame is None:
                continue
            if frame.shape[:2] != (height, width):
                import cv2
                frame = cv2.resize(frame, (width, height),
                                   interpolation=cv2.INTER_AREA)
            proc.stdin.write(frame.tobytes())
            n += 1
        proc.stdin.close()
    except BrokenPipeError:
        # ffmpeg died early; retrieve stderr below
        pass
    stderr = (proc.stderr.read() if proc.stderr else b"").decode(errors="replace")
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(
            f"ffmpeg exit {rc} after {n} frames: {stderr.strip()[:400]}")
    size = out_path.stat().st_size if out_path.exists() else 0
    log.info("MP4 written: %s  frames=%d  size=%.2f MB",
             out_path, n, size / 1e6)
    return out_path
