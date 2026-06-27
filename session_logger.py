"""v3 CSV logger — one row per video-clip session, multi-camera aware.

New in v3 vs v2:
  * ``camera_id`` column (the short id, distinct from the legacy ``camera``
    name kept for back-compat).
  * ``classifier`` and ``scenario_label`` columns from classifiers.py.
  * Per-camera session sequence: ``CAM01-001``, ``CAM02-001`` etc.
    (CAM<numeric-suffix-of-id>-NNN). If the camera id has no numeric tail
    we slug it: ``cam-shelf-A`` -> ``CAMSHELFA-001``.
  * Daily report grouped by classifier with per-classifier counters.
"""
from __future__ import annotations

import csv
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


FIELDS = [
    "session_id", "camera_id", "camera",
    "classifier", "scenario_label",
    "start_time", "end_time", "duration_sec",
    "handover_occurred", "item_presented", "item_count",
    "customer_description", "items_handed_over",
    "confidence", "flag_for_review",
    "num_people", "per_person_json", "narrative",
    "objects_detected",
    "merged_from", "merged_count",
    "snapshot_path", "mp4_path",
]


@dataclass
class DayCounts:
    total: int = 0
    handovers: int = 0
    none: int = 0
    flagged: int = 0
    low_conf: int = 0
    presented: int = 0  # alias of handovers, kept for /stats back-compat


@dataclass
class ClassifierCounts:
    total: int = 0
    events: int = 0       # rows with handover_occurred=true
    flagged: int = 0
    by_camera: dict[str, int] = field(default_factory=dict)


def _slugify_cam_id(cam_id: str) -> str:
    """Build the session-id prefix for a camera.

    ``cam_01`` -> ``CAM01``; ``cam-shelf-A`` -> ``CAMSHELFA``;
    ``store_42`` -> ``STORE42``; falls back to ``CAM`` if id is empty.
    """
    if not cam_id:
        return "CAM"
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", cam_id).upper()
    return cleaned or "CAM"


class SessionLogger:
    def __init__(self, log_dir: str, video_dir: Optional[str] = None,
                 snapshot_dir: Optional[str] = None):
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.video_dir = Path(video_dir) if video_dir else self.dir / "videos"
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir = Path(snapshot_dir) if snapshot_dir else self.dir / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        self.lock = threading.Lock()
        self.date = datetime.now().strftime("%Y-%m-%d")
        self.csv_path = self.dir / f"sessions_{self.date}.csv"
        self.report_path = self.dir / f"report_{self.date}.txt"

        # Per-camera-prefix sequence counters (CAM01 -> 17, CAM02 -> 4, ...)
        self._seq_by_prefix: dict[str, int] = {}
        self.counts = DayCounts()
        self.by_classifier: dict[str, ClassifierCounts] = {}

        new = not self.csv_path.exists()
        if not new:
            self._rehydrate()
            self._migrate_header_if_needed()
        self._f = self.csv_path.open("a", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=FIELDS)
        if new:
            self._w.writeheader()
            self._f.flush()

    # ------------------------------------------------------------------
    # Schema migration / rehydrate
    # ------------------------------------------------------------------

    def _migrate_header_if_needed(self) -> None:
        with self.csv_path.open("r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
        if header == FIELDS:
            return
        with self.csv_path.open("r", newline="") as f:
            old_rows = list(csv.DictReader(f))
        with self.csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for row in old_rows:
                w.writerow({k: row.get(k, "") for k in FIELDS})

    def _rehydrate(self) -> None:
        with self.csv_path.open("r", newline="") as f:
            for row in csv.DictReader(f):
                self._tally(row)
                # Continue per-camera session sequence from where we left off.
                sid = row.get("session_id", "") or ""
                m = re.match(r"^([A-Z0-9]+)-(\d+)$", sid)
                if m:
                    prefix, seq = m.group(1), int(m.group(2))
                    self._seq_by_prefix[prefix] = max(
                        self._seq_by_prefix.get(prefix, 0), seq)
                else:
                    # Legacy ``SESSION-NNN`` -> book under "SESSION".
                    m2 = re.search(r"(\d+)$", sid)
                    if m2:
                        self._seq_by_prefix["SESSION"] = max(
                            self._seq_by_prefix.get("SESSION", 0),
                            int(m2.group(1)))

    def _tally(self, row: dict) -> None:
        self.counts.total += 1
        handover = str(row.get("handover_occurred")
                       or row.get("item_presented", "")).lower() == "true"
        if handover:
            self.counts.handovers += 1
            self.counts.presented += 1
        else:
            self.counts.none += 1
        if str(row.get("flag_for_review", "")).lower() == "true":
            self.counts.flagged += 1
        if str(row.get("confidence", "")).lower() == "low":
            self.counts.low_conf += 1
        cls = (row.get("classifier") or "").strip().lower() or "unknown"
        bucket = self.by_classifier.setdefault(cls, ClassifierCounts())
        bucket.total += 1
        if handover:
            bucket.events += 1
        if str(row.get("flag_for_review", "")).lower() == "true":
            bucket.flagged += 1
        cam = row.get("camera_id") or row.get("camera") or "unknown"
        bucket.by_camera[cam] = bucket.by_camera.get(cam, 0) + 1

    # ------------------------------------------------------------------
    # Session id allocation
    # ------------------------------------------------------------------

    def next_session_id(self, camera_id: str = "") -> str:
        prefix = _slugify_cam_id(camera_id)
        with self.lock:
            self._seq_by_prefix[prefix] = self._seq_by_prefix.get(prefix, 0) + 1
            return f"{prefix}-{self._seq_by_prefix[prefix]:03d}"

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(self, *, camera: str, session_id: str,
            start_time: datetime, end_time: datetime,
            result: dict, snapshot_path: str = "",
            mp4_path: str = "", merged_from: Optional[list] = None,
            camera_id: str = "", classifier: str = "",
            scenario_label: str = "") -> None:
        duration = max(0.0, (end_time - start_time).total_seconds())
        try:
            item_count = int(result.get("item_count") or 0)
        except (TypeError, ValueError):
            item_count = 0
        people = result.get("people") or []
        handover = bool(result.get("handover_occurred",
                                   result.get("item_presented", False)))
        items_handed = result.get("items_handed_over") \
            or result.get("objects_detected") or []
        customer_desc = result.get("customer_description") or (
            people[0]["description"] if people else ""
        )
        row = {
            "session_id": session_id,
            "camera_id": camera_id or camera,
            "camera": camera,
            "classifier": (classifier or "").lower(),
            "scenario_label": scenario_label,
            "start_time": start_time.isoformat(timespec="seconds"),
            "end_time": end_time.isoformat(timespec="seconds"),
            "duration_sec": f"{duration:.1f}",
            "handover_occurred": handover,
            "item_presented": handover,
            "item_count": item_count,
            "customer_description": str(customer_desc)[:400],
            "items_handed_over": "|".join(str(x) for x in items_handed),
            "confidence": str(result.get("confidence", "low")).lower(),
            "flag_for_review": bool(result.get("flag_for_review", False)),
            "num_people": len(people),
            "per_person_json": json.dumps(people, ensure_ascii=False),
            "narrative": str(result.get("narrative", ""))[:2000],
            "objects_detected": "|".join(str(x) for x in items_handed),
            "merged_from": "|".join(str(x) for x in (merged_from or [])),
            "merged_count": len(merged_from or []),
            "snapshot_path": snapshot_path,
            "mp4_path": mp4_path,
        }
        with self.lock:
            self._w.writerow(row)
            self._f.flush()
            self._tally({
                "handover_occurred": str(handover),
                "flag_for_review": str(row["flag_for_review"]),
                "confidence": row["confidence"],
                "classifier": row["classifier"],
                "camera_id": row["camera_id"],
                "camera": row["camera"],
            })

        icon = "[OK]" if handover else "[--]"
        label = "Event" if handover else "no-event"
        cls_disp = scenario_label or row["classifier"] or "?"
        print(f"{icon} {end_time.strftime('%H:%M:%S')}  {session_id}  "
              f"[{cls_disp}]  {label:<10} dur={duration:5.1f}s  "
              f"items={item_count}  [{row['confidence'].upper()}]",
              flush=True)

    # ------------------------------------------------------------------
    # Daily report
    # ------------------------------------------------------------------

    def write_daily_report(self) -> str:
        c = self.counts
        lines = [
            f"SCO Vision — daily report {self.date}",
            "=" * 60,
            f"total sessions         : {c.total}",
            f"events observed        : {c.handovers}",
            f"no-event sessions      : {c.none}",
            f"flagged for review     : {c.flagged}",
            f"low-confidence (review): {c.low_conf}",
            f"csv log                : {self.csv_path}",
            f"videos                 : {self.video_dir}",
            "",
            "By classifier",
            "-" * 60,
        ]
        if not self.by_classifier:
            lines.append("(no sessions yet)")
        else:
            for cls in sorted(self.by_classifier.keys()):
                bucket = self.by_classifier[cls]
                lines.append(
                    f"  {cls:<14}  total={bucket.total:<5}  "
                    f"events={bucket.events:<5}  flagged={bucket.flagged:<5}  "
                    f"cameras={dict(sorted(bucket.by_camera.items()))}"
                )
        text = "\n".join(lines)
        with self.lock:
            self.report_path.write_text(text + "\n")
        return text

    def close(self) -> None:
        with self.lock:
            try:
                self._f.close()
            except Exception:
                pass


# ----------------------------------------------------------------------
# Retention
# ----------------------------------------------------------------------

def cleanup_retention(*, video_dir: Path, snapshot_dir: Path,
                      log_dir: Path, retention_days: int) -> dict:
    cutoff = time.time() - retention_days * 86400
    deleted = {"mp4": 0, "snapshots": 0, "csv": 0}

    for p in Path(video_dir).glob("*.mp4"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink(missing_ok=True)
                deleted["mp4"] += 1
        except FileNotFoundError:
            continue
        except Exception:
            log.exception("retention: failed to delete %s", p)

    for p in Path(snapshot_dir).glob("*"):
        try:
            if p.is_file() and p.stat().st_mtime < cutoff:
                p.unlink(missing_ok=True)
                deleted["snapshots"] += 1
        except FileNotFoundError:
            continue
        except Exception:
            log.exception("retention: failed to delete %s", p)

    csv_cutoff = time.time() - 2 * retention_days * 86400
    for p in (list(Path(log_dir).glob("sessions_*.csv")) +
              list(Path(log_dir).glob("report_*.txt"))):
        try:
            if p.stat().st_mtime < csv_cutoff:
                p.unlink(missing_ok=True)
                deleted["csv"] += 1
        except FileNotFoundError:
            continue
        except Exception:
            log.exception("retention: failed to delete %s", p)

    log.info("retention sweep: %s (cutoff=%s)", deleted,
             datetime.fromtimestamp(cutoff).isoformat(timespec="seconds"))
    return deleted


class RetentionJanitor(threading.Thread):
    def __init__(self, *, video_dir: Path, snapshot_dir: Path, log_dir: Path,
                 retention_days: int, interval_hours: float,
                 stop_evt: threading.Event):
        super().__init__(daemon=True, name="retention-janitor")
        self.video_dir = Path(video_dir)
        self.snapshot_dir = Path(snapshot_dir)
        self.log_dir = Path(log_dir)
        self.retention_days = retention_days
        self.interval_sec = max(60.0, interval_hours * 3600.0)
        self.stop_evt = stop_evt

    def run(self) -> None:
        try:
            cleanup_retention(
                video_dir=self.video_dir,
                snapshot_dir=self.snapshot_dir,
                log_dir=self.log_dir,
                retention_days=self.retention_days,
            )
        except Exception:
            log.exception("retention initial sweep failed")
        while not self.stop_evt.wait(self.interval_sec):
            try:
                cleanup_retention(
                    video_dir=self.video_dir,
                    snapshot_dir=self.snapshot_dir,
                    log_dir=self.log_dir,
                    retention_days=self.retention_days,
                )
            except Exception:
                log.exception("retention sweep failed")
