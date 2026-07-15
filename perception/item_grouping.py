"""SCO item de-duplication (v1.1).

Falcon emits multiple labels per physical item — the POS-derived
``sco_item_NNN`` query and the generic ``sco_generic_*`` catch-all
can both fire on the same on-screen object. The tracker may also
emit multiple short tracks for the same item across frames. Without
de-dup, the VLM and the SCO policy see "one extra candidate" for
what was actually a re-detection of the POS item.

This module folds raw detections + tracks into a small, stable list
of **canonical item groups**. Each group carries:
  * the POS bill line it matched (or None for extras),
  * every source label / track that fed it (audit-friendly),
  * a representative bbox + frame for evidence overlay,
  * a confidence tier.

Person and receipt tracks are intentionally ignored — they are
identity / paperwork, not items to count.

The grouping logic is deliberately conservative: a generic-product
detection only merges into a POS-matched group when it is BOTH
spatially close (IoU above a threshold) AND temporally overlapping
(track time ranges intersect, possibly across a small gap).
Otherwise it becomes its own group and is reported as an extra
candidate.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Iterable, Optional


log = logging.getLogger(__name__)


# Defaults tuned to be loose enough to fold re-detections of the
# same physical object but tight enough that two distinct items
# sitting side-by-side on a bagging shelf are not collapsed.
DEFAULT_IOU_THRESHOLD: float = 0.3
DEFAULT_TIME_GAP_SEC: float = 5.0

# Reserved Falcon labels that are NOT items (these come from
# FalconClient.DEFAULT_CATEGORIES). Tracks/detections carrying these
# are skipped before grouping.
_NON_ITEM_LABELS: frozenset[str] = frozenset({"person", "receipt"})

# Pattern: sco_item_NNN. Captures the line index so the group can
# resolve back to the POS basket entry.
_POS_ITEM_RE = re.compile(r"^sco_item_(\d+)$")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def group_sco_items(detections: list[dict],
                    tracks: list[dict],
                    *,
                    pos_basket: Optional[Iterable[dict]] = None,
                    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
                    time_gap_sec: float = DEFAULT_TIME_GAP_SEC,
                    ) -> list[dict]:
    """Fold raw Falcon detections/tracks into canonical item groups.

    Returns a list of group dicts shaped:

        {
          "group_id": "sco_group_001",
          "matched_pos_item": "<basket line desc>" | None,
          "matched_pos_index": <0-based int> | None,
          "source_labels": ["sco_item_000", "sco_generic_products"],
          "track_ids": [...],
          "first_seen_ts": "<ISO>",
          "last_seen_ts": "<ISO>",
          "representative_bbox": [x1, y1, x2, y2],
          "representative_frame_id": "...",
          "confidence": "high" | "medium" | "low",
          "is_extra_candidate": bool,
        }

    Non-item tracks (person, receipt) and tracks with no resolvable
    bbox are skipped. The function is pure / never raises on
    malformed input.
    """
    basket = list(pos_basket or [])
    indexed_detections = list(detections or [])
    track_summaries = [
        s for s in (_summarise_track(t, indexed_detections)
                    for t in (tracks or []))
        if s is not None
    ]

    # Phase 1: one group PER DISTINCT POS LINE (not per track). Falcon +
    # the tracker fragment a single physical item into many
    # ``sco_item_NNN`` tracks that all carry the SAME ``pos_index``;
    # seeding a group per track inflates the "distinct item" count that
    # the VLM prompt is told to treat as authoritative (it then reports a
    # false count mismatch and every case becomes REVIEW). Fold all tracks
    # sharing a ``pos_index`` into ONE group — consistent with
    # ``count_audit_zone_items`` which already collapses matched groups to
    # distinct POS lines.
    pos_groups: list[dict] = []
    pos_by_index: dict[int, dict] = {}
    for s in track_summaries:
        idx = _pos_index(s["label"])
        if idx is None:
            continue
        existing = pos_by_index.get(idx)
        if existing is None:
            g = _seed_group(s, pos_index=idx, basket=basket)
            pos_by_index[idx] = g
            pos_groups.append(g)
        else:
            _merge_into(existing, s)

    # Phase 2: fold generic / default-item tracks into the closest
    # overlapping POS group (greedy by IoU). If none, the track
    # becomes its own group (potential extra candidate).
    extra_groups: list[dict] = []
    for s in track_summaries:
        if _pos_index(s["label"]) is not None:
            continue
        best_idx, best_iou = _best_overlap(s, pos_groups,
                                            iou_threshold=iou_threshold,
                                            time_gap_sec=time_gap_sec)
        if best_idx is not None:
            _merge_into(pos_groups[best_idx], s)
        else:
            extra_groups.append(_seed_group(s, pos_index=None,
                                             basket=basket))

    # Phase 2.5: de-fragment the EXTRA groups among themselves. Falcon
    # over-detects and splits one physical extra object into several
    # tracks, each of which seeded its own group; collapse fragments that
    # overlap in space (IoU) AND time so the count the VLM sees reflects
    # distinct physical objects, not detector noise. Same criterion
    # (COUNT_MERGE_IOU + time overlap) used by ``count_audit_zone_items``.
    extra_groups = _defragment_extra_groups(extra_groups,
                                            time_gap_sec=time_gap_sec)

    # Phase 3: stable id + confidence + extra-candidate flag.
    all_groups = pos_groups + extra_groups
    for i, g in enumerate(all_groups, start=1):
        g["group_id"] = f"sco_group_{i:03d}"
        g["confidence"] = _confidence_for(g)
        g["is_extra_candidate"] = (g["matched_pos_item"] is None)
    return all_groups


def matched_groups(groups: list[dict]) -> list[dict]:
    """Convenience: groups with a POS line attached."""
    return [g for g in groups if not g.get("is_extra_candidate")]


def extra_groups(groups: list[dict]) -> list[dict]:
    """Convenience: groups that did NOT match any POS line."""
    return [g for g in groups if g.get("is_extra_candidate")]


# IoU above which two extra groups are treated as fragments of the SAME
# physical object for COUNTING (looser than the POS-merge threshold — the
# goal here is deliberate over-merging so the count isn't inflated by
# Falcon's duplicate boxes).
COUNT_MERGE_IOU: float = 0.35


def count_audit_zone_items(groups: list[dict],
                           *, merge_iou: float = COUNT_MERGE_IOU,
                           time_gap_sec: float = DEFAULT_TIME_GAP_SEC) -> dict:
    """Deterministic distinct-item count for the SCO audit zone — an
    INDEPENDENT Falcon signal, computed WITHOUT the VLM and WITHOUT mutating
    ``groups`` (which still feed the VLM prompt unchanged).

    Each POS-matched group counts once. Extra (non-POS) groups are collapsed
    among themselves when they overlap in space AND time, because Falcon
    over-detects and fragments one physical object into several tracks —
    those must not each add to the count. Returns::

        {"count": int,           # matched + de-fragmented extras
         "matched_count": int,   # distinct POS-matched items
         "extra_count": int,     # distinct extra items (after collapse)
         "extra_raw": int}       # extra groups BEFORE collapse (transparency)

    Pure; never raises on malformed input.
    """
    gs = list(groups or [])
    matched = [g for g in gs if not g.get("is_extra_candidate")]
    extras = [g for g in gs if g.get("is_extra_candidate")]

    # Matched groups collapse to DISTINCT POS lines, not distinct fragments:
    # Falcon splits one POS item into many ``sco_item_NNN`` tracks, each of
    # which seeds its own matched group. Count the POS line once.
    matched_pos = set()
    for g in matched:
        idx = g.get("matched_pos_index")
        matched_pos.add(idx if idx is not None else id(g))
    matched_count = len(matched_pos)

    def _area(g) -> float:
        bb = g.get("representative_bbox")
        if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
            return 0.0
        return max(0.0, float(bb[2]) - float(bb[0])) \
            * max(0.0, float(bb[3]) - float(bb[1]))

    kept: list[dict] = []
    for g in sorted(extras, key=_area, reverse=True):
        bb = g.get("representative_bbox")
        if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
            continue
        gf, gl = _coerce_dt(g.get("first_seen_ts")), _coerce_dt(
            g.get("last_seen_ts"))
        is_fragment = False
        for k in kept:
            if _iou(bb, k["representative_bbox"]) >= merge_iou and \
                    _time_overlaps(gf, gl,
                                   _coerce_dt(k.get("first_seen_ts")),
                                   _coerce_dt(k.get("last_seen_ts")),
                                   gap_sec=time_gap_sec):
                is_fragment = True
                break
        if not is_fragment:
            kept.append(g)

    return {
        "count": matched_count + len(kept),
        "matched_count": matched_count,
        "extra_count": len(kept),
        "extra_raw": len(extras),
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _summarise_track(track, detections) -> Optional[dict]:
    """Pull (label, first, last, repr bbox, repr frame, repr score)
    out of a Track dict. Returns None if the track is non-item or has
    no resolvable bbox."""
    if not isinstance(track, dict):
        return None
    label = str(track.get("label") or "").strip()
    if not label:
        return None
    label_lower = label.lower()
    if label_lower in _NON_ITEM_LABELS:
        return None
    # Skip Falcon's generic "person*"-style track variants (e.g. hand,
    # arm) — same intent as the episode selector's _is_person_label.
    if "person" in label_lower and not label_lower.startswith("sco_"):
        return None

    first = _coerce_dt(track.get("first_seen_ts"))
    last = _coerce_dt(track.get("last_seen_ts"))
    track_id = str(track.get("track_id") or "") or None
    det_indices = list(track.get("detections") or [])

    # Resolve representative detection: highest-scored detection
    # belonging to this track. Tolerate missing/garbage indices.
    repr_det = None
    for i in det_indices:
        if not isinstance(i, int) or i < 0 or i >= len(detections):
            continue
        d = detections[i]
        if not isinstance(d, dict):
            continue
        if repr_det is None or float(d.get("score") or 0.0) > \
                float(repr_det.get("score") or 0.0):
            repr_det = d

    # Fall back to ANY detection sharing the same label if the track's
    # own detections list was empty/broken.
    if repr_det is None:
        for d in detections:
            if isinstance(d, dict) and d.get("label") == label:
                if repr_det is None or float(d.get("score") or 0.0) > \
                        float(repr_det.get("score") or 0.0):
                    repr_det = d

    if repr_det is None:
        return None
    bbox = repr_det.get("bbox_xyxy")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    score = float(repr_det.get("score") or 0.0)
    frame_id = str(repr_det.get("frame_id") or "")

    # Fall back to representative-detection ts when the track itself
    # didn't carry a usable timestamp.
    repr_ts = _coerce_dt(repr_det.get("ts"))
    if first is None:
        first = repr_ts
    if last is None:
        last = repr_ts

    return {
        "label": label,
        "track_id": track_id,
        "first": first,
        "last": last,
        "bbox": [float(x) for x in bbox],
        "score": score,
        "frame_id": frame_id,
    }


def _seed_group(s: dict, *, pos_index: Optional[int],
                basket: list[dict]) -> dict:
    matched_pos = None
    if pos_index is not None and 0 <= pos_index < len(basket):
        item = basket[pos_index]
        if isinstance(item, dict):
            matched_pos = (item.get("description") or item.get("name")
                           or item.get("item_description")
                           or item.get("sku") or None)
    track_ids = [s["track_id"]] if s.get("track_id") else []
    return {
        "matched_pos_item": matched_pos,
        "matched_pos_index": pos_index,
        "source_labels": [s["label"]],
        "track_ids": track_ids,
        "first_seen_ts": _iso(s["first"]),
        "last_seen_ts": _iso(s["last"]),
        "representative_bbox": list(s["bbox"]),
        "representative_frame_id": s["frame_id"],
        "_repr_score": s["score"],
    }


def _merge_group_into(dst: dict, src: dict) -> None:
    """Fold group ``src`` into group ``dst`` (group-to-group merge).

    Unions source labels + track ids, widens the [first,last] span, and
    keeps the higher-scored representative bbox/frame. Used to collapse
    extra-group fragments of one physical object.
    """
    for lbl in (src.get("source_labels") or []):
        if lbl not in dst["source_labels"]:
            dst["source_labels"].append(lbl)
    for tid in (src.get("track_ids") or []):
        if tid and tid not in dst["track_ids"]:
            dst["track_ids"].append(tid)
    cur_first = _coerce_dt(dst.get("first_seen_ts"))
    cur_last = _coerce_dt(dst.get("last_seen_ts"))
    s_first = _coerce_dt(src.get("first_seen_ts"))
    s_last = _coerce_dt(src.get("last_seen_ts"))
    if cur_first is None or (s_first is not None and s_first < cur_first):
        dst["first_seen_ts"] = src.get("first_seen_ts")
    if cur_last is None or (s_last is not None and s_last > cur_last):
        dst["last_seen_ts"] = src.get("last_seen_ts")
    if float(src.get("_repr_score") or 0.0) > float(dst.get("_repr_score") or 0.0):
        dst["representative_bbox"] = list(src.get("representative_bbox") or
                                          dst.get("representative_bbox") or [])
        dst["representative_frame_id"] = src.get("representative_frame_id")
        dst["_repr_score"] = src.get("_repr_score")


def _defragment_extra_groups(extras: list[dict], *,
                             merge_iou: float = COUNT_MERGE_IOU,
                             time_gap_sec: float = DEFAULT_TIME_GAP_SEC
                             ) -> list[dict]:
    """Collapse extra-group fragments of the SAME physical object.

    Largest-first greedy: an extra group is a fragment of a kept group
    when their representative bboxes overlap (IoU >= ``merge_iou``) AND
    their time spans overlap within ``time_gap_sec``. Mirrors the count
    logic in ``count_audit_zone_items`` but actually merges the group
    records so the surviving list = distinct physical extras.
    """
    def _area(g) -> float:
        bb = g.get("representative_bbox")
        if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
            return 0.0
        return max(0.0, float(bb[2]) - float(bb[0])) \
            * max(0.0, float(bb[3]) - float(bb[1]))

    kept: list[dict] = []
    for g in sorted(extras, key=_area, reverse=True):
        bb = g.get("representative_bbox")
        if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
            kept.append(g)
            continue
        gf, gl = _coerce_dt(g.get("first_seen_ts")), _coerce_dt(
            g.get("last_seen_ts"))
        merged = False
        for k in kept:
            if _iou(bb, k.get("representative_bbox")) >= merge_iou and \
                    _time_overlaps(gf, gl,
                                   _coerce_dt(k.get("first_seen_ts")),
                                   _coerce_dt(k.get("last_seen_ts")),
                                   gap_sec=time_gap_sec):
                _merge_group_into(k, g)
                merged = True
                break
        if not merged:
            kept.append(g)
    return kept


def _merge_into(group: dict, s: dict) -> None:
    """Fold ``s`` into an existing group, widening its time range
    and recording the source label/track."""
    if s["label"] not in group["source_labels"]:
        group["source_labels"].append(s["label"])
    if s.get("track_id") and s["track_id"] not in group["track_ids"]:
        group["track_ids"].append(s["track_id"])
    # Widen time range
    cur_first = _coerce_dt(group["first_seen_ts"])
    cur_last = _coerce_dt(group["last_seen_ts"])
    if cur_first is None or (s["first"] is not None
                              and s["first"] < cur_first):
        group["first_seen_ts"] = _iso(s["first"])
    if cur_last is None or (s["last"] is not None
                             and s["last"] > cur_last):
        group["last_seen_ts"] = _iso(s["last"])
    # Promote representative bbox if the merged detection had a
    # higher score.
    if s["score"] > float(group.get("_repr_score") or 0.0):
        group["representative_bbox"] = list(s["bbox"])
        group["representative_frame_id"] = s["frame_id"]
        group["_repr_score"] = s["score"]


def _best_overlap(s: dict, pos_groups: list[dict],
                  *, iou_threshold: float, time_gap_sec: float
                  ) -> tuple[Optional[int], float]:
    """Pick the POS group with the highest IoU above threshold AND
    overlapping time range. Returns (index, iou) or (None, 0.0)."""
    best_idx = None
    best_iou = 0.0
    for i, g in enumerate(pos_groups):
        if not _time_overlaps(s["first"], s["last"],
                              _coerce_dt(g["first_seen_ts"]),
                              _coerce_dt(g["last_seen_ts"]),
                              gap_sec=time_gap_sec):
            continue
        iou = _iou(s["bbox"], g["representative_bbox"])
        if iou >= iou_threshold and iou > best_iou:
            best_iou = iou
            best_idx = i
    return best_idx, best_iou


def _confidence_for(group: dict) -> str:
    """Heuristic confidence tier:
        high   — POS-matched group with corroborating non-POS label
        medium — POS-matched group with a single label
        low    — extra candidate (generic-only, no POS match)
    """
    if group["matched_pos_item"] is None:
        return "low"
    if len(group["source_labels"]) >= 2:
        return "high"
    return "medium"


def _pos_index(label: str) -> Optional[int]:
    m = _POS_ITEM_RE.match(label)
    return int(m.group(1)) if m else None


def _iou(a: list[float], b: list[float]) -> float:
    if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))
            and len(a) == 4 and len(b) == 4):
        return 0.0
    x1, y1, x2, y2 = (float(v) for v in a)
    X1, Y1, X2, Y2 = (float(v) for v in b)
    ix1, iy1 = max(x1, X1), max(y1, Y1)
    ix2, iy2 = min(x2, X2), min(y2, Y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0.0:
        return 0.0
    a_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    b_area = max(0.0, X2 - X1) * max(0.0, Y2 - Y1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def _time_overlaps(a_first: Optional[datetime], a_last: Optional[datetime],
                   b_first: Optional[datetime], b_last: Optional[datetime],
                   *, gap_sec: float) -> bool:
    if None in (a_first, a_last, b_first, b_last):
        # If we can't tell time-wise, fall back to spatial-only check.
        return True
    gap = timedelta(seconds=gap_sec)
    return max(a_first, b_first) <= min(a_last, b_last) + gap


def _coerce_dt(v) -> Optional[datetime]:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v)
        except ValueError:
            return None
    return None


def _iso(dt: Optional[datetime]) -> str:
    if dt is None:
        return ""
    return dt.isoformat()
