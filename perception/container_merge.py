"""Post-SAM3 container de-duplication / merge.

Sits AFTER the canonical grouper and BEFORE the VLM. The grouper
already de-duplicates label-overlap on the same physical object
(two labels firing on one bbox). What it does NOT do is collapse
fragmented identities for the same physical container across
time — e.g. one hand-held takeaway box appearing as
``sam3_obj_0003`` for frames 0..5 and ``sam3_obj_0017`` later in
the episode after a brief occlusion.

For SCO checkout the fragmentation pattern matters: raw SAM3
object_id count is NOT a reliable basket count. This module emits
merged container groups with explicit count confidence so the
prompt and policy stages can avoid auto-flagging
``sco_basket_mismatch`` purely on raw_id_count != pos_count.

Inputs:
  * ``canonical_groups`` from ``perception.item_grouping.group_sco_items``
  * ``detections``        from the perception result (used to test
                          simultaneous visibility per frame).

Output:
  ``ContainerMergeResult``:
    .merged_groups[]   one per likely physical container
    .count_min / .count_max
    .count_confidence  ``high`` / ``medium`` / ``low``
    .fragmentation_suspected
    .missed_container_possible
    .merge_audit[]     human-readable trail per merge decision

Rules (kept conservative — when in doubt, DO NOT merge):

  R1. Merge only within the container family. Source labels that
      look like ``food_container`` / ``plastic_food_box`` /
      ``takeaway_container`` / ``sco_item_*`` / ``item`` all
      qualify; ``bag`` / ``person`` / ``receipt`` do not.

  R2. Two groups never merge if they are SIMULTANEOUSLY visible
      in the same sampled frame at spatially-separated locations
      (IoU < 0.3 on shared frame). The grouper feeds us
      detections so we can check this without re-running SAM3.

  R3. Two groups can merge if:
        * same container family (R1),
        * NO frame where both fire with separated bboxes (R2),
        * compatible bbox size: max area / min area <= 4.0,
        * compatible aspect ratio: |ratio_a - ratio_b| / max <= 0.5,
        * temporal continuity: either time ranges overlap, OR
          the temporal gap between them is <= MAX_FRAGMENT_GAP_SEC,
        * spatial continuity: representative centers within
          MAX_FRAGMENT_PIXEL_DISTANCE (treats a hand-carried box
          moving short distances as the same identity).

  R4. ``missed_container_possible`` is True when only POS-specific
      groups (matched_pos_item != None) survived AND the merged
      container count is strictly less than the POS basket size
      AND the episode coverage isn't tiny — meaning the system
      saw the customer but didn't see one of the items.
      (POS basket size comes in via the optional ``pos_basket_size``
      argument; when omitted, this flag stays False.)

  R5. Count confidence:
        * high   — no fragmentation, no missed-suspect, all groups
                   POS-tied or all extra-candidate with sharp counts.
        * medium — fragmentation suspected or missed_container_possible
                   or any merge happened with R3 mid-range.
        * low    — no canonical groups at all, or every merge
                   decision was borderline, or the source data has
                   <3 sampled frames in the episode.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable, Optional


log = logging.getLogger(__name__)


CONTAINER_FAMILY_TOKENS = (
    "food_container",
    "plastic_food_box",
    "takeaway_container",
    "hot_food",
    "container",
    "sco_item_",
    "item",
)

# Tunables (operator-overridable via case_runner). Tight defaults so
# we under-merge in the borderline cases rather than over-merge.
MAX_FRAGMENT_GAP_SEC: float = 8.0
MAX_FRAGMENT_PIXEL_DISTANCE: float = 250.0
MIN_AREA_RATIO: float = 4.0      # max(area)/min(area) must be <= this
MAX_ASPECT_DELTA: float = 0.5    # |a_ratio - b_ratio| / max <= this
SIMULTANEOUS_IOU_FLOOR: float = 0.3


@dataclass
class ContainerMergeResult:
    merged_groups: list[dict]
    count_min: int
    count_max: int
    count_confidence: str           # high|medium|low
    fragmentation_suspected: bool
    missed_container_possible: bool
    merge_audit: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "merged_groups": self.merged_groups,
            "count_min": self.count_min,
            "count_max": self.count_max,
            "count_confidence": self.count_confidence,
            "fragmentation_suspected": self.fragmentation_suspected,
            "missed_container_possible": self.missed_container_possible,
            "merge_audit": self.merge_audit,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def merge_sam3_containers(canonical_groups: list[dict],
                          detections: list[dict],
                          *,
                          pos_basket_size: Optional[int] = None,
                          episode_coverage_ratio: float = 0.0,
                          max_fragment_gap_sec: float = MAX_FRAGMENT_GAP_SEC,
                          max_fragment_pixel_distance: float
                              = MAX_FRAGMENT_PIXEL_DISTANCE,
                          ) -> ContainerMergeResult:
    """Apply the merge rules to canonical groups. Pure function.

    Never raises. If input is empty/malformed, returns an empty
    result with ``count_confidence="low"``.
    """
    audit: list[str] = []
    candidates = [g for g in (canonical_groups or [])
                  if _is_container_family(g)]
    other = [g for g in (canonical_groups or [])
             if not _is_container_family(g)]

    if not candidates:
        return ContainerMergeResult(
            merged_groups=list(other), count_min=0, count_max=0,
            count_confidence="low",
            fragmentation_suspected=False,
            missed_container_possible=bool(
                pos_basket_size and pos_basket_size > 0
                and episode_coverage_ratio >= 0.05),
            merge_audit=["no container-family groups in canonical input"],
        )

    # Pre-compute simultaneous-frame conflicts: for any pair (i, j)
    # of candidate groups, did SAM3 ever emit them in the SAME frame
    # at spatially-separated bboxes? If yes, they NEVER merge (R2).
    conflicts = _simultaneous_conflicts(candidates, detections)
    for (i, j) in sorted(conflicts):
        audit.append(
            f"R2: groups {candidates[i]['group_id']} and "
            f"{candidates[j]['group_id']} co-exist in same frame "
            f"with low IoU — never merged.")

    # Greedy merging with rule R3. Two passes are enough in practice
    # (container fragments tend to be short chains).
    merged_clusters: list[list[int]] = [[i] for i in range(len(candidates))]
    changed = True
    pass_no = 0
    while changed and pass_no < 4:
        changed = False
        pass_no += 1
        i = 0
        while i < len(merged_clusters):
            j = i + 1
            while j < len(merged_clusters):
                if _can_merge_clusters(
                        merged_clusters[i], merged_clusters[j],
                        candidates, conflicts,
                        max_fragment_gap_sec=max_fragment_gap_sec,
                        max_fragment_pixel_distance=max_fragment_pixel_distance,
                        audit=audit):
                    merged_clusters[i] = merged_clusters[i] + merged_clusters[j]
                    merged_clusters.pop(j)
                    changed = True
                else:
                    j += 1
            i += 1

    merged_groups = [
        _merge_groups([candidates[idx] for idx in cluster])
        for cluster in merged_clusters
    ]

    fragmentation = any(len(c) > 1 for c in merged_clusters)
    if fragmentation:
        audit.append(
            f"R3: {sum(len(c) for c in merged_clusters)} raw "
            f"container-family groups merged into "
            f"{len(merged_groups)} physical containers.")

    # Append non-container groups back so the VLM still sees them
    # (e.g. bags, person evidence).
    merged_groups_out = list(merged_groups) + list(other)

    count_min = max(0, len(merged_groups))
    # Upper bound = raw count (no merging). If fragmentation_suspected,
    # the upper bound is meaningful evidence for the VLM/policy.
    count_max = max(count_min, len(candidates))
    missed = bool(
        pos_basket_size is not None
        and pos_basket_size > 0
        and count_max < pos_basket_size
        and episode_coverage_ratio >= 0.05)
    if missed:
        audit.append(
            f"R4: merged container count {count_max} < POS basket size "
            f"{pos_basket_size}; missed_container_possible=True.")

    # Confidence per R5.
    if not candidates:
        conf = "low"
    elif (fragmentation or missed
          or any("R3" in line for line in audit)):
        conf = "medium"
    else:
        conf = "high"
    if count_max == count_min and not fragmentation and not missed \
            and all(g.get("confidence") == "high" for g in merged_groups):
        conf = "high"

    return ContainerMergeResult(
        merged_groups=merged_groups_out,
        count_min=count_min,
        count_max=count_max,
        count_confidence=conf,
        fragmentation_suspected=fragmentation,
        missed_container_possible=missed,
        merge_audit=audit,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_container_family(group: dict) -> bool:
    labels = (group.get("source_labels") or [])
    for lab in labels:
        lo = str(lab).lower()
        if any(tok in lo for tok in CONTAINER_FAMILY_TOKENS):
            # Exclude bags / receipts / persons explicitly even if
            # the label string contains "container".
            if "bag" in lo or "receipt" in lo or "person" in lo:
                continue
            return True
    return False


def _simultaneous_conflicts(candidates: list[dict],
                            detections: list[dict]) -> set:
    """Return set of (i, j) candidate index pairs that co-fire in the
    same SAM3 sampled frame with low spatial IoU."""
    by_frame: dict[int, list[tuple[int, list[float]]]] = {}
    # Map SAM3 object_id -> candidate index.
    sam3_id_to_cand: dict[int, int] = {}
    for i, g in enumerate(candidates):
        for tid in (g.get("track_ids") or []):
            # tracks look like "sam3_obj_0003"
            try:
                obj_id = int(str(tid).rsplit("_", 1)[-1])
            except (ValueError, TypeError):
                continue
            sam3_id_to_cand[obj_id] = i
    for d in (detections or []):
        if not isinstance(d, dict):
            continue
        obj_id = d.get("sam3_object_id")
        if obj_id is None:
            continue
        cand = sam3_id_to_cand.get(int(obj_id))
        if cand is None:
            continue
        frame = d.get("frame_idx")
        bbox = d.get("bbox_xyxy")
        if frame is None or not isinstance(bbox, (list, tuple)) \
                or len(bbox) != 4:
            continue
        by_frame.setdefault(int(frame), []).append(
            (cand, [float(x) for x in bbox]))
    conflicts: set = set()
    for frame, items in by_frame.items():
        # Pairwise IoU check.
        for a in range(len(items)):
            ia, ba = items[a]
            for b in range(a + 1, len(items)):
                ib, bb = items[b]
                if ia == ib:
                    continue
                if _iou(ba, bb) < SIMULTANEOUS_IOU_FLOOR:
                    key = (min(ia, ib), max(ia, ib))
                    conflicts.add(key)
    return conflicts


def _can_merge_clusters(cluster_a: list[int], cluster_b: list[int],
                        candidates: list[dict],
                        conflicts: set,
                        *,
                        max_fragment_gap_sec: float,
                        max_fragment_pixel_distance: float,
                        audit: list[str]) -> bool:
    # R2: any pair across the clusters that co-exist → no merge.
    for i in cluster_a:
        for j in cluster_b:
            key = (min(i, j), max(i, j))
            if key in conflicts:
                return False

    a = _cluster_summary(cluster_a, candidates)
    b = _cluster_summary(cluster_b, candidates)
    if a is None or b is None:
        return False

    # Size / aspect compatibility.
    if _bad_area_ratio(a["area"], b["area"]):
        return False
    if _bad_aspect_delta(a["aspect"], b["aspect"]):
        return False

    # Temporal continuity.
    gap = _temporal_gap_sec(a["first"], a["last"], b["first"], b["last"])
    if gap is None or gap > max_fragment_gap_sec:
        return False

    # Spatial continuity.
    dist = _center_distance(a["bbox"], b["bbox"])
    if dist > max_fragment_pixel_distance:
        return False

    audit.append(
        f"R3 merge: clusters {[candidates[i]['group_id'] for i in cluster_a]} "
        f"and {[candidates[i]['group_id'] for i in cluster_b]} "
        f"(gap={gap:.1f}s dist={dist:.0f}px).")
    return True


def _cluster_summary(cluster: list[int], candidates: list[dict]
                     ) -> Optional[dict]:
    if not cluster:
        return None
    members = [candidates[i] for i in cluster]
    bboxes = [m.get("representative_bbox") for m in members
              if isinstance(m.get("representative_bbox"), list)
              and len(m["representative_bbox"]) == 4]
    if not bboxes:
        return None
    # Aggregate bbox = arithmetic mean of all member bboxes (rough
    # but enough for size/aspect compatibility).
    n = len(bboxes)
    avg = [sum(b[k] for b in bboxes) / n for k in range(4)]
    firsts = [_coerce_dt(m.get("first_seen_ts")) for m in members]
    lasts = [_coerce_dt(m.get("last_seen_ts")) for m in members]
    firsts = [d for d in firsts if d is not None]
    lasts = [d for d in lasts if d is not None]
    return {
        "bbox": avg,
        "area": max(0.0, avg[2] - avg[0]) * max(0.0, avg[3] - avg[1]),
        "aspect": _aspect(avg),
        "first": min(firsts) if firsts else None,
        "last": max(lasts) if lasts else None,
    }


def _merge_groups(groups: list[dict]) -> dict:
    """Collapse a cluster of canonical groups into one merged
    container record. The strongest representative bbox/frame is
    promoted. POS match is preserved if ANY member has one."""
    if len(groups) == 1:
        # Keep as-is but stamp merged_from = [self] for audit.
        g = dict(groups[0])
        g["merged_from"] = [g.get("group_id")]
        g["raw_group_count"] = 1
        return g
    # Multi-member.
    matched_pos = next(
        (g.get("matched_pos_item") for g in groups
         if g.get("matched_pos_item")),
        None)
    matched_idx = next(
        (g.get("matched_pos_index") for g in groups
         if g.get("matched_pos_index") is not None),
        None)
    src_labels: list[str] = []
    track_ids: list[str] = []
    for g in groups:
        for lab in (g.get("source_labels") or []):
            if lab not in src_labels:
                src_labels.append(lab)
        for tid in (g.get("track_ids") or []):
            if tid not in track_ids:
                track_ids.append(tid)
    # Time range.
    firsts = [_coerce_dt(g.get("first_seen_ts")) for g in groups]
    lasts = [_coerce_dt(g.get("last_seen_ts")) for g in groups]
    firsts = [d for d in firsts if d is not None]
    lasts = [d for d in lasts if d is not None]
    first_ts = min(firsts).isoformat() if firsts else ""
    last_ts = max(lasts).isoformat() if lasts else ""
    # Representative bbox/frame = highest score.
    repr_g = max(groups, key=lambda g: float(g.get("_repr_score") or 0.0))
    merged_ids = [g.get("group_id") for g in groups]
    return {
        # New stable ID. Keep "sco_group_..." prefix so downstream
        # rendering doesn't need to special-case.
        "group_id": "+".join(str(x) for x in merged_ids if x),
        "matched_pos_item": matched_pos,
        "matched_pos_index": matched_idx,
        "source_labels": src_labels,
        "track_ids": track_ids,
        "first_seen_ts": first_ts,
        "last_seen_ts": last_ts,
        "representative_bbox": list(repr_g.get("representative_bbox") or []),
        "representative_frame_id": repr_g.get("representative_frame_id", ""),
        "_repr_score": float(repr_g.get("_repr_score") or 0.0),
        "confidence": "medium",   # any merge → medium not high
        "is_extra_candidate": matched_pos is None,
        "merged_from": merged_ids,
        "raw_group_count": len(groups),
    }


def _bad_area_ratio(a: float, b: float) -> bool:
    if a <= 0 or b <= 0:
        return False
    return max(a, b) / max(min(a, b), 1e-6) > MIN_AREA_RATIO


def _bad_aspect_delta(a: float, b: float) -> bool:
    if a <= 0 or b <= 0:
        return True
    return abs(a - b) / max(a, b) > MAX_ASPECT_DELTA


def _temporal_gap_sec(a_first: Optional[datetime], a_last: Optional[datetime],
                      b_first: Optional[datetime], b_last: Optional[datetime]
                      ) -> Optional[float]:
    if None in (a_first, a_last, b_first, b_last):
        return None
    if max(a_first, b_first) <= min(a_last, b_last):
        return 0.0  # overlap
    return (max(a_first, b_first) - min(a_last, b_last)).total_seconds()


def _center_distance(a: list[float], b: list[float]) -> float:
    if len(a) != 4 or len(b) != 4:
        return float("inf")
    ax = (a[0] + a[2]) / 2.0; ay = (a[1] + a[3]) / 2.0
    bx = (b[0] + b[2]) / 2.0; by = (b[1] + b[3]) / 2.0
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _aspect(b: list[float]) -> float:
    w = max(0.0, b[2] - b[0]); h = max(0.0, b[3] - b[1])
    return w / max(h, 1e-6)


def _iou(a: list[float], b: list[float]) -> float:
    if len(a) != 4 or len(b) != 4:
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


def _coerce_dt(v) -> Optional[datetime]:
    if v is None or v == "":
        return None
    if isinstance(v, datetime):
        return v
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v)
        except ValueError:
            return None
    return None
