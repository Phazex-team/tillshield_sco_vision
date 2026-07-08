"""Deterministic Falcon audit-zone item count (independent of the VLM)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perception.item_grouping import count_audit_zone_items  # noqa: E402


def _g(gid, bbox, *, extra, t0="2026-07-08T08:00:00", t1="2026-07-08T08:00:05"):
    return {"group_id": gid, "representative_bbox": bbox,
            "is_extra_candidate": extra,
            "first_seen_ts": t0, "last_seen_ts": t1}


def test_matched_counts_once_each():
    groups = [_g("g1", [0, 0, 10, 10], extra=False),
              _g("g2", [50, 50, 60, 60], extra=False)]
    r = count_audit_zone_items(groups)
    assert r["count"] == 2 and r["matched_count"] == 2 and r["extra_count"] == 0


def test_extra_fragments_collapse():
    # Three heavily-overlapping extra boxes (same object, same time) -> 1.
    groups = [
        _g("e1", [100, 100, 200, 200], extra=True),
        _g("e2", [102, 101, 198, 202], extra=True),  # ~same box
        _g("e3", [101, 99, 201, 199], extra=True),   # ~same box
    ]
    r = count_audit_zone_items(groups)
    assert r["extra_raw"] == 3
    assert r["extra_count"] == 1   # collapsed to one physical item
    assert r["count"] == 1


def test_distinct_extras_kept():
    # Two well-separated extras -> both counted.
    groups = [_g("e1", [0, 0, 20, 20], extra=True),
              _g("e2", [300, 300, 320, 320], extra=True)]
    r = count_audit_zone_items(groups)
    assert r["extra_count"] == 2 and r["count"] == 2


def test_same_spot_different_time_not_merged():
    # Same location but non-overlapping time = two distinct sequential items.
    groups = [
        _g("e1", [100, 100, 200, 200], extra=True,
           t0="2026-07-08T08:00:00", t1="2026-07-08T08:00:03"),
        _g("e2", [100, 100, 200, 200], extra=True,
           t0="2026-07-08T08:00:30", t1="2026-07-08T08:00:33"),
    ]
    r = count_audit_zone_items(groups)
    assert r["extra_count"] == 2


def test_matched_plus_collapsed_extras():
    groups = [
        _g("m1", [0, 0, 10, 10], extra=False),
        _g("e1", [100, 100, 200, 200], extra=True),
        _g("e2", [103, 102, 197, 201], extra=True),  # fragment of e1
    ]
    r = count_audit_zone_items(groups)
    assert r == {"count": 2, "matched_count": 1, "extra_count": 1,
                 "extra_raw": 2}


def test_matched_fragments_collapse_by_pos_index():
    # Falcon splits POS items into many tracks -> many matched groups, but
    # they map to few POS lines. Count DISTINCT POS lines, not fragments.
    groups = [{"is_extra_candidate": False, "matched_pos_index": i % 3,
               "representative_bbox": [i, i, i + 5, i + 5]}
              for i in range(30)]   # 30 fragment groups, 3 POS lines
    r = count_audit_zone_items(groups)
    assert r["matched_count"] == 3
    assert r["count"] == 3


def test_empty():
    assert count_audit_zone_items([])["count"] == 0
