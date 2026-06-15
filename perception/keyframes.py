"""Keyframe selection (PRODUCTION_SPEC §10).

Picks frames that matter to a human reviewer: first appearance of each
track, the moment it enters the counter zone, handover candidate
moments, receipt frames, and the final state. Pure logic over Track +
Detection lists.
"""
from __future__ import annotations

from typing import Optional

from .schemas import Detection, Keyframe, Track


def select_keyframes(tracks: list[Track],
                     detections: list[Detection]
                     ) -> list[Keyframe]:
    keyframes: list[Keyframe] = []
    for t in tracks:
        if not t.detections:
            continue
        # First appearance
        first = detections[t.detections[0]]
        keyframes.append(Keyframe(
            frame_id=first.frame_id, frame_idx=first.frame_idx,
            ts=first.ts, role="first_appearance", track_id=t.track_id,
        ))
        # Counter placement: first detection inside counter_zone.
        for det_idx in t.detections:
            det = detections[det_idx]
            if "counter_zone" in t.zones and \
                    "counter_zone" in t.events[:1] + [f for f in t.events]:
                keyframes.append(Keyframe(
                    frame_id=det.frame_id, frame_idx=det.frame_idx,
                    ts=det.ts, role="counter_placement",
                    track_id=t.track_id,
                ))
                break
        # Handover candidate
        if "handover_candidate" in t.events and t.detections:
            mid = detections[t.detections[len(t.detections) // 2]]
            keyframes.append(Keyframe(
                frame_id=mid.frame_id, frame_idx=mid.frame_idx,
                ts=mid.ts, role="handover_candidate",
                track_id=t.track_id,
            ))
        # Receipt visible
        if t.receipt_candidate and t.detections:
            r = detections[t.detections[0]]
            keyframes.append(Keyframe(
                frame_id=r.frame_id, frame_idx=r.frame_idx,
                ts=r.ts, role="receipt_visible",
                track_id=t.track_id,
            ))
        # Final state
        last = detections[t.detections[-1]]
        keyframes.append(Keyframe(
            frame_id=last.frame_id, frame_idx=last.frame_idx,
            ts=last.ts, role="final_state", track_id=t.track_id,
        ))
    return keyframes
