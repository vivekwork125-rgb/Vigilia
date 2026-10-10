"""Evaluation-only deterministic object association from cached runtime detections.

A track is a sequence of compatible detector boxes, not a verified physical identity.
No MEVA annotations are imported or consulted in this module.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import pairwise
from math import hypot


def box_iou(a, b):
    ix = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0., a[2]-a[0]) * max(0., a[3]-a[1])
    area_b = max(0., b[2]-b[0]) * max(0., b[3]-b[1])
    return ix*iy/max(1., area_a+area_b-ix*iy)


def deduplicate_frame(detections, overlap=.6):
    """Suppress same-class duplicate boxes; preserve separate classes for audit."""
    kept = []
    for det in sorted(detections, key=lambda d: (-d["confidence"], d["detection_id"])):
        if not any(det["class"] == old["class"] and box_iou(det["box"], old["box"]) >= overlap
                   for old in kept):
            kept.append(det)
    return sorted(kept, key=lambda d: d["detection_id"])


def _compatibility(a, b, max_gap_s):
    if a["video_id"] != b["video_id"] or a["class"] != b["class"]:
        return None
    gap = b["t"] - a["t"]
    if gap <= 0 or gap > max_gap_s:
        return None
    aw, ah = a["box"][2]-a["box"][0], a["box"][3]-a["box"][1]
    bw, bh = b["box"][2]-b["box"][0], b["box"][3]-b["box"][1]
    if min(aw, bw) <= 0 or min(ah, bh) <= 0:
        return None
    if min(aw, bw)/max(aw, bw) < .5 or min(ah, bh)/max(ah, bh) < .5:
        return None
    ac = ((a["box"][0]+a["box"][2])/2, (a["box"][1]+a["box"][3])/2)
    bc = ((b["box"][0]+b["box"][2])/2, (b["box"][1]+b["box"][3])/2)
    norm_dist = hypot(bc[0]-ac[0], bc[1]-ac[1])/max(ah, bh, 1.)
    iou = box_iou(a["box"], b["box"])
    if norm_dist > .9 * max(1., gap/.5) or (iou < .05 and norm_dist > .5):
        return None
    return (1.-iou) + .4*norm_dist + .05*gap


def build_object_tracks(detections, *, max_gap_s=1.1, min_samples=2):
    """One-to-one, class-consistent association; never links across videos."""
    if max_gap_s <= 0 or min_samples < 1:
        raise ValueError("max_gap_s and min_samples must be positive")
    frames = defaultdict(list)
    for det in detections:
        if det["frame"] < 0 or det["t"] < 0 or len(det["box"]) != 4:
            raise ValueError("Invalid detection frame/time/box")
        frames[(det["video_id"], det["frame"])].append(det)
    tracks = []
    active = []
    for video_id, frame in sorted(frames):
        now = frames[(video_id, frame)][0]["t"]
        active = [track for track in active if track["video_id"] == video_id
                  and now - track["detections"][-1]["t"] <= max_gap_s]
        current = deduplicate_frame(frames[(video_id, frame)])
        edges = []
        for ti, track in enumerate(active):
            for di, det in enumerate(current):
                cost = _compatibility(track["detections"][-1], det, max_gap_s)
                if cost is not None:
                    edges.append((cost, track["track_id"], det["detection_id"], ti, di))
        used_tracks, used_detections = set(), set()
        # Close competing links are ambiguous: start a new provisional track
        # instead of asserting that two crossing objects kept their identities.
        ambiguous_detections = {di for cost, _tid, _did, ti, di in edges
                                if any(other_ti != ti and abs(other_cost-cost) < .12
                                       for other_cost, _otid, _odid, other_ti, other_di in edges
                                       if other_di == di)}
        for _cost, _tid, _did, ti, di in sorted(edges):
            if di in ambiguous_detections:
                continue
            if ti not in used_tracks and di not in used_detections:
                active[ti]["detections"].append(current[di])
                used_tracks.add(ti)
                used_detections.add(di)
        for di, det in enumerate(current):
            if di not in used_detections:
                track = {"track_id": f"OBT-{video_id}-{len(tracks)+1:06d}",
                         "video_id": video_id, "class": det["class"],
                         "association_method": "greedy one-to-one class/box/temporal",
                         "ambiguous_predecessor": di in ambiguous_detections,
                         "detections": [det]}
                tracks.append(track)
                active.append(track)
    for track in tracks:
        hits = track["detections"]
        track["sample_count"] = len(hits)
        track["persistent"] = len(hits) >= min_samples
        track["start_seconds"] = hits[0]["t"]
        track["end_seconds"] = hits[-1]["t"]
        track["duration_seconds"] = hits[-1]["t"]-hits[0]["t"]
        track["gap_count"] = sum(b["t"]-a["t"] > .75 for a,b in pairwise(hits))
        track["source_sha256"] = hits[0]["source_sha256"]
        track["camera_id"] = hits[0]["camera_id"]
    return tracks
