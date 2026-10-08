"""Evaluation-quality selective small-object manipulation pipeline.

Implements the complete chain:
  object detection -> object tracklet -> person-object association ->
  temporal interaction (pickup / placement) -> evidence grounding.

Consults ground-truth annotations ONLY after inference and event extraction
for evaluation. Never creates runtime ROIs or events from annotations.
"""

import statistics
from dataclasses import dataclass, field
from math import hypot
from typing import Any

from app.temporal_events import PORTABLE

SKIP_CLASSES = {"person", "car", "truck", "bus", "motorcycle", "bicycle"}


@dataclass
class ObjectTracklet:
    tracklet_id: str
    object_class: str
    detections: list[dict[str, Any]]
    mean_confidence: float
    start_time: float
    end_time: float
    sample_count: int
    is_stable: bool
    is_portable: bool


@dataclass
class ManipulationHypothesis:
    kind: str  # "picked_up" or "placed_object"
    object_tracklet_id: str
    object_class: str
    person_track_id: str
    start_time: float
    end_time: float
    confidence: float
    details: dict[str, Any] = field(default_factory=dict)
    verified: bool = False
    failure_stage: str | None = None


def center(box: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def build_tracklets(detections_by_frame: dict[int, list[dict[str, Any]]], fps: float = 30.0, max_gap: float = 1.0) -> list[ObjectTracklet]:
    """Form temporal object tracklets with limited gaps and class consistency."""
    active: list[list[dict[str, Any]]] = []
    tracklets: list[list[dict[str, Any]]] = []

    for frame in sorted(detections_by_frame):
        t_current = frame / fps
        for det in detections_by_frame[frame]:
            candidates = []
            for trk in active:
                prev = trk[-1]
                gap = det["t"] - prev["t"]
                if det["class"] != prev["class"] or gap > max_gap:
                    continue
                a, b = prev["box"], det["box"]
                ca = center(a)
                cb = center(b)
                scale = max(1.0, max(a[3] - a[1], b[3] - b[1]))
                dist = hypot(cb[0] - ca[0], cb[1] - ca[1]) / scale
                if dist <= 1.0:
                    candidates.append((dist, trk))

            if candidates:
                _, chosen = min(candidates, key=lambda item: item[0])
                if chosen[-1]["frame"] != frame:
                    chosen.append(det)
                    continue

            new_trk = [det]
            tracklets.append(new_trk)
            active.append(new_trk)

        # Prune inactive tracks
        active = [trk for trk in active if t_current - trk[-1]["t"] <= max_gap]

    results = []
    for idx, hits in enumerate(tracklets):
        confs = [h["confidence"] for h in hits]
        mean_conf = float(statistics.mean(confs)) if confs else 0.0
        is_stable = len(hits) >= 2
        is_port = hits[0]["class"] in PORTABLE
        results.append(ObjectTracklet(
            tracklet_id=f"OBJ-{idx+1:04d}",
            object_class=hits[0]["class"],
            detections=hits,
            mean_confidence=round(mean_conf, 4),
            start_time=hits[0]["t"],
            end_time=hits[-1]["t"],
            sample_count=len(hits),
            is_stable=is_stable,
            is_portable=is_port,
        ))
    return results


def analyze_motion_coupling(
    tracklet: ObjectTracklet,
    person_samples: list[dict[str, Any]],
    tolerance: float = 0.3,
) -> dict[str, Any]:
    """Compute detailed kinematic coupling between an object tracklet and a person track."""
    # Align samples in time
    pairs = []
    for hit in tracklet.detections:
        matching_person = min(
            person_samples,
            key=lambda p: abs(p["t"] - hit["t"]),
            default=None,
        )
        if matching_person and abs(matching_person["t"] - hit["t"]) <= tolerance:
            scale = max(1.0, matching_person["box"][3] - matching_person["box"][1])
            ox, oy = center(hit["box"])
            px, py = center(matching_person["box"])
            dist = hypot(ox - px, oy - py) / scale
            pairs.append((hit, matching_person, dist, scale))

    if len(pairs) < 2:
        return {"paired": False, "reason": "insufficient_aligned_samples"}

    distances = [p[2] for p in pairs]
    residuals = []
    cosines = []
    relative_speeds = []
    obj_speeds = []

    for i in range(len(pairs) - 1):
        h1, p1, _d1, s1 = pairs[i]
        h2, p2, _d2, _s2 = pairs[i + 1]
        dt = max(0.01, h2["t"] - h1["t"])
        if dt > 1.5:
            continue

        c_o1, c_o2 = center(h1["box"]), center(h2["box"])
        c_p1, c_p2 = center(p1["box"]), center(p2["box"])

        ov = (c_o2[0] - c_o1[0], c_o2[1] - c_o1[1])
        pv = (c_p2[0] - c_p1[0], c_p2[1] - c_p1[1])

        on = hypot(*ov)
        pn = hypot(*pv)

        obj_speed_norm = on / s1 / dt
        obj_speeds.append(obj_speed_norm)

        diff = hypot(ov[0] - pv[0], ov[1] - pv[1])
        residual = diff / max(on, pn, 1.0)
        residuals.append(residual)

        cos = (ov[0] * pv[0] + ov[1] * pv[1]) / max(on * pn, 1e-9)
        cosines.append(cos)

        rel_speed = diff / s1 / dt
        relative_speeds.append(rel_speed)

    approach = max(0.0, distances[0] - distances[-1])
    separation = max(0.0, distances[-1] - distances[0])

    # Coupled motion requires:
    # 1. Close proximity: mean distance <= 1.5 person heights
    # 2. Vector agreement: cosine >= 0.7 and residual <= 0.4
    is_coupled = (
        len(residuals) >= 1
        and statistics.mean(distances) <= 1.5
        and any(c >= 0.7 for c in cosines)
        and min(residuals) <= 0.4
    )

    return {
        "paired": True,
        "sample_count": len(pairs),
        "mean_distance": round(statistics.mean(distances), 4),
        "min_distance": round(min(distances), 4),
        "distances": [round(d, 4) for d in distances],
        "approach": round(approach, 4),
        "separation": round(separation, 4),
        "mean_residual": round(statistics.mean(residuals), 4) if residuals else None,
        "mean_cosine": round(statistics.mean(cosines), 4) if cosines else None,
        "obj_speeds": [round(s, 4) for s in obj_speeds],
        "is_coupled": is_coupled,
    }


def evaluate_pickup(
    tracklet: ObjectTracklet,
    person_track_id: str,
    person_samples: list[dict[str, Any]],
) -> tuple[bool, str, dict[str, Any]]:
    """Evaluate whether evidence supports a physical pickup event."""
    if not tracklet.is_stable:
        return False, "tracklet_not_stable", {}

    coupling = analyze_motion_coupling(tracklet, person_samples)
    if not coupling.get("paired"):
        return False, coupling.get("reason", "not_paired"), coupling

    # Stage 1: Object exists & initially stable/stationary
    # Initial object speed should be low (still or low movement)
    obj_speeds = coupling.get("obj_speeds", [])
    if obj_speeds and obj_speeds[0] > 0.4:
        return False, "object_not_initially_stationary", coupling

    # Stage 2: Person approaches
    # Distance should decrease or start very close (<= 1.5)
    if coupling["mean_distance"] > 1.5 and coupling["approach"] < 0.2:
        return False, "no_approach_evidence", coupling

    # Stage 3: Coupling established
    if not coupling["is_coupled"]:
        return False, "no_coupled_motion", coupling

    return True, "pickup_verified", coupling


def evaluate_placement(
    tracklet: ObjectTracklet,
    person_track_id: str,
    person_samples: list[dict[str, Any]],
) -> tuple[bool, str, dict[str, Any]]:
    """Evaluate whether evidence supports a physical placement event."""
    if not tracklet.is_stable:
        return False, "tracklet_not_stable", {}

    coupling = analyze_motion_coupling(tracklet, person_samples)
    if not coupling.get("paired"):
        return False, coupling.get("reason", "not_paired"), coupling

    # Stage 1: Initially close and coupled
    if coupling["distances"][0] > 1.5:
        return False, "not_initially_close", coupling

    # Stage 2: Person separates
    if coupling["separation"] < 0.3 and coupling["distances"][-1] < 1.8:
        return False, "no_separation_evidence", coupling

    # Stage 3: Object becomes stationary
    obj_speeds = coupling.get("obj_speeds", [])
    if obj_speeds and obj_speeds[-1] > 0.25:
        return False, "object_not_stationary_after_placement", coupling

    return True, "placement_verified", coupling
