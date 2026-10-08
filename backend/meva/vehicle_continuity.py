"""Evaluation-quality and inference layer for same-camera vehicle track continuity.

Finds candidate track pairs separated by short detection gaps using ONLY
runtime-derived evidence (same camera, temporal proximity, class compatibility,
spatial continuity, velocity continuity, box geometry). Never uses ground-truth
annotations to form candidate merges. Preserves original observation provenance.
"""

import math
from dataclasses import dataclass, field
from math import hypot
from typing import Any

VEHICLES = {"car", "truck", "bus", "motorcycle"}


@dataclass
class ContinuityCandidate:
    track_a_id: str
    track_b_id: str
    camera_id: str
    video_id: str
    gap_seconds: float
    score: float
    features: dict[str, Any] = field(default_factory=dict)
    rejected: bool = False
    rejection_reason: str | None = None


def center(box: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def box_dimensions(box: list[float]) -> tuple[float, float, float]:
    x1, y1, x2, y2 = box
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    return w, h, w / h


def motion_vector(samples: list[dict[str, Any]], tail: bool = True, max_span: float = 1.5) -> tuple[float, float, float]:
    """Estimate velocity (vx, vy in px/s) and normalized speed (box_heights/s)."""
    if len(samples) < 2:
        return 0.0, 0.0, 0.0
    window = samples[-3:] if tail else samples[:3]
    first, last = window[0], window[-1]
    dt = max(0.01, last["t"] - first["t"])
    if dt > max_span:
        # If samples are too far apart, take adjacent pair
        first, last = (samples[-2], samples[-1]) if tail else (samples[0], samples[1])
        dt = max(0.01, last["t"] - first["t"])
    c1, c2 = center(first["box"]), center(last["box"])
    vx = (c2[0] - c1[0]) / dt
    vy = (c2[1] - c1[1]) / dt
    scale = max(1.0, (first["box"][3] - first["box"][1] + last["box"][3] - last["box"][1]) / 2.0)
    speed_norm = hypot(vx, vy) / scale
    return vx, vy, speed_norm


def score_continuity(
    track_a: dict[str, Any],
    track_b: dict[str, Any],
    max_gap: float = 3.0,
) -> ContinuityCandidate | None:
    """Deterministic, interpretable scoring for candidate same-camera track continuity.

    Requires:
    - Same camera and video
    - Non-overlapping temporal order: 0 < track_b.start - track_a.end <= max_gap
    - Compatible vehicle classes
    - Plausible geometry (width/height ratio >= 0.5)
    - Plausible displacement and motion kinematics
    """
    if track_a.get("video_id") != track_b.get("video_id"):
        return None
    if track_a.get("camera_id") != track_b.get("camera_id"):
        return None

    type_a = track_a.get("object_type")
    type_b = track_b.get("object_type")
    if type_a not in VEHICLES or type_b not in VEHICLES:
        return None

    samples_a = track_a["boxes"]
    samples_b = track_b["boxes"]
    if not samples_a or not samples_b:
        return None

    last_a = samples_a[-1]
    first_b = samples_b[0]
    gap = first_b["t"] - last_a["t"]
    if gap <= 0.0 or gap > max_gap:
        return None

    box_a = last_a["box"]
    box_b = first_b["box"]
    w_a, h_a, ar_a = box_dimensions(box_a)
    w_b, h_b, ar_b = box_dimensions(box_b)

    # Geometry continuity: width ratio, height ratio, aspect ratio
    w_ratio = min(w_a, w_b) / max(w_a, w_b)
    h_ratio = min(h_a, h_b) / max(h_a, h_b)
    ar_ratio = min(ar_a, ar_b) / max(ar_a, ar_b)

    scale = max(1.0, (h_a + h_b) / 2.0)
    ca = center(box_a)
    cb = center(box_b)
    dx = cb[0] - ca[0]
    dy = cb[1] - ca[1]
    displacement = hypot(dx, dy)
    norm_displacement = displacement / scale
    gap_speed = norm_displacement / max(0.01, gap)

    # Velocity before and after gap
    vx_a, vy_a, speed_a = motion_vector(samples_a, tail=True)
    vx_b, vy_b, speed_b = motion_vector(samples_b, tail=False)

    # Predicted positions
    pred_cb_x = ca[0] + vx_a * gap
    pred_cb_y = ca[1] + vy_a * gap
    pred_err_a = hypot(cb[0] - pred_cb_x, cb[1] - pred_cb_y) / scale

    pred_ca_x = cb[0] - vx_b * gap
    pred_ca_y = cb[1] - vy_b * gap
    pred_err_b = hypot(pred_ca_x - ca[0], pred_ca_y - ca[1]) / scale

    # Direction consistency
    dir_cos = 1.0
    v_mag_a = hypot(vx_a, vy_a)
    v_mag_b = hypot(vx_b, vy_b)
    if v_mag_a > 2.0 and v_mag_b > 2.0:
        dir_cos = (vx_a * vx_b + vy_a * vy_b) / (v_mag_a * v_mag_b)

    features = {
        "gap_seconds": round(gap, 4),
        "width_ratio": round(w_ratio, 4),
        "height_ratio": round(h_ratio, 4),
        "aspect_ratio_ratio": round(ar_ratio, 4),
        "norm_displacement": round(norm_displacement, 4),
        "gap_speed_norm": round(gap_speed, 4),
        "speed_a_norm": round(speed_a, 4),
        "speed_b_norm": round(speed_b, 4),
        "pred_err_a": round(pred_err_a, 4),
        "pred_err_b": round(pred_err_b, 4),
        "direction_cosine": round(dir_cos, 4),
        "class_match": (type_a == type_b),
    }

    # Safety rejection gates (A6)
    if w_ratio < 0.5 or h_ratio < 0.5 or ar_ratio < 0.5:
        return ContinuityCandidate(
            track_a_id=track_a["track_id"],
            track_b_id=track_b["track_id"],
            camera_id=track_a["camera_id"],
            video_id=track_a["video_id"],
            gap_seconds=gap,
            score=0.0,
            features=features,
            rejected=True,
            rejection_reason="geometry_abrupt_change",
        )

    # Maximum plausible physical speed (2.5 box heights/second)
    if gap_speed > 2.5:
        return ContinuityCandidate(
            track_a_id=track_a["track_id"],
            track_b_id=track_b["track_id"],
            camera_id=track_a["camera_id"],
            video_id=track_a["video_id"],
            gap_seconds=gap,
            score=0.0,
            features=features,
            rejected=True,
            rejection_reason="implausible_gap_speed",
        )

    # Incompatible velocity direction when both are moving
    if v_mag_a > 4.0 and v_mag_b > 4.0 and dir_cos < -0.3:
        return ContinuityCandidate(
            track_a_id=track_a["track_id"],
            track_b_id=track_b["track_id"],
            camera_id=track_a["camera_id"],
            video_id=track_a["video_id"],
            gap_seconds=gap,
            score=0.0,
            features=features,
            rejected=True,
            rejection_reason="direction_inconsistent",
        )

    # Class compatibility: identical classes preferred, cross-vehicle penalized
    class_factor = 1.0 if type_a == type_b else 0.75

    # Geometry score
    geom_score = (w_ratio + h_ratio + ar_ratio) / 3.0

    # Motion / spatial fit:
    # If vehicle was moving, use min prediction error; if stationary, use normalized displacement
    best_spatial_err = min(norm_displacement, pred_err_a, pred_err_b)
    spatial_score = math.exp(-best_spatial_err)

    # Direction score
    dir_score = max(0.0, (dir_cos + 1.0) / 2.0) if (v_mag_a > 2.0 and v_mag_b > 2.0) else 1.0

    # Temporal score (shorter gap has higher continuity prior)
    temporal_score = max(0.0, 1.0 - (gap / max_gap) * 0.5)

    composite_score = (
        0.35 * spatial_score
        + 0.25 * geom_score
        + 0.20 * temporal_score
        + 0.10 * dir_score
        + 0.10 * class_factor
    )

    return ContinuityCandidate(
        track_a_id=track_a["track_id"],
        track_b_id=track_b["track_id"],
        camera_id=track_a["camera_id"],
        video_id=track_a["video_id"],
        gap_seconds=gap,
        score=round(composite_score, 4),
        features=features,
        rejected=False,
    )


def find_safe_merges(
    tracks: list[dict[str, Any]],
    max_gap: float = 2.0,
    min_score: float = 0.65,
    competition_margin: float = 0.15,
) -> list[ContinuityCandidate]:
    """Find reciprocal, non-competing, safe track merges within the same camera."""
    tracks_by_video = {}
    for t in tracks:
        tracks_by_video.setdefault(t["video_id"], []).append(t)

    all_accepted = []

    for vtracks in tracks_by_video.values():
        # Sort by start time
        vtracks = sorted(vtracks, key=lambda x: x["boxes"][0]["t"])

        candidates_from_a = {}  # track_a_id -> list of valid candidates
        candidates_to_b = {}    # track_b_id -> list of valid candidates

        for i, trk_a in enumerate(vtracks):
            for j in range(i + 1, len(vtracks)):
                trk_b = vtracks[j]
                gap = trk_b["boxes"][0]["t"] - trk_a["boxes"][-1]["t"]
                if gap > max_gap:
                    # Because tracks are sorted by start time, we can break if trk_b.start is far beyond trk_a.end
                    # Note: a later track might start earlier, but start is sorted, so if gap > max_gap and trk_b.start > trk_a.end + max_gap:
                    if trk_b["boxes"][0]["t"] > trk_a["boxes"][-1]["t"] + max_gap:
                        break
                    continue
                if gap <= 0:
                    continue

                cand = score_continuity(trk_a, trk_b, max_gap=max_gap)
                if cand is not None and not cand.rejected and cand.score >= min_score:
                    candidates_from_a.setdefault(trk_a["track_id"], []).append(cand)
                    candidates_to_b.setdefault(trk_b["track_id"], []).append(cand)

        # Ambiguity / Competition resolution:
        # A must have a unique clear best B, and B must have a unique clear best A
        best_for_a = {}
        for aid, cands in candidates_from_a.items():
            cands.sort(key=lambda c: c.score, reverse=True)
            if len(cands) == 1 or (cands[0].score - cands[1].score >= competition_margin):
                best_for_a[aid] = cands[0]

        best_for_b = {}
        for bid, cands in candidates_to_b.items():
            cands.sort(key=lambda c: c.score, reverse=True)
            if len(cands) == 1 or (cands[0].score - cands[1].score >= competition_margin):
                best_for_b[bid] = cands[0]

        # Reciprocal agreement: best_for_a[a].track_b_id == b and best_for_b[b].track_a_id == a
        for aid, cand in best_for_a.items():
            bid = cand.track_b_id
            if bid in best_for_b and best_for_b[bid].track_a_id == aid:
                # Check for crossing track active during gap in same region
                box_a = next(t for t in vtracks if t["track_id"] == aid)["boxes"][-1]["box"]
                box_b = next(t for t in vtracks if t["track_id"] == bid)["boxes"][0]["box"]
                t_start = next(t for t in vtracks if t["track_id"] == aid)["boxes"][-1]["t"]
                t_end = next(t for t in vtracks if t["track_id"] == bid)["boxes"][0]["t"]
                ca = center(box_a)
                cb = center(box_b)
                mid_x, mid_y = (ca[0] + cb[0]) / 2.0, (ca[1] + cb[1]) / 2.0
                radius = max(hypot(cb[0] - ca[0], cb[1] - ca[1]) / 2.0, 50.0)

                has_crossing = False
                for other in vtracks:
                    if other["track_id"] in (aid, bid):
                        continue
                    # Check if other has any samples between t_start and t_end inside the gap corridor
                    for s in other["boxes"]:
                        if t_start <= s["t"] <= t_end:
                            sc = center(s["box"])
                            if hypot(sc[0] - mid_x, sc[1] - mid_y) <= radius:
                                has_crossing = True
                                break
                    if has_crossing:
                        break

                if not has_crossing:
                    all_accepted.append(cand)

    return all_accepted


def build_continuous_tracks(
    tracks: list[dict[str, Any]],
    merges: list[ContinuityCandidate],
) -> list[dict[str, Any]]:
    """Chain accepted merges into continuous tracks while retaining full provenance."""
    # Build forward and backward mapping
    next_track = {}
    prev_track = {}
    merge_info = {}
    for m in merges:
        next_track[m.track_a_id] = m.track_b_id
        prev_track[m.track_b_id] = m.track_a_id
        merge_info[(m.track_a_id, m.track_b_id)] = m

    track_by_id = {t["track_id"]: t for t in tracks}
    visited = set()
    continuous = []

    for t in tracks:
        tid = t["track_id"]
        if tid in visited:
            continue
        # Find root of chain
        curr = tid
        while curr in prev_track:
            curr = prev_track[curr]

        # Traverse chain
        chain_ids = [curr]
        chain_merges = []
        visited.add(curr)
        while curr in next_track:
            nxt = next_track[curr]
            chain_ids.append(nxt)
            chain_merges.append(merge_info[(curr, nxt)])
            visited.add(nxt)
            curr = nxt

        if len(chain_ids) == 1:
            continuous.append(t)
        else:
            # Combine samples from all tracks in the chain
            combined_boxes = []
            obs_ids = []
            for cid in chain_ids:
                trk = track_by_id[cid]
                combined_boxes.extend(trk["boxes"])
                if trk.get("observation_id"):
                    obs_ids.append(trk["observation_id"])

            # Deduplicate or sort by time
            combined_boxes.sort(key=lambda s: s["t"])

            # Preserved provenance
            first_trk = track_by_id[chain_ids[0]]
            merged_track = {
                "track_id": f"CONT-{'+'.join(chain_ids)}",
                "object_type": first_trk["object_type"],
                "video_id": first_trk["video_id"],
                "camera_id": first_trk["camera_id"],
                "boxes": combined_boxes,
                "is_continuous": True,
                "constituent_track_ids": chain_ids,
                "constituent_observation_ids": obs_ids,
                "continuity_merges": [
                    {
                        "track_a": m.track_a_id,
                        "track_b": m.track_b_id,
                        "gap_seconds": m.gap_seconds,
                        "score": m.score,
                        "features": m.features,
                    }
                    for m in chain_merges
                ],
            }
            continuous.append(merged_track)

    return continuous
