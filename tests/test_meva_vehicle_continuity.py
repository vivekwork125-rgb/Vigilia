"""Regression tests for same-camera vehicle track continuity.

Tests the 10 mandated continuity edge cases:
1. same vehicle short gap
2. same vehicle longer gap
3. two nearby vehicles
4. crossing vehicles
5. competing candidate fragments
6. incompatible classes
7. inconsistent velocity
8. vehicle entering/leaving frame
9. stationary vehicle
10. ambiguous gap
"""

from meva.vehicle_continuity import (
    build_continuous_tracks,
    find_safe_merges,
    score_continuity,
)


def _make_track(
    track_id: str,
    start_t: float,
    end_t: float,
    box_start: list,
    box_end: list,
    object_type: str = "car",
    video_id: str = "VID-1",
    camera_id: str = "CAM-1",
    fps: float = 2.0,
):
    frames = round((end_t - start_t) * fps) + 1
    boxes = []
    for i in range(frames):
        t = start_t + i / fps
        frac = i / max(1, frames - 1)
        b = [
            box_start[0] + frac * (box_end[0] - box_start[0]),
            box_start[1] + frac * (box_end[1] - box_start[1]),
            box_start[2] + frac * (box_end[2] - box_start[2]),
            box_start[3] + frac * (box_end[3] - box_start[3]),
        ]
        boxes.append({"frame": int(t * 30), "t": round(t, 2), "box": b})
    return {
        "track_id": track_id,
        "object_type": object_type,
        "video_id": video_id,
        "camera_id": camera_id,
        "boxes": boxes,
        "observation_id": f"OBS-{track_id}",
    }


def test_1_same_vehicle_short_gap():
    """1. Same vehicle separated by a 1.0s gap with consistent motion merges safely."""
    trk_a = _make_track("A", 0.0, 4.0, [100, 100, 180, 160], [200, 100, 280, 160])
    trk_b = _make_track("B", 5.0, 9.0, [225, 100, 305, 160], [325, 100, 405, 160])
    cand = score_continuity(trk_a, trk_b, max_gap=2.0)
    assert cand is not None
    assert not cand.rejected
    assert cand.score >= 0.70

    merges = find_safe_merges([trk_a, trk_b], max_gap=2.0)
    assert len(merges) == 1
    assert merges[0].track_a_id == "A"
    assert merges[0].track_b_id == "B"

    continuous = build_continuous_tracks([trk_a, trk_b], merges)
    assert len(continuous) == 1
    assert continuous[0]["constituent_track_ids"] == ["A", "B"]


def test_2_same_vehicle_longer_gap():
    """2. Same vehicle separated by a 5.0s gap exceeds max_gap (3.0s) and is rejected."""
    trk_a = _make_track("A", 0.0, 4.0, [100, 100, 180, 160], [200, 100, 280, 160])
    trk_b = _make_track("B", 9.0, 13.0, [325, 100, 405, 160], [425, 100, 505, 160])
    cand = score_continuity(trk_a, trk_b, max_gap=3.0)
    assert cand is None

    merges = find_safe_merges([trk_a, trk_b], max_gap=3.0)
    assert len(merges) == 0


def test_3_two_nearby_vehicles():
    """3. Two distinct vehicles in different lanes are not merged across a gap."""
    trk_a = _make_track("A", 0.0, 4.0, [100, 100, 180, 160], [200, 100, 280, 160])
    # Lane 2 is 300px away vertically (5 box heights displacement)
    trk_other = _make_track("OTHER", 5.0, 9.0, [200, 400, 280, 460], [300, 400, 380, 460])
    cand = score_continuity(trk_a, trk_other, max_gap=2.0)
    # Either rejected or score below threshold
    assert cand is None or cand.rejected or cand.score < 0.60
    merges = find_safe_merges([trk_a, trk_other], max_gap=2.0)
    assert len(merges) == 0


def test_4_crossing_vehicles():
    """4. A third vehicle crossing the spatial gap between A and B causes merge rejection."""
    trk_a = _make_track("A", 0.0, 4.0, [100, 200, 180, 260], [200, 200, 280, 260])
    trk_b = _make_track("B", 6.0, 10.0, [250, 200, 330, 260], [350, 200, 430, 260])
    # Crossing vehicle occupies (220, 200) at t=5.0s (during the gap)
    trk_cross = _make_track("CROSS", 4.5, 5.5, [210, 150, 270, 210], [210, 250, 270, 310])

    merges = find_safe_merges([trk_a, trk_b, trk_cross], max_gap=3.0)
    assert len(merges) == 0


def test_5_competing_candidate_fragments():
    """5. Track A competing with two equally plausible successors B1 and B2 is rejected (abstention)."""
    trk_a = _make_track("A", 0.0, 4.0, [100, 100, 180, 160], [200, 100, 280, 160])
    trk_b1 = _make_track("B1", 5.0, 9.0, [225, 95, 305, 155], [325, 95, 405, 155])
    trk_b2 = _make_track("B2", 5.0, 9.0, [225, 105, 305, 165], [325, 105, 405, 165])

    merges = find_safe_merges([trk_a, trk_b1, trk_b2], max_gap=2.0, competition_margin=0.15)
    assert len(merges) == 0


def test_6_incompatible_classes():
    """6. Incompatible object classes (e.g. car vs person) are rejected."""
    trk_a = _make_track("A", 0.0, 4.0, [100, 100, 180, 160], [200, 100, 280, 160], object_type="car")
    trk_b = _make_track("B", 5.0, 9.0, [225, 100, 305, 160], [325, 100, 405, 160], object_type="person")

    cand = score_continuity(trk_a, trk_b, max_gap=2.0)
    assert cand is None


def test_7_inconsistent_velocity():
    """7. Opposing motion vectors (heading North vs heading South) are rejected."""
    # Track A moving North (vy negative)
    trk_a = _make_track("A", 0.0, 4.0, [100, 300, 180, 360], [100, 100, 180, 160])
    # Track B moving South (vy positive)
    trk_b = _make_track("B", 5.0, 9.0, [100, 110, 180, 170], [100, 310, 180, 370])

    cand = score_continuity(trk_a, trk_b, max_gap=2.0)
    assert cand is not None
    assert cand.rejected
    assert cand.rejection_reason == "direction_inconsistent"


def test_8_vehicle_entering_leaving_frame():
    """8. Vehicle disappearing at edge and reappearing across frame at impossible speed is rejected."""
    # Exiting left edge at x=10
    trk_a = _make_track("A", 0.0, 4.0, [150, 100, 230, 160], [10, 100, 90, 160])
    # Reappearing at right edge at x=1800 1s later (teleporting across 1700px in 1s = 28 box heights/s)
    trk_b = _make_track("B", 5.0, 9.0, [1800, 100, 1880, 160], [1700, 100, 1780, 160])

    cand = score_continuity(trk_a, trk_b, max_gap=2.0)
    assert cand is not None
    assert cand.rejected
    assert cand.rejection_reason == "implausible_gap_speed"


def test_9_stationary_vehicle():
    """9. Parked vehicle with small detection gap and slight jitter merges cleanly."""
    trk_a = _make_track("A", 0.0, 10.0, [200, 200, 280, 260], [202, 199, 282, 259])
    trk_b = _make_track("B", 11.5, 20.0, [201, 200, 281, 260], [200, 201, 280, 261])

    cand = score_continuity(trk_a, trk_b, max_gap=2.0)
    assert cand is not None
    assert not cand.rejected
    assert cand.score >= 0.70

    merges = find_safe_merges([trk_a, trk_b], max_gap=2.0)
    assert len(merges) == 1
    assert merges[0].track_a_id == "A"
    assert merges[0].track_b_id == "B"


def test_10_ambiguous_gap():
    """10. Track B having two predecessor candidates A1 and A2 with close scores abstains."""
    trk_a1 = _make_track("A1", 0.0, 4.0, [100, 95, 180, 155], [200, 95, 280, 155])
    trk_a2 = _make_track("A2", 0.0, 4.0, [100, 105, 180, 165], [200, 105, 280, 165])
    trk_b = _make_track("B", 5.0, 9.0, [225, 100, 305, 160], [325, 100, 405, 160])

    merges = find_safe_merges([trk_a1, trk_a2, trk_b], max_gap=2.0, competition_margin=0.15)
    assert len(merges) == 0
