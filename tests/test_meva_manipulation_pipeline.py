"""Regression tests for selective small-object manipulation pipeline.

Tests:
1. Object tracklet formation with stability and class consistency.
2. Motion coupling calculation between person and object.
3. Pickup event requiring initial stability, approach, and coupled motion.
4. Placement event requiring initial coupling, separation, and post-placement stability.
5. Abstention on unsupported classes (e.g. cup, cell phone).
6. Ambiguous person association abstention.
"""

from app.temporal_events import PORTABLE
from meva.selective_manipulation_pipeline import (
    ObjectTracklet,
    analyze_motion_coupling,
    build_tracklets,
    evaluate_pickup,
    evaluate_placement,
)
from meva.selective_objects import associate


def _make_det(frame: int, x: float, y: float, w: float = 20, h: float = 20, cls: str = "backpack", conf: float = 0.8):
    return {
        "class": cls,
        "confidence": conf,
        "box": [x, y, x + w, y + h],
        "frame": frame,
        "t": frame / 30.0,
    }


def _make_person(frame: int, x: float, y: float, w: float = 40, h: float = 100, track_id: str = "P1"):
    return {
        "track_id": track_id,
        "box": [x, y, x + w, y + h],
        "frame": frame,
        "t": frame / 30.0,
    }


def test_1_tracklet_formation_and_stability():
    """Tracklet requires multiple class-consistent detections within gap tolerance."""
    # 2 detections at 15-frame interval (0.5s gap) -> stable tracklet
    dets_by_frame = {
        0: [_make_det(0, 100, 100)],
        15: [_make_det(15, 102, 100)],
    }
    tracklets = build_tracklets(dets_by_frame, fps=30.0, max_gap=1.0)
    assert len(tracklets) == 1
    assert tracklets[0].is_stable
    assert tracklets[0].sample_count == 2
    assert tracklets[0].is_portable

    # Single isolated detection -> unstable tracklet
    isolated = {0: [_make_det(0, 100, 100)]}
    t_iso = build_tracklets(isolated, fps=30.0, max_gap=1.0)
    assert len(t_iso) == 1
    assert not t_iso[0].is_stable

    # Changed class across frames -> splits into 2 tracklets
    changed = {
        0: [_make_det(0, 100, 100, cls="backpack")],
        15: [_make_det(15, 102, 100, cls="cup")],
    }
    t_ch = build_tracklets(changed, fps=30.0, max_gap=1.0)
    assert len(t_ch) == 2
    assert not t_ch[0].is_stable
    assert not t_ch[1].is_stable


def test_2_motion_coupling_calculation():
    """Motion coupling measures vector alignment, proximity, and distance change."""
    # Person and backpack moving together horizontally
    dets = [
        _make_det(0, 100, 100),
        _make_det(15, 120, 100),
        _make_det(30, 140, 100),
    ]
    tracklet = ObjectTracklet(
        tracklet_id="OBJ-01",
        object_class="backpack",
        detections=dets,
        mean_confidence=0.8,
        start_time=0.0,
        end_time=1.0,
        sample_count=3,
        is_stable=True,
        is_portable=True,
    )
    person = [
        _make_person(0, 90, 80),
        _make_person(15, 110, 80),
        _make_person(30, 130, 80),
    ]

    coupling = analyze_motion_coupling(tracklet, person)
    assert coupling["paired"]
    assert coupling["is_coupled"]
    assert coupling["mean_distance"] < 1.0


def test_3_pickup_requires_initial_stillness_and_coupling():
    """Pickup requires object initially stationary, person approach, and transition to coupling."""
    # Initially stationary object at (100, 100), person approaches from (50, 50) to (90, 80)
    # Then object moves with person
    dets = [
        _make_det(0, 100, 100),      # stationary
        _make_det(15, 100, 100),     # stationary
        _make_det(30, 115, 100),     # coupled movement
        _make_det(45, 130, 100),     # coupled movement
    ]
    tracklet = ObjectTracklet(
        tracklet_id="OBJ-01",
        object_class="backpack",
        detections=dets,
        mean_confidence=0.8,
        start_time=0.0,
        end_time=1.5,
        sample_count=4,
        is_stable=True,
        is_portable=True,
    )
    person = [
        _make_person(0, 20, 50),     # distant
        _make_person(15, 90, 80),    # approaches
        _make_person(30, 105, 80),   # moving together
        _make_person(45, 120, 80),   # moving together
    ]

    success, reason, _details = evaluate_pickup(tracklet, "P1", person)
    assert success
    assert reason == "pickup_verified"

    # Negative test: Object already moving fast before approach -> fails initial stillness
    fast_dets = [
        _make_det(0, 100, 100),
        _make_det(15, 180, 100),     # moving fast
        _make_det(30, 200, 100),
    ]
    t_fast = ObjectTracklet(
        tracklet_id="OBJ-02",
        object_class="backpack",
        detections=fast_dets,
        mean_confidence=0.8,
        start_time=0.0,
        end_time=1.0,
        sample_count=3,
        is_stable=True,
        is_portable=True,
    )
    fail_ok, fail_reason, _ = evaluate_pickup(t_fast, "P1", person[:3])
    assert not fail_ok
    assert fail_reason == "object_not_initially_stationary"


def test_4_placement_requires_coupling_separation_and_stillness():
    """Placement requires coupled motion, followed by person separation and object remaining stationary."""
    # First 2 frames coupled movement, then person walks away while object stays at (120, 100)
    dets = [
        _make_det(0, 100, 100),
        _make_det(15, 120, 100),
        _make_det(30, 120, 100),     # stationary
        _make_det(45, 120, 100),     # stationary
    ]
    tracklet = ObjectTracklet(
        tracklet_id="OBJ-01",
        object_class="backpack",
        detections=dets,
        mean_confidence=0.8,
        start_time=0.0,
        end_time=1.5,
        sample_count=4,
        is_stable=True,
        is_portable=True,
    )
    person = [
        _make_person(0, 90, 80),     # close
        _make_person(15, 110, 80),   # close
        _make_person(30, 250, 80),   # separates
        _make_person(45, 380, 80),   # separates further
    ]

    success, reason, _details = evaluate_placement(tracklet, "P1", person)
    assert success
    assert reason == "placement_verified"

    # Negative test: Person never separates -> fails
    person_stays = [
        _make_person(0, 90, 80),
        _make_person(15, 110, 80),
        _make_person(30, 110, 80),
        _make_person(45, 110, 80),
    ]
    fail_ok, fail_reason, _ = evaluate_placement(tracklet, "P1", person_stays)
    assert not fail_ok
    assert fail_reason == "no_separation_evidence"


def test_5_unsupported_class_is_flagged():
    """Cup or cell phone detections are not in production PORTABLE class set."""
    tracklet_cup = ObjectTracklet(
        tracklet_id="OBJ-CUP",
        object_class="cup",
        detections=[_make_det(0, 100, 100, cls="cup"), _make_det(15, 100, 100, cls="cup")],
        mean_confidence=0.7,
        start_time=0.0,
        end_time=0.5,
        sample_count=2,
        is_stable=True,
        is_portable=False,
    )
    assert not tracklet_cup.is_portable
    assert tracklet_cup.object_class not in PORTABLE


def test_6_ambiguous_two_person_proximity_abstains():
    """When two people are equidistant to the object, association abstains."""
    dets = [_make_det(0, 100, 100), _make_det(15, 100, 100)]
    persons_by_frame = {
        0: [
            _make_person(0, 95, 80, track_id="P1"),
            _make_person(0, 105, 80, track_id="P2"),
        ],
        15: [
            _make_person(15, 95, 80, track_id="P1"),
            _make_person(15, 105, 80, track_id="P2"),
        ],
    }
    owner = associate(dets, persons_by_frame)
    # Because P1 and P2 are within 0.25 margin of normalized distance, associate returns None
    assert owner is None
