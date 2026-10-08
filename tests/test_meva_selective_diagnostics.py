"""Regression checks for abstention and source-derived diagnostic geometry."""

from meva.selective_objects import (
    associate,
    context_frames,
    deduplicate,
    link_objects,
    person_roi,
    relationship_metrics,
)
from meva.vehicle_rule_experiment import offline_transitions
from meva.vehicle_state_analysis import state_trace


def vehicle(points):
    return [
        {"t": i * 0.5, "frame": i * 15, "box": [x, 20, x + 30, 40]}
        for i, x in enumerate(points)
    ]


def test_offline_vehicle_rules_reject_parked_jitter_and_one_sample_shift():
    parked = vehicle([100, 100.5, 100, 100.4, 100, 100.5, 100, 100.3, 100])
    for method in ("velocity_transition", "change_point", "multi_window"):
        assert offline_transitions("car", parked, method) == []
    assert state_trace(parked, 0.7)[0]["confidence_source"].startswith("track_mean")


def test_source_derived_roi_scales_and_clips():
    person = [5, 10, 25, 50]
    small = person_roi(person, 100, 100, "expanded", 1.25)
    large = person_roi(person, 100, 100, "expanded", 2.0)
    assert small[0] >= 0 and large[0] == 0
    assert (large[2] - large[0]) > (small[2] - small[0])
    assert person_roi(person, 100, 100, "lower", 1.5)[1] > small[1]
    assert context_frames([0, 30], 61, 30, 1.0) == {0, 15, 30, 45, 60}


def detection(frame, x, label="backpack"):
    return {
        "class": label, "confidence": 0.8, "box": [x, 20, x + 12, 32],
        "frame": frame, "t": frame / 30,
    }


def test_object_tracklet_requires_repeat_detection_and_class_consistency():
    tracks = link_objects({0: [detection(0, 10)], 15: [detection(15, 11)]})
    assert len(tracks) == 1 and len(tracks[0]) == 2
    isolated = link_objects({0: [detection(0, 10)], 45: [detection(45, 11)]})
    assert all(len(track) == 1 for track in isolated)
    occluded = link_objects({0: [detection(0, 10)], 15: [], 30: [detection(30, 11)]})
    assert len(occluded) == 1 and len(occluded[0]) == 2
    changed = link_objects({0: [detection(0, 10)], 15: [detection(15, 11, "cup")]})
    assert len(changed) == 2


def test_duplicate_crops_and_ambiguous_person_association_abstain():
    assert len(deduplicate([detection(0, 10), detection(0, 11)])) == 1
    tracklet = [detection(0, 10), detection(15, 11)]
    one = {
        0: [{"track_id": "p", "box": [0, 0, 40, 60]}],
        15: [{"track_id": "p", "box": [0, 0, 40, 60]}],
    }
    assert associate(tracklet, one) == "p"
    metrics = relationship_metrics(tracklet, one)["p"]
    assert metrics["persistence_samples"] == 2
    assert metrics["median_relative_speed"] is not None
    both = {frame: people + [{"track_id": "q", "box": [0, 0, 40, 60]}] for frame, people in one.items()}
    assert associate(tracklet, both) is None
    assert associate(tracklet[:1], one) is None
