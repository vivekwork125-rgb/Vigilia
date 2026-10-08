"""Evaluation tooling must keep exact sample frames and class-aware coverage."""

from meva.perception_experiment import longest_run, summarize


def test_longest_run_does_not_bridge_a_missed_sample():
    assert longest_run([0, 15, 45, 60], [0, 15, 30, 45, 60]) == 2


def test_generic_other_overlap_does_not_count_as_event_supported_object():
    actor = {
        "activity_type": "person_puts_down_object",
        "annotation_actor_type": "other",
        "sample_frames": [0],
        "geometry_sample_count": 1,
        "hits": [{"frame": 0, "class": "laptop", "confidence": 0.35}],
        "longest_detection_run": 1,
        "median_box_area_fraction": 0.001,
    }
    stats = summarize([actor], 1, 1, 0, 1.0)
    assert stats["activity"]["person_puts_down_object"]["covered"] == 1
    assert stats["activity"]["person_puts_down_object"]["event_supported_object_covered"] == 0
