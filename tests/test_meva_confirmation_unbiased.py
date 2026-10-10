"""Regression tests for full-video confirmation evaluation semantics."""
from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/experiment_meva_confirmation_unbiased.py"
spec = importlib.util.spec_from_file_location("experiment_meva_confirmation_unbiased", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _prediction(start=10., end=12., box=(0, 0, 20, 20)):
    return {
        "candidate_id": "c", "video_id": "v", "event_type": "pickup",
        "start_seconds": start, "end_seconds": end,
        "person_boxes": [{"frame": 300, "box": box}],
        "features": {"localized_motion": .5, "relative_motion_coupling": .7,
                     "appearance_change": .4, "region_persistence": .5,
                     "separation_evidence": .6},
    }


def _truth(start=10., end=12.):
    return {"key": "v:1:0", "video_id": "v",
            "activity_type": "person_picks_up_object", "start_seconds": start,
            "end_seconds": end, "under_40px": True}


def test_matching_requires_actor_box_and_correct_type():
    truth = _truth()
    geometry = {truth["key"]: {300: (0, 0, 20, 20)}}
    assert module.eligible(_prediction(), truth, geometry)
    assert not module.eligible(_prediction(box=(50, 50, 70, 70)), truth, geometry)
    assert not module.eligible({**_prediction(), "event_type": "placement"}, truth, geometry)
    assert not module.eligible(_prediction(start=20., end=22.), truth, geometry)


def test_one_to_one_matching_prevents_duplicate_credit():
    truth = _truth()
    geometry = {truth["key"]: {300: (0, 0, 20, 20)}}
    p1 = _prediction()
    p2 = {**_prediction(), "candidate_id": "duplicate"}
    _, pairs, breakdown = module.evaluate_set([p1, p2], [truth], geometry, "baseline", 0.)
    assert len(pairs) == 1
    assert breakdown["pickup"]["gt"] == 1
    assert breakdown["pickup"]["candidates"] == 2
    assert breakdown["pickup"]["matches"] == 1
    assert breakdown["pickup"]["precision"] == .5


def test_no_single_endpoint_matching_loophole():
    truth = _truth()
    geometry = {truth["key"]: {300: (0, 0, 20, 20)}}
    # Start is within tolerance but the interval is far too long and has tiny IoU.
    assert not module.eligible(_prediction(start=10., end=100.), truth, geometry)


def test_threshold_suppresses_candidate_without_changing_truth():
    truth = _truth()
    geometry = {truth["key"]: {300: (0, 0, 20, 20)}}
    pred = _prediction()
    assert module.score(pred, "relative_motion") == .7
    _, pairs, low = module.evaluate_set([pred], [truth], geometry, "relative_motion", .6)
    _, _, high = module.evaluate_set([pred], [truth], geometry, "relative_motion", .8)
    assert len(pairs) == 1
    assert low["pickup"]["matches"] == 1
    assert high["pickup"]["matches"] == 0
    assert low["pickup"]["gt"] == high["pickup"]["gt"] == 1
