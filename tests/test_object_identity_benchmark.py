"""Ground-truth integrity and one-to-one matching tests for the manual pilot."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest
from meva.object_identity_benchmark import (
    clip_index,
    frame_record,
    load_annotations,
    save_annotations,
    validate_dataset,
    validate_record,
)
from meva.object_identity_metrics import (
    class_compatible,
    match_frame,
    target_track_diagnostics,
)


@pytest.fixture
def pilot(tmp_path):
    video = {"video_id": "v1", "camera_id": "c1", "sha256": "abc",
             "width": 200, "height": 100, "fps": 30.0, "frame_count": 300,
             "filename": "v1.avi", "local_path": "${MEVA_VIDEO_ROOT}/v1.avi",
             "annotation_path": "${MEVA_ANNOTATION_ROOT}/v1.yml"}
    source_path = tmp_path / "manifest.json"
    source_path.write_text(json.dumps({"videos": [video]}))
    clip = {"clip_id": "c1-0000", "video_id": "v1", "camera_id": "c1",
            "source_video": video["local_path"],
            "source_sha256": "abc", "width": 200, "height": 100, "fps": 30.0,
            "sample_fps": 2.0, "start_seconds": 0.0, "duration_seconds": 1.5,
            "frame_indices": [0, 15, 30]}
    clips_path = tmp_path / "clips.json"
    clips_path.write_text(json.dumps({"dataset_version": "1.0", "clips": [clip]}))
    return source_path, clips_path, clip, tmp_path / "annotations.jsonl"


def ann(object_id="OBJ-1", *, box=None, visibility="visible", cls="bag"):
    return {"object_id": object_id, "bbox_xyxy": box if box is not None else [10, 20, 40, 60],
            "class_label": cls, "visibility": visibility, "occlusion": "none",
            "identity_confidence": "high", "annotator_note": "", "reidentification_note": ""}


def test_clip_manifest_provenance_and_consecutive_sampling(pilot):
    source, path, clip, _ = pilot
    _, indexed = clip_index(path, source)
    assert indexed[clip["clip_id"]]["frame_indices"] == [0, 15, 30]
    bad = deepcopy(clip)
    bad["source_sha256"] = "different"
    path.write_text(json.dumps({"dataset_version": "1.0", "clips": [bad]}))
    with pytest.raises(ValueError, match="source_sha256"):
        clip_index(path, source)


@pytest.mark.parametrize("frames", [[0, 30], [0, 15, 15], [0, 15, 999]])
def test_clip_manifest_rejects_skips_duplicates_and_out_of_range(pilot, frames):
    source, path, clip, _ = pilot
    clip["frame_indices"] = frames
    path.write_text(json.dumps({"dataset_version": "1.0", "clips": [clip]}))
    with pytest.raises(ValueError):
        clip_index(path, source)


def test_duplicate_clip_definition_fails(pilot):
    source, path, clip, _ = pilot
    path.write_text(json.dumps({"dataset_version": "1.0", "clips": [clip, clip]}))
    with pytest.raises(ValueError, match="Duplicate clip"):
        clip_index(path, source)


def test_clip_clock_and_source_reference_are_verified(pilot):
    source, path, clip, _ = pilot
    clip["source_video"] = "${MEVA_VIDEO_ROOT}/other.avi"
    path.write_text(json.dumps({"dataset_version": "1.0", "clips": [clip]}))
    with pytest.raises(ValueError, match="source video reference"):
        clip_index(path, source)
    clip["source_video"] = "${MEVA_VIDEO_ROOT}/v1.avi"
    clip["duration_seconds"] = 2.0
    path.write_text(json.dumps({"dataset_version": "1.0", "clips": [clip]}))
    with pytest.raises(ValueError, match="clock/duration"):
        clip_index(path, source)


def test_source_box_timestamp_frame_and_duplicate_id_validation(pilot):
    _, _, clip, _ = pilot
    clips = {clip["clip_id"]: clip}
    good = frame_record(clip, 15, annotations=[ann()])
    validate_record(good, clips)
    bad = deepcopy(good)
    bad["annotations"][0]["bbox_xyxy"] = [-1, 20, 40, 60]
    with pytest.raises(ValueError, match="box"):
        validate_record(bad, clips)
    bad = deepcopy(good)
    bad["timestamp_seconds"] = .7
    with pytest.raises(ValueError, match="timestamp"):
        validate_record(bad, clips)
    bad["timestamp_seconds"] = True
    with pytest.raises(ValueError, match="timestamp"):
        validate_record(bad, clips)
    bad = deepcopy(good)
    bad["frame_index"] = 16
    with pytest.raises(ValueError, match="frame"):
        validate_record(bad, clips)
    bad = deepcopy(good)
    bad["annotations"].append(ann())
    with pytest.raises(ValueError, match="Duplicate object ID"):
        validate_record(bad, clips)


def test_absence_uncertainty_and_visibility_rules(pilot):
    _, _, clip, _ = pilot
    clips = {clip["clip_id"]: clip}
    absent = ann(visibility="not_visible")
    absent["bbox_xyxy"] = None
    validate_record(frame_record(clip, 0, annotations=[absent]), clips)
    absent["bbox_xyxy"] = [10, 20, 40, 60]
    with pytest.raises(ValueError, match="Absent object"):
        validate_record(frame_record(clip, 0, annotations=[absent]), clips)
    uncertain = ann("UNRESOLVED-1")
    with pytest.raises(ValueError, match="Unresolved"):
        validate_record(frame_record(clip, 0, annotations=[uncertain]), clips)
    uncertain["identity_confidence"] = "uncertain"
    validate_record(frame_record(clip, 0, annotations=[uncertain]), clips)
    uncertain["visibility"] = "invented"
    with pytest.raises(ValueError, match="visibility"):
        validate_record(frame_record(clip, 0, annotations=[uncertain]), clips)


def test_complete_empty_frame_requires_explicit_confirmation(pilot):
    _, _, clip, _ = pilot
    clips = {clip["clip_id"]: clip}
    with pytest.raises(ValueError, match="EMPTY_SCENE_CONFIRMED"):
        validate_record(frame_record(clip, 0, review_state="complete"), clips)
    validate_record(frame_record(clip, 0, review_state="complete",
                                 note="EMPTY_SCENE_CONFIRMED"), clips)


def test_atomic_save_load_and_interrupted_session(pilot):
    _, _, clip, path = pilot
    first = frame_record(clip, 0, annotations=[ann()])
    records = {(clip["clip_id"], 0): first}
    save_annotations(path, records)
    assert load_annotations(path) == records
    path.with_suffix(".jsonl.tmp").write_text("interrupted incomplete snapshot")
    assert load_annotations(path) == records
    records[(clip["clip_id"], 15)] = frame_record(clip, 15, annotations=[ann()])
    save_annotations(path, records)
    assert set(load_annotations(path)) == set(records)


def test_reappearance_without_note_and_unresolved_id_reuse_fail(pilot):
    source, clips_path, clip, path = pilot
    records = {(clip["clip_id"], 0): frame_record(clip, 0, annotations=[ann()]),
               (clip["clip_id"], 30): frame_record(clip, 30, annotations=[ann()])}
    save_annotations(path, records)
    report = validate_dataset(clips_path, path, source)
    assert any("reappearance" in e for e in report["errors"])
    records[(clip["clip_id"], 30)]["annotations"][0]["reidentification_note"] = "Same marked bag"
    save_annotations(path, records)
    assert validate_dataset(clips_path, path, source)["valid_schema"]
    for row in records.values():
        row["annotations"][0]["object_id"] = "UNRESOLVED-1"
        row["annotations"][0]["identity_confidence"] = "uncertain"
    save_annotations(path, records)
    assert any("unresolved ID reused" in e for e in validate_dataset(
        clips_path, path, source)["errors"])


def test_missing_frames_keep_pilot_incomplete(pilot):
    source, clips_path, clip, path = pilot
    save_annotations(path, {(clip["clip_id"], 0): frame_record(
        clip, 0, annotations=[ann()], review_state="complete")})
    report = validate_dataset(clips_path, path, source)
    assert report["valid_schema"]
    assert not report["complete_pilot"]
    assert report["complete_frames"] == 1
    assert report["planned_frames"] == 3


def test_one_to_one_matching_and_unknown_class_policy():
    targets = [ann("A"), ann("B", box=[12, 20, 42, 60])]
    detections = [{"detection_id": "d1", "class": "suitcase", "box": [10, 20, 40, 60]}]
    pairs = match_frame(targets, detections, iou_threshold=.5)
    assert len(pairs) == 1
    assert pairs[0]["annotation_index"] == 0
    assert class_compatible(ann(cls="unknown_object"), detections[0])
    assert not class_compatible(ann(cls="bag"), {"class": "chair"})
    assert match_frame([ann(visibility="not_visible")], detections) == []


def test_target_track_changes_are_named_provisional_not_identity_truth():
    a = ann("A")
    rows = [{"clip_id": "clip", "frame_index": frame, "annotations": [a],
             "detections": [{"detection_id": ident}],
             "matches": [{"annotation_index": 0, "detection_index": 0}]}
            for frame, ident in [(0, "d1"), (15, "d2"), (30, "d3")]]
    result = target_track_diagnostics(rows, {"d1": "T1", "d2": "T1", "d3": "T2"})
    assert result["clip:A"]["provisional_id_changes_among_matched_pairs"] == 1
    assert result["clip:A"]["matched_frames"] == 3
