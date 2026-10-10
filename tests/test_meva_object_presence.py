"""Behavioral tests for evaluation-only object detection association and scoring."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from meva.object_presence import build_object_tracks, deduplicate_frame


def det(idx, frame, x, *, cls="cup", video="v", confidence=.8):
    return {"detection_id": f"{video}:{frame}:{idx}", "video_id": video,
            "camera_id": "c", "source_sha256": "source-hash", "frame": frame,
            "t": frame/30, "box": [x, 0, x+10, 10],
            "class": cls, "confidence": confidence}


def test_empty_and_single_detection_abstain_from_persistence():
    assert build_object_tracks([]) == []
    tracks = build_object_tracks([det(0, 0, 0)])
    assert len(tracks) == 1
    assert not tracks[0]["persistent"]
    assert tracks[0]["duration_seconds"] == 0


def test_duplicate_overlap_is_suppressed_without_deleting_other_class():
    hits = [det(0, 0, 0, confidence=.8), det(1, 0, .5, confidence=.5),
            det(2, 0, 0, cls="bottle")]
    kept = deduplicate_frame(hits)
    assert len(kept) == 2
    assert {d["class"] for d in kept} == {"cup", "bottle"}
    assert any(d["detection_id"] == "v:0:0" for d in kept)


def test_short_missing_sample_links_but_long_gap_fragments():
    short = build_object_tracks([det(0, 0, 0), det(0, 30, 1)])
    assert len(short) == 1
    assert short[0]["persistent"]
    assert short[0]["gap_count"] == 1
    long = build_object_tracks([det(0, 0, 0), det(0, 45, 1)])
    assert len(long) == 2
    assert all(not t["persistent"] for t in long)


def test_class_change_and_video_boundary_never_preserve_identity():
    tracks = build_object_tracks([det(0, 0, 0), det(0, 15, 1, cls="bottle"),
                                  det(0, 15, 1, video="other")])
    assert len(tracks) == 3
    assert len({t["video_id"] for t in tracks}) == 2


def test_near_tied_crossing_abstains_instead_of_assigning_identity():
    tracks = build_object_tracks([det(0, 0, 0), det(1, 0, 16), det(0, 15, 8)])
    assert len(tracks) == 3
    assert tracks[-1]["ambiguous_predecessor"]
    assert all(len(t["detections"]) == 1 for t in tracks)


def test_source_time_and_track_ids_are_deterministic():
    hits = [det(0, 0, 0), det(0, 15, 1)]
    first = build_object_tracks(hits)
    second = build_object_tracks(list(reversed(hits)))
    assert first == second
    assert first[0]["source_sha256"] == "source-hash"
    assert first[0]["camera_id"] == "c"
    assert [d["frame"] for d in first[0]["detections"]] == [0, 15]
    assert first[0]["end_seconds"] == .5


def test_invalid_source_frame_fails_loudly():
    bad = det(0, -1, 0)
    with pytest.raises(ValueError):
        build_object_tracks([bad])


def test_evaluation_frame_matching_is_one_to_one():
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_meva_object_presence.py"
    spec = importlib.util.spec_from_file_location("object_eval_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    detections = [det(0, 0, 0)]
    references = [{"box": [0, 0, 10, 10]}, {"box": [1, 0, 11, 10]}]
    pairs = module.assign_frame(detections, references, threshold=.3)
    assert len(pairs) == 1
    assert pairs[0][0] == 0
    assert pairs[0][2] >= .3


def test_full_video_extraction_is_annotation_free_and_resumable(tmp_path, monkeypatch):
    import argparse

    import numpy as np

    path = Path(__file__).resolve().parents[1] / "scripts/experiment_meva_object_presence.py"
    spec = importlib.util.spec_from_file_location("object_extract_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    video_file = tmp_path / "source.avi"
    video_file.write_bytes(b"fixture")
    model_file = tmp_path / "model.pt"
    model_file.write_bytes(b"fixture")
    video = {"video_id": "v", "camera_id": "c", "local_path": "runtime-video",
             "sha256": "source-hash", "fps": 2.0, "frame_count": 2,
             "duration": 1., "width": 64, "height": 64}
    monkeypatch.setattr(module, "load_manifest", lambda _: {"videos": [video]})
    monkeypatch.setattr(module, "resolve", lambda _: video_file)
    monkeypatch.setattr(module, "sha256", lambda _: "model-hash")

    class Capture:
        def __init__(self, _path):
            pass
        def isOpened(self):
            return True
        def set(self, *_args):
            pass
        def read(self):
            return True, np.zeros((64, 64, 3), dtype=np.uint8)
        def release(self):
            pass

    class Tensor:
        def __init__(self, value):
            self.value = np.asarray(value)
            self.device = "cpu"
        def cpu(self):
            return self
        def numpy(self):
            return self.value

    calls = []
    class Model:
        def __init__(self, _path):
            self.names = {0: "cup"}
        def predict(self, *_args, **_kwargs):
            calls.append(1)
            boxes = type("Boxes", (), {
                "xyxy": Tensor([[1, 2, 11, 12]]),
                "cls": Tensor([0]),
                "conf": Tensor([.8]),
                "data": Tensor([[1, 2, 11, 12]]),
            })()
            return [type("Result", (), {"boxes": boxes, "speed": {"inference": 1.}})()]

    monkeypatch.setattr(module.cv2, "VideoCapture", Capture)
    monkeypatch.setattr(module, "YOLO", Model)
    args = argparse.Namespace(manifest=tmp_path/"manifest.json", model=model_file,
                              size=640, confidence=.3, sample_fps=2.,
                              run_dir=tmp_path/"run", video_id=None, force=False,
                              roi_strategy=None, roi_factor=2., database=tmp_path/"db")
    module.extract(args)
    artifact = __import__("json").loads((args.run_dir/"v.json").read_text())
    assert artifact["sampled_frames"] == 2
    assert len(artifact["detections"]) == 2
    assert artifact["detections"][0]["source_sha256"] == "source-hash"
    assert artifact["detections"][0]["box"] == [1., 2., 11., 12.]
    assert calls == [1, 1]
    module.extract(args)
    assert calls == [1, 1]  # cached inference was not repeated
    args.confidence = .4
    with pytest.raises(RuntimeError, match="configuration differs"):
        module.extract(args)
