"""Prepare source-linked, separated review clips without detector or GT input."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from meva.dataset import load_manifest, resolve
from meva.object_identity_benchmark import DATASET_VERSION, clip_index

# Chosen by inspecting actual frames at regular intervals, not activity labels
# or detector predictions. Each clip spans 5 s on the source's 30 FPS clock.
CAMERAS = {
    "G421": "Indoor lounge: small table/bag objects, people and frequent occlusion.",
    "G331": "Bus waiting room: visible bags, ordinary behavior and near/far people.",
    "G424": "Outdoor parking: moving and stationary vehicles at several distances.",
}
START_SECONDS = {
    "G421": (0, 60, 120, 180, 240),
    "G331": (0, 60, 120, 180, 240),
    "G424": (30, 60, 120, 180, 240),
}
CLIP_SECONDS = 5
SAMPLE_FPS = 2


def prepare(manifest_path, output):
    manifest = load_manifest(manifest_path)
    selected = {v["dataset_camera_id"]: v for v in manifest["videos"]}
    clips = []
    for camera, rationale in CAMERAS.items():
        video = selected[camera]
        path = resolve(video["local_path"])
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {path}")
        try:
            step = round(video["fps"] / SAMPLE_FPS)
            if abs(video["fps"] / step - SAMPLE_FPS) > 1e-6:
                raise ValueError(f"2 FPS is not aligned for {camera}")
            for seconds in START_SECONDS[camera]:
                first = round(seconds * video["fps"])
                frames = list(range(first, first + CLIP_SECONDS*round(video["fps"]), step))
                for frame in (frames[0], frames[-1]):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                    ok, image = cap.read()
                    if not ok or image.shape[1] != video["width"] or image.shape[0] != video["height"]:
                        raise RuntimeError(f"Cannot decode {camera} frame {frame}")
                clips.append({
                    "clip_id": f"{camera}-{seconds:04d}",
                    "video_id": video["video_id"],
                    "camera_id": video["camera_id"],
                    "source_video": video["local_path"],
                    "source_sha256": video["sha256"],
                    "width": video["width"], "height": video["height"],
                    "fps": video["fps"], "sample_fps": SAMPLE_FPS,
                    "start_seconds": seconds, "duration_seconds": CLIP_SECONDS,
                    "frame_indices": frames,
                    "selection_reason": rationale,
                    "split": "pilot_review_only",
                })
        finally:
            cap.release()
    result = {
        "dataset_version": DATASET_VERSION,
        "dataset": "MEVA object identity manual-review pilot",
        "source_manifest": "datasets/meva/manifest.json",
        "source_license": manifest["license"],
        "annotation_provenance": "human review of source video; no model proposals",
        "evaluation_split": "pilot_review_only; no held-out claim",
        "clip_selection_exclusions": [
            ("G424 0–5 s was visually reviewed and excluded from this manual-identity "
             "pilot because most sampled frames render green/corrupted; the source video "
             "and historical full-video benchmark remain unchanged.")
        ],
        "clips": clips,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    clip_index(output, manifest_path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/meva/manifest.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "datasets/meva/object-identity/clips.json")
    args = parser.parse_args()
    result = prepare(args.manifest, args.output)
    print(f"Prepared {len(result['clips'])} clips, "
          f"{sum(len(c['frame_indices']) for c in result['clips'])} sampled frames")


if __name__ == "__main__":
    main()
