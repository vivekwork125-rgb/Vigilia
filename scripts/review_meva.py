#!/usr/bin/env python3
"""Create source-frame contact sheets for benchmark errors; never synthesize footage."""

import argparse
import json
import sys
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from meva.dataset import DEFAULT_MANIFEST, ROOT, load_manifest, resolve


def contact_sheet(manifest, run_dir, limit=12):
    sources = {v["video_id"]: v for v in manifest["videos"]}
    errors = json.loads((run_dir / "error-analysis.json").read_text())
    selected, seen = [], set()
    # Show diverse activity/camera failures first, then fill available cells.
    for error in errors:
        ref = (
            error["ground_truth"]
            if error["kind"] == "FALSE_NEGATIVE"
            else error["prediction"]
        )
        key = (
            error["kind"],
            ref.get("activity_type", ref.get("event_type")),
            ref["camera_id"],
        )
        if key not in seen:
            seen.add(key)
            selected.append(error)
        if len(selected) >= limit:
            break
    if not selected:
        return None
    cell_w, cell_h = 480, 320
    canvas = Image.new(
        "RGB", (cell_w * 3, cell_h * ((len(selected) + 2) // 3)), "#101820"
    )
    draw = ImageDraw.Draw(canvas)
    refs = []
    for index, error in enumerate(selected):
        ref = (
            error["ground_truth"]
            if error["kind"] == "FALSE_NEGATIVE"
            else error["prediction"]
        )
        video = sources[ref["video_id"]]
        frame = ref["start_frame"]
        cap = cv2.VideoCapture(str(resolve(video["local_path"])))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, pixels = cap.read()
        finally:
            cap.release()
        if not ok:
            raise ValueError(
                f"Review source frame cannot decode: {video['filename']} frame {frame}"
            )
        picture = Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
        picture.thumbnail((cell_w, 265))
        x, y = (index % 3) * cell_w, (index // 3) * cell_h
        canvas.paste(picture, (x + (cell_w - picture.width) // 2, y))
        label = f"{error['kind']} | {ref.get('activity_type', ref.get('event_type'))}\n{video['dataset_camera_id']} | frame {frame} | {frame / video['fps']:.3f}s"
        draw.text((x + 6, y + 268), label, fill="white")
        refs.append(
            {
                "cell": index,
                "kind": error["kind"],
                "video_id": video["video_id"],
                "source_sha256": video["sha256"],
                "frame": frame,
                "seconds": frame / video["fps"],
                "event_id": (error["prediction"] or {}).get("event_id"),
                "annotation_id": (error["ground_truth"] or {}).get("annotation_id"),
            }
        )
    path = run_dir / "error-contact-sheet.jpg"
    canvas.save(path, quality=90)
    (run_dir / "error-contact-sheet.json").write_text(json.dumps(refs, indent=2) + "\n")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "data/meva/current")
    args = parser.parse_args()
    print(contact_sheet(load_manifest(args.manifest), args.run_dir))
