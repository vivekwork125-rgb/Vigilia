"""Full-video, resumable MEVA object detections. No activity labels enter extraction."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import torch
import ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from meva.dataset import load_manifest, resolve, sha256
from meva.selective_objects import person_roi, person_samples


def extract(args):
    manifest = load_manifest(args.manifest)
    model_path = args.model.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    model_hash = sha256(model_path)
    started = time.perf_counter()
    model = YOLO(str(model_path))
    model_load_s = time.perf_counter() - started
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    people = person_samples(args.database) if args.roi_strategy else None
    config = {
        "schema_version": 1,
        "protocol": "full-video sampled runtime-only detection; no activity annotations",
        "model": model_path.name,
        "model_sha256": model_hash,
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "imgsz": args.size,
        "confidence": args.confidence,
        "sample_fps": args.sample_fps,
        "roi_strategy": args.roi_strategy,
        "roi_factor": args.roi_factor if args.roi_strategy else None,
        "model_load_seconds": round(model_load_s, 4),
        "source_manifest": str(args.manifest),
        "available_cpu_count": os.cpu_count(),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
    }
    config_path = run_dir / "config.json"
    if config_path.exists() and not args.force:
        saved = json.loads(config_path.read_text())
        keys = ("model_sha256", "imgsz", "confidence", "sample_fps", "roi_strategy", "roi_factor")
        if any(saved.get(k) != config.get(k) for k in keys):
            raise RuntimeError("Existing run configuration differs; choose a new run directory")
        config = saved
    else:
        config_path.write_text(json.dumps(config, indent=2))
    for video in manifest["videos"]:
        video_id = video["video_id"]
        if args.video_id and video_id != args.video_id:
            continue
        out = run_dir / f"{video_id}.json"
        if out.exists() and not args.force:
            old = json.loads(out.read_text())
            if (old.get("model_sha256") != model_hash or old.get("imgsz") != args.size
                    or old.get("confidence") != args.confidence or old.get("sample_fps") != args.sample_fps
                    or old.get("source_sha256") != video["sha256"]
                    or old.get("roi_strategy") != args.roi_strategy):
                raise RuntimeError(f"Existing {out} has different configuration; choose a new run directory")
            print(f"SKIP {video_id}: cached {old['sampled_frames']} frames", flush=True)
            continue
        path = resolve(video["local_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot decode {path}")
        step = round(video["fps"] / args.sample_fps)
        if step <= 0 or abs(video["fps"] / step - args.sample_fps) > .02:
            raise ValueError("Requested sampling rate does not align to source FPS")
        frames = []
        detections = []
        inference_ms = []
        begun = time.perf_counter()
        device = None
        try:
            for frame_no in range(0, video["frame_count"], step):
                frames.append(frame_no)
                runtime_people = people.get(video_id, {}).get(frame_no, []) if people is not None else None
                if runtime_people is not None and not runtime_people:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                ok, image = cap.read()
                if not ok:
                    raise RuntimeError(f"Failed to decode {video_id} frame {frame_no}")
                if runtime_people is None:
                    crops = [(image, 0, 0)]
                else:
                    crops = []
                    for person in runtime_people:
                        x1, y1, x2, y2 = person_roi(
                            person["box"], video["width"], video["height"],
                            strategy=args.roi_strategy, factor=args.roi_factor)
                        if x2-x1 >= 16 and y2-y1 >= 16:
                            crops.append((image[y1:y2, x1:x2], x1, y1))
                for crop, ox, oy in crops:
                    if crop.size == 0:
                        continue
                    result = model.predict(crop, imgsz=args.size, conf=args.confidence, verbose=False)[0]
                    inference_ms.append(float(result.speed.get("inference", 0.0)))
                    if device is None and result.boxes is not None:
                        device = str(result.boxes.data.device)
                    if result.boxes is None:
                        continue
                    for box, cls, conf in zip(
                        result.boxes.xyxy.cpu().numpy(),
                        result.boxes.cls.cpu().numpy(),
                        result.boxes.conf.cpu().numpy(),
                    ):
                        if runtime_people is not None and model.names[int(cls)] == "person":
                            continue
                        detections.append({
                            "detection_id": f"{video_id}:{frame_no}:{len(detections)}",
                            "video_id": video_id,
                            "camera_id": video["camera_id"],
                            "source_sha256": video["sha256"],
                            "frame": frame_no,
                            "t": frame_no / video["fps"],
                            "box": [round(float(box[0]+ox), 2), round(float(box[1]+oy), 2),
                                    round(float(box[2]+ox), 2), round(float(box[3]+oy), 2)],
                            "class": model.names[int(cls)],
                            "confidence": round(float(conf), 5),
                        })
                if len(frames) % 100 == 0:
                    print(f"{video_id}: {len(frames)} frames", flush=True)
        finally:
            cap.release()
        data = {
            "video_id": video_id,
            "camera_id": video["camera_id"],
            "source_sha256": video["sha256"],
            "source_path": video["local_path"],
            "fps": video["fps"],
            "duration": video["duration"],
            "frame_count": video["frame_count"],
            "sample_fps": args.sample_fps,
            "roi_strategy": args.roi_strategy,
            "roi_factor": args.roi_factor if args.roi_strategy else None,
            "inference_calls": len(inference_ms),
            "sampled_frames": len(frames),
            "frame_indices": frames,
            "model": model_path.name,
            "model_sha256": model_hash,
            "imgsz": args.size,
            "confidence": args.confidence,
            "device": device,
            "inference_seconds": round(sum(inference_ms) / 1000, 3),
            "mean_inference_ms": round(sum(inference_ms) / len(inference_ms), 3) if inference_ms else None,
            "wall_seconds": round(time.perf_counter() - begun, 3),
            "detections": detections,
        }
        temp = out.with_suffix(".json.tmp")
        temp.write_text(json.dumps(data))
        temp.replace(out)
        print(f"DONE {video_id}: {len(frames)} frames, {len(detections)} detections, {data['wall_seconds']} s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/meva/manifest.json")
    parser.add_argument("--model", type=Path, default=ROOT / "models/yolo11n.pt")
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=.30)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "data/meva/object-presence-yolo11n-640-c030")
    parser.add_argument("--video-id")
    parser.add_argument("--roi-strategy", choices=("expanded", "lower", "interaction", "scene_nearby"))
    parser.add_argument("--roi-factor", type=float, default=2.0)
    parser.add_argument("--database", type=Path, default=ROOT / "data/meva/improved-2fps/vigilia.db")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.size < 64 or not 0 < args.confidence < 1 or not 0 < args.sample_fps <= 10 or args.roi_factor < 1:
        parser.error("Invalid size, confidence, or sample FPS")
    extract(args)


if __name__ == "__main__":
    main()
