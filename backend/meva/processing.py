"""Benchmark controller: register source metadata, invoke unmodified production pipeline.

No annotation parser is called here. Only video/camera metadata reaches app.process.
Runs use their own DB and immutable configuration fingerprint; resume never silently
mixes predictions from different code or model configurations.
"""

import importlib.metadata
import json
import os
import platform
import subprocess
import secrets
from pathlib import Path

from .dataset import ROOT, resolve, sha256


def configure(run_dir):
    directory = Path(run_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["DATA_DIR"] = str(directory)
    os.environ["DATABASE_URL"] = "sqlite:///" + str(directory / "vigilia.db")
    os.environ["DEMO_MODE"] = "false"
    token_file = directory / ".api-token"
    if not token_file.exists():
        token_file.write_text(secrets.token_urlsafe(32))
        token_file.chmod(0o600)
    os.environ.setdefault("API_TOKEN", token_file.read_text().strip())
    os.environ.setdefault("PERCEPTION_BACKEND", "yolo")
    os.environ.setdefault("YOLO_MODEL", str(ROOT / "models/yolo11n.pt"))
    os.environ.setdefault("SAMPLE_FPS", "2")
    os.environ.setdefault("DETECTION_SIZE", "640")
    os.environ.setdefault("DETECTION_CONFIDENCE", "0.3")
    os.environ.setdefault("ENABLE_CLIP", "false")
    os.environ.setdefault("ENABLE_OCR", "false")
    return directory


def configuration(manifest):
    if os.environ["PERCEPTION_BACKEND"] != "yolo":
        raise ValueError(
            "The MEVA baseline requires YOLO + ByteTrack; HOG is forbidden"
        )
    model = Path(os.environ["YOLO_MODEL"])
    if not model.is_file():
        raise ValueError(f"Required local YOLO model missing: {model}")
    code_files = sorted((ROOT / "backend/app").glob("*.py"))
    import hashlib

    code_hash = hashlib.sha256(
        b"".join(p.name.encode() + p.read_bytes() for p in code_files)
    ).hexdigest()
    return {
        "detector": "YOLO",
        "tracker": "ByteTrack",
        "model": model.name,
        "model_sha256": sha256(model),
        "sample_fps": float(os.environ["SAMPLE_FPS"]),
        "detection_size": int(os.environ["DETECTION_SIZE"]),
        "confidence_threshold": float(os.environ["DETECTION_CONFIDENCE"]),
        "enable_clip": os.environ["ENABLE_CLIP"],
        "enable_ocr": os.environ["ENABLE_OCR"],
        "unattended_seconds": float(os.environ.get("UNATTENDED_SECONDS", "5")),
        "software": {
            p: importlib.metadata.version(p)
            for p in (
                "ultralytics",
                "torch",
                "opencv-python",
                "numpy",
                "sqlalchemy",
                "PyYAML",
            )
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git_revision": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "production_code_sha256": code_hash,
        "production_diff": subprocess.check_output(
            ["git", "-C", str(ROOT), "diff", "--", "backend/app"], text=True
        ),
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest(),
        "cross_camera_identity": "not evaluated; identities remain camera scoped; no annotation IDs enter runtime",
        "detector_tracker_timing": "combined model.track wall time; separate tracker time unavailable",
    }


def process_manifest(manifest, directory):
    config = configuration(manifest)
    config_file = directory / "run-config.json"

    def fingerprint(value):
        return {
            k: v
            for k, v in value.items()
            if k not in ("git_revision", "production_diff")
        }

    if config_file.exists() and fingerprint(
        json.loads(config_file.read_text())
    ) != fingerprint(config):
        raise ValueError(
            "Run configuration/code changed. Choose a fresh --run-dir; the recorded baseline is immutable."
        )
    if not config_file.exists():
        config_file.write_text(json.dumps(config, indent=2) + "\n")
    # Import only after selecting the isolated database.
    from app.db import Camera, Location, Video, initialize, session
    from app.processing import process

    initialize()
    timing_file = directory / "processing-results.json"
    timings = json.loads(timing_file.read_text()) if timing_file.exists() else []
    for source in manifest["videos"]:
        path = resolve(source["local_path"])
        if not path.is_file() or sha256(path) != source["sha256"]:
            raise ValueError(f"Missing/changed source video {source['filename']}")
        with session() as s:
            video = s.get(Video, source["video_id"])
            if video:
                if video.sha256 != source["sha256"]:
                    raise ValueError("Existing DB source hash mismatch")
                if video.status == "completed":
                    print(f"Resume: {video.id} already completed", flush=True)
                    continue
            else:
                location_id = "MEVA-" + source["filename"].split(".")[-4]
                if not s.get(Location, location_id):
                    s.add(Location(id=location_id, name=location_id))
                    s.flush()
                if not s.get(Camera, source["camera_id"]):
                    s.add(
                        Camera(
                            id=source["camera_id"],
                            location_id=location_id,
                            name=source["dataset_camera_id"],
                            connections={},
                        )
                    )
                    s.flush()
                fields = {
                    k: source[k]
                    for k in (
                        "filename",
                        "camera_id",
                        "recording_start",
                        "duration",
                        "fps",
                        "frame_count",
                        "width",
                        "height",
                        "sha256",
                    )
                }
                s.add(Video(id=source["video_id"], path=str(path), **fields))
        print(f"Processing real source: {source['filename']}", flush=True)
        try:
            timing = process(source["video_id"])
        except Exception as exc:
            with session() as s:
                video = s.get(Video, source["video_id"])
                video.status, video.error = "failed", str(exc)
            raise
        timings.append(timing)
        if timing["decoded_frames"] != source["frame_count"]:
            raise ValueError(f"Incomplete source decoding: {source['filename']}")
        timing_file.write_text(json.dumps(timings, indent=2) + "\n")
        print(json.dumps(timing), flush=True)
    return timings
