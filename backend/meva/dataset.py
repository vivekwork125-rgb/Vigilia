"""External paths, metadata and fail-closed dataset validation."""

import hashlib
import json
import math
import os
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "datasets/meva/manifest.json"


def roots():
    base = Path(os.environ.get("MEVA_ROOT", "~/VIGILIA_DATA")).expanduser()
    return {
        "MEVA_VIDEO_ROOT": Path(
            os.environ.get("MEVA_VIDEO_ROOT", str(base / "videos/meva"))
        ).expanduser(),
        "MEVA_ANNOTATION_ROOT": Path(
            os.environ.get(
                "MEVA_ANNOTATION_ROOT",
                str(base / "meva-data-repo/annotation/DIVA-phase-2/MEVA/kitware"),
            )
        ).expanduser(),
    }


def resolve(reference):
    for name, root in roots().items():
        prefix = "${" + name + "}/"
        if reference.startswith(prefix):
            relative = Path(reference[len(prefix) :])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe manifest path: {reference}")
            return root / relative
    raise ValueError(
        f"Manifest path must use MEVA_VIDEO_ROOT or MEVA_ANNOTATION_ROOT: {reference}"
    )


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata(path):
    if not Path(path).is_file():
        raise ValueError(f"Missing required video: {path}")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Video cannot open: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width, height = (
            cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        )
        if not all(math.isfinite(x) and x > 0 for x in (fps, count, width, height)):
            raise ValueError(f"Unreadable video metadata: {path}")
        for frame in (0, int(count) - 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, _ = cap.read()
            if not ok:
                raise ValueError(f"Video frame {frame} cannot decode: {path}")
        return {
            "fps": fps,
            "frame_count": int(count),
            "duration": count / fps,
            "width": int(width),
            "height": int(height),
        }
    finally:
        cap.release()


def load_manifest(path=DEFAULT_MANIFEST):
    manifest = json.loads(Path(path).read_text())
    videos = manifest.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ValueError("Manifest has no videos")
    for field in ("video_id", "filename", "local_path", "camera_id", "annotation_path"):
        values = [v[field] for v in videos]
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate manifest {field}")
    return manifest


def validate(manifest, geometry=True):
    from .annotations import parse_activities, parse_geometry, parse_types

    errors, rows = [], []
    for video in manifest["videos"]:
        try:
            actual = metadata(resolve(video["local_path"]))
            for key, value in actual.items():
                if not math.isclose(value, video[key], rel_tol=1e-6):
                    raise ValueError(
                        f"Metadata mismatch {key}: recorded={video[key]}, actual={value}"
                    )
            for path_key, hash_key in (
                ("local_path", "sha256"),
                ("annotation_path", "annotation_sha256"),
                ("types_path", "types_sha256"),
                ("geometry_path", "geometry_sha256"),
            ):
                path = resolve(video[path_key])
                if not path.is_file():
                    raise ValueError(f"Missing required file: {path}")
                if video.get(hash_key) and sha256(path) != video[hash_key]:
                    raise ValueError(f"SHA-256 mismatch: {path}")
            types = parse_types(resolve(video["types_path"]))
            activities = parse_activities(
                resolve(video["annotation_path"]), video, types
            )
            if geometry:
                parse_geometry(resolve(video["geometry_path"]), video, types)
            rows.append(
                {
                    "video_id": video["video_id"],
                    "duration": actual["duration"],
                    "activities": len(activities),
                    "valid": True,
                }
            )
        except (ValueError, KeyError, OSError) as exc:
            errors.append(f"{video['filename']}: {exc}")
    if errors:
        raise ValueError("MEVA validation failed:\n" + "\n".join(errors))
    return {
        "valid": True,
        "videos": rows,
        "total_duration": sum(r["duration"] for r in rows),
        "activity_count": sum(r["activities"] for r in rows),
        "geometry_validated": geometry,
    }
