"""Generate portable manifest for the explicitly selected drops-123-r13 subset."""

import json
import subprocess
from datetime import datetime
from pathlib import Path

from .dataset import DEFAULT_MANIFEST, metadata, roots, sha256

SELECTED = (
    "2018-03-15.15-55-00.16-00-00.school.G421.r13.avi",
    "2018-03-15.15-55-00.16-00-00.school.G336.r13.avi",
    "2018-03-15.15-55-00.16-00-00.school.G424.r13.avi",
    "2018-03-15.15-55-01.16-00-01.school.G328.r13.avi",
    "2018-03-15.15-55-01.16-00-01.school.G639.r13.avi",
    "2018-03-15.15-55-00.16-00-00.school.G638.r13.avi",
    "2018-03-15.15-55-00.16-00-00.bus.G331.r13.avi",
    "2018-03-15.15-55-07.16-00-07.hospital.G436.r13.avi",
)


def prepare(output=DEFAULT_MANIFEST):
    directories, videos = roots(), []
    for name in SELECTED:
        path = directories["MEVA_VIDEO_ROOT"] / "2018-03-15-1555" / name
        stem = Path(name).stem.removesuffix(".r13")
        matches = list(
            directories["MEVA_ANNOTATION_ROOT"].rglob(stem + ".activities.yml")
        )
        if len(matches) != 1:
            raise ValueError(
                f"Expected one annotation for {name}, found {len(matches)}"
            )
        annotation = matches[0]
        camera = stem.split(".")[-1]
        video = {
            "video_id": "MEVA-" + camera + "-20180315-1555",
            "filename": name,
            "local_path": "${MEVA_VIDEO_ROOT}/2018-03-15-1555/" + name,
            "camera_id": "MEVA-" + camera,
            "dataset_camera_id": camera,
            "recording_start": datetime.strptime(
                ".".join(stem.split(".")[:2]), "%Y-%m-%d.%H-%M-%S"
            ).isoformat(),
            "recording_timezone": "unspecified (dataset local clock; no timezone conversion)",
            "source": "drops-123-r13",
            "dataset_split": "selected evaluation subset; not training",
            **metadata(path),
            "sha256": sha256(path),
        }
        for suffix, key in (
            ("activities", "annotation"),
            ("types", "types"),
            ("geom", "geometry"),
        ):
            file = annotation.with_name(stem + "." + suffix + ".yml")
            video[key + "_path"] = (
                "${MEVA_ANNOTATION_ROOT}/"
                + file.relative_to(directories["MEVA_ANNOTATION_ROOT"]).as_posix()
            )
            video[key + "_sha256"] = sha256(file)
        videos.append(video)
    revision = subprocess.check_output(
        ["git", "-C", str(directories["MEVA_ANNOTATION_ROOT"]), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    manifest = {
        "schema_version": 1,
        "dataset": "MEVA",
        "license": "CC-BY-4.0",
        "attribution": "MEVA by Kitware Inc. and IARPA",
        "dataset_url": "https://mevadata.org/",
        "annotation_repository": "https://gitlab.kitware.com/meva/meva-data-repo",
        "annotation_revision": revision,
        "frame_convention": "zero-based inclusive tsr0 endpoints",
        "videos": videos,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
