"""A synthetic *video* fixture indexed through the ordinary perception pipeline.

The color-contour adapter is intentionally limited to this generated artwork.
It detects rendered pixels; scene coordinates and action labels are never passed
to the processor. This fixture is a regression demonstration, not field accuracy.
"""

import hashlib
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
from sqlalchemy import select

from .config import MEDIA
from .db import (
    Camera,
    Event,
    EventEntity,
    Evidence,
    Location,
    Relationship,
    Video,
    session,
    uid,
)
from .processing import process

ID = "VID-PIPELINE-DEMO-03"
CAMERA = "CAM-PIPELINE-DEMO"
FPS, DURATION, WIDTH, HEIGHT = 10, 30, 320, 180


class SyntheticPixelDetector:
    name = "synthetic pixel contours (fixture only)"

    def detect(self, frame):
        found = []
        for ident, kind, low, high in [
            (1, "person", (180, 0, 180), (255, 70, 255)),
            (2, "backpack", (0, 180, 0), (70, 255, 70)),
            (3, "car", (180, 180, 0), (255, 255, 70)),
        ]:
            mask = cv2.inRange(frame, np.array(low, np.uint8), np.array(high, np.uint8))
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            if w * h >= 80:
                found.append(
                    {
                        "track_id": ident,
                        "box": [x, y, x + w, y + h],
                        "object_type": kind,
                        "confidence": None,
                    }
                )
        return found


def frame_at(t):
    frame = np.full((HEIGHT, WIDTH, 3), (32, 40, 44), dtype=np.uint8)
    cv2.putText(
        frame,
        "SYNTHETIC PIXEL-INDEXED FIXTURE",
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (215, 215, 215),
        1,
    )
    cv2.putText(
        frame,
        f"{t:04.1f}s",
        (8, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (215, 215, 215),
        1,
    )
    if t <= 5:
        person_x, bag_x = 28 + 9 * t, 43 + 9 * t
    elif t <= 12:
        person_x, bag_x = 73 + 15 * (t - 5), 88
    elif t <= 17:
        person_x, bag_x = 178, 88
    elif t <= 22:
        person_x, bag_x = 178 - 21 * (t - 17), 88
    else:
        person_x, bag_x = 73 + 10 * (t - 22), 88 + 10 * (t - 22)
    px, bx = round(person_x), round(bag_x)
    cv2.rectangle(frame, (px, 52), (px + 20, 94), (240, 0, 240), -1)
    cv2.rectangle(frame, (bx, 78), (bx + 14, 98), (0, 240, 0), -1)
    car_x = 237 if t <= 17 else round(237 + 13 * (t - 17))
    cv2.rectangle(
        frame, (car_x, 110), (min(WIDTH - 1, car_x + 55), 139), (240, 240, 0), -1
    )
    return frame


def create_fixture():
    path = MEDIA / "pipeline-synthetic-fixture-v3.mp4"
    if path.exists():
        return path
    encoder = subprocess.Popen(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{WIDTH}x{HEIGHT}",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-crf",
            "17",
            "-movflags",
            "+faststart",
            str(path),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    for i in range(FPS * DURATION):
        encoder.stdin.write(frame_at(i / FPS).tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        path.unlink(missing_ok=True)
        raise RuntimeError("Pipeline fixture video encoding failed")
    return path


def seed_pipeline_demo():
    with session() as s:
        existing = s.get(Video, ID)
        if existing and existing.status == "completed":
            # A fixture indexed by an earlier release may predate persisted
            # relationship rows. Derive them from its recorded event evidence.
            for event in s.scalars(
                select(Event).join(Evidence).where(Evidence.video_id == ID)
            ).all():
                if s.scalar(
                    select(Relationship).where(
                        Relationship.evidence_id == event.evidence_id
                    )
                ):
                    continue
                entities = s.scalars(
                    select(EventEntity.entity_id).where(
                        EventEntity.event_id == event.id
                    )
                ).all()
                if len(entities) == 2:
                    s.add(
                        Relationship(
                            id=uid("REL"),
                            source_id=entities[0],
                            target_id=entities[1],
                            evidence_id=event.evidence_id,
                            relation=event.event_type,
                            category=event.category,
                            signals={"method": "recorded pixel-indexed event"},
                            explanation="Track-derived fixture event; association remains an inference.",
                        )
                    )
            return
    path = create_fixture()
    with session() as s:
        if not s.get(Location, "LOC-PIPELINE-DEMO"):
            s.add(Location(id="LOC-PIPELINE-DEMO", name="Pipeline test scene"))
        if not s.get(Camera, CAMERA):
            s.add(
                Camera(
                    id=CAMERA,
                    location_id="LOC-PIPELINE-DEMO",
                    name="Pipeline test scene",
                    connections={
                        "zones": [
                            {"name": "Transfer zone", "rect": [0.2, 0.15, 0.6, 0.7]}
                        ]
                    },
                )
            )
        if not s.get(Video, ID):
            s.add(
                Video(
                    id=ID,
                    camera_id=CAMERA,
                    filename=path.name,
                    path=str(path),
                    recording_start="2026-10-06T19:00:00+05:30",
                    duration=DURATION,
                    fps=FPS,
                    frame_count=FPS * DURATION,
                    width=WIDTH,
                    height=HEIGHT,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    status="uploaded",
                    progress=0,
                    is_demo=True,
                    pipeline="pending",
                )
            )
    process(ID, model_override=SyntheticPixelDetector())
