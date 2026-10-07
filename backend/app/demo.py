"""Deterministic, explicitly synthetic footage and separately authored retrieval labels."""

import hashlib
import math
import subprocess
from pathlib import Path
import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .config import MEDIA
from .db import (
    BenchmarkQuery,
    Camera,
    Embedding,
    Entity,
    Event,
    Finding,
    GroundTruth,
    Investigation,
    Location,
    Observation,
    Relationship,
    Track,
    Video,
    session,
)
from .evidence import create_event
from .vision import appearance

FPS, DURATION, WIDTH, HEIGHT = 10, 40, 768, 432
SCENES = [
    ("CAM-01", "East entrance", "2026-10-06T18:04:00+05:30"),
    ("CAM-02", "West plaza", "2026-10-06T18:00:00+05:30"),
    ("CAM-03", "Parking", "2026-10-06T18:07:00+05:30"),
]
COLORS = {
    "red": (184, 66, 58),
    "blue": (63, 116, 170),
    "black": (32, 36, 42),
    "white": (190, 197, 199),
}


def actors(camera, t):
    a = []
    if camera == "CAM-01":
        if 2 <= t <= 31:
            x = (
                90 + (min(t, 12) - 2) * 26
                if t <= 12
                else (350 if t <= 15 else 350 + (t - 15) * 23)
            )
            a.append(("P-E01", "person", "red", [x - 14, 200, x + 14, 260]))
        if 7 <= t <= 37:
            a.append(
                (
                    "P-E02",
                    "person",
                    "blue",
                    [660 - (t - 7) * 11, 280, 688 - (t - 7) * 11, 340],
                )
            )
        if 20 <= t <= 38:
            a.append(
                (
                    "P-E03",
                    "person",
                    "red",
                    [85 + (t - 20) * 12, 315, 113 + (t - 20) * 12, 375],
                )
            )
        if 12 <= t <= 30:
            a.append(("B-E01", "backpack", "black", [358, 252, 381, 278]))
    elif camera == "CAM-02":
        if 2 <= t <= 26:
            a.append(("P-W01", "person", "red", [100 + t * 12, 240, 128 + t * 12, 300]))
        if 10 <= t <= 38:
            a.append(
                (
                    "P-W02",
                    "person",
                    "red",
                    [610 - (t - 10) * 9, 300, 638 - (t - 10) * 9, 360],
                )
            )
        if 5 <= t <= 34:
            a.append(
                ("P-W03", "person", "blue", [140 + t * 10, 140, 168 + t * 10, 200])
            )
    else:
        if t <= 36:
            x = 420 + max(0, t - 24) * 26
            a.append(("V-C01", "car", "white", [x, 195, x + 110, 246]))
        if 3 <= t <= 24:
            x = 130 + (min(t, 17) - 3) * 19
            a.append(("P-C01", "person", "red", [x, 267, x + 28, 327]))
        if 12 <= t <= 36:
            a.append(
                (
                    "P-C02",
                    "person",
                    "blue",
                    [650 - (t - 12) * 8, 345, 678 - (t - 12) * 8, 405],
                )
            )
    return a


def font(size):
    for path in [
        "/System/Library/Fonts/Monaco.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def render(camera, t):
    image = Image.new("RGB", (WIDTH, HEIGHT), (38, 47, 51))
    d = ImageDraw.Draw(image)
    for y in range(85, HEIGHT, 45):
        d.line((0, y, WIDTH, y), fill=(48, 57, 60))
    for x in range(0, WIDTH, 70):
        d.line((x, 85, x, HEIGHT), fill=(48, 57, 60))
    if camera == "CAM-01":
        d.rectangle((0, 65, WIDTH, 143), fill=(69, 76, 77))
        d.rectangle((316, 75, 465, 144), fill=(22, 31, 34))
        for x in [330, 398, 466]:
            d.line((x, 76, x, 145), fill=(101, 115, 118), width=3)
        d.text((30, 93), "EAST ENTRANCE", font=font(15), fill=(154, 171, 172))
        d.rectangle((292, 158, 482, 300), outline=(123, 133, 95), width=2)
        d.text((303, 167), "ZONE A", font=font(10), fill=(162, 164, 128))
        for x in [38, 670]:
            d.rectangle((x, 170, x + 42, 225), fill=(55, 66, 58))
            d.ellipse((x - 5, 155, x + 47, 205), fill=(53, 78, 66))
    elif camera == "CAM-02":
        d.rectangle((0, 65, WIDTH, 106), fill=(71, 79, 79))
        d.text((28, 79), "WEST PLAZA / WALKWAY", font=font(15), fill=(156, 171, 174))
        d.rectangle((290, 138, 470, 203), fill=(55, 64, 65))
        d.ellipse((320, 125, 442, 213), fill=(47, 72, 63))
        d.rectangle((325, 350, 456, 366), fill=(94, 91, 78))
        # An occluder is drawn over a later actor region to demonstrate partial visibility.
    else:
        d.rectangle((0, 60, WIDTH, 135), fill=(65, 73, 74))
        d.text((27, 83), "PARKING / SOUTH ACCESS", font=font(15), fill=(157, 173, 175))
        for x in range(50, WIDTH, 140):
            d.line((x, 168, x, 270), fill=(131, 137, 125), width=2)
        d.line((0, 303, WIDTH, 303), fill=(139, 139, 116), width=3)
        d.line((0, 320, WIDTH, 320), fill=(139, 139, 116), width=3)
        for x in range(30, WIDTH, 85):
            d.rectangle((x, 373, x + 40, 378), fill=(113, 119, 113))
    for eid, kind, color, box in actors(camera, t):
        x1, y1, x2, y2 = box
        c = COLORS[color]
        d.ellipse((x1 - 4, y2 - 7, x2 + 10, y2 + 6), fill=(24, 31, 34))
        if kind == "person":
            stride = math.sin(t * 7) * 4
            d.line((x1 + 9, y2 - 22, x1 + 7 + stride, y2), fill=(29, 33, 39), width=7)
            d.line((x2 - 9, y2 - 22, x2 - 7 - stride, y2), fill=(29, 33, 39), width=7)
            d.rounded_rectangle((x1, y1 + 13, x2, y2 - 20), radius=8, fill=c)
            d.ellipse((x1 + 5, y1, x2 - 5, y1 + 18), fill=(181, 158, 135))
            if eid in ("P-E01", "P-W01") and (camera != "CAM-01" or t < 12):
                d.rounded_rectangle(
                    (x2 - 7, y1 + 23, x2 + 7, y1 + 42), radius=4, fill=(29, 32, 36)
                )
        elif kind == "car":
            d.rounded_rectangle(tuple(box), radius=12, fill=c)
            d.rectangle((x1 + 27, y1 + 5, x2 - 28, y2 - 5), fill=(53, 74, 83))
            d.rectangle((x1 + 8, y1 + 8, x1 + 15, y2 - 8), fill=(152, 169, 176))
        else:
            d.rounded_rectangle(tuple(box), radius=4, fill=c)
            d.arc(
                (x1 + 5, y1 - 6, x2 - 5, y1 + 10), 180, 360, fill=(84, 89, 91), width=3
            )
    if camera == "CAM-02":
        d.rectangle((499, 106, 523, 305), fill=(89, 96, 95))
    if camera == "CAM-03":
        # Deliberately lower illumination, included in fixture case labels.
        image = Image.fromarray((np.array(image) * 0.8).astype("uint8"))
        d = ImageDraw.Draw(image)
    d.rectangle((0, 0, WIDTH, 43), fill=(16, 24, 28))
    d.ellipse((19, 16, 27, 24), fill=(100, 215, 167))
    d.text(
        (38, 13),
        camera + "  /  SYNTHETIC RECONSTRUCTION",
        font=font(12),
        fill=(179, 201, 203),
    )
    d.text(
        (WIDTH - 128, 13),
        f"00:{int(t):02d}.{int(t % 1 * 10)}  10 FPS",
        font=font(11),
        fill=(159, 177, 179),
    )
    d.rectangle((0, HEIGHT - 24, WIDTH, HEIGHT), fill=(16, 24, 28))
    d.text(
        (17, HEIGHT - 18),
        "VIGILIA SIMULATION • NO REAL PEOPLE • NOT DETECTOR OUTPUT",
        font=font(10),
        fill=(146, 167, 170),
    )
    return np.array(image)


def create_video(camera):
    path = MEDIA / f"{camera}-synthetic.mp4"
    if path.exists():
        return path
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    process = subprocess.Popen(
        [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
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
            "fast",
            "-crf",
            "25",
            "-movflags",
            "+faststart",
            str(path),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    for i in range(FPS * DURATION):
        process.stdin.write(render(camera, i / FPS).tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("Synthetic video encoding failed")
    return path


# Authored independently of the query parser; event labels correspond to scripted footage.
EVENTS = [
    (
        "E01",
        "CAM-01",
        ["P-E01"],
        "entered_zone",
        "Red-clothed person enters east entrance",
        3,
        8,
        "OBSERVED",
    ),
    (
        "E02",
        "CAM-01",
        ["P-E01", "B-E01"],
        "carried",
        "Person carrying a black backpack near east entrance",
        8,
        11,
        "OBSERVED",
    ),
    (
        "E03",
        "CAM-01",
        ["P-E01", "B-E01"],
        "placed_object",
        "Person places a backpack near the east entrance",
        12,
        15,
        "OBSERVED",
    ),
    (
        "E04",
        "CAM-01",
        ["B-E01"],
        "unattended",
        "Backpack may have been left unattended at east entrance",
        16,
        29,
        "INFERRED",
    ),
    (
        "E05",
        "CAM-01",
        ["P-E01"],
        "exited",
        "Red-clothed person leaves east entrance",
        28,
        31,
        "OBSERVED",
    ),
    (
        "E06",
        "CAM-01",
        ["P-E02"],
        "entered_zone",
        "Blue-clothed person enters east entrance",
        7,
        12,
        "OBSERVED",
    ),
    (
        "E07",
        "CAM-01",
        ["P-E02", "B-E01"],
        "picked_up",
        "Blue-clothed person may have picked up the backpack",
        30,
        32,
        "INFERRED",
    ),
    (
        "E08",
        "CAM-01",
        ["P-E03"],
        "entered_zone",
        "Second red-clothed person enters east entrance",
        20,
        24,
        "OBSERVED",
    ),
    (
        "E09",
        "CAM-01",
        ["P-E03"],
        "exited",
        "Second red-clothed person leaves east entrance",
        35,
        38,
        "OBSERVED",
    ),
    (
        "W01",
        "CAM-02",
        ["P-W01"],
        "entered_zone",
        "Red-clothed person enters west plaza",
        2,
        6,
        "OBSERVED",
    ),
    (
        "W02",
        "CAM-02",
        ["P-W01"],
        "carried",
        "Person carrying a backpack through west plaza",
        6,
        18,
        "OBSERVED",
    ),
    (
        "W03",
        "CAM-02",
        ["P-W01"],
        "exited",
        "Red-clothed person exits west plaza after partial occlusion",
        23,
        26,
        "OBSERVED",
    ),
    (
        "W04",
        "CAM-02",
        ["P-W02"],
        "entered_zone",
        "Similar-looking red-clothed person enters west plaza",
        10,
        15,
        "OBSERVED",
    ),
    (
        "W05",
        "CAM-02",
        ["P-W02"],
        "exited",
        "Similar-looking person leaves west plaza",
        34,
        38,
        "OBSERVED",
    ),
    (
        "W06",
        "CAM-02",
        ["P-W03"],
        "entered_zone",
        "Blue-clothed person enters west plaza",
        5,
        10,
        "OBSERVED",
    ),
    (
        "W07",
        "CAM-02",
        ["P-W03"],
        "exited",
        "Blue-clothed person exits west plaza",
        31,
        34,
        "OBSERVED",
    ),
    (
        "C01",
        "CAM-03",
        ["V-C01"],
        "stopped",
        "White car stopped in parking for 24 seconds",
        0,
        24,
        "OBSERVED",
    ),
    (
        "C02",
        "CAM-03",
        ["P-C01"],
        "entered_zone",
        "Red-clothed person enters parking in low light",
        3,
        9,
        "OBSERVED",
    ),
    (
        "C03",
        "CAM-03",
        ["P-C01", "V-C01"],
        "approached_vehicle",
        "Person approaches a white vehicle in parking",
        10,
        18,
        "OBSERVED",
    ),
    (
        "C04",
        "CAM-03",
        ["V-C01"],
        "exited",
        "White vehicle exits parking",
        24,
        36,
        "OBSERVED",
    ),
    (
        "C05",
        "CAM-03",
        ["P-C02"],
        "entered_zone",
        "Blue-clothed person enters parking",
        12,
        16,
        "OBSERVED",
    ),
    (
        "C06",
        "CAM-03",
        ["P-C01"],
        "disappeared",
        "Person no longer visible near the vehicle",
        22,
        24,
        "INFERRED",
    ),
]
QUERIES = [
    ("Find the person who left an object near the east entrance", ["E03"], "placement"),
    ("Show objects placed at the east gate", ["E03"], "placement"),
    ("Find a red person who placed a bag", ["E03"], "attributes"),
    ("Find the person carrying backpacks near the east entrance", ["E02"], "carried"),
    ("Show people carrying a backpack in west plaza", ["W02"], "carried"),
    ("Find objects left unattended", ["E04"], "uncertainty"),
    ("Show unattended bags near east entrance", ["E04"], "uncertainty"),
    ("Find a person who picked up a bag", ["E07"], "uncertainty"),
    ("Show blue people who picked up the backpack", ["E07"], "attributes"),
    ("Find red people entering east entrance", ["E01", "E08"], "similar appearance"),
    ("Find blue people entering east entrance", ["E06"], "attributes"),
    ("Find people who entered west plaza", ["W01", "W04", "W06"], "zones"),
    ("Find red people who entered west plaza", ["W01", "W04"], "similar appearance"),
    ("Find blue people entering west plaza", ["W06"], "attributes"),
    ("Find people who entered parking", ["C02", "C05"], "low light"),
    ("Find red people entering parking", ["C02"], "low light"),
    ("Show blue people entering parking", ["C05"], "low light"),
    ("Show white vehicles stopped in parking", ["C01"], "duration"),
    ("Show vehicles stopped for more than 20 seconds", ["C01"], "duration"),
    ("Find vehicles that stopped for more than two minutes", [], "negative duration"),
    ("Find the person who approached a vehicle", ["C03"], "interaction"),
    ("Find red people who approached a car in parking", ["C03"], "interaction"),
    ("Find vehicles that exited parking", ["C04"], "exit"),
    ("Find red people who exited east entrance", ["E05", "E09"], "similar appearance"),
    ("Show blue people who exited west plaza", ["W07"], "exit"),
    (
        "Find people who left the scene at west plaza",
        ["W03", "W05", "W07"],
        "occlusion",
    ),
    ("Find people entering Camera 1", ["E01", "E06", "E08"], "camera"),
    ("Find people entering Camera 2", ["W01", "W04", "W06"], "camera"),
    ("Find people entering Camera 3", ["C02", "C05"], "camera"),
    ("Find red people entering after 6:03 PM", ["E01", "E08", "C02"], "time"),
    ("Find people entering before 6:01 PM", ["W01", "W04", "W06"], "time"),
    ("Find people entering between 6:04 PM and 6:05 PM", ["E01", "E06", "E08"], "time"),
    (
        "Find people who approached a vehicle within two minutes before it left",
        ["C03"],
        "temporal relation",
    ),
    ("Show a yellow bus", [], "negative object"),
    ("Find people at Camera 5 between 7 and 8 PM", [], "camera gap"),
    ("Find people wearing purple", [], "negative color"),
    ("Find bicycles entering the entrance", [], "negative object"),
    ("Show objects placed before 6:05 PM", ["E03"], "time"),
    ("Find a backpack placed at Camera 1", ["E03"], "objects"),
    ("Show cars stopped at Camera 3", ["C01"], "camera"),
]


def seed():
    with session() as s:
        if s.get(Video, "VID-DEMO-01"):
            return
    videos = {camera: create_video(camera) for camera, _, _ in SCENES}
    with session() as s:
        obs_by_entity = {}
        entities_by_camera = {}
        for camera, location, start in SCENES:
            location_id = "LOC-" + camera[-2:]
            s.add(Location(id=location_id, name=location))
            s.flush()
            links = {c: [90, 600] for c, _, _ in SCENES if c != camera}
            s.add(
                Camera(
                    id=camera, location_id=location_id, name=location, connections=links
                )
            )
            s.flush()
            path = videos[camera]
            video = Video(
                id="VID-DEMO-" + camera[-2:],
                camera_id=camera,
                filename=path.name,
                path=str(path),
                recording_start=start,
                duration=DURATION,
                fps=FPS,
                frame_count=FPS * DURATION,
                width=WIDTH,
                height=HEIGHT,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                status="completed",
                progress=100,
                is_demo=True,
                pipeline="Synthetic geometry + authored annotations",
            )
            s.add(video)
            s.flush()
            sampled = {}
            for frame in range(0, FPS * DURATION, 5):
                for eid, kind, color, box in actors(camera, frame / FPS):
                    sampled.setdefault(
                        eid, {"kind": kind, "color": color, "samples": []}
                    )["samples"].append(
                        {
                            "frame": frame,
                            "t": frame / FPS,
                            "box": [round(v) for v in box],
                        }
                    )
            entities_by_camera[camera] = sampled
            for eid, data in sampled.items():
                attrs = {
                    "color": data["color"],
                    "source": "synthetic scene definition",
                    "orientation": "unknown",
                }
                s.add(
                    Entity(
                        id=eid,
                        camera_id=camera,
                        object_type=data["kind"],
                        attributes=attrs,
                    )
                )
                s.flush()
                tid = "TRK-" + eid
                s.add(
                    Track(
                        id=tid,
                        video_id=video.id,
                        entity_id=eid,
                        tracker="synthetic trajectories",
                    )
                )
                s.flush()
                samples = data["samples"]
                oid = "OBS-" + eid
                s.add(
                    Observation(
                        id=oid,
                        video_id=video.id,
                        entity_id=eid,
                        track_id=tid,
                        start=samples[0]["t"],
                        end=samples[-1]["t"],
                        boxes=samples,
                        confidence=None,
                        attributes=attrs,
                    )
                )
                s.flush()
                middle = samples[len(samples) // 2]
                frame = render(camera, middle["t"])
                x1, y1, x2, y2 = middle["box"]
                crop = frame[max(0, y1) : min(HEIGHT, y2), max(0, x1) : min(WIDTH, x2)]
                s.add(
                    Embedding(
                        id="EMB-" + eid,
                        observation_id=oid,
                        model="HSV-48",
                        values=appearance(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)),
                    )
                )
                obs_by_entity[eid] = oid
        for eid, camera, entities, kind, title, start, end, category in EVENTS:
            create_event(
                s,
                s.get(Video, "VID-DEMO-" + camera[-2:]),
                entities,
                kind,
                title,
                start,
                end,
                category=category,
                observation=obs_by_entity[entities[0]],
                method="Authored synthetic annotation; not a model prediction",
                event_id="EVT-" + eid,
            )
        for i, (query, expected, case) in enumerate(QUERIES):
            qid = f"BQ-{i + 1:03d}"
            s.add(BenchmarkQuery(id=qid, query=query, case=case))
            s.flush()
            for eid in expected:
                s.add(GroundTruth(query_id=qid, event_id="EVT-" + eid))
        ev = s.get(Event, "EVT-E03")
        s.add(
            Relationship(
                id="REL-PLACED",
                source_id="P-E01",
                target_id="B-E01",
                evidence_id=ev.evidence_id,
                relation="placed",
                category="OBSERVED",
                signals={"method": "authored synthetic annotation"},
                explanation="Placement is part of the authored synthetic scenario. Open frames 120–150.",
            )
        )
        inv = Investigation(
            id="INV-DEMO", title="Object left at east entrance", query=QUERIES[0][0]
        )
        s.add(inv)
        s.flush()
        s.add(Finding(investigation_id=inv.id, event_id="EVT-E03"))
