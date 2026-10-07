"""Persistent single-worker job queue. Jobs are video rows, recovered after restart."""

import logging
import os
import threading
from collections import defaultdict
import cv2
import numpy as np
from sqlalchemy import select, update
from .config import SAMPLE_FPS
from .db import Camera, Embedding, Entity, Observation, Track, Video, session, uid
from .evidence import create_event
from .vision import detector, dominant_color, appearance, ClipEncoder

log = logging.getLogger("vigilia.worker")
stop = threading.Event()


def progress(video_id, state, value):
    with session() as s:
        s.execute(
            update(Video)
            .where(Video.id == video_id)
            .values(status=state, progress=value)
        )


def extract_events(samples, duration):
    """Conservative track rules. State changes are hypotheses; detections are observed."""
    result = [("appeared", samples[0]["t"], samples[0]["t"], "OBSERVED")]
    if samples[-1]["t"] < duration - 2 / SAMPLE_FPS:
        result.append(("disappeared", samples[-1]["t"], samples[-1]["t"], "INFERRED"))
    stationary_start = None
    for a, b in zip(samples, samples[1:]):
        center_a = np.array(
            [(a["box"][0] + a["box"][2]) / 2, (a["box"][1] + a["box"][3]) / 2]
        )
        center_b = np.array(
            [(b["box"][0] + b["box"][2]) / 2, (b["box"][1] + b["box"][3]) / 2]
        )
        scale = max(1, a["box"][3] - a["box"][1])
        speed = np.linalg.norm(center_b - center_a) / scale / max(0.01, b["t"] - a["t"])
        if speed < 0.08:
            stationary_start = a["t"] if stationary_start is None else stationary_start
        elif stationary_start is not None:
            if a["t"] - stationary_start >= 3:
                result.append(("stopped", stationary_start, a["t"], "INFERRED"))
            stationary_start = None
    if stationary_start is not None and samples[-1]["t"] - stationary_start >= 3:
        result.append(("stopped", stationary_start, samples[-1]["t"], "INFERRED"))
    return result


def process(video_id):
    with session() as s:
        video = s.get(Video, video_id)
        if not video:
            return
        path, duration = video.path, video.duration
    progress(video_id, "processing", 2)
    model = detector()
    clip = (
        ClipEncoder() if os.getenv("ENABLE_CLIP", "false").lower() == "true" else None
    )
    reader = None
    if os.getenv("ENABLE_OCR", "false").lower() == "true":
        import easyocr

        reader = easyocr.Reader(["en"], gpu=False)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(
            "Video cannot be decoded; unsupported codec or corrupt source"
        )
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, round(fps / SAMPLE_FPS))
    tracks = defaultdict(list)
    crops = {}
    ocr = []
    previous = None
    frame_idx = 0
    scene = 0
    try:
        while not stop.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % step == 0:
                t = frame_idx / fps
                # Cut resets keep tracker identities scoped to continuous scenes.
                gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36))
                if (
                    previous is not None
                    and np.mean(np.abs(gray.astype(float) - previous)) > 70
                ):
                    model = detector()
                    scene += 1
                elif previous is None:
                    scene = 0
                previous = gray.astype(float)
                for item in model.detect(frame):
                    key = f"{scene}-{item['track_id']}"
                    x1, y1, x2, y2 = item["box"]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    item.update(t=t, frame=frame_idx, box=[x1, y1, x2, y2])
                    tracks[key].append(item)
                    # Select largest visible crop per track, not every-frame embeddings.
                    crop = frame[y1:y2, x1:x2]
                    if key not in crops or crop.size > crops[key].size:
                        crops[key] = crop.copy()
                if reader is not None and frame_idx % (step * 10) == 0:
                    for box, text, confidence in reader.readtext(frame):
                        if confidence >= 0.6:
                            ocr.append((t, text, float(confidence)))
                progress(video_id, "detecting", min(85, 5 + 80 * t / max(duration, 1)))
            frame_idx += 1
        if stop.is_set():
            raise RuntimeError("Worker interrupted; retry indexing")
        if frame_idx < video.frame_count * 0.95:
            raise RuntimeError(
                "Decoding stopped before the end of the source; video may be truncated"
            )
    finally:
        cap.release()
    progress(video_id, "indexing", 90)
    with session() as s:
        video = s.get(Video, video_id)
        camera = s.get(Camera, video.camera_id)
        track_entities = {}
        for key, samples in tracks.items():
            if len(samples) < 2:
                continue
            crop = crops[key]
            # Upper-half color is a measured coarse cue, not a verified clothing label.
            color = dominant_color(crop[: max(1, crop.shape[0] // 2)])
            typ = samples[0]["object_type"]
            attrs = {
                "color": color,
                "color_method": "crop HSV median",
                "orientation": "unknown",
                "carried_object": "unknown",
            }
            entity = Entity(
                id=uid("P" if typ == "person" else "OBJ"),
                camera_id=video.camera_id,
                object_type=typ,
                attributes=attrs,
            )
            s.add(entity)
            s.flush()
            track = Track(
                id=uid("TRK"),
                video_id=video.id,
                entity_id=entity.id,
                tracker=model.name,
            )
            s.add(track)
            s.flush()
            confidence = [
                x["confidence"] for x in samples if x["confidence"] is not None
            ]
            obs = Observation(
                id=uid("OBS"),
                video_id=video.id,
                entity_id=entity.id,
                track_id=track.id,
                start=samples[0]["t"],
                end=samples[-1]["t"],
                boxes=[
                    {"frame": x["frame"], "t": x["t"], "box": x["box"]} for x in samples
                ],
                confidence=float(np.mean(confidence)) if confidence else None,
                attributes=attrs,
            )
            s.add(obs)
            s.flush()
            s.add(
                Embedding(
                    id=uid("EMB"),
                    observation_id=obs.id,
                    model="HSV-48",
                    values=appearance(crop),
                )
            )
            if clip:
                s.add(
                    Embedding(
                        id=uid("EMB"),
                        observation_id=obs.id,
                        model="OpenCLIP-ViT-B-32",
                        values=clip.image(crop),
                    )
                )
            track_entities[key] = (entity, obs)
            for kind, start, end, category in extract_events(samples, duration):
                create_event(
                    s,
                    video,
                    [entity.id],
                    kind,
                    f"{color.title()} {typ} {kind.replace('_', ' ')}",
                    start,
                    end,
                    confidence=obs.confidence,
                    category=category,
                    attributes=attrs,
                    observation=obs.id,
                    method=model.name + " + geometric rules",
                )
            # User-defined rectangular normalized camera zones.
            for zone in camera.connections.get("zones", []):
                inside = []
                for sample in samples:
                    x1, y1, x2, y2 = sample["box"]
                    x, y = (x1 + x2) / 2 / video.width, y2 / video.height
                    if (
                        zone["rect"][0] <= x <= zone["rect"][2]
                        and zone["rect"][1] <= y <= zone["rect"][3]
                    ):
                        inside.append(sample)
                if inside:
                    create_event(
                        s,
                        video,
                        [entity.id],
                        "entered_zone",
                        f"{typ.title()} detected in {zone['name']}",
                        inside[0]["t"],
                        inside[-1]["t"],
                        category="INFERRED",
                        attributes={**attrs, "zone": zone["name"]},
                        observation=obs.id,
                        method="bounding-box footpoint in configured rectangle",
                    )
        # Pairwise approach hypothesis; never interprets proximity as actual contact.
        keys = list(track_entities)
        for person_key in keys:
            person, obs = track_entities[person_key]
            if person.object_type != "person":
                continue
            for other_key in keys:
                other, _ = track_entities[other_key]
                if other.object_type not in ("car", "truck", "bus", "motorcycle"):
                    continue
                other_frames = {x["frame"]: x for x in tracks[other_key]}
                distances = []
                for x in tracks[person_key]:
                    if x["frame"] not in other_frames:
                        continue
                    a, b = x["box"], other_frames[x["frame"]]["box"]
                    d = np.linalg.norm(
                        np.array([(a[0] + a[2]) / 2, (a[1] + a[3]) / 2])
                        - np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
                    ) / max(1, video.width)
                    distances.append((x["t"], d))
                if (
                    len(distances) >= 3
                    and distances[0][1] - distances[-1][1] > 0.12
                    and distances[-1][1] < 0.2
                ):
                    create_event(
                        s,
                        video,
                        [person.id, other.id],
                        "approached_vehicle",
                        "Person may have approached a vehicle",
                        distances[0][0],
                        distances[-1][0],
                        category="INFERRED",
                        observation=obs.id,
                        method="normalized center-distance decrease",
                    )
        for t, text, conf in ocr:
            create_event(
                s,
                video,
                [],
                "readable_text",
                f"OCR candidate: {text}",
                t,
                t,
                confidence=conf,
                category="INFERRED",
                attributes={"text": text, "ocr_confidence": conf},
                method="EasyOCR; text requires visual verification",
            )
        video.status = "completed"
        video.progress = 100
        video.pipeline = model.name
        video.error = None
    log.info("Indexed %s with %d tracks", video_id, len(track_entities))


def worker_loop():
    # Run one worker process only. Processing is atomic; interrupted jobs are recovered.
    with session() as s:
        s.execute(
            update(Video)
            .where(Video.status.in_(["processing", "detecting", "indexing"]))
            .values(status="queued", progress=0)
        )
    while not stop.is_set():
        with session() as s:
            item = s.scalar(
                select(Video)
                .where(Video.status == "queued")
                .order_by(Video.created_at)
                .limit(1)
            )
            video_id = item.id if item else None
        if video_id:
            try:
                process(video_id)
            except Exception as exc:
                log.exception("Indexing failed: %s", video_id)
                with session() as s:
                    video = s.get(Video, video_id)
                    video.status = "failed"
                    video.error = str(exc)[:800]
        else:
            stop.wait(0.5)


def start_worker():
    stop.clear()
    thread = threading.Thread(target=worker_loop, name="vigilia-indexer", daemon=True)
    thread.start()
    return thread
