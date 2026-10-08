"""Persistent single-worker job queue. Jobs are video rows, recovered after restart."""

import logging
import os
import threading
import time
from collections import defaultdict
import cv2
import numpy as np
from sqlalchemy import select, update
from .config import SAMPLE_FPS
from .db import (
    Camera,
    Embedding,
    Entity,
    Observation,
    Relationship,
    Track,
    Video,
    session,
    uid,
)
from .evidence import create_event
from .vision import detector, dominant_color, appearance, ClipEncoder
from .temporal_events import continuous_tracks, track_events

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
    """Per-track appearance/disappearance. Motion rules have one canonical engine."""
    result = [("appeared", samples[0]["t"], samples[0]["t"], "OBSERVED")]
    if samples[-1]["t"] < duration - 2 / SAMPLE_FPS:
        result.append(("disappeared", samples[-1]["t"], samples[-1]["t"], "INFERRED"))
    return result


def process(video_id, model_override=None):
    begun = time.perf_counter()
    decode_seconds = detector_tracking_seconds = 0.0
    sampled_frames = 0
    with session() as s:
        video = s.get(Video, video_id)
        if not video:
            return
        path, duration = video.path, video.duration
    progress(video_id, "processing", 2)
    model = model_override or detector()
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
            tick = time.perf_counter()
            ok, frame = cap.read()
            decode_seconds += time.perf_counter() - tick
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
                    model = model_override or detector()
                    scene += 1
                elif previous is None:
                    scene = 0
                previous = gray.astype(float)
                tick = time.perf_counter()
                detections = model.detect(frame)
                detector_tracking_seconds += time.perf_counter() - tick
                sampled_frames += 1
                for item in detections:
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
    tracks = continuous_tracks(tracks, max_gap=max(1.5, 3 / SAMPLE_FPS))
    indexing_start = time.perf_counter()
    with session() as s:
        video = s.get(Video, video_id)
        camera = s.get(Camera, video.camera_id)
        track_entities = {}
        for key, samples in tracks.items():
            if len(samples) < 2:
                continue
            crop = crops[key.split("@")[0]]
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
                    confidence=obs.confidence if category == "OBSERVED" else None,
                    category=category,
                    attributes=attrs,
                    observation=obs.id,
                    method=model.name + " + geometric rules",
                )
        # Every interaction is derived from the same detector/tracker samples.
        tick = time.perf_counter()
        candidates = track_events(
            tracks,
            video.width,
            video.height,
            zones=camera.connections.get("zones", []),
        )
        event_extraction_seconds = time.perf_counter() - tick
        for candidate in candidates:
            primary = track_entities.get(candidate.keys[0])
            if primary is None:
                continue
            entities = [
                track_entities[key][0].id
                for key in candidate.keys
                if key in track_entities
            ]
            if len(entities) != len(candidate.keys):
                continue
            labels = [track_entities[key][0].object_type for key in candidate.keys]
            event = create_event(
                s,
                video,
                entities,
                candidate.kind,
                f"{' and '.join(labels).title()} {candidate.kind.replace('_', ' ')}",
                candidate.start,
                candidate.end,
                category=candidate.category,
                attributes=candidate.attributes,
                observation=primary[1].id,
                method=model.name + " + " + candidate.method,
            )
            if len(entities) == 2:
                s.add(
                    Relationship(
                        id=uid("REL"),
                        source_id=entities[0],
                        target_id=entities[1],
                        evidence_id=event.evidence_id,
                        relation=candidate.kind,
                        category=candidate.category,
                        signals={
                            "method": candidate.method,
                            "source_interval": [candidate.start, candidate.end],
                        },
                        explanation="Track geometry supports this hypothesis; proximity does not establish identity, possession, or intent.",
                    )
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
    wall = time.perf_counter() - begun
    return {
        "video_id": video_id,
        "video_duration_seconds": duration,
        "wall_seconds": wall,
        "decoded_frames": frame_idx,
        "sampled_frames": sampled_frames,
        "processing_fps": frame_idx / wall,
        "sampled_processing_fps": sampled_frames / wall,
        "decode_seconds": decode_seconds,
        "detector_and_tracker_seconds": detector_tracking_seconds,
        "event_extraction_seconds": event_extraction_seconds,
        "indexing_seconds_including_event_extraction": time.perf_counter()
        - indexing_start,
        "tracks": len(track_entities),
    }


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
