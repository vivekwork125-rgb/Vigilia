"""Provenance construction and serialization shared by every retrieval route."""

import hashlib
import json
from datetime import datetime, timedelta
from sqlalchemy import select
from .db import (
    Camera,
    Entity,
    Event,
    EventEntity,
    Evidence,
    Location,
    Observation,
    Video,
    row,
    uid,
)


def absolute(video, seconds):
    return (
        datetime.fromisoformat(video.recording_start) + timedelta(seconds=seconds)
    ).isoformat()


def source_frame(seconds, fps, frame_count):
    """Nearest decoded source frame; avoid float truncation before a sample."""
    return min(frame_count - 1, max(0, round(seconds * fps)))


def create_event(
    s,
    video,
    entities,
    kind,
    title,
    start,
    end,
    *,
    confidence=None,
    category="OBSERVED",
    attributes=None,
    observation=None,
    method="rule-engine",
    event_id=None,
):
    if not 0 <= start <= end <= video.duration:
        raise ValueError("Event interval must lie inside the source video")
    a = source_frame(start, video.fps, video.frame_count)
    b = source_frame(end, video.fps, video.frame_count)
    payload = {
        "video": video.id,
        "source_sha256": video.sha256,
        "frame_start": a,
        "frame_end": b,
        "method": method,
    }
    ev = Evidence(
        id=uid("EVD"),
        video_id=video.id,
        observation_id=observation,
        frame_start=a,
        frame_end=b,
        timestamp_start=start,
        timestamp_end=end,
        sha256=hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
        method=method,
    )
    s.add(ev)
    s.flush()
    event = Event(
        id=event_id or uid("EVT"),
        evidence_id=ev.id,
        camera_id=video.camera_id,
        event_type=kind,
        title=title,
        description=title,
        start=absolute(video, start),
        end=absolute(video, end),
        confidence=confidence,
        category=category,
        attributes=attributes or {},
    )
    s.add(event)
    s.flush()
    for entity in entities:
        s.add(EventEntity(event_id=event.id, entity_id=entity))
    return event


def serialize_event(s, event):
    result = row(event)
    evidence = s.get(Evidence, event.evidence_id)
    video = s.get(Video, evidence.video_id)
    camera = s.get(Camera, event.camera_id)
    entities = s.scalars(
        select(Entity)
        .join(EventEntity, Entity.id == EventEntity.entity_id)
        .where(EventEntity.event_id == event.id)
    ).all()
    primary_observation = (
        s.get(Observation, evidence.observation_id) if evidence.observation_id else None
    )
    if primary_observation:
        entities.sort(key=lambda entity: entity.id != primary_observation.entity_id)
    result.update(
        entities=[row(x) for x in entities],
        evidence=row(evidence),
        video_id=video.id,
        is_demo=video.is_demo,
        location=s.get(Location, camera.location_id).name,
        camera_name=camera.name,
        source_sha256=video.sha256,
        media_url=f"/api/videos/{video.id}/media",
        thumbnail_url=f"/api/evidence/{evidence.id}/thumbnail",
    )
    return result


def entity_timeline(s, entity_id):
    events = s.scalars(
        select(Event)
        .join(EventEntity)
        .where(EventEntity.entity_id == entity_id)
        .order_by(Event.start)
    ).all()
    return [serialize_event(s, x) for x in events]
