import hashlib
import hmac
import logging
import subprocess
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal
import cv2
import imageio_ffmpeg
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, Header
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from . import config
from .db import (
    Camera,
    Embedding,
    Entity,
    EvaluationRun,
    Event,
    EventEntity,
    Evidence,
    Finding,
    Investigation,
    Location,
    Observation,
    Relationship,
    ReviewTiming,
    Video,
    initialize,
    now,
    row,
    session,
    uid,
)
from .demo import seed
from .demo_pipeline import seed_pipeline_demo
from .evidence import serialize_event, entity_timeline
from .evaluation import run_evaluation
from .processing import start_worker, stop
from .retrieval import search, associations
from .vision import appearance

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app):
    if not config.DEMO_MODE and len(config.TOKEN) < 24:
        raise RuntimeError(
            "Non-demo mode requires API_TOKEN with at least 24 characters."
        )
    initialize()
    if config.DEMO_MODE:
        seed()
        seed_pipeline_demo()
    worker = start_worker()
    yield
    stop.set()
    worker.join(timeout=3)


async def authorize(authorization: str | None = Header(default=None)):
    if config.TOKEN and not hmac.compare_digest(
        authorization or "", "Bearer " + config.TOKEN
    ):
        raise HTTPException(401, "Valid access token required")


app = FastAPI(
    title="VIGILIA Evidence API",
    version="0.1.0",
    lifespan=lifespan,
    dependencies=[Depends(authorize)],
)


def require(s, model, key):
    obj = s.get(model, key)
    if obj is None:
        raise HTTPException(404, f"{model.__name__} not found")
    return obj


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    camera: str | None = None
    category: Literal["OBSERVED", "CORRELATED", "INFERRED"] | None = None
    entity_id: str | None = None
    reference_event_id: str | None = None
    limit: int = Field(default=30, ge=1, le=100)


class InvestigationRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    query: str = Field(max_length=2000)
    event_ids: list[str] = Field(default_factory=list, max_length=100)


class FindingRequest(BaseModel):
    event_id: str


class TimingRequest(BaseModel):
    task: str = Field(min_length=1, max_length=200)
    mode: Literal["manual", "assisted"]
    seconds: float = Field(gt=0, le=86400)


class ZoneRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    rect: list[float] = Field(min_length=4, max_length=4)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "demo_mode": config.DEMO_MODE,
        "access_control": bool(config.TOKEN),
    }


@app.get("/overview")
def overview():
    with session() as s:
        videos = s.scalars(select(Video)).all()
        return {
            "cameras": s.scalar(select(func.count()).select_from(Camera)),
            "videos": len(videos),
            "indexed_seconds": sum(
                v.duration for v in videos if v.status == "completed"
            ),
            "events": s.scalar(select(func.count()).select_from(Event)),
            "entities": s.scalar(select(func.count()).select_from(Entity)),
            "demo_mode": config.DEMO_MODE,
            "recent_events": [
                serialize_event(s, e)
                for e in s.scalars(
                    select(Event).order_by(Event.start.desc()).limit(6)
                ).all()
            ],
            "investigations": [
                row(x)
                for x in s.scalars(
                    select(Investigation).order_by(Investigation.created_at.desc())
                ).all()
            ],
            "pipeline": __import__("os").getenv("PERCEPTION_BACKEND", "hog"),
        }


@app.get("/cameras")
def cameras():
    with session() as s:
        return [
            {**row(c), "location": s.get(Location, c.location_id).name}
            for c in s.scalars(select(Camera)).all()
        ]


@app.post("/cameras/{camera_id}/zones")
def zone(camera_id: str, body: ZoneRequest):
    a, b, c, d = body.rect
    if not (0 <= a < c <= 1 and 0 <= b < d <= 1):
        raise HTTPException(
            422, "Zone rectangle must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1"
        )
    with session() as s:
        camera = require(s, Camera, camera_id)
        data = dict(camera.connections)
        data["zones"] = [*data.get("zones", []), body.model_dump()]
        camera.connections = data
        return row(camera)


@app.get("/videos")
def videos():
    with session() as s:
        return [
            {k: v for k, v in row(x).items() if k != "path"}
            for x in s.scalars(select(Video).order_by(Video.created_at.desc())).all()
        ]


@app.post("/videos/upload", status_code=201)
async def upload(
    file: UploadFile = File(...),
    camera_id: str = Form(...),
    recording_start: str = Form(...),
    location: str = Form("Unspecified"),
):
    if len(camera_id) > 80 or len(location) > 160:
        raise HTTPException(422, "Camera ID or location is too long")
    if not camera_id.strip():
        raise HTTPException(422, "Camera ID is required")
    try:
        start = datetime.fromisoformat(recording_start)
        if start.tzinfo is None:
            raise ValueError()
    except ValueError:
        raise HTTPException(422, "Recording start must be ISO 8601 with timezone")
    ext = Path(file.filename or "").suffix.lower()
    if ext not in (".mp4", ".mov", ".avi", ".mkv"):
        raise HTTPException(415, "Supported extensions: MP4, MOV, AVI, MKV")
    vid = uid("VID")
    path = config.MEDIA / (vid + ext)
    size = 0
    digest = hashlib.sha256()
    try:
        with path.open("wb") as dest:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > config.MAX_UPLOAD:
                    raise HTTPException(413, "Upload exceeds configured size limit")
                digest.update(chunk)
                dest.write(chunk)
        cap = cv2.VideoCapture(str(path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ok, _ = cap.read()
        finally:
            cap.release()
        if not ok or not np.isfinite(fps) or fps <= 0 or frames <= 0:
            raise HTTPException(
                422, "Corrupted video or unsupported codec; no frames could be decoded"
            )
        if frames / fps > 7200 or width * height > 3840 * 2160:
            raise HTTPException(
                422, "Prototype limit: two hours per file and 4K resolution"
            )
        with session() as s:
            camera = s.get(Camera, camera_id)
            if not camera:
                loc = Location(id=uid("LOC"), name=location)
                s.add(loc)
                s.flush()
                s.add(
                    Camera(
                        id=camera_id, name=location, location_id=loc.id, connections={}
                    )
                )
                s.flush()
            video = Video(
                id=vid,
                camera_id=camera_id,
                filename=Path(file.filename).name,
                path=str(path),
                recording_start=start.isoformat(),
                duration=frames / fps,
                fps=fps,
                frame_count=frames,
                width=width,
                height=height,
                sha256=digest.hexdigest(),
                status="uploaded",
                progress=0,
                is_demo=False,
                pipeline="pending",
            )
            s.add(video)
            s.flush()
            return {k: v for k, v in row(video).items() if k != "path"}
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()


@app.get("/videos/{video_id}")
@app.get("/videos/{video_id}/status")
def video_status(video_id: str):
    with session() as s:
        return {
            k: v for k, v in row(require(s, Video, video_id)).items() if k != "path"
        }


@app.post("/videos/{video_id}/index", status_code=202)
def queue_video(video_id: str):
    with session() as s:
        video = require(s, Video, video_id)
        if video.status in (
            "completed",
            "queued",
            "processing",
            "detecting",
            "indexing",
        ):
            return {
                "id": video.id,
                "status": video.status,
                "message": "Already indexed or scheduled; no duplicate job created",
            }
        video.status = "queued"
        video.error = None
        video.progress = 0
        return {"id": video.id, "status": "queued"}


@app.get("/videos/{video_id}/media")
def media(video_id: str):
    with session() as s:
        video = require(s, Video, video_id)
        if not Path(video.path).exists():
            raise HTTPException(404, "Source artifact is missing")
        # Transcode non-browser codecs on demand; original hash still references the upload.
        path = Path(video.path)
        if not video.is_demo:
            playable = config.MEDIA / (video.id + "-playback.mp4")
            if not playable.exists():
                temporary = config.MEDIA / (uid("PLAYBACK") + ".mp4")
                try:
                    subprocess.run(
                        [
                            imageio_ffmpeg.get_ffmpeg_exe(),
                            "-y",
                            "-i",
                            str(path),
                            "-an",
                            "-c:v",
                            "libx264",
                            "-preset",
                            "veryfast",
                            "-pix_fmt",
                            "yuv420p",
                            "-movflags",
                            "+faststart",
                            str(temporary),
                        ],
                        check=True,
                        capture_output=True,
                        timeout=300,
                    )
                    temporary.replace(playable)
                except (subprocess.SubprocessError, OSError):
                    temporary.unlink(missing_ok=True)
                    raise HTTPException(
                        422,
                        "Browser playback conversion failed; original source retained",
                    )
            path = playable
        return FileResponse(path, media_type="video/mp4")


@app.post("/search")
def search_route(body: SearchRequest):
    with session() as s:
        if body.entity_id:
            require(s, Entity, body.entity_id)
        if body.reference_event_id:
            require(s, Event, body.reference_event_id)
        return search(s, **body.model_dump())


@app.post("/search/image")
async def image_search(file: UploadFile = File(...)):
    payload = await file.read(10 * 1024 * 1024 + 1)
    await file.close()
    if len(payload) > 10 * 1024 * 1024:
        raise HTTPException(413, "Reference image must be under 10 MB")
    frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(422, "Unable to decode reference image")
    vector = appearance(frame)
    results = []
    with session() as s:
        for emb in s.scalars(
            select(Embedding).where(Embedding.model == "HSV-48")
        ).all():
            similarity = max(0, float(np.dot(vector, emb.values)))
            if similarity < 0.45:
                continue
            obs = s.get(Observation, emb.observation_id)
            event = s.scalar(
                select(Event)
                .join(Evidence, Event.evidence_id == Evidence.id)
                .where(Evidence.observation_id == obs.id)
                .order_by(Event.start)
            )
            if event:
                results.append(
                    {
                        **serialize_event(s, event),
                        "score": round(similarity, 4),
                        "signals": {"appearance_cosine": round(similarity, 4)},
                        "explanations": [
                            "Measured HSV color-histogram similarity; not identity verification."
                        ],
                    }
                )
        results.sort(key=lambda x: -x["score"])
        return {
            "results": results[:30],
            "total": len(results),
            "query": "Reference image",
            "parsed": {
                "method": "HSV-48 appearance baseline",
                "warnings": [
                    "Crop the reference to the object. Color similarity alone cannot identify a person."
                ],
            },
            "candidates_evaluated": s.scalar(
                select(func.count()).select_from(Embedding)
            ),
            "elapsed_ms": None,
            "score_notice": "Appearance similarity only; no identity claim.",
        }


@app.get("/events/{event_id}")
def get_event(event_id: str):
    with session() as s:
        return serialize_event(s, require(s, Event, event_id))


@app.get("/observations/{observation_id}")
def observation(observation_id: str):
    with session() as s:
        return row(require(s, Observation, observation_id))


@app.get("/entities/{entity_id}/timeline")
def timeline(entity_id: str):
    with session() as s:
        require(s, Entity, entity_id)
        return entity_timeline(s, entity_id)


@app.get("/entities/{entity_id}/relationships")
def relationships(entity_id: str):
    with session() as s:
        require(s, Entity, entity_id)
        direct = [
            row(r)
            for r in s.scalars(
                select(Relationship).where(
                    (Relationship.source_id == entity_id)
                    | (Relationship.target_id == entity_id)
                )
            ).all()
        ]
        return {"direct": direct, "associations": associations(s, entity_id)}


@app.get("/graph")
def graph(event_id: str | None = None):
    with session() as s:
        entities = s.scalars(select(Entity)).all()
        events = (
            [require(s, Event, event_id)]
            if event_id
            else s.scalars(select(Event).order_by(Event.start).limit(200)).all()
        )
        cameras = s.scalars(select(Camera)).all()
        locations = s.scalars(select(Location)).all()
        relationships = s.scalars(
            select(Relationship).where(
                Relationship.evidence_id == events[0].evidence_id
            )
            if event_id
            else select(Relationship)
        ).all()
        event_by_evidence = {
            evidence_id: event_id
            for event_id, evidence_id in s.execute(
                select(Event.id, Event.evidence_id)
            ).all()
        }
        nodes = [
            {
                "id": e.id,
                "label": e.id,
                "kind": e.object_type,
                "category": "OBSERVED",
                "camera_id": e.camera_id,
            }
            for e in entities
        ]
        edges = []
        nodes.extend(
            {
                "id": c.id,
                "label": c.name,
                "kind": "camera",
                "category": "OBSERVED",
                "camera_id": c.id,
            }
            for c in cameras
        )
        nodes.extend(
            {
                "id": loc.id,
                "label": loc.name,
                "kind": "location",
                "category": "OBSERVED",
                "camera_id": "",
            }
            for loc in locations
        )
        edges.extend(
            {
                "source": c.id,
                "target": c.location_id,
                "label": "located_at",
                "category": "OBSERVED",
                "evidence_id": None,
                "event_id": None,
            }
            for c in cameras
        )
        edges.extend(
            {
                "source": e.id,
                "target": e.camera_id,
                "label": "observed_at",
                "category": "OBSERVED",
                "evidence_id": None,
                "event_id": None,
            }
            for e in entities
        )
        for e in events:
            nodes.append(
                {
                    "id": e.id,
                    "label": e.title,
                    "kind": "event",
                    "category": e.category,
                    "camera_id": e.camera_id,
                }
            )
            for eid in s.scalars(
                select(EventEntity.entity_id).where(EventEntity.event_id == e.id)
            ).all():
                edges.append(
                    {
                        "source": eid,
                        "target": e.id,
                        "label": e.event_type,
                        "category": e.category,
                        "evidence_id": e.evidence_id,
                    }
                )
        for rel in relationships:
            edges.append(
                {
                    "source": rel.source_id,
                    "target": rel.target_id,
                    "label": rel.relation,
                    "category": rel.category,
                    "evidence_id": rel.evidence_id,
                    "event_id": event_by_evidence.get(rel.evidence_id),
                }
            )
        return {"nodes": nodes, "edges": edges}


@app.get("/evidence/{evidence_id}")
def evidence(evidence_id: str):
    with session() as s:
        item = require(s, Evidence, evidence_id)
        observation = (
            s.get(Observation, item.observation_id) if item.observation_id else None
        )
        source_boxes = (
            [
                box
                for box in observation.boxes
                if item.frame_start <= box["frame"] <= item.frame_end
            ]
            if observation
            else []
        )
        event = s.scalar(select(Event).where(Event.evidence_id == item.id))
        entity_ids = (
            s.scalars(
                select(EventEntity.entity_id).where(EventEntity.event_id == event.id)
            ).all()
            if event
            else []
        )
        supporting = (
            s.scalars(
                select(Observation).where(
                    Observation.video_id == item.video_id,
                    Observation.entity_id.in_(entity_ids),
                )
            ).all()
            if entity_ids
            else []
        )
        return {
            **row(item),
            "observation": row(observation) if observation else None,
            "track_id": observation.track_id if observation else None,
            "source_boxes": source_boxes,
            "supporting_tracks": [
                {
                    "entity_id": obs.entity_id,
                    "track_id": obs.track_id,
                    "boxes": [
                        box
                        for box in obs.boxes
                        if item.frame_start <= box["frame"] <= item.frame_end
                    ],
                }
                for obs in supporting
            ],
            "source": {
                k: v for k, v in row(s.get(Video, item.video_id)).items() if k != "path"
            },
        }


@app.get("/evidence/{evidence_id}/thumbnail")
def thumbnail(evidence_id: str):
    with session() as s:
        item = require(s, Evidence, evidence_id)
        video = s.get(Video, item.video_id)
        path = config.MEDIA / (item.id + ".jpg")
        if not path.exists():
            cap = cv2.VideoCapture(video.path)
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, item.frame_start)
                ok, frame = cap.read()
            finally:
                cap.release()
            if not ok:
                raise HTTPException(422, "Source frame unavailable")
            cv2.imwrite(str(path), frame)
        return FileResponse(path, media_type="image/jpeg")


@app.get("/evidence/{evidence_id}/clip")
def clip(evidence_id: str):
    with session() as s:
        item = require(s, Evidence, evidence_id)
        video = s.get(Video, item.video_id)
        path = config.MEDIA / (item.id + "-clip.mp4")
        if not path.exists():
            temporary = config.MEDIA / (uid("CLIP") + ".mp4")
            try:
                subprocess.run(
                    [
                        imageio_ffmpeg.get_ffmpeg_exe(),
                        "-y",
                        "-ss",
                        str(max(0, item.timestamp_start - 2)),
                        "-i",
                        video.path,
                        "-t",
                        str(max(3, item.timestamp_end - item.timestamp_start + 4)),
                        "-an",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        str(temporary),
                    ],
                    check=True,
                    capture_output=True,
                    timeout=90,
                )
                temporary.replace(path)
            except (subprocess.SubprocessError, OSError):
                temporary.unlink(missing_ok=True)
                raise HTTPException(
                    422, "Clip generation failed; original source retained"
                )
        return FileResponse(path, media_type="video/mp4", filename=f"{evidence_id}.mp4")


@app.post("/investigations", status_code=201)
def create_investigation(body: InvestigationRequest):
    with session() as s:
        for eid in body.event_ids:
            require(s, Event, eid)
        inv = Investigation(id=uid("INV"), title=body.title, query=body.query)
        s.add(inv)
        s.flush()
        for eid in set(body.event_ids):
            s.add(Finding(investigation_id=inv.id, event_id=eid))
        return row(inv)


@app.get("/investigations")
def investigations():
    with session() as s:
        return [
            row(x)
            for x in s.scalars(
                select(Investigation).order_by(Investigation.created_at.desc())
            ).all()
        ]


@app.get("/investigations/{investigation_id}")
def investigation(investigation_id: str):
    with session() as s:
        inv = require(s, Investigation, investigation_id)
        return {
            **row(inv),
            "findings": [
                serialize_event(s, require(s, Event, x))
                for x in s.scalars(
                    select(Finding.event_id).where(Finding.investigation_id == inv.id)
                ).all()
            ],
        }


@app.post("/investigations/{investigation_id}/findings")
def add_finding(investigation_id: str, body: FindingRequest):
    with session() as s:
        require(s, Investigation, investigation_id)
        require(s, Event, body.event_id)
        if not s.get(Finding, (investigation_id, body.event_id)):
            s.add(Finding(investigation_id=investigation_id, event_id=body.event_id))
        return {"status": "saved"}


@app.post("/investigations/{investigation_id}/report")
def report(investigation_id: str):
    data = investigation(investigation_id)
    lines = [
        "# VIGILIA — Investigation report",
        "",
        f"Investigation: {data['title']}",
        f"Query: {data['query']}",
        f"Generated: {now()}",
        "",
        "Scores indicate retrieval relevance or detector output, not identity probability.",
    ]
    for category in ("OBSERVED", "CORRELATED", "INFERRED"):
        lines += ["", f"## {category}"]
        items = [e for e in data["findings"] if e["category"] == category]
        if not items:
            lines.append("No findings collected in this category.")
        for item in sorted(items, key=lambda x: x["start"]):
            e = item["evidence"]
            lines += [
                f"- {item['start']} — {item['title']}",
                f"  Event: {item['id']}; evidence: {e['id']}; camera: {item['camera_id']}; video: {item['video_id']}",
                f"  Frames: {e['frame_start']}–{e['frame_end']}; seconds: {e['timestamp_start']}–{e['timestamp_end']}",
                f"  Entities: {', '.join(x['id'] for x in item['entities'])}; detector confidence: {item['confidence'] if item['confidence'] is not None else 'not measured (annotation or uncalibrated detector)'}",
                f"  Source SHA-256: {item['source_sha256']}",
                f"  Method: {e['method']}; synthetic footage: {item['is_demo']}",
            ]
    lines += [
        "",
        "## Unresolved ambiguities",
        "Cross-camera similarity does not establish identity. Missing camera intervals prevent continuous tracking claims. Inferred events require visual verification. Synthetic annotations do not validate perception accuracy.",
    ]
    return {"markdown": "\n".join(lines), "investigation_id": investigation_id}


@app.post("/evaluation/run")
def evaluate():
    with session() as s:
        return run_evaluation(s)


@app.get("/evaluation/results")
def evaluations():
    with session() as s:
        return [
            row(x)
            for x in s.scalars(
                select(EvaluationRun)
                .order_by(EvaluationRun.created_at.desc())
                .limit(10)
            ).all()
        ]


@app.post("/evaluation/timings", status_code=201)
def timing(body: TimingRequest):
    with session() as s:
        item = ReviewTiming(id=uid("TIME"), **body.model_dump())
        s.add(item)
        s.flush()
        return row(item)
