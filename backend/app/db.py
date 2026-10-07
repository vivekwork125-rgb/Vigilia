from contextlib import contextmanager
from datetime import datetime, timezone
from uuid import uuid4
from sqlalchemy import (
    JSON,
    Boolean,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from .config import DATABASE_URL


def uid(prefix):
    return f"{prefix}-{uuid4().hex[:12].upper()}"


def now():
    return datetime.now(timezone.utc).isoformat()


class Base(DeclarativeBase):
    pass


class Location(Base):
    __tablename__ = "locations"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)


class Camera(Base):
    __tablename__ = "cameras"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    location_id: Mapped[str] = mapped_column(ForeignKey("locations.id"), index=True)
    name: Mapped[str] = mapped_column(String)
    connections: Mapped[dict] = mapped_column(JSON, default=dict)


class Video(Base):
    __tablename__ = "videos"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id"), index=True)
    filename: Mapped[str] = mapped_column(String)
    path: Mapped[str] = mapped_column(String)
    recording_start: Mapped[str] = mapped_column(String, index=True)
    duration: Mapped[float] = mapped_column(Float, default=0)
    fps: Mapped[float] = mapped_column(Float, default=0)
    frame_count: Mapped[int] = mapped_column(Integer, default=0)
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    sha256: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="uploaded", index=True)
    progress: Mapped[float] = mapped_column(Float, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)
    pipeline: Mapped[str] = mapped_column(String, default="pending")
    created_at: Mapped[str] = mapped_column(String, default=now)


class Entity(Base):
    __tablename__ = "entities"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id"), index=True)
    object_type: Mapped[str] = mapped_column(String, index=True)
    attributes: Mapped[dict] = mapped_column(JSON, default=dict)


class Track(Base):
    __tablename__ = "tracks"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id"), index=True)
    entity_id: Mapped[str] = mapped_column(ForeignKey("entities.id"), index=True)
    tracker: Mapped[str] = mapped_column(String)


class Observation(Base):
    __tablename__ = "observations"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id"), index=True)
    entity_id: Mapped[str] = mapped_column(ForeignKey("entities.id"), index=True)
    track_id: Mapped[str] = mapped_column(ForeignKey("tracks.id"), index=True)
    start: Mapped[float] = mapped_column(Float)
    end: Mapped[float] = mapped_column(Float)
    boxes: Mapped[list] = mapped_column(JSON, default=list)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    attributes: Mapped[dict] = mapped_column(JSON, default=dict)


class Embedding(Base):
    __tablename__ = "embeddings"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    observation_id: Mapped[str] = mapped_column(
        ForeignKey("observations.id"), index=True
    )
    model: Mapped[str] = mapped_column(String, index=True)
    values: Mapped[list] = mapped_column(JSON)


class Evidence(Base):
    __tablename__ = "evidence"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id"), index=True)
    observation_id: Mapped[str | None] = mapped_column(
        ForeignKey("observations.id"), nullable=True, index=True
    )
    frame_start: Mapped[int] = mapped_column(Integer)
    frame_end: Mapped[int] = mapped_column(Integer)
    timestamp_start: Mapped[float] = mapped_column(Float)
    timestamp_end: Mapped[float] = mapped_column(Float)
    sha256: Mapped[str] = mapped_column(String)
    method: Mapped[str] = mapped_column(String)


class Event(Base):
    __tablename__ = "events"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    evidence_id: Mapped[str] = mapped_column(ForeignKey("evidence.id"), index=True)
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id"), index=True)
    event_type: Mapped[str] = mapped_column(String, index=True)
    title: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(Text)
    start: Mapped[str] = mapped_column(String, index=True)
    end: Mapped[str] = mapped_column(String)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    category: Mapped[str] = mapped_column(String, default="OBSERVED")
    attributes: Mapped[dict] = mapped_column(JSON, default=dict)


class EventEntity(Base):
    __tablename__ = "event_entities"
    event_id: Mapped[str] = mapped_column(ForeignKey("events.id"), primary_key=True)
    entity_id: Mapped[str] = mapped_column(ForeignKey("entities.id"), primary_key=True)


class Relationship(Base):
    __tablename__ = "relationships"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    source_id: Mapped[str] = mapped_column(ForeignKey("entities.id"), index=True)
    target_id: Mapped[str] = mapped_column(ForeignKey("entities.id"), index=True)
    evidence_id: Mapped[str] = mapped_column(ForeignKey("evidence.id"), index=True)
    relation: Mapped[str] = mapped_column(String)
    category: Mapped[str] = mapped_column(String)
    signals: Mapped[dict] = mapped_column(JSON)
    explanation: Mapped[str] = mapped_column(Text)


class Investigation(Base):
    __tablename__ = "investigations"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(String)
    query: Mapped[str] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(String, default=now)


class Finding(Base):
    __tablename__ = "investigation_findings"
    investigation_id: Mapped[str] = mapped_column(
        ForeignKey("investigations.id"), primary_key=True
    )
    event_id: Mapped[str] = mapped_column(ForeignKey("events.id"), primary_key=True)


class BenchmarkQuery(Base):
    __tablename__ = "benchmark_queries"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    query: Mapped[str] = mapped_column(Text)
    case: Mapped[str] = mapped_column(String)


class GroundTruth(Base):
    __tablename__ = "benchmark_ground_truth"
    query_id: Mapped[str] = mapped_column(
        ForeignKey("benchmark_queries.id"), primary_key=True
    )
    event_id: Mapped[str] = mapped_column(ForeignKey("events.id"), primary_key=True)


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[str] = mapped_column(String, default=now)
    results: Mapped[dict] = mapped_column(JSON)


class ReviewTiming(Base):
    __tablename__ = "review_timings"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    task: Mapped[str] = mapped_column(String)
    mode: Mapped[str] = mapped_column(String)
    seconds: Mapped[float] = mapped_column(Float)


engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
    if DATABASE_URL.startswith("sqlite")
    else {},
    pool_pre_ping=True,
)
if DATABASE_URL.startswith("sqlite"):

    @event.listens_for(engine, "connect")
    def sqlite_pragmas(conn, _):
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=15000")


Session = sessionmaker(engine, expire_on_commit=False)


@contextmanager
def session():
    with Session() as s:
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise


def initialize():
    Base.metadata.create_all(engine)


def row(obj):
    return {c.name: getattr(obj, c.name) for c in obj.__table__.columns}


# Native vector storage on PostgreSQL, JSON on SQLite. The canonical embedding
# record retains its model identifier so incompatible spaces are never mixed.
from pgvector.sqlalchemy import Vector


class EmbeddingVector(Base):
    __tablename__ = "embedding_vectors"
    embedding_id: Mapped[str] = mapped_column(
        ForeignKey("embeddings.id"), primary_key=True
    )
    model: Mapped[str] = mapped_column(String, index=True)
    vector: Mapped[list] = mapped_column(JSON().with_variant(Vector(), "postgresql"))


@event.listens_for(Embedding, "after_insert")
def store_native_vector(mapper, connection, target):
    connection.execute(
        EmbeddingVector.__table__.insert().values(
            embedding_id=target.id, model=target.model, vector=target.values
        )
    )
