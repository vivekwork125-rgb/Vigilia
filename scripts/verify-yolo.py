"""Real-model smoke test using Ultralytics' packaged bus example.
A repeated still frame checks integration, not surveillance accuracy.
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data" / "yolo-smoke"
RUN.mkdir(parents=True, exist_ok=True)
os.environ["DATA_DIR"] = str(RUN)
os.environ["DATABASE_URL"] = f"sqlite:///{RUN / 'smoke.db'}"
os.environ["DEMO_MODE"] = "false"
os.environ["PERCEPTION_BACKEND"] = "yolo"
os.environ["YOLO_MODEL"] = str(ROOT / "models" / "yolo11n.pt")
os.environ["YOLO_CONFIG_DIR"] = str(RUN / "config")
os.environ["MPLCONFIGDIR"] = str(RUN / "matplotlib")
sys.path.insert(0, str(ROOT / "backend"))
import cv2
import hashlib
import ultralytics
from sqlalchemy import select
from app.db import (
    initialize,
    session,
    Location,
    Camera,
    Video,
    Observation,
    Event,
    Evidence,
    uid,
)
from app.processing import process
from app.retrieval import search

source = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
frame = cv2.imread(str(source))
if frame is None:
    raise RuntimeError("Ultralytics packaged bus example not found")
frame = cv2.resize(frame, (608, 810))
vid = uid("SMOKE")
path = RUN / f"{vid}.avi"
writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (608, 810))
for _ in range(50):
    writer.write(frame)
writer.release()
initialize()
with session() as s:
    if not s.get(Location, "SMOKE-LOC"):
        s.add(Location(id="SMOKE-LOC", name="Packaged detector example"))
        s.flush()
        s.add(
            Camera(
                id="SMOKE-CAM",
                name="Static replay",
                location_id="SMOKE-LOC",
                connections={},
            )
        )
        s.flush()
    s.add(
        Video(
            id=vid,
            camera_id="SMOKE-CAM",
            filename=path.name,
            path=str(path),
            recording_start="2026-10-06T18:00:00+05:30",
            duration=5,
            fps=10,
            frame_count=50,
            width=608,
            height=810,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            status="uploaded",
            is_demo=False,
            pipeline="pending",
        )
    )
process(vid)
with session() as s:
    observations = s.scalars(
        select(Observation).where(Observation.video_id == vid)
    ).all()
    events = s.scalars(
        select(Event)
        .join(Evidence, Event.evidence_id == Evidence.id)
        .where(Evidence.video_id == vid)
    ).all()
    result = search(s, "person stopped", camera="SMOKE-CAM")
    assert observations and events and result["results"]
    report = {
        "test": "YOLO11n + ByteTrack / actual pretrained model integration",
        "source": "Ultralytics packaged assets/bus.jpg, repeated for 5 seconds",
        "not_a_video_accuracy_benchmark": True,
        "video_id": vid,
        "observations": len(observations),
        "events": len(events),
        "person_stopped_results": len(result["results"]),
        "pipeline": s.get(Video, vid).pipeline,
        "event_types": sorted(set(x.event_type for x in events)),
    }
(ROOT / "docs" / "model-smoke-results.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report, indent=2))
