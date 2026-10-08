"""Transition rules and a pixel-derived upload-to-evidence regression."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
from app.db import Observation, Track, session
from app.demo_pipeline import ID as PIPELINE_ID
from app.processing import process
from app.retrieval import parse_query
from app.temporal import matches
from app.temporal_events import track_events
from sqlalchemy import select


REAL_VEHICLE_BOXES = json.loads(
    (Path(__file__).parent / "fixtures/meva_vehicle_transitions.json").read_text()
)["cases"]


def test_real_meva_deceleration_interval_precedes_verified_stationary_state():
    events = track_events({"car": REAL_VEHICLE_BOXES["stop_after_deceleration"]}, 1920, 1080)
    stops = [event for event in events if event.kind == "stopped"]
    assert len(stops) == 1
    assert 164 <= stops[0].start <= 166
    assert 168 <= stops[0].end <= 169
    assert stops[0].category == "INFERRED"


def test_real_meva_start_after_stationary_with_short_detection_gap():
    events = track_events({"car": REAL_VEHICLE_BOXES["start_after_stationary"]}, 1920, 1072)
    starts = [event for event in events if event.kind == "started_moving"]
    assert len(starts) == 1
    assert 18 <= starts[0].start <= 20
    assert 20 <= starts[0].end <= 22


def test_real_meva_parked_car_jitter_is_not_a_transition():
    events = track_events({"car": REAL_VEHICLE_BOXES["parked_jitter"]}, 1920, 1080)
    assert not {"started_moving", "stopped"} & {event.kind for event in events}


def test_vehicle_stop_followed_by_start_needs_sustained_state():
    positions = [
        10 + 20 * t if t <= 4 else 90 if t <= 8 else 90 + 20 * (t - 8)
        for t in range(15)
    ]
    car = [sample(t, x, "car", h=40, w=60) for t, x in enumerate(positions)]
    events = track_events({"car": car}, 500, 160)
    assert len([e for e in events if e.kind == "stopped"]) == 1
    assert len([e for e in events if e.kind == "started_moving"]) == 1
    assert all(e.category == "INFERRED" for e in events if e.kind in {"stopped", "started_moving"})


def test_person_passing_stationary_object_does_not_imply_pickup():
    person = [sample(t, 10 + 18 * t, "person") for t in range(12)]
    bag = [sample(t, 110, "backpack", 55, 15, 20) for t in range(12)]
    events = track_events({"person": person, "bag": bag}, 400, 160)
    assert not any(e.kind in {"picked_up", "placed_object"} for e in events)


def sample(t, x, kind, y=40, w=20, h=40):
    return {
        "t": float(t),
        "frame": t * 10,
        "box": [x, y, x + w, y + h],
        "object_type": kind,
        "confidence": None,
    }


def test_placement_and_unattended_require_motion_departure_and_duration():
    person = [
        sample(t, 100 + 10 * t if t <= 5 else 150 + 15 * (t - 5), "person")
        for t in range(16)
    ]
    bag = [
        sample(t, 115 + 10 * t if t <= 5 else 165, "backpack", 55, 15, 20)
        for t in range(16)
    ]
    events = track_events({"p": person, "o": bag}, 400, 160, unattended_seconds=3)
    placed = next(e for e in events if e.kind == "placed_object")
    unattended = next(e for e in events if e.kind == "unattended")
    assert placed.keys == ("p", "o") and placed.start == 5
    assert unattended.keys == ("o", "p")
    assert unattended.attributes["unattended_seconds"] >= 3
    assert unattended.category == "INFERRED"
    assert not any(
        e.kind == "placed_object"
        for e in track_events(
            {"p": person, "o": bag[5:]}, 400, 160, unattended_seconds=3
        )
    )
    assert not any(
        e.kind == "unattended"
        for e in track_events({"p": person, "o": bag}, 400, 160, unattended_seconds=20)
    )
    guard = [sample(t, 155, "person") for t in range(16)]
    assert not any(
        e.kind == "unattended"
        for e in track_events(
            {"p": person, "g": guard, "o": bag}, 400, 160, unattended_seconds=3
        )
    )


def test_pickup_needs_stationary_then_coupled_motion():
    person = [
        sample(t, 125 if t <= 5 else 125 + 8 * (t - 5), "person") for t in range(12)
    ]
    bag = [
        sample(t, 140 if t <= 5 else 140 + 8 * (t - 5), "backpack", 55, 15, 20)
        for t in range(12)
    ]
    events = track_events({"p": person, "o": bag}, 400, 160)
    assert any(e.kind == "picked_up" and e.start == 5 for e in events)
    assert any(e.kind == "started_moving" for e in events)
    assert not any(
        e.kind == "picked_up"
        for e in track_events(
            {"p": [sample(t, 10, "person") for t in range(12)], "o": bag}, 400, 160
        )
    )


def test_person_approach_and_move_away_from_another_person():
    first = [
        sample(t, 10 + 12 * t if t <= 6 else 82 + 20 * (t - 6), "person")
        for t in range(13)
    ]
    second = [sample(t, 100, "person") for t in range(13)]
    events = track_events({"first": first, "second": second}, 320, 160)
    assert any(
        e.kind == "approached_person" and e.keys == ("first", "second") for e in events
    )
    assert any(e.kind == "moved_away" and e.keys == ("first", "second") for e in events)


def test_zone_transitions_and_dwell():
    person = [
        sample(t, x, "person", y=40, h=40)
        for t, x in enumerate([0, 10, 30, 50, 70, 80, 80, 80, 80, 80, 80, 150])
    ]
    events = track_events(
        {"p": person}, 200, 160, zones=[{"name": "Gate", "rect": [0.25, 0.3, 0.6, 0.8]}]
    )
    assert {"entered_zone", "exited_zone", "remained_in_zone"} <= {
        e.kind for e in events
    }
    assert all(e.attributes.get("zone") == "Gate" for e in events if "zone" in e.kind)


def test_temporal_interval_operators_and_query_window():
    base = datetime(2026, 10, 6, tzinfo=timezone.utc)
    at = lambda n: base + timedelta(seconds=n)
    assert matches(at(0), at(5), at(10), at(20), "before", 5)
    assert matches(at(0), at(5), at(10), at(20), "followed_by", 5)
    assert not matches(at(0), at(5), at(10), at(20), "before", 4)
    assert matches(at(10), at(20), at(0), at(5), "after", 5)
    assert matches(at(12), at(15), at(10), at(20), "during")
    assert matches(at(0), at(12), at(10), at(20), "overlaps")
    assert matches(at(0), at(5), at(10), at(20), "near", 5)
    assert not matches(at(0), at(5), at(20), at(30), "within", 10)
    parsed = parse_query(
        "Find people who approached a vehicle within two minutes before it left"
    )
    assert (parsed["temporal_relation"], parsed["window_seconds"]) == ("before", 120)


def test_pipeline_demo_events_come_from_decoded_video(client):
    videos = client.get("/videos").json()
    video = next(v for v in videos if v["id"] == PIPELINE_ID)
    assert video["status"] == "completed" and "pixel contours" in video["pipeline"]
    result = client.post(
        "/search", json={"query": "unattended bag", "camera": "CAM-PIPELINE-DEMO"}
    ).json()
    assert any(e["event_type"] == "unattended" for e in result["results"])
    for kind, query in (
        ("placed_object", "person placed bag"),
        ("unattended", "unattended bag"),
        ("picked_up", "person picked up bag"),
    ):
        results = client.post(
            "/search", json={"query": query, "camera": "CAM-PIPELINE-DEMO"}
        ).json()["results"]
        event = next(
            e
            for e in results
            if e["video_id"] == PIPELINE_ID and e["event_type"] == kind
        )
        provenance = client.get(f"/evidence/{event['evidence']['id']}").json()
        assert provenance["track_id"] and provenance["source_boxes"]
        assert len(provenance["supporting_tracks"]) == 2
        assert all(
            track["track_id"] and track["boxes"]
            for track in provenance["supporting_tracks"]
        )
        assert "synthetic pixel contours" in provenance["method"]
        graph = client.get("/graph").json()
        assert any(
            edge.get("event_id") == event["id"]
            and edge["evidence_id"] == event["evidence"]["id"]
            for edge in graph["edges"]
        )


class PixelDetector:
    """Test adapter: contours are measured from decoded video pixels, not a script."""

    name = "pixel-color contour test adapter"

    def detect(self, frame):
        found = []
        for ident, kind, low, high in [
            (1, "person", (180, 0, 180), (255, 70, 255)),
            (2, "backpack", (0, 180, 0), (70, 255, 70)),
        ]:
            mask = cv2.inRange(frame, np.array(low, np.uint8), np.array(high, np.uint8))
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                if w * h > 80:
                    found.append(
                        {
                            "track_id": ident,
                            "object_type": kind,
                            "box": [x, y, x + w, y + h],
                            "confidence": None,
                        }
                    )
        return found


def test_uploaded_pixel_video_to_placement_search_and_source(
    client, tmp_path, monkeypatch
):
    monkeypatch.setattr("app.processing.detector", PixelDetector)
    monkeypatch.setenv("UNATTENDED_SECONDS", "2")
    path = tmp_path / "moving.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 160))
    for frame_number in range(160):
        t = frame_number / 10
        person_x = round(30 + 8 * t if t <= 5 else 70 + 16 * (t - 5))
        bag_x = round(45 + 8 * t if t <= 5 else 85)
        frame = np.zeros((160, 320, 3), dtype=np.uint8)
        cv2.rectangle(frame, (person_x, 40), (person_x + 20, 80), (240, 0, 240), -1)
        cv2.rectangle(frame, (bag_x, 66), (bag_x + 14, 86), (0, 240, 0), -1)
        writer.write(frame)
    writer.release()
    payload = path.read_bytes()
    response = client.post(
        "/videos/upload",
        data={
            "camera_id": "PIXEL-TEST",
            "recording_start": "2026-10-06T21:00:00+05:30",
            "location": "Test lab",
        },
        files={"file": ("moving.avi", payload)},
    )
    assert response.status_code == 201, response.text
    video_id = response.json()["id"]
    process(video_id)
    with session() as s:
        obs = s.scalars(
            select(Observation).where(Observation.video_id == video_id)
        ).all()
        assert {s.get(Track, o.track_id).tracker for o in obs} == {PixelDetector.name}
        assert len(obs) == 2 and all(o.boxes for o in obs)
    result = client.post(
        "/search", json={"query": "person placed bag", "camera": "PIXEL-TEST"}
    ).json()
    assert result["results"], result
    hit = result["results"][0]
    assert hit["event_type"] == "placed_object" and hit["category"] == "INFERRED"
    assert hit["confidence"] is None and hit["video_id"] == video_id
    ev = client.get(f"/evidence/{hit['evidence']['id']}").json()
    assert ev["observation"]["track_id"] and ev["frame_start"] >= 0
    assert (
        client.get(
            hit["media_url"].removeprefix("/api"), headers={"Range": "bytes=0-99"}
        ).status_code
        == 206
    )
    timeline = client.get(f"/entities/{hit['entities'][0]['id']}/timeline").json()
    assert any(e["id"] == hit["id"] for e in timeline)
    inv = client.post(
        "/investigations",
        json={
            "title": "Pixel video",
            "query": result["query"],
            "event_ids": [hit["id"]],
        },
    ).json()
    report = client.post(f"/investigations/{inv['id']}/report").json()["markdown"]
    assert hit["evidence"]["id"] in report and "INFERRED" in report


def test_pickup_uses_closest_person_not_every_nearby_person():
    person = [
        sample(t, 125 if t <= 5 else 125 + 8 * (t - 5), "person") for t in range(12)
    ]
    nearby = [
        sample(t, 145 if t <= 5 else 145 + 2 * (t - 5), "person") for t in range(12)
    ]
    bag = [
        sample(t, 140 if t <= 5 else 140 + 8 * (t - 5), "backpack", 55, 15, 20)
        for t in range(12)
    ]
    events = track_events({"p": person, "nearby": nearby, "o": bag}, 400, 160)
    pickups = [e for e in events if e.kind == "picked_up"]
    assert len(pickups) == 1
    assert pickups[0].keys == ("p", "o")


def test_pickup_ignores_detector_jitter_as_object_motion():
    person = [sample(t, 125 + 8 * max(0, t - 5), "person") for t in range(12)]
    jitter = [140, 141, 139, 141, 140, 142, 139, 141, 140, 141, 139, 140]
    bag = [sample(t, x, "backpack", 55, 15, 20) for t, x in enumerate(jitter)]
    events = track_events({"p": person, "o": bag}, 400, 160)
    assert not any(e.kind == "picked_up" for e in events)


def test_real_meva_static_vehicle_and_detection_gap_do_not_establish_stop():
    import json
    from pathlib import Path
    from app.temporal_events import continuous_tracks
    from app.processing import extract_events

    fixture = json.loads(
        (Path(__file__).parent / "fixtures/meva_vehicle_gap.json").read_text()
    )
    samples = fixture["samples"]
    segments = continuous_tracks({"car": samples})
    assert len(segments) == 2
    for segment in segments.values():
        assert not any(e[0] == "stopped" for e in extract_events(segment, 15))
        assert not any(
            e.kind in ("stopped", "started_moving")
            for e in track_events({"car": segment}, 1920, 1072)
        )


def test_pickup_rejects_opposite_motion_and_ambiguous_person():
    bag = [
        sample(t, 140 + 8 * max(0, t - 5), "backpack", 55, 15, 20) for t in range(12)
    ]
    opposite = [sample(t, 165 - 8 * max(0, t - 5), "person") for t in range(12)]
    assert not any(
        e.kind == "picked_up" for e in track_events({"p": opposite, "o": bag}, 400, 160)
    )
    person = [sample(t, 125 + 8 * max(0, t - 5), "person") for t in range(12)]
    assert not any(
        e.kind == "picked_up"
        for e in track_events({"p": person, "q": person, "o": bag}, 400, 160)
    )


def test_placement_requires_prior_coupling_and_sustained_separation():
    bag = [sample(t, 115 + 10 * min(t, 5), "backpack", 55, 15, 20) for t in range(16)]
    bystander = [
        sample(t, 150 if t <= 5 else 150 + 15 * (t - 5), "person") for t in range(16)
    ]
    assert not any(
        e.kind == "placed_object"
        for e in track_events({"p": bystander, "o": bag}, 400, 160)
    )
