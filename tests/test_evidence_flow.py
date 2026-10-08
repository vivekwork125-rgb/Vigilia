import hashlib
import time
import cv2
import numpy as np
import pytest
from sqlalchemy import select
from app.db import session, Event, Evidence, Video, Observation
from app.retrieval import parse_query
from app.processing import extract_events, process, sampled_frame
from app.vision import iou
from app.evidence import source_frame


def test_meva_sample_times_round_to_the_actual_source_frame():
    # These are measured frames from the 3.75 FPS diagnostic. Truncation put
    # their evidence one frame before the source observation.
    assert source_frame(32.8, 30, 9001) == 984
    assert source_frame(65.86666666666666, 30, 9001) == 1976
    assert source_frame(135.2, 30, 9001) == 4056


def test_sampling_schedule_uses_requested_rate_when_fps_ratio_is_fractional():
    assert [sampled_frame(n, 30, 4) for n in range(6)] == [0, 8, 15, 22, 30, 38]
    assert sampled_frame(1200, 30, 4) == 9000
    assert [sampled_frame(n, 30, 2) for n in range(4)] == [0, 15, 30, 45]


def test_search_to_source_and_report(client):
    result = client.post(
        "/search",
        json={"query": "Find the person who left an object near the east entrance"},
    ).json()
    assert result["results"][0]["id"] == "EVT-E03"
    hit = result["results"][0]
    source = client.get(f"/evidence/{hit['evidence']['id']}").json()
    assert source["frame_start"] == 120 and source["frame_end"] == 150
    assert source["source"]["sha256"] == hit["source_sha256"]
    response = client.get(
        hit["media_url"].removeprefix("/api"), headers={"Range": "bytes=0-99"}
    )
    assert response.status_code == 206 and len(response.content) == 100
    timeline = client.get("/entities/P-E01/timeline").json()
    assert [x["id"] for x in timeline] == ["EVT-E01", "EVT-E02", "EVT-E03", "EVT-E05"]
    inv = client.post(
        "/investigations",
        json={"title": "Test case", "query": result["query"], "event_ids": [hit["id"]]},
    ).json()
    report = client.post(f"/investigations/{inv['id']}/report").json()["markdown"]
    assert (
        hit["evidence"]["id"] in report
        and "Source SHA-256" in report
        and "synthetic footage: True" in report
    )


@pytest.mark.parametrize(
    "query,expected",
    [
        ("Find a purple bus", []),
        ("Find people at Camera 5 between 7 and 8 PM", []),
        ("Vehicles stopped for more than two minutes", []),
        (
            "Find people who approached a vehicle within two minutes before it left",
            ["EVT-C03"],
        ),
        ("Find red people who entered before 6:01 PM", ["EVT-W01", "EVT-W04"]),
    ],
)
def test_constraints(client, query, expected):
    response = client.post("/search", json={"query": query})
    assert response.status_code == 200
    assert sorted(x["id"] for x in response.json()["results"]) == sorted(expected)


def test_negative_coverage_does_not_claim_absence(client):
    result = client.post(
        "/search", json={"query": "Find people at Camera 1 between 7 and 8 PM"}
    ).json()
    assert result["results"] == []
    assert result["coverage"]["footage_available"] is True
    assert result["coverage"]["requested_interval_covered"] is False
    assert result["coverage"]["absence_confidence"] == "not established"


def test_preceding_context(client):
    result = client.post(
        "/search",
        json={
            "query": "What happened immediately before this event?",
            "reference_event_id": "EVT-E03",
            "camera": "CAM-01",
        },
    ).json()
    assert any(x["id"] == "EVT-E02" for x in result["results"])
    assert all(x["end"] <= "2026-10-06T18:04:12+05:30" for x in result["results"])


def test_every_event_has_valid_provenance(client):
    with session() as s:
        for event in s.scalars(select(Event)).all():
            evidence = s.get(Evidence, event.evidence_id)
            assert evidence
            video = s.get(Video, evidence.video_id)
            assert video
            assert 0 <= evidence.frame_start <= evidence.frame_end < video.frame_count
            assert (
                0
                <= evidence.timestamp_start
                <= evidence.timestamp_end
                <= video.duration
            )
            assert len(evidence.sha256) == 64
            assert event.category in ("OBSERVED", "INFERRED", "CORRELATED")
            assert event.confidence is None


def test_ambiguous_cross_camera_candidates(client):
    result = client.get("/entities/P-E01/relationships").json()["associations"]
    assert len(result) >= 2
    assert any(x["status"] == "ambiguous" for x in result)
    assert all(
        x["entity_id"] != "P-E01" and x["category"] == "CORRELATED" for x in result
    )


def test_benchmark_calculated_and_no_fake_timings(client):
    response = client.post("/evaluation/run")
    assert response.status_code == 200
    result = response.json()["results"]
    assert result["query_count"] == 40
    expected = (
        sum(q["correct_top1"] for q in result["queries"] if q["expected"])
        / result["positive_queries"]
    )
    assert result["metrics"]["precision_at_1"] == round(expected, 4)
    assert result["metrics"]["manual_seconds"] is None
    assert result["metrics"]["temporal_iou"] is None
    assert result["metrics"]["negative_query_false_positive_rate"] == 0


def test_measured_review_time(client):
    assert (
        client.post(
            "/evaluation/timings",
            json={"task": "paired test", "mode": "manual", "seconds": 100},
        ).status_code
        == 201
    )
    assert (
        client.post(
            "/evaluation/timings",
            json={"task": "paired test", "mode": "assisted", "seconds": 40},
        ).status_code
        == 201
    )
    metrics = client.post("/evaluation/run").json()["results"]["metrics"]
    assert metrics["time_reduction"] == 0.6
    assert (
        client.post(
            "/evaluation/timings", json={"task": "x", "mode": "manual", "seconds": 0}
        ).status_code
        == 422
    )


def test_upload_rejects_bad_source_and_metadata(client):
    data = {"camera_id": "TEST", "recording_start": "2026-10-06T18:00:00+05:30"}
    assert (
        client.post(
            "/videos/upload", data=data, files={"file": ("bad.exe", b"test")}
        ).status_code
        == 415
    )
    assert (
        client.post(
            "/videos/upload", data=data, files={"file": ("bad.mp4", b"not a video")}
        ).status_code
        == 422
    )
    data["recording_start"] = "2026-10-06T18:00:00"
    assert (
        client.post(
            "/videos/upload", data=data, files={"file": ("video.mp4", b"test")}
        ).status_code
        == 422
    )


def make_video(path, count=20):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 160))
    for _ in range(count):
        writer.write(np.zeros((160, 160, 3), dtype=np.uint8))
    writer.release()
    return path.read_bytes()


def test_actual_upload_and_cpu_index(client, tmp_path):
    payload = make_video(tmp_path / "empty.avi")
    result = client.post(
        "/videos/upload",
        data={
            "camera_id": "TEST-UPLOAD",
            "recording_start": "2026-10-06T20:00:00+05:30",
            "location": "Test lab",
        },
        files={"file": ("empty.avi", payload)},
    )
    assert result.status_code == 201
    v = result.json()
    assert v["sha256"] == hashlib.sha256(payload).hexdigest() and v["is_demo"] is False
    assert client.post(f"/videos/{v['id']}/index").status_code == 202
    for _ in range(100):
        status = client.get(f"/videos/{v['id']}/status").json()
        if status["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)
    assert status["status"] == "completed", status
    with session() as s:
        assert (
            s.scalars(select(Observation).where(Observation.video_id == v["id"])).all()
            == []
        )
    assert client.post(f"/videos/{v['id']}/index").json()["status"] == "completed"


def test_indexing_integration_creates_grounded_events(client, tmp_path, monkeypatch):
    # Detector contract test, not a measurement of detector accuracy.
    payload = make_video(tmp_path / "tracked.avi", 50)
    result = client.post(
        "/videos/upload",
        data={
            "camera_id": "TRACK-TEST",
            "recording_start": "2026-10-06T21:00:00+05:30",
        },
        files={"file": ("tracked.avi", payload)},
    ).json()

    class DeterministicDetector:
        name = "test detector contract"

        def detect(self, frame):
            return [
                {
                    "track_id": 1,
                    "box": [20, 20, 60, 100],
                    "object_type": "person",
                    "confidence": 0.8,
                }
            ]

    monkeypatch.setattr("app.processing.detector", DeterministicDetector)
    process(result["id"])
    response = client.post(
        "/search", json={"query": "person stationary", "camera": "TRACK-TEST"}
    ).json()
    assert response["results"]
    event = response["results"][0]
    assert (
        event["category"] == "INFERRED"
        and event["evidence"]["video_id"] == result["id"]
        and event["is_demo"] is False
    )


def test_image_search_and_clip(client):
    hit = client.get("/events/EVT-E03").json()
    thumbnail = client.get(hit["thumbnail_url"].removeprefix("/api"))
    assert thumbnail.status_code == 200
    assert (
        client.post(
            "/search/image", files={"file": ("invalid.png", b"invalid")}
        ).status_code
        == 422
    )
    image = client.post(
        "/search/image", files={"file": ("reference.jpg", thumbnail.content)}
    )
    assert (
        image.status_code == 200
        and image.json()["parsed"]["method"] == "HSV-48 appearance baseline"
    )
    clip = client.get(f"/evidence/{hit['evidence']['id']}/clip")
    assert clip.status_code == 200 and len(clip.content) > 1000


def test_unknown_ids_and_zone_validation(client):
    assert client.get("/events/no-such-event").status_code == 404
    assert (
        client.post(
            "/search", json={"query": "person", "entity_id": "unknown"}
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/cameras/CAM-01/zones", json={"name": "Invalid", "rect": [0.8, 0, 0.1, 1]}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/cameras/CAM-01/zones",
            json={"name": "Zone A", "rect": [0.3, 0.3, 0.6, 0.8]},
        ).status_code
        == 200
    )
    assert client.post("/search", json={"query": ""}).status_code == 422


def test_query_parser():
    p = parse_query(
        "Find the person with a red shirt who approached a vehicle near the east entrance after 6 PM"
    )
    assert (p["entity"], p["color"], p["event"], p["after"]) == (
        "person",
        "red",
        "approached_vehicle",
        1080,
    )
    assert parse_query("between 7 and 8 PM")["after"] == 1140
    assert parse_query("after 99:00")["warnings"]


def test_event_rules_and_iou():
    samples = [{"t": t, "box": [10, 10, 30, 70]} for t in range(6)]
    events = extract_events(samples, 10)
    assert not any(e[0] == "stopped" for e in events)
    assert events[0][0] == "appeared" and events[0][3] == "OBSERVED"
    assert (
        iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1
        and iou([0, 0, 5, 5], [6, 6, 10, 10]) == 0
    )


def test_all_evidence_routes_require_configured_token(client, monkeypatch):
    from app import config

    monkeypatch.setattr(config, "TOKEN", "test-token-for-verification-only")
    for route in ["/videos", "/videos/VID-DEMO-01/media", "/search", "/graph"]:
        response = (
            client.post(route, json={"query": "person"})
            if route == "/search"
            else client.get(route)
        )
        assert response.status_code == 401
    assert (
        client.get(
            "/videos",
            headers={"Authorization": "Bearer test-token-for-verification-only"},
        ).status_code
        == 200
    )
