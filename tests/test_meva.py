"""KPF parsing, independent matching and runtime isolation regression tests."""

import ast
from pathlib import Path

import pytest
import yaml
from meva.annotations import parse_activities, parse_geometry, parse_types
from meva.dataset import load_manifest, resolve, validate
from meva.evaluation import match, spatial_compatibility
from meva.mapping import mapping


def annotation_files(tmp_path, span=(10, 20), ident=71):
    types = tmp_path / "types.yml"
    types.write_text(
        yaml.safe_dump(
            [
                {"types": {"id1": 7, "cset3": {"person": 1}}},
                {"types": {"id1": 8, "cset3": {"other": 1}}},
            ]
        )
    )
    activities = tmp_path / "activities.yml"
    record = {
        "act": {
            "id2": ident,
            "act2": {"person_picks_up_object": 1},
            "timespan": [{"tsr0": list(span)}],
            "actors": [{"id1": k, "timespan": [{"tsr0": list(span)}]} for k in (7, 8)],
        }
    }
    activities.write_text(yaml.safe_dump([record]))
    return types, activities, record


def video(fps=25):
    return {
        "fps": fps,
        "frame_count": 100,
        "video_id": "clip",
        "camera_id": "camera",
        "annotation_path": "${MEVA_ANNOTATION_ROOT}/activities.yml",
    }


def test_kpf_preserves_ids_frames_and_actual_fps(tmp_path):
    types, activities, _ = annotation_files(tmp_path)
    rows = parse_activities(activities, video(), parse_types(types))
    row = rows[0]
    assert (row["annotation_id"], row["actor_id"], row["object_id"]) == (71, 7, 8)
    assert (row["start_frame"], row["end_frame"]) == (10, 20)
    assert (row["start_seconds"], row["end_seconds"], row["end_exclusive_seconds"]) == (
        0.4,
        0.8,
        0.84,
    )


@pytest.mark.parametrize("span", [(20, 10), (-1, 20), (0, 100)])
def test_rejects_invalid_temporal_ranges(tmp_path, span):
    types, activities, _ = annotation_files(tmp_path, span)
    with pytest.raises(ValueError):
        parse_activities(activities, video(), parse_types(types))


def test_rejects_duplicate_activity_ids_and_malformed_yaml(tmp_path):
    types, activities, record = annotation_files(tmp_path)
    activities.write_text(yaml.safe_dump([record, record]))
    with pytest.raises(ValueError, match="duplicate activity"):
        parse_activities(activities, video(), parse_types(types))
    activities.write_text("- act: [broken")
    with pytest.raises(ValueError, match="invalid annotation YAML"):
        parse_activities(activities, video(), parse_types(types))


def test_rejects_duplicate_geometry_frames(tmp_path):
    file = tmp_path / "geom.yml"
    file.write_text(
        yaml.safe_dump(
            [
                {"geom": {"id0": i, "id1": 7, "ts0": 10, "g0": "0 0 20 20"}}
                for i in (1, 2)
            ]
        )
    )
    with pytest.raises(ValueError, match="duplicate actor/frame"):
        parse_geometry(file, video(), {7: "person"})


def prediction(start, end, ident="pred"):
    return {
        "event_id": ident,
        "video_id": "v",
        "camera_id": "c",
        "event_type": "picked_up",
        "entity_types": ["person", "backpack"],
        "start_seconds": start,
        "end_seconds": end,
        "spatial_scores": {"label": 1.0},
    }


def truth(start, end):
    return {
        "key": "label",
        "annotation_id": "unrelated-label-id",
        "video_id": "v",
        "camera_id": "c",
        "activity_type": "person_picks_up_object",
        "event_type": "picked_up",
        "start_seconds": start,
        "end_seconds": end,
    }


def test_matching_never_compares_event_ids_and_is_one_to_one():
    p = [prediction(1, 2), prediction(1, 2, "duplicate")]
    assert len(match(p, [truth(1, 2)])) == 1
    p[0]["camera_id"] = "other"
    assert match(p[:1], [truth(1, 2)]) == []
    p[0]["camera_id"] = "c"
    p[0]["spatial_scores"] = {"label": 0}
    assert match(p[:1], [truth(1, 2)]) == []


def test_matching_maximizes_cardinality_before_overlap():
    a, b = truth(1, 2), truth(2, 3)
    a["key"], b["key"] = "a", "b"
    p, q = prediction(1, 3), prediction(1, 2)
    p["spatial_scores"], q["spatial_scores"] = {"a": 1, "b": 1}, {"a": 1}
    assert set(match([p, q], [a, b], tolerance=0)) == {(0, 1), (1, 0)}


def test_spatial_match_requires_both_person_and_object():
    p = prediction(1, 2)
    p["supporting_tracks"] = [
        {
            "object_type": "person",
            "boxes": [{"t": 1, "frame": 25, "box": [0, 0, 20, 40]}],
        }
    ]
    g = truth(1, 2)
    g.update(actor_ids=[7, 8], actor_types={"7": "person", "8": "other"})
    geometry = {7: {25: [0, 0, 20, 40]}, 8: {25: [10, 20, 20, 40]}}
    assert spatial_compatibility(p, g, geometry, 25, 2) == 0
    p["supporting_tracks"].append(
        {
            "object_type": "backpack",
            "boxes": [{"t": 1, "frame": 25, "box": [10, 20, 20, 40]}],
        }
    )
    assert spatial_compatibility(p, g, geometry, 25, 2) == 1


def test_missing_manifest_data_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.setenv("MEVA_ROOT", str(tmp_path))
    manifest = {
        "videos": [
            {"filename": "missing.avi", "local_path": "${MEVA_VIDEO_ROOT}/missing.avi"}
        ]
    }
    with pytest.raises(ValueError, match="Missing required video"):
        validate(manifest)
    with pytest.raises(ValueError, match="Unsafe manifest path"):
        resolve("${MEVA_VIDEO_ROOT}/../escape.avi")


def test_unknown_activity_is_unsupported_and_runtime_cannot_import_meva():
    assert mapping("person_talks_on_phone")["status"] == "UNSUPPORTED"
    assert mapping("unknown_new_activity")["status"] == "UNSUPPORTED"
    assert mapping("person_enters_scene_through_structure")["status"] == "APPROXIMATE"
    root = Path(__file__).resolve().parents[1]
    for path in (root / "backend/app").glob("*.py"):
        tree = ast.parse(path.read_text())
        imports = [
            n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
        ]
        imports += [
            a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
        ]
        assert all("meva" not in name.lower() for name in imports), path
    manifest = load_manifest()
    assert len(manifest["videos"]) == 8
    assert all(not Path(v["local_path"]).is_absolute() for v in manifest["videos"])
