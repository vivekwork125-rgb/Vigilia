"""Guardrails for external labels and measured timing records."""

import csv
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    path = ROOT / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_independent_label_guard_and_interval_match(tmp_path):
    tool = load("evaluate_independent")
    path = tmp_path / "labels.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tool.FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "query_id": "Q1",
                "query": "placed bag",
                "camera_id": "C1",
                "video_id": "V1",
                "event_type": "placed_object",
                "start_seconds": 5,
                "end_seconds": 8,
                "tolerance_seconds": 1,
                "entity_id": "",
                "negative": "false",
                "case": "placement",
            }
        )
    with pytest.raises(ValueError, match="at least 30"):
        tool.read_labels(path)
    assert len(tool.read_labels(path, allow_small=True)) == 1
    hit = {
        "video_id": "V1",
        "event_type": "placed_object",
        "entities": [],
        "evidence": {"timestamp_start": 5.5, "timestamp_end": 8.5},
    }
    row = next(iter(tool.read_labels(path, allow_small=True).values()))[0]
    assert tool.label_match(hit, row)
    hit["evidence"]["timestamp_end"] = 12
    assert not tool.label_match(hit, row)


def test_timing_study_requires_real_complete_pair(tmp_path):
    tool = load("timing_study")
    path = tmp_path / "timings.csv"
    assert tool.summarize(path)["metrics"] is None
    base = {
        "task": "bag case",
        "participant": "P1",
        "trial": "1",
        "first_relevant": 20.0,
        "correct": 30.0,
        "total": 50.0,
        "notes": "measured",
    }
    tool.record(path, SimpleNamespace(**base, mode="manual"))
    assert tool.summarize(path)["metrics"] is None
    assisted = {**base, "first_relevant": 10.0, "correct": 15.0, "total": 25.0}
    tool.record(path, SimpleNamespace(**assisted, mode="assisted"))
    summary = tool.summarize(path)
    assert summary["paired_trials"] == 1
    assert summary["metrics"]["total_seconds"]["mean_paired_reduction"] == 0.5
    with pytest.raises(ValueError):
        tool.record(path, SimpleNamespace(**{**base, "correct": 10.0}, mode="manual"))
