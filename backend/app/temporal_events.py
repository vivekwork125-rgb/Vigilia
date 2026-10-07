"""Conservative, track-derived event hypotheses; no authored scene labels.

Inputs are detector/tracker samples with source frame, time and pixel box. Rules
emit intervals and supporting track keys, never probabilities for an action.
"""

import os
from dataclasses import dataclass, field
from itertools import pairwise
from math import hypot

VEHICLES = {"car", "truck", "bus", "motorcycle"}
PORTABLE = {"backpack", "handbag", "suitcase", "bag", "briefcase", "bottle"}


@dataclass(frozen=True)
class Candidate:
    kind: str
    keys: tuple[str, ...]
    start: float
    end: float
    category: str = "INFERRED"
    attributes: dict = field(default_factory=dict)
    method: str = "track geometry and temporal state transitions"


def center(sample):
    x1, y1, x2, y2 = sample["box"]
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def speed(a, b, scale):
    x1, y1 = center(a)
    x2, y2 = center(b)
    return hypot(x2 - x1, y2 - y1) / max(1, scale) / max(0.01, b["t"] - a["t"])


def moving(samples, scale, threshold=0.08):
    return [speed(a, b, scale) >= threshold for a, b in pairwise(samples)]


def intervals(states, samples, value):
    """Contiguous state runs with their source sample endpoints."""
    start = None
    for i, state in enumerate(states):
        if state == value and start is None:
            start = i
        if state != value and start is not None:
            yield samples[start]["t"], samples[i]["t"]
            start = None
    if start is not None:
        yield samples[start]["t"], samples[min(len(states), len(samples) - 1)]["t"]


def aligned(a, b, tolerance=0.3):
    j = 0
    for first in a:
        while j + 1 < len(b) and b[j + 1]["t"] <= first["t"]:
            j += 1
        options = b[max(0, j - 1) : min(len(b), j + 2)]
        other = min(options, key=lambda x: abs(x["t"] - first["t"]), default=None)
        if other and abs(other["t"] - first["t"]) <= tolerance:
            yield first, other


def separation(a, b, scale):
    x1, y1 = center(a)
    x2, y2 = center(b)
    return hypot(x2 - x1, y2 - y1) / max(1, scale)


def track_events(tracks, width, height, zones=(), unattended_seconds=None):
    """Extract repeatable hypotheses from actual sampled tracks.

    `tracks` maps tracker keys to sample sequences; frames must be time ordered.
    Camera zones are normalized rectangles. No identity is inferred across gaps.
    """
    unattended_seconds = (
        float(os.getenv("UNATTENDED_SECONDS", "5"))
        if unattended_seconds is None
        else unattended_seconds
    )
    if unattended_seconds <= 0:
        raise ValueError("UNATTENDED_SECONDS must be positive")
    out = []
    valid = {key: samples for key, samples in tracks.items() if len(samples) >= 3}
    for key, samples in valid.items():
        typ = samples[0]["object_type"]
        scale = max(1, samples[0]["box"][3] - samples[0]["box"][1])
        motion = moving(samples, scale)
        for start, end in intervals(motion, samples, False):
            if end - start >= 2:
                out.append(
                    Candidate(
                        "stopped"
                        if typ in VEHICLES | {"person"}
                        else "became_stationary",
                        (key,),
                        start,
                        end,
                    )
                )
        for start, end in intervals(motion, samples, True):
            if end - start >= 1 and start > samples[0]["t"]:
                out.append(Candidate("started_moving", (key,), start, end))
        for zone in zones:
            rect = zone.get("rect", ())
            if len(rect) != 4:
                continue
            inside = []
            for sample in samples:
                x1, _, x2, y2 = sample["box"]
                x, y = (x1 + x2) / 2 / width, y2 / height
                inside.append(rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3])
            for i, (a, b) in enumerate(pairwise(inside), 1):
                if not a and b:
                    out.append(
                        Candidate(
                            "entered_zone",
                            (key,),
                            samples[i]["t"],
                            samples[i]["t"],
                            attributes={"zone": zone["name"]},
                        )
                    )
                elif a and not b:
                    out.append(
                        Candidate(
                            "exited_zone",
                            (key,),
                            samples[i]["t"],
                            samples[i]["t"],
                            attributes={"zone": zone["name"]},
                        )
                    )
            for start, end in intervals(inside, samples, True):
                if end - start >= 3:
                    out.append(
                        Candidate(
                            "remained_in_zone",
                            (key,),
                            start,
                            end,
                            attributes={"zone": zone["name"]},
                        )
                    )

    persons = {k: v for k, v in valid.items() if v[0]["object_type"] == "person"}
    others = valid
    for person_key, person in persons.items():
        person_scale = max(1, person[0]["box"][3] - person[0]["box"][1])
        for other_key, obj in others.items():
            if other_key == person_key:
                continue
            pairs = list(aligned(person, obj))
            if len(pairs) < 3:
                continue
            distances = [(a["t"], separation(a, b, person_scale)) for a, b in pairs]
            typ = obj[0]["object_type"]
            near = [i for i, (_, d) in enumerate(distances) if d <= 1.5]
            if not near:
                continue
            first, last = near[0], near[-1]
            if first >= 2 and distances[0][1] - distances[first][1] >= 0.7:
                kind = (
                    "approached_vehicle"
                    if typ in VEHICLES
                    else (
                        "approached_person" if typ == "person" else "approached_object"
                    )
                )
                out.append(
                    Candidate(
                        kind,
                        (person_key, other_key),
                        distances[0][0],
                        distances[first][0],
                    )
                )
            if (
                last + 2 < len(distances)
                and distances[-1][1] - distances[last][1] >= 0.7
            ):
                out.append(
                    Candidate(
                        "moved_away",
                        (person_key, other_key),
                        distances[last][0],
                        distances[-1][0],
                    )
                )
            if typ in PORTABLE:
                # Require an observed object motion phase, a sustained stationary
                # phase, proximity, then separation. A newly appearing static bag
                # cannot by itself establish placement or pickup.
                obj_scale = max(1, obj[0]["box"][3] - obj[0]["box"][1])
                object_moves = moving(obj, obj_scale, threshold=0.15)
                for start, end in intervals(object_moves, obj, False):
                    if end - start < 2:
                        continue
                    before = [(t, d) for t, d in distances if start - 3 <= t < start]
                    close_at_start = [
                        (t, d)
                        for t, d in distances
                        if start <= t <= start + 2 and d <= 1.5
                    ]
                    far_after = [
                        (t, d)
                        for t, d in distances
                        if close_at_start
                        and close_at_start[0][0] < t <= end
                        and d >= 2.2
                    ]
                    moving_before = any(
                        obj[i]["t"] < start and obj[i]["t"] >= start - 3 and state
                        for i, state in enumerate(object_moves)
                    )
                    moving_after = any(
                        obj[i]["t"] >= end and obj[i]["t"] <= end + 3 and state
                        for i, state in enumerate(object_moves)
                    )
                    if (
                        moving_before
                        and before
                        and min(d for _, d in before) <= 1.5
                        and far_after
                    ):
                        out.append(
                            Candidate(
                                "placed_object",
                                (person_key, other_key),
                                start,
                                far_after[0][0],
                                attributes={
                                    "association": "spatial-temporal hypothesis"
                                },
                            )
                        )
                    close_at_end = [
                        (t, d)
                        for t, d in distances
                        if end - 2 <= t <= end + 2 and d <= 1.5
                    ]
                    co_moving = [
                        (t, d) for t, d in distances if end < t <= end + 3 and d <= 1.5
                    ]
                    if moving_after and close_at_end and len(co_moving) >= 2:
                        out.append(
                            Candidate(
                                "picked_up",
                                (person_key, other_key),
                                end,
                                co_moving[-1][0],
                                attributes={
                                    "association": "spatial-temporal hypothesis"
                                },
                            )
                        )
                    # A visible stationary object with a person moving away is an
                    # unattended *hypothesis*, ending when that person returns.
                    if far_after:
                        left_at = far_after[0][0]
                        returned = next(
                            (
                                t
                                for t, d in distances
                                if left_at < t <= end and d <= 1.5
                            ),
                            None,
                        )
                        unattended_end = returned if returned is not None else end
                        another_person_near = any(
                            left_at <= bag_sample["t"] <= unattended_end
                            and separation(bag_sample, other_sample, person_scale)
                            <= 1.5
                            for other_person_key, other_person in persons.items()
                            if other_person_key != person_key
                            for bag_sample, other_sample in aligned(obj, other_person)
                        )
                        if (
                            unattended_end - left_at >= unattended_seconds
                            and not another_person_near
                        ):
                            out.append(
                                Candidate(
                                    "unattended",
                                    (other_key, person_key),
                                    left_at,
                                    unattended_end,
                                    attributes={
                                        "person_left_at": left_at,
                                        "unattended_seconds": round(
                                            unattended_end - left_at, 2
                                        ),
                                        "association": "uncertain; proximity does not prove ownership",
                                    },
                                )
                            )
    # Prevent repeated hypotheses from multiple runs sharing exact source interval.
    unique = {}
    for event in out:
        unique[
            (event.kind, event.keys, event.start, event.end, str(event.attributes))
        ] = event
    return sorted(unique.values(), key=lambda e: (e.start, e.kind, e.keys))
