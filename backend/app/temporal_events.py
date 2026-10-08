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


def continuous_tracks(tracks, max_gap=1.5):
    """A detection gap is unknown coverage, never evidence of continuous motion."""
    output = {}
    for key, samples in tracks.items():
        groups = [[]]
        for item in samples:
            if groups[-1] and item["t"] - groups[-1][-1]["t"] > max_gap:
                groups.append([])
            groups[-1].append(item)
        for index, group in enumerate(groups):
            if group:
                output[key if len(groups) == 1 else f"{key}@{index}"] = group
    return output


def coupling(person, obj, start, end):
    """Common pixel-coordinate motion vectors, proximity and sustained displacement.

    Comparing speeds normalized by different box heights can assign the wrong
    person. Direction and relative vector agreement must both support coupling.
    """
    pairs = [(a, b) for a, b in aligned(person, obj) if start <= a["t"] <= end]
    good, residuals, distances = 0, [], []
    for (a, b), (c, d) in pairwise(pairs):
        if c["t"] - a["t"] > 1.5:
            continue
        scale = max(1, a["box"][3] - a["box"][1])
        ax, ay = center(a)
        bx, by = center(b)
        cx, cy = center(c)
        dx, dy = center(d)
        pv, ov = (cx - ax, cy - ay), (dx - bx, dy - by)
        pn, on = hypot(*pv), hypot(*ov)
        residual = hypot(pv[0] - ov[0], pv[1] - ov[1]) / max(pn, on, 1)
        cosine = (pv[0] * ov[0] + pv[1] * ov[1]) / max(pn * on, 1e-9)
        distance = separation(c, d, scale)
        if (
            pn >= 2
            and on >= 2
            and cosine >= 0.8
            and residual <= 0.35
            and distance <= 1.5
        ):
            good += 1
            residuals.append(residual)
            distances.append(distance)
    if good < 2 or not pairs:
        return None
    displacement = hypot(
        center(pairs[-1][1])[0] - center(pairs[0][1])[0],
        center(pairs[-1][1])[1] - center(pairs[0][1])[1],
    )
    person_height = max(1, pairs[0][0]["box"][3] - pairs[0][0]["box"][1])
    if displacement < max(4, 0.2 * person_height):
        return None
    return sum(residuals) / good + 0.1 * sum(distances) / good


def object_interactions(tracks, persons, unattended_seconds):
    """Resolve one person-object hypothesis per transition; ambiguous ties abstain."""
    output = []
    for object_key, obj in tracks.items():
        if obj[0]["object_type"] not in PORTABLE:
            continue
        scale = max(1, obj[0]["box"][3] - obj[0]["box"][1])
        motion = moving(obj, scale, threshold=0.25)
        for start, end in intervals(motion, obj, False):
            if end - start < 2:
                continue
            for kind, window in (
                ("picked_up", (end, end + 3)),
                ("placed_object", (start - 3, start)),
            ):
                candidates = []
                for person_key, person in persons.items():
                    quality = coupling(person, obj, *window)
                    if quality is None:
                        continue
                    person_scale = max(1, person[0]["box"][3] - person[0]["box"][1])
                    distances = [
                        (a["t"], separation(a, b, person_scale))
                        for a, b in aligned(person, obj)
                    ]
                    if kind == "picked_up":
                        prior_near = [
                            (t, d)
                            for t, d in distances
                            if end - 2 <= t <= end and d <= 1.5
                        ]
                        after = [
                            (t, d)
                            for t, d in distances
                            if end < t <= end + 3 and d <= 1.5
                        ]
                        # Contact-region proximity before movement followed by consistent coupled motion.
                        if prior_near and len(after) >= 2:
                            candidates.append(
                                (quality, person_key, after[-1][0], distances)
                            )
                    else:
                        close = [
                            (t, d)
                            for t, d in distances
                            if start <= t <= start + 2 and d <= 1.5
                        ]
                        separated = [
                            (t, d)
                            for t, d in distances
                            if close and close[0][0] < t <= end and d >= 2.2
                        ]
                        if not separated:
                            continue
                        left = separated[0][0]
                        sustained = [
                            (t, d)
                            for t, d in distances
                            if left <= t <= min(end, left + 2)
                        ]
                        if (
                            sustained
                            and sustained[-1][0] - left >= 2
                            and all(d >= 2.2 for _, d in sustained)
                        ):
                            candidates.append((quality, person_key, left, distances))
                candidates.sort(key=lambda item: item[0])
                if not candidates or (
                    len(candidates) > 1 and candidates[1][0] - candidates[0][0] < 0.05
                ):
                    continue
                quality, person_key, finish, distances = candidates[0]
                output.append(
                    Candidate(
                        kind,
                        (person_key, object_key),
                        end if kind == "picked_up" else start,
                        finish,
                        attributes={
                            "association": "uncertain coupled-motion hypothesis; proximity does not prove possession",
                            "coupling_residual_score": round(quality, 4),
                        },
                    )
                )
            # Visible stationary object, sustained departure and continuous absence of nearby-person evidence.
            departures = []
            for person_key, person in persons.items():
                person_scale = max(1, person[0]["box"][3] - person[0]["box"][1])
                distances = [
                    (a["t"], separation(a, b, person_scale))
                    for a, b in aligned(person, obj)
                    if start <= a["t"] <= end
                ]
                near = [t for t, d in distances if d <= 1.5]
                if not near:
                    continue
                left = next((t for t, d in distances if t > near[0] and d >= 2.2), None)
                if left is None:
                    continue
                returned = next((t for t, d in distances if t > left and d <= 1.5), end)
                if returned - left >= unattended_seconds:
                    departures.append((left, returned, person_key))
            for left, finish, person_key in sorted(departures):
                nearby = any(
                    left <= a["t"] <= finish
                    and separation(a, b, max(1, b["box"][3] - b["box"][1])) <= 1.5
                    for key, person in persons.items()
                    if key != person_key
                    for a, b in aligned(obj, person)
                )
                if not nearby:
                    output.append(
                        Candidate(
                            "unattended",
                            (object_key, person_key),
                            left,
                            finish,
                            attributes={
                                "person_left_at": left,
                                "unattended_seconds": round(finish - left, 2),
                                "association": "visible-camera hypothesis only; absence and ownership not established",
                            },
                        )
                    )
                    break
    return output


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
    tracks = continuous_tracks(
        tracks, max_gap=max(1.5, 3 / float(os.getenv("SAMPLE_FPS", "2")))
    )
    valid = {key: samples for key, samples in tracks.items() if len(samples) >= 3}
    for key, samples in valid.items():
        typ = samples[0]["object_type"]
        scale = max(1, samples[0]["box"][3] - samples[0]["box"][1])
        motion = moving(samples, scale)
        stationary = list(intervals(motion, samples, False))
        movements = list(intervals(motion, samples, True))
        for start, end in stationary:
            if end - start >= 2:
                prior_motion = any(b == start and b - a >= 1 for a, b in movements)
                kind = (
                    ("stopped" if typ in VEHICLES | {"person"} else "became_stationary")
                    if prior_motion
                    else "stationary"
                )
                output_end = min(end, start + 2) if prior_motion else end
                out.append(
                    Candidate(
                        kind,
                        (key,),
                        start,
                        output_end,
                        attributes={
                            "state_verified_until": end,
                            "prior_motion_observed": prior_motion,
                        },
                    )
                )
        for start, end in movements:
            if end - start >= 1 and any(
                b == start and b - a >= 2 for a, b in stationary
            ):
                out.append(
                    Candidate(
                        "started_moving",
                        (key,),
                        start,
                        min(end, start + 1),
                        attributes={
                            "motion_verified_until": end,
                            "prior_stationary_observed": True,
                        },
                    )
                )
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
    out.extend(object_interactions(valid, persons, unattended_seconds))
    # Prevent repeated hypotheses from multiple runs sharing exact source interval.
    unique = {}
    for event in out:
        unique[
            (event.kind, event.keys, event.start, event.end, str(event.attributes))
        ] = event
    return sorted(unique.values(), key=lambda e: (e.start, e.kind, e.keys))
