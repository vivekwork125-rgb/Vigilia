"""Strict KPF parsing. Annotation identities exist only in this evaluation layer."""

import math
from pathlib import Path

import yaml

Loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def records(path, packet):
    try:
        rows = yaml.load(Path(path).read_text(), Loader=Loader)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"{path}: invalid annotation YAML: {exc}") from exc
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a KPF record list")
    for number, row in enumerate(rows, 1):
        if isinstance(row, dict) and "meta" in row:
            continue
        if not isinstance(row, dict) or not isinstance(row.get(packet), dict):
            raise ValueError(f"{path}: record {number}: missing {packet} packet")
        yield row[packet]


def integer(value, context):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context}: expected a nonnegative integer, got {value!r}")
    return value


def spans(value, frame_count, context):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context}: missing timespan")
    result = []
    for span in value:
        pair = span.get("tsr0") if isinstance(span, dict) else None
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"{context}: expected tsr0 [start, end]")
        a, b = [integer(x, context) for x in pair]
        if a > b or b >= frame_count:
            raise ValueError(
                f"{context}: invalid interval {pair} for {frame_count} frames"
            )
        result.append((a, b))
    for (a, b), (c, d) in zip(sorted(result), sorted(result)[1:]):
        if c <= b:
            raise ValueError(f"{context}: overlapping activity spans")
    return result


def parse_types(path):
    types = {}
    for record in records(path, "types"):
        ident = integer(record.get("id1"), str(path))
        if ident in types:
            raise ValueError(f"{path}: duplicate type id1={ident}")
        labels = record.get("cset3")
        if not isinstance(labels, dict) or len(labels) != 1:
            raise ValueError(f"{path}: id1={ident}: expected one cset3 type")
        types[ident] = next(iter(labels))
    return types


def parse_activities(path, video, types):
    fps = video["fps"]
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"{path}: invalid video FPS {fps}")
    result, seen = [], set()
    for record in records(path, "act"):
        ident = integer(record.get("id2"), str(path))
        if ident in seen:
            raise ValueError(f"{path}: duplicate activity id2={ident}")
        seen.add(ident)
        context = f"{path}: activity id2={ident}"
        labels = record.get("act2")
        if not isinstance(labels, dict) or len(labels) != 1:
            raise ValueError(f"{context}: expected one act2 activity")
        intervals = spans(record.get("timespan"), video["frame_count"], context)
        actors = record.get("actors")
        if not isinstance(actors, list) or not actors:
            raise ValueError(f"{context}: missing actors")
        ids, actor_spans = [], {}
        for actor in actors:
            key = integer(actor.get("id1"), context)
            if key in ids or key not in types:
                raise ValueError(f"{context}: duplicate/unknown actor id1={key}")
            ids.append(key)
            actor_spans[key] = spans(
                actor.get("timespan"), video["frame_count"], context
            )
            if any(
                not any(a <= c <= d <= b for a, b in intervals)
                for c, d in actor_spans[key]
            ):
                raise ValueError(f"{context}: actor interval outside activity")
        people = [key for key in ids if types[key] == "person"]
        objects = [key for key in ids if types[key] != "person"]
        for span_index, (a, b) in enumerate(intervals):
            result.append(
                {
                    "annotation_id": ident,
                    "span_index": span_index,
                    "activity_type": next(iter(labels)),
                    "actor_id": people[0] if people else ids[0],
                    "object_id": objects[0] if objects else None,
                    "actor_ids": ids,
                    "actor_types": {str(k): types[k] for k in ids},
                    "actor_intervals": {str(k): actor_spans[k] for k in ids},
                    "start_frame": a,
                    "end_frame": b,
                    "start_seconds": a / fps,
                    "end_seconds": b / fps,
                    "end_exclusive_seconds": (b + 1) / fps,
                    "camera_id": video["camera_id"],
                    "video_id": video["video_id"],
                    "source_annotation": video["annotation_path"],
                }
            )
    return result


def parse_geometry(path, video, types):
    """Sparse lookup for evaluation-only spatial compatibility; no runtime IDs."""
    geometry, seen = {}, set()
    for record in records(path, "geom"):
        ident = integer(record.get("id0"), str(path))
        key = integer(record.get("id1"), str(path))
        frame = integer(record.get("ts0"), str(path))
        if ident in seen or frame >= video["frame_count"] or key not in types:
            raise ValueError(
                f"{path}: invalid/duplicate geom id0={ident}, id1={key}, ts0={frame}"
            )
        seen.add(ident)
        try:
            box = [float(x) for x in record["g0"].split()]
        except (KeyError, AttributeError, ValueError) as exc:
            raise ValueError(f"{path}: malformed box id0={ident}") from exc
        if (
            len(box) != 4
            or not all(math.isfinite(x) for x in box)
            or box[2] <= box[0]
            or box[3] <= box[1]
        ):
            raise ValueError(f"{path}: invalid box id0={ident}: {box}")
        target = geometry.setdefault(key, {})
        if frame in target:
            raise ValueError(f"{path}: duplicate actor/frame {key}/{frame}")
        target[frame] = box
    return geometry
