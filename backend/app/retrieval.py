"""Auditable query decomposition and hybrid retrieval. No generative source of truth."""

import re
import time
from datetime import datetime, timedelta
from functools import lru_cache
import numpy as np
from sqlalchemy import select
from sklearn.feature_extraction.text import TfidfVectorizer
from .db import (
    Camera,
    Embedding,
    Entity,
    Event,
    EventEntity,
    Evidence,
    Location,
    Observation,
    Video,
)
from .config import WEIGHTS
from .evidence import serialize_event
from .temporal import matches as temporal_matches

COLORS = ("red", "blue", "green", "yellow", "white", "black", "gray", "purple")
EVENT_PATTERNS = [
    (
        "placed_object",
        r"left (?:an? |the )?(?:object|bag|backpack|suitcase)|placed|put down|drop(?:ped)?|abandon",
    ),
    ("picked_up", r"pick(?:ed)? up|collected|retrieved (?:the )?bag"),
    ("unattended", r"unattended"),
    (
        "approached_vehicle",
        r"approach(?:ed|ing)? (?:a |the )?(?:vehicle|car)|interacted with (?:a |the )?(?:car|vehicle)",
    ),
    ("entered_zone", r"enter(?:ed|ing)?|crossed (?:the )?(?:zone|boundary)"),
    (
        "exited",
        r"exit(?:ed|ing)?|depart(?:ed|ure)?|left (?:the )?(?:scene|entrance|parking)|drove away",
    ),
    ("stopped", r"stop(?:ped)?|stationary|parked"),
    ("carried", r"carrying|carried|with (?:a )?backpack"),
]
SYNONYMS = [
    (r"\bpeople\b|\bman\b|\bwoman\b", "person"),
    (r"\bvehicles?\b", "car"),
    (r"\bbackpacks?\b|\bsuitcase\b", "bag"),
    (r"\bwearing\b|\bshirt\b|\bclothing\b", ""),
    (r"\bentry\b|\bgate\b", "entrance"),
]


def normalize(text):
    text = text.lower()
    for pattern, replacement in SYNONYMS:
        text = re.sub(pattern, replacement, text)
    for kind, pattern in EVENT_PATTERNS:
        if re.search(pattern, text):
            text += " " + kind.replace("_", " ")
    return text


def parse_clock(value):
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", value.strip(), re.I)
    if not m:
        return None
    h, minute, ap = int(m[1]), int(m[2] or 0), m[3]
    if minute > 59 or (ap and not 1 <= h <= 12) or (not ap and h > 23):
        return None
    if ap:
        h = h % 12 + (12 if ap.lower() == "pm" else 0)
    return h * 60 + minute


def parse_query(query):
    q = query.lower()
    kinds = [kind for kind, pattern in EVENT_PATTERNS if re.search(pattern, q)]
    entities = [
        x
        for x in ("person", "car", "truck", "bus", "bicycle", "motorcycle", "bag")
        if re.search(r"\b" + x + r"s?\b", normalize(q))
    ]
    result = {
        "entity": "person"
        if "person" in entities
        else (entities[0] if entities else None),
        "color": next((c for c in COLORS if re.search(r"\b" + c + r"\b", q)), None),
        "event": kinds[0] if kinds else None,
        "camera": None,
        "location": None,
        "after": None,
        "before": None,
        "min_duration": None,
        "temporal_relation": None,
        "window_seconds": None,
        "warnings": [],
    }
    camera = re.search(r"(?:camera|cam)[ _-]*0?(\d+)", q)
    if camera:
        result["camera"] = f"CAM-{int(camera[1]):02d}"
    for label, aliases in [
        ("east entrance", ["east entrance", "east gate"]),
        ("west plaza", ["west plaza", "plaza"]),
        ("parking", ["parking", "car park"]),
    ]:
        if any(a in q for a in aliases):
            result["location"] = label
    for direction in ("after", "before"):
        match = re.search(
            r"\b" + direction + r"\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b", q
        )
        if match:
            result[direction] = parse_clock(match[1])
            if result[direction] is None:
                result["warnings"].append("Invalid clock time; use HH:MM or 6 PM.")
    between = re.search(
        r"between\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s+and\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)",
        q,
    )
    if between:
        first, second = between[1], between[2]
        suffix = re.search(r"(am|pm)", second)
        if suffix and not re.search(r"am|pm", first):
            first += suffix[1]
        result["after"], result["before"] = parse_clock(first), parse_clock(second)
    duration = re.search(
        r"(?:more than|over|at least|for)\s+(\d+|one|two|three|five)\s*(second|minute)",
        q,
    )
    if duration:
        value = {"one": 1, "two": 2, "three": 3, "five": 5}.get(
            duration[1], duration[1]
        )
        result["min_duration"] = int(value) * (60 if duration[2] == "minute" else 1)
    temporal = re.search(
        r"within\s+(\d+|one|two|three|five)\s*(second|minute).*?\b(before|after)\b", q
    )
    if temporal:
        result["temporal_relation"] = temporal[3]
        result["window_seconds"] = int(
            {"one": 1, "two": 2, "three": 3, "five": 5}.get(temporal[1], temporal[1])
        ) * (60 if temporal[2] == "minute" else 1)
    if "immediately before" in q:
        result["temporal_relation"] = "before"
        result["window_seconds"] = 300
    if "immediately after" in q:
        result["temporal_relation"] = "after"
        result["window_seconds"] = 300
    if "during this event" in q:
        result["temporal_relation"] = "during"
        result["window_seconds"] = 0
    if "near this event" in q:
        result["temporal_relation"] = "near"
        result["window_seconds"] = 120
    if "overlap" in q and "this event" in q:
        result["temporal_relation"] = "overlaps"
        result["window_seconds"] = 0
    if "followed by this event" in q:
        result["temporal_relation"] = "followed_by"
        result["window_seconds"] = 300
    if not any(
        result[k] is not None
        for k in ("entity", "color", "event", "camera", "location", "after", "before")
    ):
        result["warnings"].append(
            "No structured constraints recognized; results use lexical evidence similarity."
        )
    return result


def event_matches(kind, wanted):
    if wanted == "exited":
        return kind in ("exited", "disappeared")
    if wanted == "entered_zone":
        return kind in ("entered_zone", "appeared")
    return kind == wanted


def clock_matches(start, after, before):
    minute = start.hour * 60 + start.minute
    if after is not None and before is not None and after > before:
        return minute >= after or minute <= before
    return (after is None or minute >= after) and (before is None or minute <= before)


def coverage(s, parsed, has_match=False):
    videos = s.scalars(select(Video)).all()
    if parsed.get("camera"):
        videos = [v for v in videos if v.camera_id == parsed["camera"]]
    if parsed.get("location"):
        videos = [
            v
            for v in videos
            if parsed["location"]
            in s.get(Location, s.get(Camera, v.camera_id).location_id).name.lower()
        ]
    return {
        "camera_operational": "unknown — no live health telemetry",
        "footage_available": bool(videos),
        "requested_interval_covered": "unknown"
        if parsed["after"] is None or parsed["before"] is None
        else _covered(videos, parsed["after"], parsed["before"]),
        "intervals": [
            {
                "camera_id": v.camera_id,
                "start": v.recording_start,
                "end": (
                    datetime.fromisoformat(v.recording_start)
                    + timedelta(seconds=v.duration)
                ).isoformat(),
                "indexed": v.status == "completed",
            }
            for v in videos
        ],
        "conclusion": (
            "Matching indexed events were retrieved; source evidence still requires review."
            if has_match
            else "No matching indexed event under the evaluated filters and threshold. This does not establish absence."
        ),
        "absence_confidence": "not established",
    }


def _covered(videos, after, before):
    # Union per camera and recording date; never bridge different days or camera gaps.
    groups = {}
    if before < after:
        return False
    for v in videos:
        if v.status != "completed":
            continue
        d = datetime.fromisoformat(v.recording_start)
        a = d.hour * 60 + d.minute + d.second / 60
        groups.setdefault((v.camera_id, d.date()), []).append((a, a + v.duration / 60))
    for intervals in groups.values():
        cursor = float(after)
        for a, b in sorted(intervals):
            if a > cursor:
                break
            cursor = max(cursor, b)
        if cursor >= before:
            return True
    return False


@lru_cache(maxsize=1)
def clip_encoder():
    from .vision import ClipEncoder

    return ClipEncoder()


def search(
    s,
    query,
    camera=None,
    category=None,
    entity_id=None,
    reference_event_id=None,
    limit=30,
    demo_only=False,
):
    begun = time.perf_counter()
    parsed = parse_query(query)
    if camera:
        parsed["camera"] = camera
    statement = select(Event)
    if demo_only:
        statement = (
            statement.join(Evidence, Event.evidence_id == Evidence.id)
            .join(Video, Evidence.video_id == Video.id)
            .where(Video.id.like("VID-DEMO-%"))
        )
    if parsed["camera"]:
        statement = statement.where(Event.camera_id == parsed["camera"])
    if category:
        statement = statement.where(Event.category == category)
    if entity_id:
        statement = statement.join(EventEntity).where(
            EventEntity.entity_id == entity_id
        )
    events = s.scalars(statement.order_by(Event.start)).all()
    items = [serialize_event(s, e) for e in events]
    texts = [
        normalize(
            x["title"]
            + " "
            + x["description"]
            + " "
            + x["location"]
            + " "
            + " ".join(str(e["attributes"]) for e in x["entities"])
        )
        for x in items
    ]
    lexical = np.zeros(len(items))
    if texts:
        matrix = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform(
            texts + [normalize(query)]
        )
        lexical = (matrix[:-1] @ matrix[-1].T).toarray().ravel()
    reference = s.get(Event, reference_event_id) if reference_event_id else None
    temporal_targets = (
        [reference]
        if reference
        else [
            e
            for e in s.scalars(select(Event)).all()
            if e.event_type in ("exited", "disappeared")
        ]
    )
    use_clip = __import__("os").getenv("ENABLE_CLIP", "false").lower() == "true"
    clip_text = clip_encoder().text(query) if use_clip and items else None
    results = []
    for item, similarity in zip(items, lexical):
        entities = item["entities"]
        start = datetime.fromisoformat(item["start"])
        end = datetime.fromisoformat(item["end"])
        if parsed["entity"] and not any(
            e["object_type"] == parsed["entity"]
            or (
                parsed["entity"] == "bag"
                and e["object_type"] in ("backpack", "suitcase")
            )
            for e in entities
        ):
            continue
        if parsed["color"] and not any(
            e["attributes"].get("color") == parsed["color"] for e in entities
        ):
            continue
        if parsed["event"] and not event_matches(item["event_type"], parsed["event"]):
            continue
        if parsed["location"] and parsed["location"] not in item["location"].lower():
            continue
        if not clock_matches(start, parsed["after"], parsed["before"]):
            continue
        duration = (end - start).total_seconds()
        if parsed["min_duration"] is not None and duration < parsed["min_duration"]:
            continue
        supporting_temporal = []
        if parsed["temporal_relation"]:
            for target in temporal_targets:
                if not target or target.id == item["id"]:
                    continue
                # Non-context temporal queries require a shared entity with target.
                target_entities = set(
                    s.scalars(
                        select(EventEntity.entity_id).where(
                            EventEntity.event_id == target.id
                        )
                    ).all()
                )
                if not reference and not target_entities.intersection(
                    e["id"] for e in entities
                ):
                    continue
                compatible = temporal_matches(
                    start,
                    end,
                    target.start,
                    target.end,
                    parsed["temporal_relation"],
                    parsed["window_seconds"] or 0,
                )
                if compatible:
                    supporting_temporal.append(target.id)
            if not supporting_temporal:
                continue
        semantic = float(similarity)
        semantic_method = "TF-IDF evidence-text cosine"
        if clip_text is not None and item["evidence"]["observation_id"]:
            vector = s.scalar(
                select(Embedding).where(
                    Embedding.observation_id == item["evidence"]["observation_id"],
                    Embedding.model == "OpenCLIP-ViT-B-32",
                )
            )
            if vector:
                semantic = max(0, float(np.dot(clip_text, vector.values)))
                semantic_method = "OpenCLIP image/text cosine"
        constrained = (
            any(
                parsed[k] is not None
                for k in ("entity", "event", "color", "location", "camera")
            )
            or bool(entity_id)
            or bool(reference)
        )
        if not constrained and semantic < 0.12:
            continue
        signals = {
            "semantic": round(semantic, 4),
            "attributes": 1.0 if parsed["entity"] or parsed["color"] else None,
            "temporal": 1.0
            if any(
                parsed[k] is not None
                for k in ("after", "before", "min_duration", "temporal_relation")
            )
            else None,
            "spatial": 1.0 if parsed["camera"] or parsed["location"] else None,
            "event": 1.0 if parsed["event"] else None,
            "graph": 1.0 if entity_id or supporting_temporal else None,
        }
        active = {k: v for k, v in signals.items() if v is not None}
        score = sum(WEIGHTS[k] * v for k, v in active.items()) / max(
            1e-9, sum(WEIGHTS[k] for k in active)
        )
        explanations = [f"{semantic_method}: {semantic:.3f}"]
        if parsed["event"]:
            explanations.append(f"Event type satisfies {parsed['event']}")
        if parsed["color"]:
            explanations.append(f"Recorded color: {parsed['color']}")
        if parsed["location"]:
            explanations.append(f"Source location: {item['location']}")
        if supporting_temporal:
            explanations.append(
                "Temporal relation supported by " + ", ".join(supporting_temporal)
            )
        results.append(
            {
                **item,
                "score": round(score, 4),
                "signals": signals,
                "explanations": explanations,
                "temporal_evidence": supporting_temporal,
            }
        )
    results.sort(key=lambda x: (-x["score"], x["start"], x["id"]))
    return {
        "query": query,
        "parsed": parsed,
        "results": results[:limit],
        "total": len(results),
        "candidates_evaluated": len(items),
        "elapsed_ms": round((time.perf_counter() - begun) * 1000, 2),
        "coverage": coverage(s, parsed, bool(results)),
        "match_threshold": {
            "unconstrained_semantic_minimum": 0.12,
            "structured_query": "hard filters; no additional semantic minimum",
        },
        "score_notice": "Relevance is a retrieval score, not an identity probability. Null signals were not evaluated.",
    }


def associations(s, entity_id):
    entity = s.get(Entity, entity_id)
    if not entity:
        return []
    source = s.scalar(select(Observation).where(Observation.entity_id == entity_id))
    if not source:
        return []
    vector = s.scalar(
        select(Embedding).where(
            Embedding.observation_id == source.id, Embedding.model == "HSV-48"
        )
    )
    if not vector:
        return []
    video = s.get(Video, source.video_id)
    camera = s.get(Camera, entity.camera_id)
    result = []
    for candidate in s.scalars(
        select(Entity).where(
            Entity.camera_id != entity.camera_id,
            Entity.object_type == entity.object_type,
        )
    ).all():
        obs = s.scalar(select(Observation).where(Observation.entity_id == candidate.id))
        if not obs:
            continue
        emb = s.scalar(
            select(Embedding).where(
                Embedding.observation_id == obs.id, Embedding.model == "HSV-48"
            )
        )
        if not emb:
            continue
        target_video = s.get(Video, obs.video_id)
        source_start = datetime.fromisoformat(video.recording_start) + timedelta(
            seconds=source.start
        )
        source_end = datetime.fromisoformat(video.recording_start) + timedelta(
            seconds=source.end
        )
        target_start = datetime.fromisoformat(target_video.recording_start) + timedelta(
            seconds=obs.start
        )
        target_end = datetime.fromisoformat(target_video.recording_start) + timedelta(
            seconds=obs.end
        )
        gap = max(
            (target_start - source_end).total_seconds(),
            (source_start - target_end).total_seconds(),
        )
        bounds = camera.connections.get(candidate.camera_id)
        visual = float(np.dot(vector.values, emb.values))
        compatible = bool(bounds and bounds[0] <= gap <= bounds[1])
        color = (
            entity.attributes.get("color") == candidate.attributes.get("color")
            and entity.attributes.get("color") != "unknown"
        )
        score = 0.5 * max(0, visual) + 0.2 * color + 0.3 * compatible
        status = (
            "rejected"
            if not compatible or visual < 0.4
            else ("medium" if score >= 0.7 else "ambiguous")
        )
        result.append(
            {
                "entity_id": candidate.id,
                "camera_id": candidate.camera_id,
                "category": "CORRELATED",
                "status": status,
                "score": round(score, 4),
                "signals": {
                    "appearance_cosine": round(visual, 4),
                    "color_match": color,
                    "travel_compatible": compatible,
                    "gap_seconds": round(gap, 1),
                },
                "explanation": "Coarse color similarity and configured travel-time constraints. Identity is unverified; there is no continuous footage between cameras.",
            }
        )
    result.sort(key=lambda x: -x["score"])
    viable = [r for r in result if r["status"] != "rejected"]
    if len(viable) > 1 and viable[0]["score"] - viable[1]["score"] < 0.08:
        for r in viable:
            if viable[0]["score"] - r["score"] < 0.08:
                r["status"] = "ambiguous"
                r["explanation"] = (
                    "Ambiguous match: multiple candidates have similar measured appearance and compatible travel time. Identity is unresolved."
                )
    return result
