import statistics
from sqlalchemy import select
from .db import BenchmarkQuery, EvaluationRun, GroundTruth, ReviewTiming, row, uid
from .retrieval import search


def run_evaluation(s):
    rows = []
    p1 = []
    p5 = []
    r5 = []
    rr = []
    ap = []
    negatives = []
    queries = s.scalars(select(BenchmarkQuery).order_by(BenchmarkQuery.id)).all()
    for q in queries:
        expected = set(
            s.scalars(
                select(GroundTruth.event_id).where(GroundTruth.query_id == q.id)
            ).all()
        )
        result = search(s, q.query, limit=5, demo_only=True)
        got = [e["id"] for e in result["results"]]
        relevance = [int(e in expected) for e in got]
        if expected:
            p1.append(float(bool(got) and got[0] in expected))
            p5.append(sum(relevance) / 5)
            r5.append(sum(relevance) / len(expected))
            rr.append(next((1 / (i + 1) for i, x in enumerate(relevance) if x), 0))
            ap.append(
                sum(
                    sum(relevance[: i + 1]) / (i + 1)
                    for i, x in enumerate(relevance)
                    if x
                )
                / len(expected)
            )
        else:
            negatives.append(float(bool(got)))
        rows.append(
            {
                "id": q.id,
                "query": q.query,
                "case": q.case,
                "expected": sorted(expected),
                "retrieved": got,
                "correct_top1": (bool(got) and got[0] in expected)
                if expected
                else not got,
                "latency_ms": result["elapsed_ms"],
            }
        )
    avg = lambda values: round(statistics.mean(values), 4) if values else None
    timings = s.scalars(select(ReviewTiming)).all()
    tasks = {t.task for t in timings}
    pairs = []
    for task in tasks:
        manual = [t.seconds for t in timings if t.task == task and t.mode == "manual"]
        assisted = [
            t.seconds for t in timings if t.task == task and t.mode == "assisted"
        ]
        if manual and assisted:
            pairs.append(
                {
                    "task": task,
                    "manual_seconds": avg(manual),
                    "assisted_seconds": avg(assisted),
                }
            )
    metrics = {
        "precision_at_1": avg(p1),
        "precision_at_5": avg(p5),
        "recall_at_5": avg(r5),
        "mrr": avg(rr),
        "map_at_5": avg(ap),
        "negative_query_false_positive_rate": avg(negatives),
        "median_latency_ms": round(
            statistics.median([r["latency_ms"] for r in rows]), 2
        )
        if rows
        else None,
        "temporal_iou": None,
        "event_start_error": None,
        "event_end_error": None,
        "ambiguous_match_rate": None,
        "manual_seconds": avg([x["manual_seconds"] for x in pairs]),
        "assisted_seconds": avg([x["assisted_seconds"] for x in pairs]),
    }
    metrics["time_reduction"] = (
        round(1 - metrics["assisted_seconds"] / metrics["manual_seconds"], 4)
        if metrics["manual_seconds"]
        else None
    )
    result = {
        "dataset": "VIGILIA synthetic fixture v1",
        "scope": "Retrieval against 40 authored queries over synthetic annotations. This is not validation of real-world perception or identity.",
        "query_count": len(rows),
        "positive_queries": len(p1),
        "negative_queries": len(negatives),
        "metrics": metrics,
        "queries": rows,
        "timing_pairs": pairs,
        "limitations": [
            "P@5 uses a fixed denominator of five; relevant sets may contain fewer items.",
            "Temporal localization is unmeasured: independent model predictions and interval labels are required.",
            "Association accuracy is unmeasured; similar-looking entities are deliberately ambiguous.",
            "Time reduction is calculated only after paired manual and assisted timings are submitted.",
        ],
    }
    record = EvaluationRun(id=uid("RUN"), results=result)
    s.add(record)
    s.flush()
    return {**row(record), "results": result}
