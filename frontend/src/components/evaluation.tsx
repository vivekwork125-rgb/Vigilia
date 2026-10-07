"use client";
import { useEffect, useState } from "react";
import {
  ArrowDownToLine,
  Check,
  FlaskConical,
  Gauge,
  LoaderCircle,
  Play,
  Timer,
  X,
} from "lucide-react";
import { api, post, download, type EvalRun } from "@/lib/api";
export function Evaluation({ notify }: { notify: (s: string) => void }) {
  const [run, setRun] = useState<EvalRun | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [task, setTask] = useState("Locate the backpack placement");
  const [mode, setMode] = useState("assisted");
  const [seconds, setSeconds] = useState("");
  const [started, setStarted] = useState<number | null>(null);
  const [tick, setTick] = useState(0);
  useEffect(() => {
    let live = true;
    api<EvalRun[]>("/evaluation/results")
      .then((r) => {
        if (live) setRun(r[0] || null);
      })
      .catch((e) => {
        if (live) setError(e.message);
      });
    return () => {
      live = false;
    };
  }, []);
  useEffect(() => {
    if (started === null) return;
    const timer = setInterval(() => setTick(Date.now() - started), 100);
    return () => clearInterval(timer);
  }, [started]);
  const evaluate = async () => {
    setBusy(true);
    setError("");
    try {
      setRun(await post<EvalRun>("/evaluation/run"));
      notify("Benchmark completed using actual retrieval results");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const saveTime = async () => {
    try {
      await post("/evaluation/timings", {
        task,
        mode,
        seconds: Number(seconds),
      });
      setSeconds("");
      notify("Measured timing saved. Run benchmark to update paired results.");
    } catch (e) {
      setError((e as Error).message);
    }
  };
  const metric = (name: string, percent = true) => {
    const v = run?.results.metrics[name];
    return v == null
      ? "—"
      : percent
        ? `${(v * 100).toFixed(1)}%`
        : v.toFixed(3);
  };
  return (
    <>
      <div className="page-heading">
        <div>
          <div className="eyebrow">MEASUREMENT, NOT MARKETING</div>
          <h1>Put the evidence to the test.</h1>
          <p>Reproducible retrieval benchmarks. Transparent limitations.</p>
        </div>
        <div className="button-row">
          {run && (
            <button
              className="secondary"
              onClick={() =>
                download(
                  JSON.stringify(run, null, 2),
                  `${run.id}.json`,
                  "application/json",
                )
              }
            >
              <ArrowDownToLine size={16} />
              Export results
            </button>
          )}
          <button
            className="primary"
            disabled={busy}
            onClick={() => void evaluate()}
          >
            {busy ? (
              <LoaderCircle size={16} className="spin" />
            ) : (
              <Play size={16} />
            )}
            Run benchmark
          </button>
        </div>
      </div>
      {error && (
        <div className="error-banner" role="alert">
          {error}
        </div>
      )}
      <section className="benchmark-banner">
        <span className="benchmark-icon">
          <FlaskConical size={28} />
        </span>
        <div>
          <div className="section-kicker">CONTROLLED SYNTHETIC DATASET</div>
          <h2>40 questions. Three cameras. A verifiable answer set.</h2>
          <p>
            Includes similar appearance, occlusion, low light, temporal
            constraints, and missing camera coverage.
          </p>
        </div>
        <span className="outline-tag">FIXTURE V1</span>
      </section>
      <div className="evaluation-metrics">
        {[
          ["precision_at_1", "Precision @ 1", "Top result is relevant"],
          ["precision_at_5", "Precision @ 5", "Relevant results / 5"],
          ["recall_at_5", "Recall @ 5", "Relevant evidence recovered"],
          ["mrr", "Mean reciprocal rank", "First relevant result rank"],
        ].map(([key, label, sub]) => (
          <div className="metric" key={key}>
            <div className="metric-top">
              <span>{label}</span>
              <Gauge size={16} />
            </div>
            <strong>{metric(key, key !== "mrr")}</strong>
            <small>{sub}</small>
          </div>
        ))}
      </div>
      <div className="eval-detail-grid">
        <section className="panel">
          <h3>Additional measurements</h3>
          <dl className="metric-list">
            <dt>mAP @ 5</dt>
            <dd>{metric("map_at_5")}</dd>
            <dt>Negative-query false positive rate</dt>
            <dd>{metric("negative_query_false_positive_rate")}</dd>
            <dt>Median retrieval latency</dt>
            <dd>{run ? `${run.results.metrics.median_latency_ms} ms` : "—"}</dd>
            <dt>Temporal localization IoU</dt>
            <dd>Not measured</dd>
            <dt>Association accuracy</dt>
            <dd>Not measured</dd>
          </dl>
          <p className="fine-print">
            Retrieval uses authored annotations. Independent model predictions
            are needed to measure perception and localization accuracy.
          </p>
        </section>
        <section className="panel">
          <h3>
            <Timer size={17} /> Investigation efficiency
          </h3>
          <div className="timing-comparison">
            <div>
              <small>MANUAL REVIEW</small>
              <strong>
                {run?.results.metrics.manual_seconds
                  ? `${run.results.metrics.manual_seconds}s`
                  : "—"}
              </strong>
            </div>
            <ArrowDownToLine size={20} />
            <div>
              <small>ASSISTED REVIEW</small>
              <strong className="mint-text">
                {run?.results.metrics.assisted_seconds
                  ? `${run.results.metrics.assisted_seconds}s`
                  : "—"}
              </strong>
            </div>
          </div>
          <p className="fine-print">
            {run?.results.metrics.time_reduction != null
              ? `${(run.results.metrics.time_reduction * 100).toFixed(1)}% measured time reduction across paired tasks.`
              : "No paired timings yet. Measure the same task in both modes before reporting a reduction."}
          </p>
          <div className="timing-form">
            <label>
              Task
              <input value={task} onChange={(e) => setTask(e.target.value)} />
            </label>
            <div className="button-row">
              <select
                aria-label="Review timing mode"
                value={mode}
                onChange={(e) => setMode(e.target.value)}
              >
                <option value="assisted">VIGILIA assisted</option>
                <option value="manual">Manual review</option>
              </select>
              <input
                aria-label="Measured seconds"
                type="number"
                min="0.1"
                step="0.1"
                placeholder="Seconds"
                value={seconds}
                onChange={(e) => setSeconds(e.target.value)}
              />
              <button
                className="secondary compact"
                onClick={() => {
                  if (started === null) {
                    setStarted(Date.now());
                    setTick(0);
                  } else {
                    setSeconds((tick / 1000).toFixed(1));
                    setStarted(null);
                  }
                }}
              >
                {started !== null
                  ? `Stop ${(tick / 1000).toFixed(1)}s`
                  : "Timer"}
              </button>
            </div>
            <button
              className="secondary full-button"
              disabled={!task || Number(seconds) <= 0 || !seconds}
              onClick={() => void saveTime()}
            >
              Save measured timing
            </button>
          </div>
        </section>
      </div>
      <div className="section-heading">
        <div>
          <h2>Query-level results</h2>
          <p>
            {run
              ? `${run.results.query_count} queries · ${run.results.positive_queries} positive · ${run.results.negative_queries} negative`
              : "Run the benchmark to calculate every metric from retrieval output."}
          </p>
        </div>
        {run && <span className="mono muted">{run.id}</span>}
      </div>
      {run ? (
        <>
          <div className="panel benchmark-table">
            <div className="benchmark-head">
              <span>QUERY / HARD CASE</span>
              <span>EXPECTED</span>
              <span>RETRIEVED</span>
              <span>TOP 1</span>
            </div>
            {run.results.queries.map((q) => (
              <div className="benchmark-row" key={q.id}>
                <div>
                  <strong>{q.query}</strong>
                  <small>
                    {q.id} · {q.case} · {q.latency_ms} ms
                  </small>
                </div>
                <span>{q.expected.join(", ") || "No match"}</span>
                <span>{q.retrieved.join(", ") || "No match"}</span>
                <span className={q.correct_top1 ? "mint-text" : "amber-text"}>
                  {q.correct_top1 ? <Check size={18} /> : <X size={18} />}
                </span>
              </div>
            ))}
          </div>
          <div className="limitations">
            <h3>What these numbers establish</h3>
            <p>{run.results.scope}</p>
            {run.results.limitations.map((l) => (
              <p key={l}>• {l}</p>
            ))}
          </div>
        </>
      ) : (
        <div className="empty">
          <FlaskConical size={34} />
          <h3>No benchmark run yet</h3>
          <p>Metrics remain empty until the retrieval engine is evaluated.</p>
        </div>
      )}
    </>
  );
}
