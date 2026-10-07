"use client";
import { useEffect, useRef, useState } from "react";
import {
  ArrowDownToLine,
  ArrowLeft,
  ArrowRight,
  Bookmark,
  Check,
  ChevronRight,
  Clock3,
  FileCheck2,
  Fingerprint,
  Focus,
  GitBranch,
  Layers,
  ShieldCheck,
} from "lucide-react";
import { api, time, type Association, type EvidenceEvent } from "@/lib/api";
import { Badge } from "./workspace";
type Box = { t: number; frame: number; box: number[] };
export function EvidenceWorkspace({
  event,
  onOpen,
  onCollect,
  collected,
  onSearch,
  onError,
}: {
  event: EvidenceEvent;
  onOpen: (e: EvidenceEvent | string) => void;
  onCollect: (e: EvidenceEvent) => void;
  collected: boolean;
  onSearch: (q: string, extra: Record<string, unknown>) => void;
  onError: (e: string) => void;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const [timeline, setTimeline] = useState<EvidenceEvent[]>([]);
  const [associations, setAssociations] = useState<Association[]>([]);
  const [entity, setEntity] = useState(event.entities[0]?.id || "");
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [playback, setPlayback] = useState(event.evidence.timestamp_start);
  const [overlay, setOverlay] = useState(true);
  const [size, setSize] = useState([768, 432]);
  useEffect(() => {
    setEntity(event.entities[0]?.id || "");
  }, [event.id]);
  useEffect(() => {
    let live = true;
    setTimeline([]);
    setAssociations([]);
    if (entity)
      Promise.all([
        api<EvidenceEvent[]>(`/entities/${entity}/timeline`),
        api<{ associations: Association[] }>(
          `/entities/${entity}/relationships`,
        ),
      ])
        .then(([t, r]) => {
          if (live) {
            setTimeline(t);
            setAssociations(r.associations);
          }
        })
        .catch((e) => {
          if (live) onError(e.message);
        });
    return () => {
      live = false;
    };
  }, [entity, onError]);
  useEffect(() => {
    let live = true;
    setBoxes([]);
    if (event.evidence.observation_id)
      api<{ boxes: Box[] }>(`/observations/${event.evidence.observation_id}`)
        .then((o) => {
          if (live) setBoxes(o.boxes);
        })
        .catch((e) => {
          if (live) onError(e.message);
        });
    return () => {
      live = false;
    };
  }, [event.evidence.observation_id, onError]);
  useEffect(() => {
    if (video.current?.readyState) {
      video.current.currentTime = event.evidence.timestamp_start;
    }
    setPlayback(event.evidence.timestamp_start);
  }, [event.id, event.evidence.timestamp_start]);
  const nearest = boxes.reduce<Box | null>(
    (best, b) =>
      !best || Math.abs(b.t - playback) < Math.abs(best.t - playback)
        ? b
        : best,
    null,
  );
  const box =
    nearest && Math.abs(nearest.t - playback) < 0.55 ? nearest.box : null;
  return (
    <div className="investigation-layout">
      <div className="evidence-main">
        <section className="panel viewer-panel">
          <div className="viewer-heading">
            <span>
              <span className="status-dot" />
              {event.camera_id}{" "}
              <span className="muted">/ {event.location}</span>
            </span>
            <span className="mono">{event.video_id}</span>
          </div>
          <div className="video-stage">
            <video
              ref={video}
              key={event.video_id}
              src={event.media_url}
              controls
              preload="metadata"
              playsInline
              onLoadedMetadata={() => {
                if (video.current) {
                  video.current.currentTime = event.evidence.timestamp_start;
                  setSize([
                    video.current.videoWidth,
                    video.current.videoHeight,
                  ]);
                }
              }}
              onTimeUpdate={() => setPlayback(video.current?.currentTime || 0)}
              onError={() =>
                onError(
                  "Source video could not be played. Try downloading the evidence clip.",
                )
              }
            />
            {overlay && box && (
              <svg
                className="video-boxes"
                viewBox={`0 0 ${size[0]} ${size[1]}`}
                aria-label="Nearest sampled observation bounding box"
              >
                <rect
                  x={box[0]}
                  y={box[1]}
                  width={box[2] - box[0]}
                  height={box[3] - box[1]}
                  fill="none"
                  stroke="#b9f58c"
                  strokeWidth="2"
                />
                <text
                  x={box[0]}
                  y={Math.max(15, box[1] - 8)}
                  fill="#d9ffb8"
                  fontSize="13"
                >
                  {event.entities[0]?.id} · sampled
                </text>
              </svg>
            )}
          </div>
          <div className="viewer-controls">
            <span>
              <Clock3 size={14} />
              {playback.toFixed(1)}s{" "}
              <span className="muted">
                / evidence {event.evidence.timestamp_start}–
                {event.evidence.timestamp_end}s
              </span>
            </span>
            <button
              className={`text-button ${overlay ? "mint-text" : ""}`}
              onClick={() => setOverlay(!overlay)}
            >
              <Focus size={15} />
              Bounding box
            </button>
          </div>
          <div className="evidence-summary">
            <div className="result-tags">
              <Badge category={event.category} />
              <span className="mono">{event.id}</span>
              {event.is_demo && (
                <span className="tiny-tag">SYNTHETIC ANNOTATION</span>
              )}
            </div>
            <h2>{event.title}</h2>
            <p>{event.evidence.method}</p>
            <div className="button-row">
              <button className="primary" onClick={() => onCollect(event)}>
                {collected ? <Check size={15} /> : <Bookmark size={15} />}{" "}
                {collected ? "Collected in case" : "Collect finding"}
              </button>
              <a
                className="secondary"
                href={`/api/evidence/${event.evidence.id}/clip`}
                download
              >
                <ArrowDownToLine size={15} />
                Evidence clip
              </a>
            </div>
          </div>
        </section>
        <section className="panel timeline-panel">
          <div className="section-heading">
            <div>
              <h2>
                <Clock3 size={17} /> Entity timeline
              </h2>
              <p>Within-camera observations, ordered by source time.</p>
            </div>
            <select
              aria-label="Timeline entity"
              value={entity}
              onChange={(e) => setEntity(e.target.value)}
            >
              {event.entities.map((e) => (
                <option key={e.id} value={e.id}>
                  {e.id} · {e.object_type}
                </option>
              ))}
            </select>
          </div>
          <div className="timeline">
            {timeline.map((e, i) => (
              <button
                className={`timeline-event ${e.id === event.id ? "selected" : ""}`}
                key={e.id}
                onClick={() => onOpen(e)}
              >
                <time>{time(e.start)}</time>
                <span
                  className={`timeline-dot ${e.category === "INFERRED" ? "inferred" : ""}`}
                />
                <div>
                  <strong>{e.title}</strong>
                  <small>
                    {e.id} · {e.camera_id}
                  </small>
                </div>
                <Badge category={e.category} />
              </button>
            ))}
          </div>
          <div className="timeline-actions">
            <button
              className="text-button"
              onClick={() =>
                onSearch("What happened immediately before this event?", {
                  reference_event_id: event.id,
                  camera: event.camera_id,
                })
              }
            >
              <ArrowLeft size={14} />
              Immediately before
            </button>
            <button
              className="text-button"
              onClick={() =>
                onSearch("Show all related events", {
                  entity_id: entity,
                  camera: null,
                })
              }
            >
              Everything involving {entity}
              <ArrowRight size={14} />
            </button>
          </div>
        </section>
        <section className="panel association-panel">
          <div className="section-heading">
            <div>
              <h2>
                <GitBranch size={17} /> Possible earlier or later observations
              </h2>
              <p>Separate entities. Camera gaps remain unresolved.</p>
            </div>
            <Badge category="CORRELATED" />
          </div>
          {associations.filter((a) => a.status !== "rejected").length ? (
            associations
              .filter((a) => a.status !== "rejected")
              .map((a) => (
                <div className="association" key={a.entity_id}>
                  <div>
                    <Fingerprint size={21} />
                    <strong>{a.entity_id}</strong>
                    <span>{a.camera_id}</span>
                    <span className="ambiguity">
                      {a.status === "ambiguous"
                        ? "Ambiguous match"
                        : `${a.status} association`}
                    </span>
                  </div>
                  <p>{a.explanation}</p>
                  <div className="association-signals">
                    <span>
                      Appearance{" "}
                      {Math.round(Number(a.signals.appearance_cosine) * 100)}%
                    </span>
                    <span>Gap {a.signals.gap_seconds}s</span>
                    <span>
                      Travel{" "}
                      {a.signals.travel_compatible
                        ? "compatible"
                        : "incompatible"}
                    </span>
                    <button
                      className="text-button"
                      onClick={() =>
                        onSearch("Show related events", {
                          entity_id: a.entity_id,
                          camera: null,
                        })
                      }
                    >
                      Inspect candidate <ChevronRight size={13} />
                    </button>
                  </div>
                </div>
              ))
          ) : (
            <p className="muted">
              No sufficiently supported cross-camera candidate. Unconfigured
              camera connections are rejected.
            </p>
          )}
        </section>
      </div>
      <aside className="evidence-rail">
        <section className="panel">
          <h3>
            <ShieldCheck size={17} /> Evidence provenance
          </h3>
          <dl>
            <dt>Evidence ID</dt>
            <dd className="mono">{event.evidence.id}</dd>
            <dt>Source camera</dt>
            <dd>
              {event.camera_id} · {event.location}
            </dd>
            <dt>Recording timestamp</dt>
            <dd>{event.start.replace("T", " ")}</dd>
            <dt>Frame range</dt>
            <dd className="mono">
              {event.evidence.frame_start} — {event.evidence.frame_end}
            </dd>
            <dt>Detection confidence</dt>
            <dd>
              {event.confidence === null
                ? "Not measured"
                : `${Math.round(event.confidence * 100)}% model output`}
            </dd>
            <dt>Source SHA-256</dt>
            <dd className="hash">{event.source_sha256}</dd>
          </dl>
          <div className="integrity">
            <FileCheck2 size={16} />
            Source hash recorded at ingestion
          </div>
        </section>
        <section className="panel">
          <h3>
            <Layers size={17} /> Why this result?
          </h3>
          {event.signals ? (
            Object.entries(event.signals).map(([key, value]) => (
              <div className="signal" key={key}>
                <div>
                  <span>{key}</span>
                  <span>
                    {value === null
                      ? "Not evaluated"
                      : `${Math.round(value * 100)}%`}
                  </span>
                </div>
                <div className="signal-track">
                  <span style={{ width: `${(value || 0) * 100}%` }} />
                </div>
              </div>
            ))
          ) : (
            <p className="muted">
              Opened through the evidence chain. No search relevance score
              applies.
            </p>
          )}
          <p className="fine-print">
            Retrieval scores are measured signals, not identity probabilities.
          </p>
        </section>
        <section className="uncertainty-note">
          <Focus size={19} />
          <strong>Keep the distinction clear</strong>
          <p>
            <b>Observed</b> is supported by a source. <b>Correlated</b> suggests
            a possible link. <b>Inferred</b> needs verification.
          </p>
          {event.is_demo && (
            <p>This source is generated footage with authored annotations.</p>
          )}
        </section>
      </aside>
    </div>
  );
}
