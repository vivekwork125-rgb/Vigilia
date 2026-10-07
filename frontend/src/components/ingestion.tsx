"use client";
import { useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  Check,
  Clock3,
  Database,
  FileVideo,
  LoaderCircle,
  Play,
  Upload,
  X,
} from "lucide-react";
import { api, post, duration, type Video } from "@/lib/api";
export function Ingestion({
  videos,
  refresh,
  notify,
  onOpen,
}: {
  videos: Video[];
  refresh: () => Promise<void>;
  notify: (s: string) => void;
  onOpen: (camera: string) => void;
}) {
  const [files, setFiles] = useState<File[]>([]);
  const [camera, setCamera] = useState("CAM-04");
  const [location, setLocation] = useState("");
  const [start, setStart] = useState("2026-10-06T18:00");
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState("");
  const [drag, setDrag] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const processing = videos.some((v) =>
    ["queued", "processing", "detecting", "indexing"].includes(v.status),
  );
  useEffect(() => {
    if (!processing) return;
    const timer = setInterval(() => void refresh(), 1800);
    return () => clearInterval(timer);
  }, [processing, refresh]);
  const upload = async () => {
    if (!files.length) return;
    setBusy(true);
    setError("");
    try {
      for (let i = 0; i < files.length; i++) {
        const form = new FormData();
        form.append("file", files[i]);
        form.append("camera_id", camera);
        form.append("location", location || "Unspecified");
        form.append("recording_start", `${start}:00+05:30`);
        const result = await new Promise<Video>((resolve, reject) => {
          const xhr = new XMLHttpRequest();
          xhr.open("POST", "/api/videos/upload");
          xhr.upload.onprogress = (e) => {
            if (e.lengthComputable)
              setProgress(
                Math.round((100 * (i + e.loaded / e.total)) / files.length),
              );
          };
          xhr.onload = () => {
            try {
              const body = JSON.parse(xhr.responseText);
              if (xhr.status < 300) resolve(body);
              else
                reject(
                  new Error(
                    typeof body.detail === "string"
                      ? body.detail
                      : "Video rejected; verify metadata and format",
                  ),
                );
            } catch {
              reject(new Error("Invalid upload response"));
            }
          };
          xhr.onerror = () =>
            reject(new Error("Upload failed: connection lost"));
          xhr.send(form);
        });
        await post(`/videos/${result.id}/index`);
      }
      setFiles([]);
      setProgress(100);
      await refresh();
      notify("Footage uploaded and queued for indexing");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const retry = async (id: string) => {
    try {
      await post(`/videos/${id}/index`);
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    }
  };
  return (
    <>
      <div className="page-heading">
        <div>
          <div className="eyebrow">SOURCE MANAGEMENT</div>
          <h1>Every investigation starts here.</h1>
          <p>Bring in footage. Preserve the source. Make events searchable.</p>
        </div>
        <span className="outline-tag">
          <Database size={15} />
          {videos.length} source files
        </span>
      </div>
      <div className="ingestion-grid">
        <section className="panel upload-panel">
          <h2>Ingest footage</h2>
          <p className="muted">MP4, MOV, AVI, or MKV · up to 250 MB per file</p>
          <input
            ref={input}
            type="file"
            accept=".mp4,.mov,.avi,.mkv"
            multiple
            hidden
            onChange={(e) => setFiles(Array.from(e.target.files || []))}
          />
          <button
            className={`dropzone ${drag ? "drag" : ""}`}
            onClick={() => input.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDrag(true);
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDrag(false);
              setFiles(Array.from(e.dataTransfer.files));
            }}
          >
            <span className="upload-symbol">
              <Upload size={26} />
            </span>
            <strong>Drop your footage here</strong>
            <span>or choose files from your computer</span>
            <small>Original files are retained with SHA-256 hashes</small>
          </button>
          {files.length > 0 && (
            <div className="file-list">
              {files.map((f, i) => (
                <div key={`${f.name}-${i}`}>
                  <FileVideo size={16} />
                  <span>{f.name}</span>
                  <small>{(f.size / 1024 / 1024).toFixed(1)} MB</small>
                  <button
                    aria-label={`Remove ${f.name}`}
                    disabled={busy}
                    onClick={() =>
                      setFiles((old) => old.filter((_, n) => i !== n))
                    }
                  >
                    <X size={14} />
                  </button>
                </div>
              ))}
            </div>
          )}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              void upload();
            }}
          >
            <div className="form-grid">
              <label>
                Camera ID
                <input
                  required
                  value={camera}
                  onChange={(e) => setCamera(e.target.value)}
                  placeholder="CAM-04"
                />
              </label>
              <label>
                Location
                <input
                  value={location}
                  onChange={(e) => setLocation(e.target.value)}
                  placeholder="e.g. North entrance"
                />
              </label>
              <label className="full">
                Recording start · UTC+05:30
                <input
                  required
                  type="datetime-local"
                  value={start}
                  onChange={(e) => setStart(e.target.value)}
                />
              </label>
            </div>
            <p className="fine-print">
              For multiple files, this start time applies to each file. Upload
              sequential recordings separately with their respective timestamps.
            </p>
            {busy && (
              <div className="upload-progress">
                <div style={{ width: `${progress}%` }} />
                <span>Uploading {progress}%</span>
              </div>
            )}
            {error && (
              <p className="inline-error" role="alert">
                {error}
              </p>
            )}
            <button
              className="primary full-button"
              disabled={!files.length || busy}
            >
              {busy ? (
                <LoaderCircle className="spin" size={16} />
              ) : (
                <Upload size={16} />
              )}
              Upload & index{" "}
              {files.length
                ? `${files.length} file${files.length > 1 ? "s" : ""}`
                : ""}
            </button>
          </form>
        </section>
        <section className="pipeline-info">
          <div className="section-kicker">FROM FOOTAGE TO EVIDENCE</div>
          <h2>
            One source.
            <br />A searchable story.
          </h2>
          <div className="pipeline-steps">
            {[
              [
                "01",
                "Ingest & preserve",
                "Validate footage and record its source hash.",
              ],
              [
                "02",
                "Detect & track",
                "Sample frames and form camera-scoped tracks.",
              ],
              [
                "03",
                "Extract events",
                "Identify detections and conservative event hypotheses.",
              ],
              [
                "04",
                "Index & investigate",
                "Connect observations to frames and source video.",
              ],
            ].map(([n, title, desc]) => (
              <div key={n}>
                <span>{n}</span>
                <div>
                  <strong>{title}</strong>
                  <p>{desc}</p>
                </div>
              </div>
            ))}
          </div>
          <div className="notice">
            <FocusIcon />
            <p>
              <b>Honest by design.</b> Real uploads use the configured model.
              The CPU baseline detects people only; YOLO + ByteTrack enables
              additional object classes. Failures remain visible.
            </p>
          </div>
        </section>
      </div>
      <div className="section-heading">
        <div>
          <h2>Footage library</h2>
          <p>
            Indexing runs asynchronously. This view updates as jobs progress.
          </p>
        </div>
        <button className="text-button" onClick={() => void refresh()}>
          Refresh status
        </button>
      </div>
      <div className="panel footage-table">
        <div className="footage-head">
          <span>SOURCE FILE</span>
          <span>CAMERA</span>
          <span>DURATION</span>
          <span>INDEXING STATUS</span>
          <span />
        </div>
        {videos.map((v) => (
          <div className="footage-row" key={v.id}>
            <div className="file-cell">
              <FileVideo size={21} />
              <span>
                <strong>{v.filename}</strong>
                <small>
                  {v.is_demo ? "Synthetic · pre-indexed" : v.pipeline}
                </small>
              </span>
            </div>
            <span className="mono">{v.camera_id}</span>
            <span>{duration(v.duration)}</span>
            <div>
              <span className={`job-status ${v.status}`}>
                {v.status === "completed" ? (
                  <Check size={13} />
                ) : ["failed", "uploaded"].includes(v.status) ? (
                  <Clock3 size={13} />
                ) : (
                  <LoaderCircle size={13} className="spin" />
                )}
                {v.status}{" "}
                {v.progress > 0 && v.progress < 100
                  ? `${Math.round(v.progress)}%`
                  : ""}
              </span>
              {v.error && <p className="inline-error">{v.error}</p>}
            </div>
            <div className="button-row">
              <button
                className="icon-button"
                aria-label={`Preview ${v.filename}`}
                onClick={() => setPreview(v.id)}
              >
                <Play size={16} />
              </button>
              {v.status === "failed" || v.status === "uploaded" ? (
                <button
                  className="text-button"
                  onClick={() => void retry(v.id)}
                >
                  Index
                </button>
              ) : v.status === "completed" ? (
                <button
                  className="text-button"
                  onClick={() => onOpen(v.camera_id)}
                >
                  Explore <ArrowRight size={15} />
                </button>
              ) : null}
            </div>
          </div>
        ))}
      </div>
      {preview && (
        <div className="modal-backdrop" onClick={() => setPreview(null)}>
          <div
            className="modal"
            role="dialog"
            aria-modal="true"
            aria-label="Source footage preview"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              className="modal-close icon-button"
              aria-label="Close preview"
              onClick={() => setPreview(null)}
            >
              <X />
            </button>
            <video
              src={`/api/videos/${preview}/media`}
              controls
              autoPlay
              playsInline
            />
          </div>
        </div>
      )}
    </>
  );
}
function FocusIcon() {
  return <FileVideo size={22} />;
}
