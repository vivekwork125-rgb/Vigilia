"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  Aperture,
  ArrowDownToLine,
  ArrowRight,
  Bookmark,
  Camera as CameraIcon,
  Check,
  ChevronRight,
  CircleHelp,
  Clock3,
  Database,
  FileText,
  Fingerprint,
  Focus,
  Gauge,
  GitBranch,
  ImagePlus,
  LayoutDashboard,
  LoaderCircle,
  LockKeyhole,
  Menu,
  Play,
  Plus,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Upload,
  X,
} from "lucide-react";
import {
  api,
  post,
  download,
  duration,
  time,
  timezoneLabel,
  type Camera,
  type EvidenceEvent,
  type Investigation,
  type Overview,
  type SearchResult,
  type Video,
} from "@/lib/api";
import { EvidenceWorkspace } from "./evidence-workspace";
import { Ingestion } from "./ingestion";
import { Evaluation } from "./evaluation";
import { EvidenceGraph } from "./graph";
export const SAMPLE_QUERY =
  "Find the person who left an object near the east entrance";
type Screen =
  | "overview"
  | "search"
  | "ingestion"
  | "investigation"
  | "graph"
  | "evaluation";
const navigation = [
  { id: "overview", label: "Command center", icon: LayoutDashboard },
  { id: "search", label: "Evidence search", icon: Search },
  { id: "ingestion", label: "Footage library", icon: Database },
  { id: "investigation", label: "Investigations", icon: Fingerprint },
  { id: "graph", label: "Evidence graph", icon: GitBranch },
  { id: "evaluation", label: "Evaluation lab", icon: Gauge },
] as const;
export function Badge({ category }: { category: string }) {
  return (
    <span className={`badge ${category.toLowerCase()}`}>
      <span />
      {category.toLowerCase()}
    </span>
  );
}
export function Empty({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="empty">
      <Focus size={32} />
      <h3>{title}</h3>
      <p>{detail}</p>
    </div>
  );
}
export default function Workspace() {
  const [screen, setScreen] = useState<Screen>("overview");
  const [overview, setOverview] = useState<Overview | null>(null);
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [videos, setVideos] = useState<Video[]>([]);
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState<SearchResult | null>(null);
  const [selected, setSelected] = useState<EvidenceEvent | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [toast, setToast] = useState("");
  const [camera, setCamera] = useState("");
  const [category, setCategory] = useState("");
  const [showPlan, setShowPlan] = useState(false);
  const [investigations, setInvestigations] = useState<Investigation[]>([]);
  const [activeCase, setActiveCase] = useState("");
  const [findings, setFindings] = useState<EvidenceEvent[]>([]);
  const [sidebar, setSidebar] = useState(false);
  const [accessToken, setAccessToken] = useState("");
  const imageInput = useRef<HTMLInputElement>(null);
  const requestId = useRef(0);
  const refresh = useCallback(async () => {
    try {
      const [o, c, v, i] = await Promise.all([
        api<Overview>("/overview"),
        api<Camera[]>("/cameras"),
        api<Video[]>("/videos"),
        api<Investigation[]>("/investigations"),
      ]);
      setOverview(o);
      setCameras(c);
      setVideos(v);
      setInvestigations(i);
      setActiveCase((old) => old || i[0]?.id || "");
      setError("");
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);
  useEffect(() => {
    void refresh();
  }, [refresh]);
  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(""), 4000);
    return () => clearTimeout(timer);
  }, [toast]);
  useEffect(() => {
    if (!activeCase) {
      setFindings([]);
      return;
    }
    let live = true;
    api<Investigation>(`/investigations/${activeCase}`)
      .then((x) => {
        if (live) setFindings(x.findings || []);
      })
      .catch((e) => {
        if (live) setError(e.message);
      });
    return () => {
      live = false;
    };
  }, [activeCase]);
  const navigate = (next: Screen) => {
    setScreen(next);
    setSidebar(false);
  };
  const runSearch = async (
    text = query,
    extra: Record<string, unknown> = {},
  ) => {
    if (!text.trim()) return;
    const id = ++requestId.current;
    setBusy(true);
    setError("");
    setQuery(text);
    setScreen("search");
    try {
      const result = await post<SearchResult>("/search", {
        query: text,
        camera: camera || null,
        category: category || null,
        ...extra,
      });
      if (id === requestId.current) setResponse(result);
    } catch (e) {
      if (id === requestId.current) setError((e as Error).message);
    } finally {
      if (id === requestId.current) setBusy(false);
    }
  };
  const open = async (item: EvidenceEvent | string) => {
    try {
      setSelected(
        typeof item === "string"
          ? await api<EvidenceEvent>(`/events/${item}`)
          : item,
      );
      setScreen("investigation");
    } catch (e) {
      setError((e as Error).message);
    }
  };
  const collect = async (item: EvidenceEvent) => {
    try {
      let id = activeCase;
      if (!id) {
        const created = await post<Investigation>("/investigations", {
          title: "New evidence investigation",
          query: query || SAMPLE_QUERY,
          event_ids: [],
        });
        id = created.id;
        setActiveCase(id);
        setInvestigations((x) => [created, ...x]);
      }
      await post(`/investigations/${id}/findings`, { event_id: item.id });
      const data = await api<Investigation>(`/investigations/${id}`);
      setFindings(data.findings || []);
      setToast(`${item.id} added to case`);
    } catch (e) {
      setError((e as Error).message);
    }
  };
  const report = async () => {
    if (!activeCase) return;
    try {
      const data = await post<{ markdown: string }>(
        `/investigations/${activeCase}/report`,
      );
      download(data.markdown, `VIGILIA-${activeCase}.md`);
      setToast("Evidence-backed report downloaded");
    } catch (e) {
      setError((e as Error).message);
    }
  };
  const newCase = async () => {
    try {
      const created = await post<Investigation>("/investigations", {
        title: query
          ? query.slice(0, 100)
          : `Investigation ${investigations.length + 1}`,
        query: query || SAMPLE_QUERY,
        event_ids: [],
      });
      setInvestigations((x) => [created, ...x]);
      setActiveCase(created.id);
      setFindings([]);
      setToast("New investigation created");
    } catch (e) {
      setError((e as Error).message);
    }
  };
  const imageSearch = async (file: File) => {
    setBusy(true);
    setScreen("search");
    setError("");
    const form = new FormData();
    form.append("file", file);
    try {
      setResponse(
        await api<SearchResult>("/search/image", {
          method: "POST",
          body: form,
        }),
      );
      setQuery("Reference image search");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const searchBox = (large = false) => (
    <form
      className={`search-box ${large ? "large" : ""}`}
      onSubmit={(e) => {
        e.preventDefault();
        void runSearch();
      }}
    >
      <Search size={large ? 22 : 19} />
      <input
        aria-label="Investigation query"
        placeholder="Describe an event, person, object, or moment…"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <button
        className="image-search"
        type="button"
        title="Search with a reference image"
        aria-label="Search with a reference image"
        onClick={() => imageInput.current?.click()}
      >
        <ImagePlus size={19} />
      </button>
      <button className="primary" disabled={busy || !query.trim()}>
        {busy ? (
          <LoaderCircle className="spin" size={17} />
        ) : (
          <ArrowRight size={18} />
        )}
        <span>Search evidence</span>
      </button>
    </form>
  );
  return (
    <div className="app-shell">
      <input
        ref={imageInput}
        type="file"
        accept="image/*"
        hidden
        onChange={(e) => {
          if (e.target.files?.[0]) void imageSearch(e.target.files[0]);
          e.target.value = "";
        }}
      />
      <aside className={`sidebar ${sidebar ? "open" : ""}`}>
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            navigate("overview");
          }}
        >
          <span className="brand-mark">
            <Aperture size={27} />
          </span>
          <span>
            VIGILIA<small>EVIDENCE INTELLIGENCE</small>
          </span>
        </a>
        <div className="workspace-label">
          <span className="workspace-symbol">V</span>
          <div>
            Investigation workspace<small>Local environment</small>
          </div>
          <ChevronRight size={14} />
        </div>
        <span className="nav-label">WORKSPACE</span>
        <nav>
          {navigation.map((n) => (
            <button
              key={n.id}
              className={screen === n.id ? "active" : ""}
              onClick={() => navigate(n.id)}
            >
              <n.icon size={18} />
              {n.label}
              {n.id === "ingestion" && (
                <span className="nav-count">{videos.length}</span>
              )}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="principle">
            <ShieldCheck size={20} />
            <strong>Evidence before inference.</strong>
            <p>Every finding leads back to its source.</p>
          </div>
          <div className="user">
            <span>IN</span>
            <div>
              Investigator<small>Local workspace</small>
            </div>
            <LockKeyhole size={15} />
          </div>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <div className="breadcrumbs">
            <button
              className="icon-button mobile-toggle"
              aria-label="Toggle navigation"
              onClick={() => setSidebar(!sidebar)}
            >
              <Menu size={20} />
            </button>
            <span>Workspace</span>
            <ChevronRight size={13} />
            <strong>{navigation.find((n) => n.id === screen)?.label}</strong>
          </div>
          <div className="top-status">
            <span className={`status-dot ${overview ? "" : "offline"}`} />
            {overview ? "System ready" : "Connecting"}
            <span className="top-divider" />
            <span className="demo-badge">
              {overview?.demo_mode ? "SYNTHETIC DEMO" : "LOCAL EVIDENCE"}
            </span>
            <button
              className="icon-button"
              aria-label="Show data provenance information"
              onClick={() =>
                setToast(
                  overview?.demo_mode
                    ? "Demo footage and annotations are generated. Uploaded footage is processed independently. Every result retains its source frames."
                    : "Indexed source footage is processed from video pixels. Activity labels are used only by the separate benchmark. Every result retains its source frames.",
                )
              }
            >
              <CircleHelp size={18} />
            </button>
          </div>
        </header>
        <main>
          {error && (
            <div className="error-banner" role="alert">
              <span>{error}</span>
              <button onClick={() => void refresh()}>Retry connection</button>
              {error.includes("token") && (
                <form
                  onSubmit={async (e) => {
                    e.preventDefault();
                    try {
                      await post("/session", { token: accessToken });
                      setAccessToken("");
                      await refresh();
                    } catch (err) {
                      setError((err as Error).message);
                    }
                  }}
                >
                  <input
                    type="password"
                    value={accessToken}
                    onChange={(e) => setAccessToken(e.target.value)}
                    placeholder="Access token"
                    aria-label="Access token"
                  />
                  <button>Unlock workspace</button>
                </form>
              )}
            </div>
          )}
          {screen === "overview" && (
            <>
              <div className="page-heading">
                <div>
                  <div className="eyebrow">
                    <span /> INVESTIGATION CONTROL
                  </div>
                  <h1>Follow the evidence.</h1>
                  <p>Turn hours of footage into a clear chain of events.</p>
                </div>
                <button
                  className="secondary"
                  onClick={() => navigate("ingestion")}
                >
                  <Upload size={16} /> Ingest footage
                </button>
              </div>
              <section className="search-hero">
                <div className="section-kicker">
                  <Sparkles size={15} /> YOUR NEXT FINDING STARTS WITH A
                  QUESTION
                </div>
                {searchBox(true)}
                <div className="suggestions">
                  <span>Try asking</span>
                  {(overview?.demo_mode
                    ? ["Who left an object at the east entrance?", "People approaching a vehicle", "Vehicles stopped for more than 20 seconds"]
                    : ["Vehicle started", "Vehicle stopped", "Person approached another person"]
                  ).map((q, i) => (
                    <button
                      key={q}
                      onClick={() => void runSearch(i === 0 && overview?.demo_mode ? SAMPLE_QUERY : q)}
                    >
                      {q}
                      <ArrowRight size={12} />
                    </button>
                  ))}
                </div>
              </section>
              <div className="metrics-row">
                {[
                  {
                    icon: CameraIcon,
                    value: overview?.cameras,
                    label: "Camera sources",
                    sub: "Across indexed locations",
                  },
                  {
                    icon: Clock3,
                    value: overview ? duration(overview.indexed_seconds) : "—",
                    label: "Indexed footage",
                    sub: "Ready for investigation",
                  },
                  {
                    icon: Focus,
                    value: overview?.events,
                    label: "Searchable events",
                    sub: "Linked to source evidence",
                  },
                  {
                    icon: Fingerprint,
                    value: investigations.length,
                    label: "Investigations",
                    sub: "Evidence collections",
                  },
                ].map((m, i) => (
                  <div className="metric" key={m.label}>
                    <div className="metric-top">
                      <span>{m.label}</span>
                      <m.icon size={17} />
                    </div>
                    <strong>{m.value ?? "—"}</strong>
                    <small>
                      <span className={i === 2 ? "mint-text" : ""}>
                        {i === 2 ? "↗ " : ""}
                        {m.sub}
                      </span>
                    </small>
                  </div>
                ))}
              </div>
              <div className="section-heading">
                <div>
                  <h2>
                    Camera coverage{" "}
                    <span className="count">{cameras.length}</span>
                  </h2>
                  <p>Source footage, ready to explore.</p>
                </div>
                <button
                  className="text-button"
                  onClick={() => navigate("ingestion")}
                >
                  View footage library <ArrowRight size={15} />
                </button>
              </div>
              <div className="camera-grid">
                {cameras.slice(0, 3).map((c, index) => {
                  const v = videos.find((v) => v.camera_id === c.id);
                  const event = overview?.recent_events.find(
                    (e) => e.camera_id === c.id,
                  );
                  return (
                    <button
                      className="camera-card"
                      key={c.id}
                      onClick={() =>
                        void runSearch(`Show events at ${c.name}`, {
                          camera: c.id,
                        })
                      }
                    >
                      <div className="camera-image">
                        {v ? (
                          <video
                            src={`/api/videos/${v.id}/media#t=${index === 0 ? 13 : 8}`}
                            muted
                            preload="metadata"
                            playsInline
                            aria-label={`${c.name} preview`}
                          />
                        ) : (
                          <CameraIcon size={40} />
                        )}
                        <span className="camera-id">
                          <span />
                          {c.id}
                        </span>
                        <span className="camera-tag">
                          {v?.is_demo ? "SYNTHETIC" : "UPLOADED"}
                        </span>
                        <span className="play-circle">
                          <Play size={16} fill="currentColor" />
                        </span>
                        <span className="camera-time">
                          {v ? time(v.recording_start) : "No recording"}{" "}
                          <span>{v ? timezoneLabel(v.recording_start) : ""}</span>
                        </span>
                      </div>
                      <div className="camera-info">
                        <div>
                          <h3>{c.name}</h3>
                          <p>
                            {v
                              ? `${duration(v.duration)} available`
                              : "No footage"}{" "}
                            <span>•</span> {v?.status || "Unavailable"}
                          </p>
                        </div>
                        <ArrowRight size={17} />
                      </div>
                    </button>
                  );
                })}
              </div>
              <div className="overview-bottom">
                <section className="panel recent-panel">
                  <div className="section-heading">
                    <div>
                      <h2>Recent evidence</h2>
                      <p>Events with a verifiable source.</p>
                    </div>
                    <button
                      className="text-button"
                      onClick={() => void runSearch("person")}
                    >
                      Explore all <ArrowRight size={14} />
                    </button>
                  </div>
                  <div className="table-head">
                    <span>EVENT / SOURCE</span>
                    <span>TIME</span>
                    <span>EVIDENCE TYPE</span>
                  </div>
                  {overview?.recent_events.slice(0, 4).map((e) => (
                    <button
                      className="event-row"
                      key={e.id}
                      onClick={() => void open(e)}
                    >
                      <div>
                        <span className="event-icon">
                          <Focus size={16} />
                        </span>
                        <span>
                          <strong>{e.title}</strong>
                          <small>
                            {e.camera_id} · {e.id}
                          </small>
                        </span>
                      </div>
                      <time>{time(e.start)}</time>
                      <Badge category={e.category} />
                    </button>
                  ))}
                </section>
                <section className="demo-case">
                  <div className="section-kicker">
                    <Fingerprint size={16} /> GUIDED INVESTIGATION
                  </div>
                  <div className="case-illustration">
                    <div className="orbit one" />
                    <div className="orbit two" />
                    <Focus size={40} />
                    <span className="orbit-point" />
                  </div>
                  <h2>{overview?.demo_mode ? <>An object. An entrance.<br />A trail of evidence.</> : <>A question. A timestamp.<br />A source frame.</>}</h2>
                  <p>{overview?.demo_mode
                    ? "A person leaves a backpack near the east entrance. Find the moment. Follow the timeline. Verify the source."
                    : "Search the indexed footage. Inspect each motion hypothesis against its exact source video and track."}</p>
                  <button
                    className="primary"
                    onClick={() => void runSearch(overview?.demo_mode ? SAMPLE_QUERY : "vehicle started")}
                  >
                    Start investigation <ArrowRight size={16} />
                  </button>
                  <small>{overview?.demo_mode
                    ? "3 synthetic cameras · Source-linked annotations"
                    : `${cameras.length} indexed cameras · Source-linked predictions`}</small>
                </section>
              </div>
              <footer className="workspace-footer">
                <ShieldCheck size={14} />
                <span>
                  Observed facts, correlations, and inferences stay distinct.
                </span>
                <span>VIGILIA / PROTOTYPE 01</span>
              </footer>
            </>
          )}
          {screen === "search" && (
            <>
              <div className="page-heading">
                <div>
                  <div className="eyebrow">EVIDENCE RETRIEVAL</div>
                  <h1>Ask. Find. Verify.</h1>
                  <p>
                    Search events and relationships. Go directly to the source.
                  </p>
                </div>
              </div>
              {searchBox()}
              <div className="filters">
                <span>
                  <SlidersHorizontal size={16} /> Refine evidence
                </span>
                <select
                  aria-label="Camera filter"
                  value={camera}
                  onChange={(e) => setCamera(e.target.value)}
                >
                  <option value="">All cameras</option>
                  {cameras.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.id} · {c.name}
                    </option>
                  ))}
                </select>
                <select
                  aria-label="Evidence type filter"
                  value={category}
                  onChange={(e) => setCategory(e.target.value)}
                >
                  <option value="">All evidence types</option>
                  {["OBSERVED", "CORRELATED", "INFERRED"].map((c) => (
                    <option key={c}>{c}</option>
                  ))}
                </select>
                <button
                  className="secondary compact"
                  onClick={() => void runSearch()}
                  disabled={!query}
                >
                  Apply filters
                </button>
                <button
                  className="text-button plan-toggle"
                  onClick={() => setShowPlan(!showPlan)}
                >
                  {showPlan ? "Hide" : "Inspect"} query plan{" "}
                  <GitBranch size={15} />
                </button>
              </div>
              {showPlan && response && (
                <div className="query-plan">
                  <strong>Structured retrieval plan</strong>
                  <pre>{JSON.stringify(response.parsed, null, 2)}</pre>
                </div>
              )}
              {response && (
                <div className="results-meta">
                  <strong>{response.total} matching events</strong>
                  <span>
                    {response.candidates_evaluated} candidates evaluated{" "}
                    {response.elapsed_ms !== null
                      ? `· ${response.elapsed_ms} ms`
                      : ""}
                  </span>
                </div>
              )}
              {busy ? (
                <div className="loading">
                  <LoaderCircle className="spin" />
                  Retrieving evidence…
                </div>
              ) : response?.results.length ? (
                <>
                  <div className="results-grid">
                    {response.results.map((e, i) => (
                      <article className="result-card" key={e.id}>
                        <button
                          className="result-image"
                          onClick={() => void open(e)}
                        >
                          <img
                            src={e.thumbnail_url}
                            alt={`Source frame: ${e.title}`}
                          />
                          <span className="result-rank">
                            {String(i + 1).padStart(2, "0")}
                          </span>
                          <span className="result-camera">
                            {e.camera_id} · {time(e.start)}
                          </span>
                          <span className="play-circle">
                            <Play size={17} fill="currentColor" />
                          </span>
                        </button>
                        <div className="result-content">
                          <div className="result-tags">
                            <Badge category={e.category} />
                            <span>
                              {Math.round((e.score || 0) * 100)}% relevance
                            </span>
                          </div>
                          <h3>{e.title}</h3>
                          <p>
                            {e.location} ·{" "}
                            {e.entities.map((x) => x.id).join(", ")}
                          </p>
                          <div className="result-reason">
                            <Check size={14} />
                            {e.explanations?.[1] ||
                              e.explanations?.[0] ||
                              e.evidence.method}
                          </div>
                          <div className="result-actions">
                            <button
                              className="text-button"
                              onClick={() => void open(e)}
                            >
                              Open evidence <ArrowRight size={15} />
                            </button>
                            <button
                              className={`icon-button ${findings.some((f) => f.id === e.id) ? "mint-text" : ""}`}
                              aria-label={`Collect ${e.id}`}
                              onClick={() => void collect(e)}
                            >
                              <Bookmark size={17} />
                            </button>
                          </div>
                        </div>
                      </article>
                    ))}
                  </div>
                  <p className="fine-print">
                    {response.score_notice}{" "}
                    {response.results.some((x) => x.is_demo)
                      ? "Demo events are authored synthetic annotations, not model predictions."
                      : ""}
                  </p>
                </>
              ) : response ? (
                <>
                  <Empty
                    title={response.parsed.unsupported_activity ? "Activity not supported" : "No sufficiently strong match found"}
                    detail={response.parsed.unsupported_activity
                      ? `The current detector cannot infer ${String(response.parsed.unsupported_activity).replaceAll("_", " ")}. No observation was treated as proof of this activity.`
                      : "No indexed observation satisfies this query. This does not establish that the event or person was absent."}
                  />
                  {response.coverage && (
                    <div className="panel coverage">
                      <h3>Coverage assessment</h3>
                      <dl>
                        <dt>Footage available</dt>
                        <dd>
                          {response.coverage.footage_available ? "Yes" : "No"}
                        </dd>
                        <dt>Requested interval covered</dt>
                        <dd>
                          {String(response.coverage.requested_interval_covered)}
                        </dd>
                        <dt>Camera operational</dt>
                        <dd>{response.coverage.camera_operational}</dd>
                        <dt>Absence confidence</dt>
                        <dd>{response.coverage.absence_confidence}</dd>
                      </dl>
                    </div>
                  )}
                </>
              ) : (
                <Empty
                  title="Begin with a question"
                  detail={overview?.demo_mode
                    ? "Try “Find the person who left an object near the east entrance”, or upload a reference image."
                    : "Try “vehicle started” or “person approached another person”, then inspect the source evidence."}
                />
              )}
            </>
          )}
          {screen === "ingestion" && (
            <Ingestion
              videos={videos}
              refresh={refresh}
              notify={setToast}
              onOpen={(id) => void runSearch("Show events", { camera: id })}
            />
          )}
          {screen === "investigation" && (
            <>
              <div className="page-heading">
                <div>
                  <div className="eyebrow">INVESTIGATION WORKSPACE</div>
                  <h1>Build the evidence chain.</h1>
                  <p>
                    Inspect the source. Collect findings. Keep uncertainty
                    visible.
                  </p>
                </div>
                <div className="button-row">
                  <button className="secondary" onClick={() => void newCase()}>
                    <Plus size={16} />
                    New case
                  </button>
                  <button
                    className="primary"
                    disabled={!activeCase || !findings.length}
                    onClick={() => void report()}
                  >
                    <ArrowDownToLine size={16} />
                    Export report
                  </button>
                </div>
              </div>
              <div className="case-toolbar">
                <Fingerprint size={17} />
                <select
                  aria-label="Active investigation"
                  value={activeCase}
                  onChange={(e) => setActiveCase(e.target.value)}
                >
                  {!investigations.length && (
                    <option value="">No investigation yet</option>
                  )}
                  {investigations.map((i) => (
                    <option value={i.id} key={i.id}>
                      {i.title}
                    </option>
                  ))}
                </select>
                <span>{findings.length} findings collected</span>
              </div>
              {selected ? (
                <EvidenceWorkspace
                  event={selected}
                  onOpen={(e) => void open(e)}
                  onCollect={(e) => void collect(e)}
                  collected={findings.some((e) => e.id === selected.id)}
                  onSearch={(q, extra) => void runSearch(q, extra)}
                  onError={setError}
                />
              ) : (
                <Empty
                  title="Select evidence to investigate"
                  detail="Open a search result or a collected finding to inspect its source video, timeline, and related entities."
                />
              )}
              {findings.length > 0 && (
                <section className="panel findings">
                  <div className="section-heading">
                    <h2>
                      Collected findings{" "}
                      <span className="count">{findings.length}</span>
                    </h2>
                    <FileText size={18} />
                  </div>
                  {findings.map((f) => (
                    <button
                      className="finding"
                      key={f.id}
                      onClick={() => void open(f)}
                    >
                      <span className="mint-text">
                        <Bookmark size={17} />
                      </span>
                      <strong>{f.title}</strong>
                      <span>{time(f.start)}</span>
                      <Badge category={f.category} />
                      <ChevronRight size={16} />
                    </button>
                  ))}
                </section>
              )}
            </>
          )}
          {screen === "graph" && (
            <EvidenceGraph
              onOpen={(id) => void open(id)}
              onSearch={(id) =>
                void runSearch("Show all related events", { entity_id: id })
              }
            />
          )}
          {screen === "evaluation" && <Evaluation notify={setToast} />}
        </main>
      </div>
      {toast && (
        <div className="toast" role="status">
          <ShieldCheck size={18} />
          {toast}
          <button
            aria-label="Dismiss notification"
            onClick={() => setToast("")}
          >
            <X size={15} />
          </button>
        </div>
      )}
    </div>
  );
}
