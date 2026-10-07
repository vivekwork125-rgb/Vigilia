"use client";
import { useEffect, useState } from "react";
import { ArrowRight, GitBranch, Search, ZoomIn, ZoomOut } from "lucide-react";
import { api } from "@/lib/api";
import { Badge } from "./workspace";
interface Node {
  id: string;
  label: string;
  kind: string;
  category: string;
  camera_id: string;
}
interface Edge {
  source: string;
  target: string;
  label: string;
  category: string;
  evidence_id: string | null;
  event_id?: string | null;
}
interface Graph {
  nodes: Node[];
  edges: Edge[];
}
export function EvidenceGraph({
  onOpen,
  onSearch,
}: {
  onOpen: (id: string) => void;
  onSearch: (id: string) => void;
}) {
  const [graph, setGraph] = useState<Graph>({ nodes: [], edges: [] });
  const [entity, setEntity] = useState("");
  const [selected, setSelected] = useState<Node | null>(null);
  const [error, setError] = useState("");
  const [zoom, setZoom] = useState(1);
  useEffect(() => {
    let live = true;
    api<Graph>("/graph")
      .then((g) => {
        if (live) {
          setGraph(g);
          setEntity(g.nodes.find((n) => n.kind === "person")?.id || "");
        }
      })
      .catch((e) => {
        if (live) setError(e.message);
      });
    return () => {
      live = false;
    };
  }, []);
  const primary = graph.nodes.find((n) => n.id === entity);
  const links = graph.edges.filter(
    (e) => e.source === entity && graph.nodes.some((n) => n.id === e.target && n.kind === "event"),
  );
  const direct = graph.edges.filter(
    (e) => e.event_id && (e.source === entity || e.target === entity),
  );
  const events = links
    .map((e) => graph.nodes.find((n) => n.id === e.target))
    .filter((n): n is Node => !!n);
  const others = [
    ...new Set(
      graph.edges
        .filter(
          (e) => events.some((n) => n.id === e.target) && e.source !== entity,
        )
        .map((e) => e.source),
    ),
  ]
    .map((id) => graph.nodes.find((n) => n.id === id))
    .filter((n): n is Node => !!n);
  const height = Math.max(500, events.length * 100 + 80);
  const eventPos = (i: number) => 70 + i * 100;
  return (
    <>
      <div className="page-heading">
        <div>
          <div className="eyebrow">RELATIONSHIP EXPLORER</div>
          <h1>Context is connected.</h1>
          <p>
            Follow an entity through its events. Inspect the source behind every
            edge.
          </p>
        </div>
        <div className="button-row">
          <Badge category="OBSERVED" />
          <Badge category="INFERRED" />
        </div>
      </div>
      {error && <div className="error-banner">{error}</div>}
      <div className="graph-toolbar">
        <GitBranch size={18} />
        <label>
          Focus entity{" "}
          <select
            value={entity}
            onChange={(e) => {
              setEntity(e.target.value);
              setSelected(null);
            }}
          >
            {graph.nodes
              .filter((n) => !["event", "camera", "location"].includes(n.kind))
              .map((n) => (
                <option key={n.id} value={n.id}>
                  {n.id} · {n.kind} · {n.camera_id}
                </option>
              ))}
          </select>
        </label>
        <span>Click a node or edge to inspect evidence</span>
        <button
          className="icon-button"
          aria-label="Zoom out"
          onClick={() => setZoom((z) => Math.max(0.6, z - 0.1))}
        >
          <ZoomOut size={18} />
        </button>
        <button
          className="icon-button"
          aria-label="Zoom in"
          onClick={() => setZoom((z) => Math.min(1.5, z + 0.1))}
        >
          <ZoomIn size={18} />
        </button>
      </div>
      <div className="graph-layout">
        <div className="graph-canvas">
          <svg
            viewBox={`0 0 850 ${height}`}
            style={{ minWidth: 650 * zoom }}
            role="img"
            aria-label={`Evidence graph for ${entity}`}
          >
            <defs>
              <pattern
                id="dots"
                width="24"
                height="24"
                patternUnits="userSpaceOnUse"
              >
                <circle cx="2" cy="2" r="1" fill="#273436" />
              </pattern>
              <marker
                id="arrow"
                viewBox="0 0 10 10"
                refX="9"
                refY="5"
                markerWidth="5"
                markerHeight="5"
                orient="auto-start-reverse"
              >
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#66866f" />
              </marker>
            </defs>
            <rect width="100%" height="100%" fill="url(#dots)" />
            {events.map((n, i) => (
              <g key={n.id}>
                <path
                  d={`M 240 ${height / 2} C 340 ${height / 2}, 310 ${eventPos(i)}, 410 ${eventPos(i)}`}
                  fill="none"
                  stroke={n.category === "INFERRED" ? "#ad9871" : "#678a78"}
                  strokeWidth="1.5"
                  strokeDasharray={
                    n.category === "INFERRED" ? "5 5" : undefined
                  }
                  markerEnd="url(#arrow)"
                />
                <path
                  d={`M 240 ${height / 2} C 340 ${height / 2}, 310 ${eventPos(i)}, 410 ${eventPos(i)}`}
                  fill="none"
                  stroke="transparent"
                  strokeWidth="18"
                  className="graph-node"
                  role="button"
                  tabIndex={0}
                  aria-label={`Inspect relationship ${n.label}`}
                  onClick={() => onOpen(n.id)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") onOpen(n.id);
                  }}
                />
              </g>
            ))}
            {others.map((n, i) => (
              <g key={n.id}>
                {graph.edges
                  .filter(
                    (e) =>
                      e.source === n.id &&
                      events.some((x) => x.id === e.target),
                  )
                  .map((e) => (
                    <path
                      key={e.target}
                      d={`M 645 ${eventPos(events.findIndex((x) => x.id === e.target))} L 750 ${height / 2 + i * 85}`}
                      stroke="#5d7779"
                      strokeWidth="1.5"
                    />
                  ))}
                <g
                  className="graph-node"
                  role="button"
                  tabIndex={0}
                  aria-label={`Focus ${n.id}`}
                  onClick={() => setEntity(n.id)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") setEntity(n.id);
                  }}
                >
                  <rect
                    x="698"
                    y={height / 2 + i * 85 - 25}
                    width="112"
                    height="50"
                    rx="8"
                    fill="#202f31"
                    stroke="#547575"
                  />
                  <text
                    x="754"
                    y={height / 2 + i * 85}
                    textAnchor="middle"
                    fill="#c1dbdb"
                    fontSize="12"
                  >
                    {n.id}
                  </text>
                  <text
                    x="754"
                    y={height / 2 + i * 85 + 16}
                    textAnchor="middle"
                    fill="#7b9395"
                    fontSize="10"
                  >
                    {n.kind}
                  </text>
                </g>
              </g>
            ))}
            {primary && (
              <g
                className="graph-node"
                role="button"
                tabIndex={0}
                aria-label={`Inspect ${primary.id}`}
                onClick={() => setSelected(primary)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") setSelected(primary);
                }}
              >
                <rect
                  x="65"
                  y={height / 2 - 46}
                  width="175"
                  height="92"
                  rx="12"
                  fill="#28372a"
                  stroke="#b9e698"
                  strokeWidth="1.5"
                />
                <text
                  x="152"
                  y={height / 2 - 15}
                  textAnchor="middle"
                  fill="#9ba99b"
                  fontSize="10"
                >
                  {primary.kind.toUpperCase()} / {primary.camera_id}
                </text>
                <text
                  x="152"
                  y={height / 2 + 10}
                  textAnchor="middle"
                  fill="#d9f3c4"
                  fontSize="20"
                  fontWeight="600"
                >
                  {primary.id}
                </text>
                <text
                  x="152"
                  y={height / 2 + 29}
                  textAnchor="middle"
                  fill="#a4b998"
                  fontSize="10"
                >
                  CAMERA-SCOPED ENTITY
                </text>
              </g>
            )}
            {events.map((n, i) => (
              <g
                key={n.id}
                className="graph-node"
                role="button"
                tabIndex={0}
                aria-label={`Inspect ${n.id}`}
                onClick={() => setSelected(n)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") setSelected(n);
                }}
              >
                <rect
                  x="410"
                  y={eventPos(i) - 31}
                  width="235"
                  height="64"
                  rx="8"
                  fill={selected?.id === n.id ? "#293d32" : "#1b282a"}
                  stroke={n.category === "INFERRED" ? "#9b8560" : "#486552"}
                />
                <text
                  x="427"
                  y={eventPos(i) - 10}
                  fill={n.category === "INFERRED" ? "#d4b880" : "#a9d3b1"}
                  fontSize="10"
                >
                  {n.id} · {n.category}
                </text>
                <text x="427" y={eventPos(i) + 11} fill="#d4dedb" fontSize="11">
                  {links[i].label.replaceAll("_", " ")}
                </text>
              </g>
            ))}
          </svg>
        </div>
        <aside className="panel graph-inspector">
          <h3>Evidence inspector</h3>
          {selected ? (
            <>
              <Badge category={selected.category} />
              <h2>{selected.id}</h2>
              <p>{selected.label}</p>
              <dl>
                <dt>Source camera</dt>
                <dd>{selected.camera_id}</dd>
                <dt>Node type</dt>
                <dd>{selected.kind}</dd>
              </dl>
              <button
                className="primary full-button"
                onClick={() =>
                  selected.kind === "event"
                    ? onOpen(selected.id)
                    : onSearch(selected.id)
                }
              >
                Open {selected.kind === "event" ? "evidence" : "related events"}
                <ArrowRight size={15} />
              </button>
            </>
          ) : (
            <>
              <GitBranch size={36} className="muted" />
              <p>Select a node to inspect its source evidence.</p>
            </>
          )}
          {direct.length > 0 && (
            <div className="graph-note">
              <h4>Stored relationships</h4>
              {direct.map((edge) => (
                <button
                  className="text-button"
                  key={`${edge.source}-${edge.target}-${edge.evidence_id}`}
                  onClick={() => edge.event_id && onOpen(edge.event_id)}
                >
                  {edge.label.replaceAll("_", " ")} · {edge.source === entity ? edge.target : edge.source} · {edge.category}
                  <ArrowRight size={14} />
                </button>
              ))}
            </div>
          )}
          <div className="graph-note">
            <h4>No hidden identity links</h4>
            <p>
              Solid links connect entities to source-backed events. Dashed links
              denote inferred events. Cross-camera candidates stay separate in
              the investigation workspace.
            </p>
          </div>
        </aside>
      </div>
    </>
  );
}
