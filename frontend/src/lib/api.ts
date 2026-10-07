export type Category = "OBSERVED" | "CORRELATED" | "INFERRED";
export interface Entity {
  id: string;
  object_type: string;
  camera_id: string;
  attributes: Record<string, string>;
}
export interface Evidence {
  id: string;
  video_id: string;
  observation_id: string | null;
  frame_start: number;
  frame_end: number;
  timestamp_start: number;
  timestamp_end: number;
  sha256: string;
  method: string;
}
export interface EvidenceEvent {
  id: string;
  title: string;
  description: string;
  event_type: string;
  camera_id: string;
  camera_name: string;
  location: string;
  start: string;
  end: string;
  category: Category;
  confidence: number | null;
  evidence: Evidence;
  entities: Entity[];
  is_demo: boolean;
  media_url: string;
  thumbnail_url: string;
  video_id: string;
  source_sha256: string;
  score?: number;
  signals?: Record<string, number | null>;
  explanations?: string[];
}
export interface Video {
  id: string;
  camera_id: string;
  filename: string;
  recording_start: string;
  duration: number;
  status: string;
  progress: number;
  error: string | null;
  is_demo: boolean;
  pipeline: string;
  width: number;
  height: number;
}
export interface Camera {
  id: string;
  name: string;
  location: string;
}
export interface Investigation {
  id: string;
  title: string;
  query: string;
  created_at: string;
  findings?: EvidenceEvent[];
}
export interface Overview {
  cameras: number;
  videos: number;
  indexed_seconds: number;
  events: number;
  entities: number;
  demo_mode: boolean;
  recent_events: EvidenceEvent[];
  investigations: Investigation[];
  pipeline: string;
}
export interface SearchResult {
  query: string;
  results: EvidenceEvent[];
  total: number;
  parsed: Record<string, unknown>;
  candidates_evaluated: number;
  elapsed_ms: number | null;
  coverage?: {
    camera_operational: string;
    footage_available: boolean;
    requested_interval_covered: string | boolean;
    conclusion: string;
    absence_confidence: string;
    intervals: {
      camera_id: string;
      start: string;
      end: string;
      indexed: boolean;
    }[];
  };
  score_notice: string;
}
export interface Association {
  entity_id: string;
  camera_id: string;
  category: Category;
  status: string;
  score: number;
  explanation: string;
  signals: Record<string, number | boolean>;
}
export interface EvalRun {
  id: string;
  created_at: string;
  results: {
    dataset: string;
    scope: string;
    query_count: number;
    positive_queries: number;
    negative_queries: number;
    metrics: Record<string, number | null>;
    limitations: string[];
    queries: {
      id: string;
      query: string;
      case: string;
      expected: string[];
      retrieved: string[];
      correct_top1: boolean;
      latency_ms: number;
    }[];
  };
}
export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    ...init,
    headers: {
      ...(init?.body && !(init.body instanceof FormData)
        ? { "content-type": "application/json" }
        : {}),
      ...init?.headers,
    },
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : `Request failed (${response.status})`,
    );
  }
  return response.json();
}
export const post = <T>(path: string, body: unknown = {}) =>
  api<T>(path, { method: "POST", body: JSON.stringify(body) });
export const time = (value: string) => value.slice(11, 19);
export const duration = (seconds: number) =>
  `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
export function download(text: string, name: string, type = "text/markdown") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
