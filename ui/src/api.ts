import type {
  ChatResult,
  Health,
  IngestRecord,
  JobStatus,
  PipelineLog,
  ReportDetail,
  ReportListItem,
  StreamEvent,
  Trace,
  TraceSummary,
  Versions,
} from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      // keep the status text
    }
    throw new Error(`${response.status}: ${detail}`);
  }
  return (await response.json()) as T;
}

const withVersion = (path: string, version?: string) => (version ? `${path}?version=${encodeURIComponent(version)}` : path);

export const api = {
  health: () => request<Health>("/api/health"),
  versions: () => request<Versions>("/api/versions"),
  startChat: (question: string, k: number) =>
    request<{ job_id: string }>("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, k }),
    }),
  chatStatus: (jobId: string) => request<JobStatus>(`/api/chat/${jobId}`),
  traces: (version?: string) => request<TraceSummary[]>(withVersion("/api/traces", version)),
  trace: (id: string) => request<Trace>(`/api/traces/${id}`),
  pipelineLog: (lines: number) => request<PipelineLog>(`/api/logs/pipeline?lines=${lines}`),
  ingestLog: () => request<IngestRecord[]>("/api/logs/ingest"),
  reports: (version?: string) => request<ReportListItem[]>(withVersion("/api/reports", version)),
  report: (name: string) => request<ReportDetail>(`/api/reports/${encodeURIComponent(name)}`),
};

export async function askQuestion(question: string, k: number): Promise<ChatResult> {
  const { job_id } = await api.startChat(question, k);
  for (;;) {
    await new Promise((resolve) => setTimeout(resolve, 2000));
    const job = await api.chatStatus(job_id);
    if (job.status === "done" && job.result) return job.result;
    if (job.status === "error") throw new Error(job.error ?? "The query failed");
  }
}

/** Ask a question and receive the answer as it is written (Server-Sent Events from /api/chat/stream). `onEvent` gets every
 *  stage and token; the promise resolves with the final result (which carries ttft_ms). */
export async function streamQuestion(question: string, k: number, onEvent: (event: StreamEvent) => void): Promise<ChatResult> {
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ question, k }),
  });
  if (!response.ok || !response.body) throw new Error(`${response.status}: ${response.statusText}`);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let end: number;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      const data = block
        .split("\n")
        .filter((line) => line.startsWith("data: "))
        .map((line) => line.slice(6))
        .join("\n");
      if (!data) continue;
      const event = JSON.parse(data) as StreamEvent;
      onEvent(event);
      if (event.type === "done") return event.result;
      if (event.type === "error") throw new Error(event.error);
    }
  }
  throw new Error("The answer stream ended before the answer was complete");
}
