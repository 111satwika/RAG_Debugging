export type Chunk = {
  rank: number;
  chunk_id: string;
  doc_id: string;
  doc_type: string;
  page: number;
  score: number;
  text: string;
};

export type Validation = {
  grade: "correct" | "ambiguous" | "incorrect";
  first_grade: "correct" | "ambiguous" | "incorrect";
  best_score: number | null;
  action: "answer" | "abstain";
  reason: string;
  retries: {
    query: string;
    skipped?: string;
    search_ms?: number;
    grade_before?: string;
    grade_after?: string;
    best_retry_score?: number | null;
  }[];
};

/** One retrieved chunk with its CRAG grade (from its reranker score) and whether the answer model read it. */
export type ContextChunk = {
  chunk_id: string;
  doc_id: string | null;
  page: number | null;
  score: number | null;
  grade: "correct" | "ambiguous" | "incorrect";
  used: boolean;
};

export type ChatResult = {
  question: string;
  answer: string;
  trace_id: string;
  chunks: Chunk[];
  timings: Record<string, number>;
  total_ms: number;
  model: string | null;
  input_tokens: number | null;
  output_tokens: number | null;
  /** Cited prompts only (v2 and later): whether the documents held the answer, how well the reply followed the
   *  structured format, and the sources the answer cites. All null for the plain baseline prompt. */
  found?: boolean | null;
  answer_status?: "ok" | "flawed" | "invalid" | null;
  sources?: { source_id: string; chunk_id: string }[] | null;
  /** Versions with the validation layer (v4): the grade of the retrieved context and what was done about it. */
  validation?: Validation | null;
  /** Streamed answers only: ms from the question to the first answer character shown, and from the model call to the
   *  model's first token (earlier, because the model writes its "checks" before the answer). */
  ttft_ms?: number | null;
  model_ttft_ms?: number | null;
  /** Versions with validation: every retrieved chunk with its CRAG grade. */
  context?: ContextChunk[] | null;
};

/** One Server-Sent Event of /api/chat/stream. */
export type StreamEvent =
  | { type: "stage"; stage: string }
  | { type: "token"; text: string }
  | { type: "validation"; validation: Validation }
  | { type: "context"; context: ContextChunk[] }
  | { type: "done"; result: ChatResult }
  | { type: "error"; error: string };

export type JobStatus = {
  status: "queued" | "running" | "done" | "error";
  elapsed_s: number;
  result: ChatResult | null;
  error: string | null;
};

export type ChatMessage =
  | { id: string; role: "user"; text: string }
  | { id: string; role: "assistant"; status: "pending"; startedAt: number; stage?: string; text?: string; firstTokenAt?: number;
      validation?: Validation; context?: ContextChunk[] }
  | { id: string; role: "assistant"; status: "done"; result: ChatResult }
  | { id: string; role: "assistant"; status: "error"; error: string };

export type Health = {
  status: string;
  config: Record<string, string | number>;
  indexed_chunks: number | null;
  llm: { provider: string; model: string; reachable: boolean | null };
};

export type TraceSummary = {
  trace_id: string;
  timestamp: string;
  query: string;
  total_ms: number | null;
  error: string | null;
  answer_preview: string;
  top_score: number | null;
  llm: string | null;
  version: string;
};

export type Versions = { versions: string[]; current: string };

export type RetrievedPreview = {
  chunk_id: string;
  doc_id: string;
  page: number;
  score: number;
  preview: string;
};

export type Span = {
  ms: number;
  error?: string;
  results?: RetrievedPreview[];
  final_prompt?: string;
  chars?: number;
  model?: string;
  input_tokens?: number | null;
  output_tokens?: number | null;
  temperature?: number;
  dim?: number;
} & Partial<Validation>;

export type Golden = {
  id: string;
  type: string;
  expected_answer: string;
  source_docs: string[];
  source_section: string | null;
  must_include: string[] | null;
  must_not_include: string[] | null;
  notes: string | null;
};

export type Relevance =
  | { available: true; relevant_ids: string[]; weak: boolean; missing: string[] }
  | { available: false; reason: string };

export type Trace = {
  trace_id: string;
  timestamp: string;
  config: Record<string, string | number>;
  query: string;
  spans: Record<string, Span>;
  answer: string | null;
  total_ms: number | null;
  error: string | null;
  golden: Golden | null;
  relevance: Relevance | null;
  version: string;
};

export type LogEntry = { time: string; level: string; logger: string; message: string };
export type PipelineLog = { entries: LogEntry[]; size_bytes: number };

export type IngestRecord = {
  timestamp: string;
  file: string;
  doc_type: string;
  pages: number;
  empty_pages: number;
  chars: number;
  chunks: number;
  error: string | null;
};

export type Num = number | null;

export type Summary = {
  n: number;
  n_judged: number;
  n_retrieval: number;
  n_failed: number;
  k: number;
  recall: Num;
  precision: Num;
  context_precision?: Num;   // relevant / chunks given to the model; missing in runs made before it existed
  mrr: Num;
  ndcg: Num;
  hit: Num;
  correctness: Num;
  relevance: Num;
  completeness: Num;
  faithfulness: Num;
  keyword_include: Num;
  abstention_accuracy: Num;
  refusal_rate_unanswerable: Num;
  false_abstention_rate: Num;
  latency_p50_s: Num;
  latency_p95_s: Num;
  avg_input_tokens: Num;
  avg_output_tokens: Num;
  /** Cited prompts only (v2 and later); null for runs made with the plain baseline prompt. */
  citation_rate?: Num;
  cites_relevant_rate?: Num;
  format_ok_rate?: Num;
  format_invalid_rate?: Num;
  found_agreement?: Num;
};

export type ReportListItem = {
  name: string;
  created: string;
  note: string;
  k: number | null;
  config: Record<string, string | number>;
  judge: string | null;
  version: string;
  overall: Summary;
};

export type Judge = {
  correctness: number;
  relevance: number;
  completeness: Num;
  faithfulness: number;
  abstained: boolean;
  reasoning: string;
  unsupported_claims: string[];
};

export type QuestionResult = {
  id: string;
  type: string;
  question: string;
  expected: string;
  answer: string | null;
  error: string | null;
  trace_id: string | null;
  retrieved: { chunk_id: string; doc_id: string; page: number; score: number }[];
  relevant_ids: string[];
  weak_labels: boolean;
  retrieval: { recall: number; precision: number; context_precision?: number; mrr: number; ndcg: number; hit: number } | null;
  keyword: { include_rate: Num; missing: string[]; forbidden: string[] } | null;
  judge: Judge | null;
  failure: { stage: string; note: string } | null;
};

export type ReportDetail = {
  meta: { name: string; created: string; note: string; k: number; config: Record<string, string | number>; judge: string };
  overall: Summary;
  by_type: Record<string, Summary>;
  results: QuestionResult[];
  markdown: string;
  version: string;
};
