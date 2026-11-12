import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { fmtInt, fmtMs } from "../format";
import type { ChatMessage, ChatResult, ContextChunk, Health, Validation } from "../types";
import { Icon } from "./Icon";

const EXAMPLES = [
  "What does error code QL-1003 mean?",
  "How many days of Sick Leave does QuickLeave allow per year?",
  "What BLEU scores did the base and big Transformer models get on WMT 2014 English-to-German?",
  "Which database does QuickLeave use?",
];

type Props = {
  messages: ChatMessage[];
  busy: boolean;
  health: Health | null;
  apiDown: boolean;
  onSend: (question: string, k: number) => void;
  onClear: () => void;
  onOpenTrace: (id: string) => void;
};

const STAGES: Record<string, string> = {
  queued: "Waiting for the previous question to finish",
  retrieve: "Searching the documents",
  validate: "Checking the retrieved context",
  refine: "Filtering the passages",
  generate: "Writing the answer",
};

type PendingMessage = Extract<ChatMessage, { status: "pending" }>;

const GRADES: Record<Validation["grade"], { label: string; tone: string; meaning: string }> = {
  correct: { label: "Correct", tone: "good", meaning: "the retrieved context is relevant" },
  ambiguous: { label: "Partial", tone: "warn", meaning: "the retrieved context is only partly relevant (CRAG: ambiguous)" },
  incorrect: { label: "Incorrect", tone: "bad", meaning: "the retrieved context is not relevant" },
};

/** "Retrieved 5: 2 correct · 1 partial · 2 incorrect": the CRAG grade of each retrieved chunk, counted. */
function CragCounts({ context }: { context: ContextChunk[] }) {
  const count = (grade: ContextChunk["grade"]) => context.filter((c) => c.grade === grade).length;
  const used = context.filter((c) => c.used).length;
  return (
    <div className="crag-counts small">
      <span className="muted">Retrieved {context.length}:</span>
      <span className="count good">{count("correct")} correct</span>
      <span className="count warn">{count("ambiguous")} partial</span>
      <span className="count bad">{count("incorrect")} incorrect</span>
      <span className="muted">· {used} read by the model</span>
    </div>
  );
}

/** The sources of an answer by name only: each retrieved chunk with its CRAG grade, score, whether the model read it and
 *  the citation id it got. Without per-chunk grades (versions without validation), the chunks the model read. */
/** The [S#] id a chunk had in the prompt: S1 for the first chunk the model read, S2 for the next (src/generate/generate.py). */
function sourceIdOf(result: ChatResult, chunkId: string): string | undefined {
  const index = result.chunks.findIndex((c) => c.chunk_id === chunkId);
  return index < 0 ? undefined : `S${index + 1}`;
}

function SourceList({ result, active, onSelect }: { result: ChatResult; active: string | null; onSelect: (sourceId: string | null) => void }) {
  const [open, setOpen] = useState(false);
  const activeRef = useRef<HTMLLIElement>(null);
  const cited = new Set(result.sources?.map((s) => s.source_id) ?? []);
  const rows: { chunk_id: string; page: number | null; score: number | null; grade?: ContextChunk["grade"]; used: boolean }[] =
    result.context ?? result.chunks.map((c) => ({ chunk_id: c.chunk_id, page: c.page, score: c.score, used: true }));

  useEffect(() => {
    // a click on a citation in the answer opens the list and brings that source into view
    if (active) {
      setOpen(true);
      requestAnimationFrame(() => activeRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" }));
    }
  }, [active]);

  return (
    <details className="sources" open={open} onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
      <summary>
        Sources ({rows.length}
        {result.sources ? `, ${result.sources.length} cited` : ""})
      </summary>
      <ul className="source-names">
        {rows.map((c) => {
          const sid = sourceIdOf(result, c.chunk_id);           // only chunks the model read have an [S#] id
          const isActive = sid !== undefined && sid === active;
          const text = result.chunks.find((x) => x.chunk_id === c.chunk_id)?.text;
          return (
            <li key={c.chunk_id} ref={isActive ? activeRef : undefined} className={[c.used ? "" : "filtered", isActive ? "active" : ""].join(" ")}>
              <div className="source-row">
                {sid ? (
                  <button className="link source-name" title="Show or hide this chunk's text" onClick={() => onSelect(isActive ? null : sid)}>
                    <code>{c.chunk_id}</code>
                  </button>
                ) : (
                  <code>{c.chunk_id}</code>
                )}
                {c.page != null && <span className="muted"> p.{c.page}</span>}
                {c.grade && <span className={`badge ${GRADES[c.grade].tone}`}>{GRADES[c.grade].label}</span>}
                {c.score != null && <span className="muted small"> {c.score.toFixed(3)}</span>}
                {sid && cited.has(sid) && <span className="badge good">cited {sid}</span>}
                {!c.used && <span className="muted small"> · not read (filtered out)</span>}
              </div>
              {isActive && text && <pre className="source-text">{text}</pre>}
            </li>
          );
        })}
      </ul>
    </details>
  );
}

/** The answer text with every [S#] citation turned into a button that opens that source. */
function AnswerText({ text, known, onCite }: { text: string; known: (sourceId: string) => boolean; onCite: (sourceId: string) => void }) {
  const linked = text.replace(/\[(S\d+)\]/g, (whole, sid: string) => (known(sid) ? `[${sid}](#cite-${sid})` : whole));
  return (
    <ReactMarkdown
      components={{
        a: ({ href, children }) =>
          href?.startsWith("#cite-") ? (
            <button className="cite" title="Show this source" onClick={() => onCite(href.slice("#cite-".length))}>
              [{children}]
            </button>
          ) : (
            <a href={href} target="_blank" rel="noreferrer">
              {children}
            </a>
          ),
      }}
    >
      {linked}
    </ReactMarkdown>
  );
}

/** A finished answer: CRAG grade, the answer with clickable citations, timings, and the sources by name. */
function FinishedAnswer({ result, onOpenTrace }: { result: ChatResult; onOpenTrace: (id: string) => void }) {
  const [active, setActive] = useState<string | null>(null);     // the [S#] whose text is shown in the sources
  const known = (sid: string) => Number(sid.slice(1)) >= 1 && Number(sid.slice(1)) <= result.chunks.length;
  return (
    <div className="bubble assistant">
      {result.validation && <CragBadge validation={result.validation} />}
      {result.context && <CragCounts context={result.context} />}
      <div className="markdown">
        <AnswerText text={result.answer} known={known} onCite={setActive} />
      </div>
      {result.found === false && (
        <div className="meta">
          <span className="badge bad" title="The model found nothing in the retrieved sources that answers the question">
            not found in the documents
          </span>
        </div>
      )}
      {result.answer_status && result.answer_status !== "ok" && (
        <div className="meta">
          <span className="badge bad" title="The reply did not fully follow the structured answer format">
            {result.answer_status === "invalid" ? "reply was not valid JSON" : "citation format issue"}
          </span>
        </div>
      )}
      <div className="meta">
        {result.ttft_ms != null && (
          <span
            title={`Time to first token: from the question to the first answer character. The model's own first token came ${fmtMs(result.model_ttft_ms ?? null)} after it was called.`}
          >
            TTFT {fmtMs(result.ttft_ms)}
          </span>
        )}
        <span>{fmtMs(result.total_ms)}</span>
        <span>{fmtInt(result.input_tokens)} tokens in</span>
        <button className="link" onClick={() => onOpenTrace(result.trace_id)}>
          Open trace {result.trace_id}
        </button>
      </div>
      <SourceList result={result} active={active} onSelect={setActive} />
    </div>
  );
}

/** The CRAG grade of the retrieved context for this query, and what the pipeline did about it. */
function CragBadge({ validation }: { validation: Validation }) {
  const grade = GRADES[validation.grade];
  const corrected = validation.retries.some((r) => r.search_ms !== undefined);
  const action =
    validation.action === "abstain"
      ? "not answered: no relevant context"
      : corrected
        ? `corrected by a second search (first graded ${GRADES[validation.first_grade].label.toLowerCase()})`
        : "answered from the first search";
  const score = validation.best_score == null ? "-" : validation.best_score.toFixed(3);
  return (
    <div className="crag">
      <span
        className={`badge ${grade.tone}`}
        title={`CRAG grade: ${grade.meaning}. Best reranker score ${score} (correct at 0.1 or more).`}
      >
        CRAG: {grade.label}
      </span>
      <span className="muted small">{action}</span>
    </div>
  );
}

/** The answer while it is being produced: the current step until the first token, then the text as it streams in. */
function Pending({ message }: { message: PendingMessage }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 250);
    return () => clearInterval(timer);
  }, []);
  const seconds = (ms: number) => `${(ms / 1000).toFixed(1)} s`;
  if (message.text) {
    return (
      <div className="bubble assistant">
        {message.validation && <CragBadge validation={message.validation} />}
        {message.context && <CragCounts context={message.context} />}
        <div className="markdown streaming">
          <ReactMarkdown>{message.text}</ReactMarkdown>
          <span className="cursor" />
        </div>
        <div className="meta">
          <span title="Time to first token: from sending the question to the first answer character">
            TTFT {seconds((message.firstTokenAt ?? now) - message.startedAt)}
          </span>
          <span>{seconds(now - message.startedAt)}</span>
        </div>
      </div>
    );
  }
  return (
    <div className="bubble assistant">
      {message.validation && <CragBadge validation={message.validation} />}
        {message.context && <CragCounts context={message.context} />}
      <div className="typing">
        <span />
        <span />
        <span />
      </div>
      <div className="muted small">
        {STAGES[message.stage ?? "queued"] ?? "Working"}... {seconds(now - message.startedAt)}
      </div>
    </div>
  );
}

export function Chat({ messages, busy, health, apiDown, onSend, onClear, onOpenTrace }: Props) {
  const [input, setInput] = useState("");
  const [k, setK] = useState(5);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (messages.length > 0) endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  const submit = (question: string) => {
    const text = question.trim();
    if (!text || busy) return;
    onSend(text, k);
    setInput("");
  };

  return (
    <div className="chat">
      <header className="page-header">
        <div>
          <h1>Ask your documents</h1>
          <p className="muted">
            Each question is answered on its own from the indexed resume, spec and paper.
            {health?.config.version && (
              <>
                {" "}
                Pipeline <span className="badge stage" title="The pipeline version that answers your questions (PIPELINE_VERSION in src/config.py)">{String(health.config.version)}</span>
              </>
            )}
          </p>
        </div>
        <div className="header-actions">
          <label className="inline-field">
            Top-k
            <select value={k} onChange={(e) => setK(Number(e.target.value))} disabled={busy}>
              {[1, 3, 5, 8, 10].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
          <button className="btn ghost" onClick={onClear} disabled={busy || messages.length === 0}>
            <Icon name="trash" size={15} /> Clear
          </button>
        </div>
      </header>

      {apiDown && <div className="banner bad">Cannot reach the API. Start it with: python -m uvicorn api.main:app --port 8765</div>}
      {!apiDown && health?.llm.reachable === false && (
        <div className="banner warn">The model server ({health.llm.provider}) is not reachable. Start it to get answers.</div>
      )}

      <div className="messages">
        {messages.length === 0 ? (
          <div className="empty">
            <h2>What would you like to know?</h2>
            <p className="muted">Try one of these, or type your own question.</p>
            <div className="chips">
              {EXAMPLES.map((example) => (
                <button key={example} className="chip" onClick={() => submit(example)} disabled={busy || apiDown}>
                  {example}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((m) => {
            if (m.role === "user") {
              return (
                <div key={m.id} className="row user">
                  <div className="bubble user">{m.text}</div>
                </div>
              );
            }
            return (
              <div key={m.id} className="row assistant">
                {m.status === "pending" && <Pending message={m} />}
                {m.status === "error" && <div className="bubble assistant error">{m.error}</div>}
                {m.status === "done" && <FinishedAnswer result={m.result} onOpenTrace={onOpenTrace} />}
              </div>
            );
          })
        )}
        <div ref={endRef} />
      </div>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          submit(input);
        }}
      >
        <textarea
          value={input}
          rows={1}
          placeholder={busy ? "Waiting for the answer..." : "Ask a question about your documents"}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit(input);
            }
          }}
          disabled={apiDown}
        />
        <button className="btn primary" type="submit" disabled={busy || apiDown || !input.trim()}>
          <Icon name="send" size={16} /> Send
        </button>
      </form>
    </div>
  );
}
