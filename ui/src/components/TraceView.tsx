import ReactMarkdown from "react-markdown";
import { fmtInt, fmtMs, fmtTime } from "../format";
import type { Trace } from "../types";
import { ChunkCard } from "./ChunkCard";

function RelevanceNote({ trace }: { trace: Trace }) {
  const rel = trace.relevance;
  if (trace.golden?.type === "unanswerable") {
    return <p className="muted small">No chunk should be relevant here: the documents do not answer this question.</p>;
  }
  if (!trace.golden) {
    return <p className="muted small">Relevant chunks can only be marked for questions that are in the golden set.</p>;
  }
  if (!rel) return null;
  if (!rel.available) return <p className="muted small">{rel.reason}</p>;
  return (
    <>
      {rel.missing.length > 0 && (
        <div className="banner warn">The chunk that holds the answer was not retrieved: {rel.missing.join(", ")}</div>
      )}
      <p className="muted small">
        Green "relevant" means the chunk holds the key facts of the expected answer. It is a text check against the golden
        set, not a model's opinion{rel.weak ? ", and a weak one here: no chunk held most of the facts" : ""}.
      </p>
    </>
  );
}

export function TraceView({ trace }: { trace: Trace }) {
  const total = trace.total_ms ?? Object.values(trace.spans).reduce((sum, s) => sum + s.ms, 0);
  const retrieve = trace.spans.retrieve;
  const llm = trace.spans.llm_call;
  const prompt = trace.spans.build_prompt;
  const relevant = new Set(trace.relevance?.available ? trace.relevance.relevant_ids : []);

  return (
    <div className="trace-view">
      <div className="trace-title">
        <h2>
          Trace <code>{trace.trace_id}</code>
        </h2>
        <span className="badge version">{trace.version}</span>
        <span className="muted">
          {fmtTime(trace.timestamp)} · total {fmtMs(trace.total_ms)}
        </span>
      </div>

      {trace.error && <div className="banner bad">{trace.error}</div>}

      <section>
        <h3>Question</h3>
        <div className="box">{trace.query}</div>
      </section>

      <section>
        <h3>Timing by stage</h3>
        <div className="timings">
          {Object.entries(trace.spans).map(([name, span]) => (
            <div key={name} className="timing-row">
              <span className="timing-name">{name}</span>
              <span className="timing-track">
                <span
                  className={`timing-fill ${span.error ? "bad" : ""}`}
                  style={{ width: `${Math.max(0.6, (span.ms / Math.max(total, 1)) * 100)}%` }}
                />
              </span>
              <span className="timing-ms">{fmtMs(span.ms)}</span>
            </div>
          ))}
        </div>
      </section>

      <section>
        <h3>Settings used</h3>
        <div className="chips static">
          {Object.entries(trace.config).map(([key, value]) => (
            <span key={key} className="tag">
              {key}: <b>{String(value)}</b>
            </span>
          ))}
        </div>
      </section>

      {trace.spans.validate && (
        <section>
          <h3>Validation</h3>
          <div className="chips static">
            <span className="tag">
              first grade: <b>{trace.spans.validate.first_grade}</b>
            </span>
            <span className="tag">
              final grade: <b>{trace.spans.validate.grade}</b>
            </span>
            <span className="tag">
              best score: <b>{trace.spans.validate.best_score?.toFixed(4) ?? "-"}</b>
            </span>
            <span className="tag">
              action: <b>{trace.spans.validate.action === "abstain" ? "stopped (not found)" : "answer"}</b>
            </span>
          </div>
          {trace.spans.validate.reason && <p className="muted small">{trace.spans.validate.reason}</p>}
          {(trace.spans.validate.retries ?? []).map((r, i) => (
            <div key={i} className="box">
              {r.skipped ? (
                <>Second search skipped: {r.skipped}.</>
              ) : (
                <>
                  Second search for <b>{r.query}</b>: graded {r.grade_before} then {r.grade_after} (best score of the new search{" "}
                  {r.best_retry_score?.toFixed(4) ?? "-"}, {fmtMs(r.search_ms ?? 0)}).
                </>
              )}
            </div>
          ))}
        </section>
      )}

      {retrieve?.results && (
        <section>
          <h3>Retrieved chunks</h3>
          <RelevanceNote trace={trace} />
          {retrieve.results.map((r, i) => (
            <ChunkCard
              key={r.chunk_id}
              rank={i + 1}
              chunkId={r.chunk_id}
              docId={r.doc_id}
              page={r.page}
              score={r.score}
              text={r.preview}
              highlight={relevant.has(r.chunk_id)}
            />
          ))}
          <p className="muted small">Previews only. The full chunk text is inside the final prompt below.</p>
        </section>
      )}

      {llm && (
        <section>
          <h3>Model call</h3>
          <div className="chips static">
            <span className="tag">
              model: <b>{llm.model ?? "-"}</b>
            </span>
            <span className="tag">
              tokens in: <b>{fmtInt(llm.input_tokens)}</b>
            </span>
            <span className="tag">
              tokens out: <b>{fmtInt(llm.output_tokens)}</b>
            </span>
            <span className="tag">
              temperature: <b>{llm.temperature ?? "-"}</b>
            </span>
          </div>
        </section>
      )}

      {prompt?.final_prompt && (
        <section>
          <details className="prompt">
            <summary>Final prompt sent to the model ({prompt.final_prompt.length.toLocaleString()} characters)</summary>
            <pre>{prompt.final_prompt}</pre>
          </details>
        </section>
      )}

      <section>
        <h3>Expected vs generated answer</h3>
        <div className="compare">
          <div>
            <div className="compare-label">
              Expected
              {trace.golden && (
                <span className="badge">
                  {trace.golden.id} · {trace.golden.type}
                </span>
              )}
            </div>
            <div className="box expected">
              {trace.golden ? trace.golden.expected_answer : <span className="muted">No expected answer. This question is not in the golden set.</span>}
            </div>
            {trace.golden && (
              <div className="muted small compare-meta">
                <div>
                  Source: {trace.golden.source_docs.length ? trace.golden.source_docs.join(", ") : "none (the system should refuse)"}
                  {trace.golden.source_section ? ` · ${trace.golden.source_section}` : ""}
                </div>
                {trace.golden.notes && <div>Note: {trace.golden.notes}</div>}
              </div>
            )}
          </div>
          <div>
            <div className="compare-label">Generated</div>
            <div className="box markdown">{trace.answer ? <ReactMarkdown>{trace.answer}</ReactMarkdown> : <span className="muted">No answer</span>}</div>
          </div>
        </div>
      </section>
    </div>
  );
}
