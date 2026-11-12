import { Fragment, useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import { api } from "../api";
import { fmtInt, fmtNum, heat } from "../format";
import type { Num, QuestionResult, ReportDetail, ReportListItem, Summary, Versions } from "../types";
import { ALL_VERSIONS, VersionFilter, versionParam } from "./VersionFilter";

type Props = {
  onOpenTrace: (id: string) => void;
  version: string;
  versions: Versions | null;
  onVersionChange: (version: string) => void;
};

function Metric({ label, value, tint, hint }: { label: string; value: string; tint?: string; hint?: string }) {
  return (
    <div className="metric" style={{ background: tint }} title={hint}>
      <div className="metric-value">{value}</div>
      <div className="metric-label">{label}</div>
    </div>
  );
}

function Score({ label, value }: { label: string; value: Num }) {
  return <Metric label={label} value={fmtNum(value)} tint={heat(value)} />;
}

function MetricGroups({ s }: { s: Summary }) {
  const judged = s.n_judged > 0;
  return (
    <div className="groups">
      <section>
        <h3>Retrieval</h3>
        <p className="muted small">Did the right chunks reach the model? ({s.n_retrieval} answerable questions)</p>
        <div className="metrics">
          <Score label={`Recall@${s.k}`} value={s.recall} />
          <Metric label={`Precision@${s.k}`} value={fmtNum(s.precision)} hint="Relevant chunks in the top k, divided by k. Capped near 1/k when a question has one relevant chunk, so it cannot show a version that sends fewer chunks." />
          <Metric label="Context precision" value={fmtNum(s.context_precision)} hint="Share of the chunks actually given to the model that are relevant. Rises when fewer, better chunks are sent (v9's cut-off)." />
          <Score label="MRR" value={s.mrr} />
          <Score label={`nDCG@${s.k}`} value={s.ndcg} />
          <Score label={`Hit@${s.k}`} value={s.hit} />
        </div>
      </section>
      <section>
        <h3>Answers</h3>
        <p className="muted small">{judged ? `Scored by the judge model (${s.n_judged} of ${s.n} answers).` : "Not judged yet."}</p>
        <div className="metrics">
          <Score label="Correctness" value={s.correctness} />
          <Score label="Relevance" value={s.relevance} />
          <Score label="Completeness" value={s.completeness} />
          <Score label="Faithfulness" value={s.faithfulness} />
          <Score label="Abstention accuracy" value={s.abstention_accuracy} />
          <Metric label="Refusal on unanswerable" value={fmtNum(s.refusal_rate_unanswerable)} />
          <Metric label="False abstention" value={fmtNum(s.false_abstention_rate)} hint="Share of answerable questions the model wrongly refused. Lower is better." />
        </div>
      </section>
      {s.citation_rate != null && (
        <section>
          <h3>Citations</h3>
          <p className="muted small">Do the answers cite their sources, and are the citations right? (cited prompts only)</p>
          <div className="metrics">
            <Score label="Citation rate" value={s.citation_rate} />
            <Score label="Cites a relevant chunk" value={s.cites_relevant_rate ?? null} />
            <Score label="Format fully valid" value={s.format_ok_rate ?? null} />
            <Metric label="Reply not valid JSON" value={fmtNum(s.format_invalid_rate ?? null)} hint="Share of replies that were not the structured answer. Lower is better." />
            <Metric label="`found` agrees with judge" value={fmtNum(s.found_agreement ?? null)} hint="Share of answers where the model's found flag matches the judge's view of whether it abstained." />
          </div>
        </section>
      )}
      <section>
        <h3>Speed and size</h3>
        <div className="metrics">
          <Metric label="Latency p50" value={s.latency_p50_s == null ? "-" : `${s.latency_p50_s.toFixed(0)} s`} />
          <Metric label="Latency p95" value={s.latency_p95_s == null ? "-" : `${s.latency_p95_s.toFixed(0)} s`} />
          <Metric label="Tokens in / query" value={fmtInt(s.avg_input_tokens)} />
          <Metric label="Tokens out / query" value={fmtInt(s.avg_output_tokens)} />
        </div>
      </section>
    </div>
  );
}

function Cell({ value }: { value: Num }) {
  return (
    <td className="num" style={{ background: heat(value) }}>
      {fmtNum(value)}
    </td>
  );
}

function RunsTable({ runs, selected, onSelect }: { runs: ReportListItem[]; selected: string | null; onSelect: (name: string) => void }) {
  return (
    <table className="table">
      <thead>
        <tr>
          <th>Run</th>
          <th>Version</th>
          <th>Change</th>
          <th className="num">Recall</th>
          <th className="num">MRR</th>
          <th className="num">nDCG</th>
          <th className="num">Correct</th>
          <th className="num">Complete</th>
          <th className="num">Faithful</th>
          <th className="num">Abstain</th>
          <th className="num">p95 (s)</th>
        </tr>
      </thead>
      <tbody>
        {runs.map((r) => (
          <tr key={r.name} className={`clickable ${r.name === selected ? "selected" : ""}`} onClick={() => onSelect(r.name)}>
            <td>
              <b>{r.name}</b>
              <div className="muted small">{r.created}</div>
            </td>
            <td>
              <span className="badge version">{r.version}</span>
            </td>
            <td className="note">{r.note || "-"}</td>
            <Cell value={r.overall.recall} />
            <Cell value={r.overall.mrr} />
            <Cell value={r.overall.ndcg} />
            <Cell value={r.overall.correctness} />
            <Cell value={r.overall.completeness} />
            <Cell value={r.overall.faithfulness} />
            <Cell value={r.overall.abstention_accuracy} />
            <td className="num">{r.overall.latency_p95_s == null ? "-" : r.overall.latency_p95_s.toFixed(0)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ByType({ byType, k }: { byType: Record<string, Summary>; k: number }) {
  return (
    <table className="table">
      <thead>
        <tr>
          <th>Question type</th>
          <th className="num">n</th>
          <th className="num">Recall@{k}</th>
          <th className="num">Prec@{k}</th>
          <th className="num">Ctx prec</th>
          <th className="num">MRR</th>
          <th className="num">nDCG@{k}</th>
          <th className="num">Correct</th>
          <th className="num">Relevant</th>
          <th className="num">Complete</th>
          <th className="num">Faithful</th>
          <th className="num">Abstain</th>
        </tr>
      </thead>
      <tbody>
        {Object.entries(byType).map(([type, s]) => (
          <tr key={type}>
            <td>{type}</td>
            <td className="num">{s.n}</td>
            <Cell value={s.recall} />
            <td className="num">{fmtNum(s.precision)}</td>
            <td className="num">{fmtNum(s.context_precision)}</td>
            <Cell value={s.mrr} />
            <Cell value={s.ndcg} />
            <Cell value={s.correctness} />
            <Cell value={s.relevance} />
            <Cell value={s.completeness} />
            <Cell value={s.faithfulness} />
            <Cell value={s.abstention_accuracy} />
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Questions({ results, onOpenTrace }: { results: QuestionResult[]; onOpenTrace: (id: string) => void }) {
  const [failuresOnly, setFailuresOnly] = useState(false);
  const [open, setOpen] = useState<string | null>(null);
  const rows = useMemo(() => (failuresOnly ? results.filter((r) => r.failure) : results), [results, failuresOnly]);

  return (
    <>
      <div className="toolbar">
        <label className="inline-field">
          <input type="checkbox" checked={failuresOnly} onChange={(e) => setFailuresOnly(e.target.checked)} /> Failures only
        </label>
        <span className="muted small">{rows.length} questions. Click a row for the answer, the judge's reasoning and the retrieved chunks.</span>
      </div>
      <table className="table">
        <thead>
          <tr>
            <th>ID</th>
            <th>Type</th>
            <th>Question</th>
            <th>Retrieval</th>
            <th className="num">Correct</th>
            <th className="num">Complete</th>
            <th className="num">Faithful</th>
            <th>Result</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <Fragment key={r.id}>
              <tr className={`clickable ${open === r.id ? "selected" : ""}`} onClick={() => setOpen(open === r.id ? null : r.id)}>
                <td>
                  <b>{r.id}</b>
                </td>
                <td>{r.type}</td>
                <td className="question">{r.question}</td>
                <td>{r.retrieval ? (r.retrieval.hit ? <span className="badge good">found</span> : <span className="badge bad">missed</span>) : <span className="muted">n/a</span>}</td>
                <Cell value={r.judge?.correctness ?? null} />
                <Cell value={r.judge?.completeness ?? null} />
                <Cell value={r.judge?.faithfulness ?? null} />
                <td>{r.failure ? <span className={`badge stage stage-${r.failure.stage}`}>{r.failure.stage}</span> : r.judge ? <span className="badge good">ok</span> : <span className="muted">-</span>}</td>
              </tr>
              {open === r.id && (
                <tr className="detail-row">
                  <td colSpan={8}>
                    <div className="detail">
                      <div>
                        <h4>Expected</h4>
                        <p>{r.expected}</p>
                        <h4>Answer</h4>
                        <div className="markdown">{r.answer ? <ReactMarkdown>{r.answer}</ReactMarkdown> : <span className="muted">{r.error ?? "No answer recorded"}</span>}</div>
                      </div>
                      <div>
                        {r.failure && (
                          <>
                            <h4>Why it failed</h4>
                            <p>
                              <span className={`badge stage stage-${r.failure.stage}`}>{r.failure.stage}</span> {r.failure.note}
                            </p>
                          </>
                        )}
                        {r.judge && (
                          <>
                            <h4>Judge</h4>
                            <p>{r.judge.reasoning}</p>
                            {r.judge.unsupported_claims.length > 0 && <p className="bad-text">Unsupported: {r.judge.unsupported_claims.join("; ")}</p>}
                          </>
                        )}
                        {r.keyword && (r.keyword.missing.length > 0 || r.keyword.forbidden.length > 0) && (
                          <p className="muted small">
                            Keyword check missing: {r.keyword.missing.join(", ") || "none"}
                            {r.keyword.forbidden.length > 0 && ` · stale values present: ${r.keyword.forbidden.join(", ")}`}
                          </p>
                        )}
                        <h4>Retrieved</h4>
                        <ul className="retrieved">
                          {r.retrieved.map((h, i) => (
                            <li key={h.chunk_id}>
                              #{i + 1} {h.chunk_id} <span className="muted">p{h.page} · {fmtNum(h.score, 3)}</span>
                              {r.relevant_ids.includes(h.chunk_id) && <span className="badge good">relevant</span>}
                            </li>
                          ))}
                        </ul>
                        {r.trace_id && (
                          <button className="btn ghost" onClick={() => onOpenTrace(r.trace_id!)}>
                            Open trace {r.trace_id}
                          </button>
                        )}
                      </div>
                    </div>
                  </td>
                </tr>
              )}
            </Fragment>
          ))}
        </tbody>
      </table>
    </>
  );
}

export function Reports({ onOpenTrace, version, versions, onVersionChange }: Props) {
  const [runs, setRuns] = useState<ReportListItem[] | null>(null);
  const [name, setName] = useState<string | null>(null);
  const [detail, setDetail] = useState<ReportDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .reports(versionParam(version))
      .then((items) => {
        setRuns(items);
        setError(null);
        // Keep the selected run if the filter still shows it; otherwise show the first one.
        setName((current) => (items.some((r) => r.name === current) ? current : (items[0]?.name ?? null)));
      })
      .catch((e: Error) => setError(e.message));
  }, [version]);

  useEffect(() => {
    if (!name) {
      setDetail(null);
      return;
    }
    let stale = false;
    api
      .report(name)
      .then((d) => !stale && setDetail(d))
      .catch((e: Error) => !stale && setError(e.message));
    return () => {
      stale = true;
    };
  }, [name]);

  const failures = detail?.results.filter((r) => r.failure) ?? [];
  const stages = failures.reduce<Record<string, number>>((acc, r) => ({ ...acc, [r.failure!.stage]: (acc[r.failure!.stage] ?? 0) + 1 }), {});

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>Reports</h1>
          <p className="muted">Evaluation runs on the golden question set. Pick a run to see where it succeeds and fails.</p>
        </div>
        <VersionFilter versions={versions} value={version} onChange={onVersionChange} />
      </header>
      {error && <div className="banner bad">{error}</div>}
      {runs?.length === 0 && (
        <div className="empty">
          <h2>{version === ALL_VERSIONS ? "No evaluation runs yet" : `No evaluation runs for version ${version}`}</h2>
          <p className="muted">
            {version === ALL_VERSIONS
              ? "Run: python -m eval.run_eval --name baseline"
              : "Run an evaluation after setting PIPELINE_VERSION in src/config.py, or pick another version."}
          </p>
        </div>
      )}
      {runs && runs.length > 0 && (
        <>
          <section>
            <h3>Runs</h3>
            <RunsTable runs={runs} selected={name} onSelect={setName} />
          </section>
          {detail && detail.meta.name === name && (
            <>
              <section>
                <div className="run-head">
                  <h2>{detail.meta.name}</h2>
                  <span className="badge version">{detail.version}</span>
                  <span className="muted">{detail.meta.created}</span>
                </div>
                {detail.meta.note && <p>{detail.meta.note}</p>}
                <div className="chips static">
                  {Object.entries(detail.meta.config).map(([key, value]) => (
                    <span key={key} className="tag">
                      {key}: <b>{String(value)}</b>
                    </span>
                  ))}
                  <span className="tag">
                    judge: <b>{detail.meta.judge}</b>
                  </span>
                </div>
              </section>
              <MetricGroups s={detail.overall} />
              <section>
                <h3>By question type</h3>
                <ByType byType={detail.by_type} k={detail.overall.k} />
              </section>
              <section>
                <h3>
                  Failures ({failures.length} of {detail.overall.n})
                </h3>
                {failures.length > 0 && (
                  <div className="chips static">
                    {Object.entries(stages).map(([stage, count]) => (
                      <span key={stage} className={`badge stage stage-${stage}`}>
                        {stage}: {count}
                      </span>
                    ))}
                  </div>
                )}
                <p className="muted small">Stages are heuristics from the retrieved chunks and the key facts. Confirm in the trace.</p>
              </section>
              <section>
                <h3>Questions</h3>
                <Questions results={detail.results} onOpenTrace={onOpenTrace} />
              </section>
            </>
          )}
        </>
      )}
    </div>
  );
}
