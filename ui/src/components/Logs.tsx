import { useCallback, useEffect, useState } from "react";
import { api } from "../api";
import { fmtInt } from "../format";
import type { IngestRecord, LogEntry } from "../types";
import { Icon } from "./Icon";

const LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"] as const;

function PipelineLog() {
  const [entries, setEntries] = useState<LogEntry[]>([]);
  const [size, setSize] = useState(0);
  const [lines, setLines] = useState(300);
  const [levels, setLevels] = useState<Set<string>>(new Set(["INFO", "WARNING", "ERROR"]));
  const [auto, setAuto] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    api
      .pipelineLog(lines)
      .then((log) => {
        setEntries(log.entries.slice().reverse());
        setSize(log.size_bytes);
        setError(null);
      })
      .catch((e: Error) => setError(e.message));
  }, [lines]);

  useEffect(load, [load]);
  useEffect(() => {
    if (!auto) return;
    const timer = setInterval(load, 5000);
    return () => clearInterval(timer);
  }, [auto, load]);

  const toggle = (level: string) =>
    setLevels((prev) => {
      const next = new Set(prev);
      if (next.has(level)) next.delete(level);
      else next.add(level);
      return next;
    });

  const shown = entries.filter((e) => levels.has(e.level));

  return (
    <>
      <div className="toolbar">
        <div className="chips static">
          {LEVELS.map((level) => (
            <button key={level} className={`chip level-${level} ${levels.has(level) ? "on" : ""}`} onClick={() => toggle(level)}>
              {level}
            </button>
          ))}
        </div>
        <div className="header-actions">
          <label className="inline-field">
            Lines
            <select value={lines} onChange={(e) => setLines(Number(e.target.value))}>
              {[100, 300, 1000].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
          <label className="inline-field">
            <input type="checkbox" checked={auto} onChange={(e) => setAuto(e.target.checked)} /> Auto-refresh
          </label>
          <button className="btn ghost" onClick={load}>
            <Icon name="refresh" size={15} /> Refresh
          </button>
        </div>
      </div>
      {error && <div className="banner bad">{error}</div>}
      <p className="muted small">
        logs/pipeline.log · {(size / 1024).toFixed(0)} KB · newest first · {shown.length} of {entries.length} lines shown. DEBUG lines hold the retrieved chunk text.
      </p>
      <div className="log-table">
        {shown.length === 0 && <p className="muted pad">No log lines for the selected levels.</p>}
        {shown.map((e, i) => (
          <div key={`${e.time}-${i}`} className="log-row">
            <span className="log-time">{e.time.slice(5)}</span>
            <span className={`level level-${e.level}`}>{e.level}</span>
            <span className="log-logger">{e.logger.replace("rag.", "")}</span>
            <span className="log-msg">{e.message}</span>
          </div>
        ))}
      </div>
    </>
  );
}

function IngestLog() {
  const [rows, setRows] = useState<IngestRecord[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .ingestLog()
      .then(setRows)
      .catch((e: Error) => setError(e.message));
  }, []);

  return (
    <>
      <p className="muted small">logs/ingest.jsonl · one line per file each time the documents were processed · newest first.</p>
      {error && <div className="banner bad">{error}</div>}
      <table className="table">
        <thead>
          <tr>
            <th>Time</th>
            <th>File</th>
            <th>Type</th>
            <th className="num">Pages</th>
            <th className="num">Empty pages</th>
            <th className="num">Characters</th>
            <th className="num">Chunks</th>
            <th>Error</th>
          </tr>
        </thead>
        <tbody>
          {rows?.map((r, i) => (
            <tr key={`${r.timestamp}-${r.file}-${i}`}>
              <td>{r.timestamp.replace("T", " ")}</td>
              <td>{r.file}</td>
              <td>{r.doc_type}</td>
              <td className="num">{r.pages}</td>
              <td className="num">{r.empty_pages}</td>
              <td className="num">{fmtInt(r.chars)}</td>
              <td className="num">{r.chunks}</td>
              <td>{r.error ? <span className="badge bad">{r.error}</span> : <span className="muted">none</span>}</td>
            </tr>
          ))}
          {rows?.length === 0 && (
            <tr>
              <td colSpan={8} className="muted">
                No ingestion records yet.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </>
  );
}

export function Logs() {
  const [view, setView] = useState<"pipeline" | "ingest">("pipeline");
  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>Logs</h1>
          <p className="muted">What the system did, step by step.</p>
        </div>
        <div className="segmented">
          <button className={view === "pipeline" ? "on" : ""} onClick={() => setView("pipeline")}>
            Pipeline
          </button>
          <button className={view === "ingest" ? "on" : ""} onClick={() => setView("ingest")}>
            Ingestion
          </button>
        </div>
      </header>
      {view === "pipeline" ? <PipelineLog /> : <IngestLog />}
    </div>
  );
}
