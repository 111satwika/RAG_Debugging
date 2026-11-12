import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { fmtMs, fmtNum, fmtTime } from "../format";
import type { Trace, TraceSummary, Versions } from "../types";
import { Icon } from "./Icon";
import { TraceView } from "./TraceView";
import { ALL_VERSIONS, VersionFilter, versionParam } from "./VersionFilter";

type Props = {
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  version: string;
  versions: Versions | null;
  onVersionChange: (version: string) => void;
};

export function Traces({ selectedId, onSelect, version, versions, onVersionChange }: Props) {
  const [list, setList] = useState<TraceSummary[] | null>(null);
  const [filter, setFilter] = useState("");
  const [trace, setTrace] = useState<Trace | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    api
      .traces(versionParam(version))
      .then((items) => {
        setList(items);
        setError(null);
      })
      .catch((e: Error) => setError(e.message));
  }, [version]);

  useEffect(refresh, [refresh]);

  // A selected trace that the version filter hides gives way to the first trace that is shown.
  const selectedVisible = selectedId !== null && (list === null || list.some((t) => t.trace_id === selectedId));
  const activeId = selectedVisible ? selectedId : (list?.[0]?.trace_id ?? null);

  useEffect(() => {
    if (!activeId) {
      setTrace(null);
      return;
    }
    let stale = false;
    api
      .trace(activeId)
      .then((t) => {
        if (!stale) {
          setTrace(t);
          setError(null);
        }
      })
      .catch((e: Error) => !stale && setError(e.message));
    return () => {
      stale = true;
    };
  }, [activeId]);

  const visible = useMemo(() => {
    const needle = filter.trim().toLowerCase();
    return (list ?? []).filter((t) => !needle || t.query.toLowerCase().includes(needle) || t.trace_id.includes(needle));
  }, [list, filter]);

  return (
    <div className="split">
      <section className="split-list">
        <header className="page-header compact">
          <h1>Traces</h1>
          <button className="btn ghost" onClick={refresh}>
            <Icon name="refresh" size={15} /> Refresh
          </button>
        </header>
        <div className="list-filters">
          <VersionFilter versions={versions} value={version} onChange={onVersionChange} />
        </div>
        <input className="search" placeholder="Filter by question or id" value={filter} onChange={(e) => setFilter(e.target.value)} />
        <div className="list">
          {list === null && !error && <p className="muted pad">Loading...</p>}
          {list?.length === 0 && (
            <p className="muted pad">
              {version === ALL_VERSIONS ? "No traces yet. Ask a question in Chat." : `No traces for version ${version}.`}
            </p>
          )}
          {visible.map((t) => (
            <button key={t.trace_id} className={`list-item ${t.trace_id === activeId ? "active" : ""}`} onClick={() => onSelect(t.trace_id)}>
              <span className="list-title">{t.query}</span>
              <span className="list-meta">
                <span className="badge version">{t.version}</span>
                <span>{fmtTime(t.timestamp)}</span>
                <span>{fmtMs(t.total_ms)}</span>
                <span title="Best similarity score">top {fmtNum(t.top_score, 2)}</span>
                {t.error && <span className="badge bad">error</span>}
              </span>
            </button>
          ))}
        </div>
      </section>
      <section className="split-detail">
        {error && <div className="banner bad">{error}</div>}
        {trace ? <TraceView trace={trace} /> : !error && <p className="muted pad">Select a trace to see its stages, chunks and prompt.</p>}
      </section>
    </div>
  );
}
