import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import { Chat } from "./components/Chat";
import { Icon, type IconName } from "./components/Icon";
import { Logs } from "./components/Logs";
import { Reports } from "./components/Reports";
import { Traces } from "./components/Traces";
import { ALL_VERSIONS } from "./components/VersionFilter";
import type { Health, Versions } from "./types";
import { useChat } from "./useChat";

type Tab = "chat" | "traces" | "logs" | "reports";

const TABS: { id: Tab; label: string; icon: IconName }[] = [
  { id: "chat", label: "Chat", icon: "chat" },
  { id: "traces", label: "Traces", icon: "traces" },
  { id: "logs", label: "Logs", icon: "logs" },
  { id: "reports", label: "Reports", icon: "reports" },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("chat");
  const [traceId, setTraceId] = useState<string | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [apiDown, setApiDown] = useState(false);
  const [version, setVersion] = useState<string>(ALL_VERSIONS);
  const [versions, setVersions] = useState<Versions | null>(null);
  const chat = useChat();

  // Refresh the version list when switching tabs, so a freshly bumped version shows up.
  useEffect(() => {
    api.versions().then(setVersions).catch(() => undefined);
  }, [tab]);

  useEffect(() => {
    const check = () =>
      api
        .health()
        .then((h) => {
          setHealth(h);
          setApiDown(false);
        })
        .catch(() => setApiDown(true));
    check();
    const timer = setInterval(check, 30_000);
    return () => clearInterval(timer);
  }, []);

  const openTrace = useCallback((id: string) => {
    setTraceId(id);
    setVersion(ALL_VERSIONS);   // the trace may belong to a version that is currently filtered out
    setTab("traces");
  }, []);

  const llm = health?.llm;
  const status = apiDown ? "down" : llm?.reachable === false ? "warn" : health ? "ok" : "idle";
  const statusText = apiDown
    ? "API offline"
    : llm?.reachable === false
      ? "Model server offline"
      : health
        ? "Ready"
        : "Connecting...";

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">R</span>
          <span>RAG Project</span>
        </div>
        <nav>
          {TABS.map((t) => (
            <button key={t.id} className={`nav-item ${tab === t.id ? "active" : ""}`} onClick={() => setTab(t.id)}>
              <Icon name={t.icon} />
              {t.label}
              {t.id === "chat" && chat.busy && <span className="nav-dot" title="Answering..." />}
            </button>
          ))}
        </nav>
        <div className="sidebar-footer">
          <div className="status">
            <span className={`dot ${status}`} />
            {statusText}
          </div>
          {llm && <div className="muted small">{llm.model}</div>}
          {health?.indexed_chunks != null && <div className="muted small">{health.indexed_chunks} chunks indexed</div>}
        </div>
      </aside>
      <main className="main">
        {tab === "chat" && (
          <Chat
            messages={chat.messages}
            busy={chat.busy}
            health={health}
            apiDown={apiDown}
            onSend={chat.send}
            onClear={chat.clear}
            onOpenTrace={openTrace}
          />
        )}
        {tab === "traces" && (
          <Traces selectedId={traceId} onSelect={setTraceId} version={version} versions={versions} onVersionChange={setVersion} />
        )}
        {tab === "logs" && <Logs />}
        {tab === "reports" && (
          <Reports onOpenTrace={openTrace} version={version} versions={versions} onVersionChange={setVersion} />
        )}
      </main>
    </div>
  );
}
