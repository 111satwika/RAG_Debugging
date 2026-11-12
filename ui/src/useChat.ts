import { useCallback, useEffect, useState } from "react";
import { streamQuestion } from "./api";
import type { ChatMessage } from "./types";

const STORAGE_KEY = "rag-chat-v1";
const MAX_STORED = 40;

function load(): ChatMessage[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    return (JSON.parse(raw) as ChatMessage[]).map((m): ChatMessage =>
      m.role === "assistant" && m.status === "pending"
        ? { id: m.id, role: "assistant", status: "error", error: "Interrupted: the page was reloaded while this was running." }
        : m,
    );
  } catch {
    return [];
  }
}

/** Chat state lives above the tabs, so switching tabs does not interrupt a running question. */
export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>(load);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(messages.slice(-MAX_STORED)));
  }, [messages]);

  const send = useCallback(async (question: string, k: number) => {
    const id = crypto.randomUUID();
    const answerId = `${id}-a`;
    const replace = (message: ChatMessage) =>
      setMessages((prev) => prev.map((m) => (m.id === message.id ? message : m)));

    setBusy(true);
    setMessages((prev) => [
      ...prev,
      { id: `${id}-q`, role: "user", text: question },
      { id: answerId, role: "assistant", status: "pending", startedAt: Date.now() },
    ]);
    // Tokens arrive faster than React needs to redraw: update the pending message with the latest text and stage.
    const update = (change: (m: Extract<ChatMessage, { status: "pending" }>) => ChatMessage) =>
      setMessages((prev) => prev.map((m) => (m.id === answerId && m.role === "assistant" && m.status === "pending" ? change(m) : m)));
    try {
      const result = await streamQuestion(question, k, (event) => {
        if (event.type === "stage") update((m) => ({ ...m, stage: event.stage }));
        if (event.type === "validation") update((m) => ({ ...m, validation: event.validation }));
        if (event.type === "context") update((m) => ({ ...m, context: event.context }));
        if (event.type === "token")
          update((m) => ({ ...m, text: (m.text ?? "") + event.text, firstTokenAt: m.firstTokenAt ?? Date.now() }));
      });
      replace({ id: answerId, role: "assistant", status: "done", result });
    } catch (err) {
      replace({ id: answerId, role: "assistant", status: "error", error: err instanceof Error ? err.message : String(err) });
    } finally {
      setBusy(false);
    }
  }, []);

  const clear = useCallback(() => {
    if (!busy) setMessages([]);
  }, [busy]);

  return { messages, busy, send, clear };
}
