export function fmtMs(ms: number | null | undefined): string {
  if (ms == null) return "-";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(ms < 10_000 ? 1 : 0)} s`;
}

export function fmtNum(value: number | null | undefined, digits = 2): string {
  return value == null ? "-" : value.toFixed(digits);
}

export function fmtInt(value: number | null | undefined): string {
  return value == null ? "-" : Math.round(value).toLocaleString();
}

export function fmtTime(iso: string): string {
  const date = new Date(iso);
  return Number.isNaN(date.getTime())
    ? iso
    : date.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

/** Soft red-to-green tint for a 0..1 score. */
export function heat(value: number | null | undefined): string | undefined {
  if (value == null) return undefined;
  const clamped = Math.max(0, Math.min(1, value));
  return `hsl(${Math.round(clamped * 120)} 65% 93%)`;
}

export function shortDoc(docId: string): string {
  return docId.length > 28 ? `${docId.slice(0, 26)}...` : docId;
}
