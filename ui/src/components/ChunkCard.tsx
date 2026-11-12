import { useState } from "react";
import { fmtNum } from "../format";

type Props = {
  chunkId: string;
  docId: string;
  page: number;
  score: number;
  text: string;
  rank?: number;
  highlight?: boolean;
  /** The source id the answer cites this chunk with (S1, S2, ...), when it cites it. */
  cited?: string;
  collapsedChars?: number;
};

/** One retrieved chunk: where it came from, its similarity score, and its text. */
export function ChunkCard({ chunkId, docId, page, score, text, rank, highlight, cited, collapsedChars = 260 }: Props) {
  const [open, setOpen] = useState(false);
  const long = text.length > collapsedChars;
  const shown = open || !long ? text : `${text.slice(0, collapsedChars).trimEnd()} ...`;
  return (
    <div className={`chunk ${highlight ? "chunk-hit" : ""}`}>
      <div className="chunk-head">
        {rank != null && <span className="rank">#{rank}</span>}
        <span className="chunk-doc" title={`${docId} · ${chunkId}`}>
          <span className="chunk-name">{chunkId}</span> <span className="muted">· page {page}</span>
        </span>
        {cited && (
          <span className="badge good" title="The answer cites this chunk">
            cited [{cited}]
          </span>
        )}
        {highlight && (
          <span className="badge good" title="This chunk holds the key facts of the expected answer">
            relevant
          </span>
        )}
        <span className="score" title="Cosine similarity to the question">
          <span className="score-bar">
            <span style={{ width: `${Math.max(0, Math.min(1, score)) * 100}%` }} />
          </span>
          {fmtNum(score, 3)}
        </span>
      </div>
      <div className="chunk-text">{shown}</div>
      {long && (
        <button className="link small" onClick={() => setOpen(!open)}>
          {open ? "Show less" : "Show full chunk"}
        </button>
      )}
    </div>
  );
}
