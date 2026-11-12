"""Measure the chunking problems that hurt retrieval, for any set of chunks.

Each problem is a heuristic count (a chunk id or question id list) so a strategy can be compared with the
baseline and you can open the exact chunks it flags.
"""
import re
from collections import Counter

from eval.metrics import SINGLE_FACT_TYPES, chunk_body, coverage, source_doc_ids

HEADING = re.compile(r"^(#{1,6})\s")
SENTENCE_END = ".!?:|;"
UNIT_START = ("#", "|", "-", "*", "\u2022")      # a chunk that starts like this begins a new unit
HUB_MIN_QUESTIONS = 3                            # a chunk that outranks the right one this often is a hub


def _lines(text: str) -> list[str]:
    return [line.rstrip() for line in text.split("\n") if line.strip()]


def _adjacent(chunks: list[dict]):
    for a, b in zip(chunks, chunks[1:]):
        if a["doc_id"] == b["doc_id"]:
            yield a, b


def _is_table_cut(a: dict, b: dict) -> bool:
    """One chunk ends inside a table and the next carries on with a row or the rest of a row.

    A table longer than the chunk budget has to be cut; that is fine when the next chunk repeats the header row.
    """
    lines, first = _lines(chunk_body(a)), _lines(chunk_body(b))[:1]
    if not lines or not first or not lines[-1].lstrip().startswith("|") or "|" not in first[0]:
        return False
    start = len(lines) - 1
    while start > 0 and lines[start - 1].lstrip().startswith("|"):
        start -= 1
    return first[0].strip() != lines[start].strip()


def _is_cut_mid_sentence(a: dict, b: dict) -> bool:
    last, first = _lines(chunk_body(a))[-1:], _lines(chunk_body(b))[:1]
    if not last or not first:
        return False
    ends_clean = last[0][-1] in SENTENCE_END or HEADING.match(last[0]) or last[0].lstrip().startswith("|")
    starts_clean = first[0].lstrip().startswith(UNIT_START) or first[0].lstrip()[:1].isupper()
    return not ends_clean and not starts_clean


def hub_chunks(results: list[dict]) -> dict[str, int]:
    """Chunks that were ranked first, wrongly, above the relevant chunk, for several questions."""
    counts = Counter()
    for r in results:
        ranked = [h["chunk_id"] for h in r["retrieved"]]
        relevant = set(r["relevant_ids"])
        if relevant and ranked and ranked[0] not in relevant and relevant & set(ranked):
            counts[ranked[0]] += 1
    return {cid: n for cid, n in counts.most_common() if n >= HUB_MIN_QUESTIONS}


def diagnose(chunks: list[dict], questions: list[dict], results: list[dict] | None = None) -> dict:
    bodies = {c["chunk_id"]: chunk_body(c) for c in chunks}
    tokens = [c["n_tokens"] for c in chunks if "n_tokens" in c]

    mixed = [cid for cid, body in bodies.items()
             if sum(1 for line in body.split("\n") if (m := HEADING.match(line)) and len(m.group(0).strip()) <= 2) >= 2]
    orphan = [c["chunk_id"] for c in chunks if _lines(bodies[c["chunk_id"]]) and HEADING.match(_lines(bodies[c["chunk_id"]])[-1])]
    split_tables = [(a["chunk_id"], b["chunk_id"]) for a, b in _adjacent(chunks) if _is_table_cut(a, b)]
    cuts = [(a["chunk_id"], b["chunk_id"]) for a, b in _adjacent(chunks) if _is_cut_mid_sentence(a, b)]
    context_free = [c["chunk_id"] for c in chunks
                    if not c.get("header") and not any(HEADING.match(line) for line in bodies[c["chunk_id"]].split("\n"))]

    split_answers = []
    for q in questions:
        keys = q.get("must_include") or []
        if q["type"] not in SINGLE_FACT_TYPES or not keys:
            continue
        sources = source_doc_ids(q)
        if max((coverage(bodies[c["chunk_id"]], keys) for c in chunks if c["doc_id"] in sources), default=0) < 1:
            split_answers.append(q["id"])

    missed_in_mixed = []
    for r in results or []:
        if r["retrieval"] and not r["retrieval"]["hit"]:
            hit = [cid for cid in r["relevant_ids"] if cid in mixed]
            if hit:
                missed_in_mixed.append((r["id"], hit[0]))

    return {
        "chunks": len(chunks),
        "avg_tokens": sum(tokens) / len(tokens) if tokens else None,
        "mixed_topic_chunks": mixed,
        "orphan_headings": orphan,
        "split_tables": split_tables,
        "mid_sentence_cuts": cuts,
        "context_free_chunks": context_free,
        "split_answers": split_answers,
        "missed_in_mixed_chunk": missed_in_mixed,
        "hub_chunks": hub_chunks(results or []),
    }
