"""Phase 4 drill: look up exact terms (a name, a spec ID, an error code) and watch where dense, BM25 and hybrid rank them.

    python -m eval.retrieval_drill

A chunk counts as the right one when its text contains the term. The report is eval/results/retrieval_drill.md.
"""
import re

from eval.run_eval import RESULTS_DIR
from src.ingest.embed import load_embedder
from src.retrieve.dense import get_collection
from src.retrieve.retriever import RetrievalSpec, Retriever

REPORT_PATH = RESULTS_DIR / "retrieval_drill.md"
K = 5
TERMS = [
    ("QL-1003", "an error code in the spec"),
    ("QL-SPEC-001", "the spec's document ID"),
    ("Valumudi Satwika", "the candidate's name"),
    ("AZ-900", "a certification"),
    ("warmup_steps", "a parameter name in the paper"),
    ("0.831", "a number from the resume"),
    ("NFR-03", "a requirement ID in the spec"),
]
SPECS = [RetrievalSpec("dense"), RetrievalSpec("bm25"), RetrievalSpec("hybrid")]


def contains(text: str, term: str) -> bool:
    squash = lambda s: re.sub(r"[\s_-]+", "", s.lower())
    return squash(term) in squash(text)


def first_rank(hits: list[dict], term: str) -> int | None:
    return next((h["rank"] for h in hits if contains(h["text"], term)), None)


def run_drill(retriever: Retriever, terms=TERMS, specs=SPECS, k: int = K) -> list[dict]:
    rows = []
    for term, what in terms:
        right = [c["chunk_id"] for c in retriever.corpus if contains(c["text"], term)]
        ranks = {spec.label: first_rank(retriever.search(spec, term, k), term) for spec in specs}
        rows.append({"term": term, "what": what, "right": right, "ranks": ranks})
    return rows


def render(rows: list[dict], specs=SPECS, k: int = K) -> str:
    labels = [s.label for s in specs]
    lines = ["# Phase 4 drill: exact-term lookups", "",
             f"The query is just the term. The cell is the rank of the first chunk that contains it in the top {k}; `-` means "
             "it was not in the top 5. Right chunks are those whose text contains the term.", "",
             "| Term | What it is | Chunks that hold it | " + " | ".join(labels) + " |", "|---|---|---|" + "---|" * len(labels)]
    for row in rows:
        cells = [str(row["ranks"][label]) if row["ranks"][label] else "-" for label in labels]
        lines.append(f"| `{row['term']}` | {row['what']} | {len(row['right'])} | " + " | ".join(cells) + " |")
    found = {label: sum(1 for r in rows if r["ranks"][label]) for label in labels}
    lines += ["", "Found in the top 5: " + ", ".join(f"{label} {n} of {len(rows)}" for label, n in found.items()) + "."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    embedder = load_embedder()
    rows = run_drill(Retriever(embedder, get_collection(embedder)))
    text = render(rows)
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(text)
