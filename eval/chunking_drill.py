"""Phase 3 drill: a question whose answer is split across chunks, and the changes that fix it.

    python -m eval.chunking_drill                    # q03 at 300 tokens, asks the local model
    python -m eval.chunking_drill --question q21 --no-llm

It chunks the documents several ways, shows where the answer's key facts land, retrieves with each strategy and
(unless --no-llm) asks the model, so you can see the failure and which change fixes it.
Writes eval/results/chunking_drill.md.
"""
import argparse
import json

import chromadb

from eval.metrics import chunk_body, contains, coverage, fact_label, keyword_check, source_doc_ids
from eval.run_eval import GOLDEN_PATH, RESULTS_DIR
from src.config import TOP_K
from src.generate.generate import build_prompt
from src.generate.llm import load_llm
from src.ingest.chunk import chunk_document, default_tokenizer
from src.ingest.chunkers import ChunkSpec
from src.ingest.embed import load_embedder
from src.ingest.index import build_index
from src.ingest.normalize import normalize_document
from src.ingest.parse import parse_all
from src.retrieve.dense import retrieve

REPORT_PATH = RESULTS_DIR / "chunking_drill.md"
DRILL_SPECS = [
    ("fixed 300 (the failure)", ChunkSpec("fixed", 300)),
    ("fixed 300 + 50-token overlap", ChunkSpec("fixed", 300, overlap=50)),
    ("recursive 300 (cuts at sentence boundaries)", ChunkSpec("recursive", 300)),
    ("structure 300 + headers", ChunkSpec("structure", 300, headers=True)),
]


def boundary_cut(chunks: list[dict], fact: str, question: dict) -> str | None:
    """Where a key fact is cut in two: it appears when two neighbouring chunks are joined, but in neither alone."""
    sources, width = source_doc_ids(question), len(fact) + 10
    for a, b in zip(chunks, chunks[1:]):
        if a["doc_id"] != b["doc_id"] or a["doc_id"] not in sources:
            continue
        tail, head = chunk_body(a)[-width:], chunk_body(b)[:width]
        if any(contains(tail + joiner + head, fact) for joiner in ("", " ")) and not contains(tail, fact) and not contains(head, fact):
            return f"...{tail[-50:]!r}  |  {head[:40]!r}..."
    return None


def run_drill(question_id: str = "q03", use_llm: bool = True, k: int = TOP_K) -> list[dict]:
    question = next(q for q in json.loads(GOLDEN_PATH.read_text(encoding="utf-8")) if q["id"] == question_id)
    keys, sources = question["must_include"], source_doc_ids(question)
    tokenizer, embedder = default_tokenizer(), load_embedder()
    docs = [normalize_document(d) for d in parse_all()]
    llm = load_llm() if use_llm else None
    client = chromadb.EphemeralClient()

    rows = []
    for label, spec in DRILL_SPECS:
        chunks = [c for doc in docs for c in chunk_document(doc, tokenizer, spec=spec)]
        collection = build_index(chunks, embedder, client)
        hits = retrieve(question["question"], k, embedder, collection)
        best_single = max((coverage(chunk_body(c), keys) for c in chunks if c["doc_id"] in sources), default=0)
        retrieved_best = max((coverage(chunk_body(h), keys) for h in hits if h["doc_id"] in sources), default=0)
        missing_keys = [key for key in keys if not any(contains(chunk_body(c), key) for c in chunks if c["doc_id"] in sources)]
        row = {"strategy": label, "chunks": len(chunks), "best_single_chunk_coverage": best_single,
               "best_retrieved_chunk_coverage": retrieved_best,
               "cut": next(filter(None, (boundary_cut(chunks, key, question) for key in missing_keys)), None),
               "answer": None, "correct": None}
        if llm:
            answer = llm.complete(build_prompt(question["question"], hits))["text"]
            check = keyword_check(answer, question)
            row.update(answer=answer, correct=check["include_rate"] == 1 and not check["forbidden"], missing=check["missing"])
        rows.append(row)
        print(f"{label:46} single-chunk coverage={best_single:.0%}  retrieved coverage={retrieved_best:.0%}"
              + (f"  correct={row['correct']}" if llm else ""), flush=True)
    REPORT_PATH.write_text(render(question, rows), encoding="utf-8")
    return rows


def render(question: dict, rows: list[dict]) -> str:
    lines = [f"# Drill: {question['id']} at a smaller chunk size", "",
             f"**Question:** {question['question']}", f"**Expected:** {question['expected_answer']}",
             f"**Key facts:** {', '.join(fact_label(f) for f in question['must_include'])}", "",
             "| Strategy | Chunks | Best single chunk holds | Best retrieved chunk holds | Model answer correct |",
             "|---|---|---|---|---|"]
    for r in rows:
        correct = "-" if r["correct"] is None else ("yes" if r["correct"] else "**no**")
        lines.append(f"| {r['strategy']} | {r['chunks']} | {r['best_single_chunk_coverage']:.0%} of the facts | "
                     f"{r['best_retrieved_chunk_coverage']:.0%} of the facts | {correct} |")
    for r in rows:
        if r["cut"]:
            lines += ["", f"**Where {r['strategy']} cuts the answer:** {r['cut']}"]
    for r in rows:
        if r["answer"]:
            lines += ["", f"**{r['strategy']} answered:** {r['answer'][:300]}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default="q03")
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()
    run_drill(args.question, not args.no_llm)
    print(f"\nreport: {REPORT_PATH}")
