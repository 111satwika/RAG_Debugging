"""Phase 3: compare chunking strategies on the golden set.

    python -m eval.chunking_experiments                          # bge-small stack, retrieval only (about a minute)
    python -m eval.chunking_experiments --stack openai           # OpenAI embeddings stack, retrieval only
    python -m eval.chunking_experiments --stack openai --only p3o-fixed-500 --full   # also answer and judge (OpenAI LLM)

Each experiment chunks the documents, embeds the chunks into a temporary in-memory index and runs the golden
questions through the normal eval harness. The main index is not touched. Every run is saved under eval/results/
with version "exp" (so the Reports tab shows it), and a comparison is written to eval/results/.

A stack fixes the tokenizer and embedding model, so a chunking change is never mixed with an embedding change:
  bge     bge-small embeddings, chunk sizes counted in bge tokens      (names p3-...)
  openai  text-embedding-3-small, chunk sizes counted in cl100k tokens (names p3o-...)
"""
import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import chromadb

from eval.chunk_diagnostics import diagnose
from eval.run_eval import GOLDEN_PATH, RESULTS_DIR, run_evaluation
from src.config import TOP_K, snapshot
from src.generate.llm import load_llm
from src.ingest.chunk import chunk_document
from src.ingest.chunkers import ChunkSpec, make_children
from src.ingest.embed import load_embedder
from src.ingest.index import build_index
from src.ingest.normalize import normalize_document
from src.ingest.parse import parse_all
from src.ingest.tokenization import load_tokenizer
from src.retrieve.dense import retrieve

EXPERIMENT_VERSION = "exp"
METRICS = ("recall", "precision", "mrr", "ndcg", "hit")


@dataclass(frozen=True)
class Experiment:
    spec: ChunkSpec
    note: str
    prefix: str = "p3"

    @property
    def name(self) -> str:
        return f"{self.prefix}-{self.spec.label}"


@dataclass(frozen=True)
class Stack:
    tokenizer: tuple[str, str]           # (provider, name)
    embedder: tuple[str, str]
    prefix: str
    report: str
    experiments: list = field(default_factory=list)


def _e(prefix, spec, note):
    return Experiment(spec, note, prefix)


# One change at a time from the baseline, except where a step needs the one before it (headers need sections).
BGE_EXPERIMENTS = [
    _e("p3", ChunkSpec("fixed", 500), "Control: the baseline setup, to check this harness reproduces the baseline numbers"),
    _e("p3", ChunkSpec("fixed", 300), "1. Smaller fixed-size chunks"),
    _e("p3", ChunkSpec("fixed", 500, overlap=50), "2. Fixed 500 with 10% overlap"),
    _e("p3", ChunkSpec("fixed", 500, overlap=100), "2. Fixed 500 with 20% overlap"),
    _e("p3", ChunkSpec("recursive", 500), "3. Recursive splitting: paragraph, then line, then sentence"),
    _e("p3", ChunkSpec("structure", 500), "4. Structure-aware: one chunk per heading section"),
    _e("p3", ChunkSpec("structure", 500, headers=True), "5. Structure-aware + contextual header (doc title > section)"),
    _e("p3", ChunkSpec("structure", 500, keep_tables=True), "6. Structure-aware + tables kept intact"),
    _e("p3", ChunkSpec("structure", 500, headers=True, keep_tables=True), "5+6. Structure-aware + header + tables kept intact"),
    _e("p3", ChunkSpec("structure", 300, headers=True), "5b. Structure-aware + header with a smaller budget: long sections are cut further"),
]

COMBINED = ChunkSpec("structure", 500, overlap=50, headers=True, keep_tables=True, min_tokens=50)
OPENAI_EXPERIMENTS = [
    _e("p3o", ChunkSpec("fixed", 500), "Control: baseline chunking with OpenAI embeddings"),
    _e("p3o", ChunkSpec("fixed", 300), "1. Fixed 300"),
    _e("p3o", ChunkSpec("fixed", 500, overlap=50), "2. Fixed 500 with 10% overlap"),
    _e("p3o", ChunkSpec("recursive", 500), "3. Recursive splitting: paragraph, then line, then sentence"),
    _e("p3o", ChunkSpec("recursive", 500, overlap=50), "3b. Recursive splitting with 10% overlap"),
    _e("p3o", ChunkSpec("structure", 500), "4. Structure-aware: one chunk per heading section"),
    _e("p3o", ChunkSpec("structure", 500, headers=True), "5. Structure-aware + contextual header"),
    _e("p3o", ChunkSpec("structure", 500, headers=True, keep_tables=True), "6. + tables kept whole"),
    _e("p3o", ChunkSpec("structure", 500, overlap=50, headers=True, keep_tables=True), "7. + 10% overlap inside long sections"),
    _e("p3o", COMBINED, "8. Combined (v2): + pieces under 50 tokens merged with a neighbour in the same section"),
    _e("p3o", ChunkSpec("structure", 500, overlap=50, headers=True, keep_tables=True, min_tokens=50, child_size=150),
       "Combined + parent-child: search 150-token children, answer with the parent section"),
]
LADDER = [e.name for e in OPENAI_EXPERIMENTS if e.spec.strategy == "structure" and not e.spec.child_size]   # each step adds one technique

STACKS = {
    "bge": Stack(("huggingface", "BAAI/bge-small-en-v1.5"), ("sentence-transformers", "BAAI/bge-small-en-v1.5"),
                 "p3", "chunking_experiments.md", BGE_EXPERIMENTS),
    "openai": Stack(("tiktoken", "cl100k_base"), ("openai", "text-embedding-3-small"),
                    "p3o", "chunking_experiments_openai.md", OPENAI_EXPERIMENTS),
}
SKIPPED = {"bge": [], "openai": []}


def experiment_config(spec: ChunkSpec, tokenizer_name: str, embedder_name: str) -> dict:
    return {**snapshot(), "version": EXPERIMENT_VERSION, "chunk_strategy": spec.strategy, "chunk_size": spec.size,
            "chunk_overlap": spec.overlap, "chunk_headers": spec.headers, "keep_tables": spec.keep_tables,
            "chunk_min_tokens": spec.min_tokens, "child_size": spec.child_size, "embed_model": embedder_name,
            "tokenizer": tokenizer_name, "retrieval_mode": "dense", "rerank_top": 0, "reranker": None, "query_transform": "none"}


def run_experiments(only: list[str] | None = None, k: int = TOP_K, results_dir: Path = RESULTS_DIR,
                    golden_path: Path = GOLDEN_PATH, experiments: list[Experiment] | None = None,
                    tokenizer=None, embedder=None, docs: list[dict] | None = None, stack: str = "bge",
                    report_name: str | None = None, full: bool = False, llm=None) -> list[dict]:
    chosen = STACKS[stack]
    experiments = [e for e in (experiments or chosen.experiments) if not only or e.name in only]
    if not experiments:
        raise ValueError(f"no experiment matches {only}")
    tokenizer = tokenizer or load_tokenizer(*chosen.tokenizer)
    embedder = embedder or load_embedder(*chosen.embedder)
    docs = docs or [normalize_document(d) for d in parse_all()]
    questions = json.loads(golden_path.read_text(encoding="utf-8"))
    client = chromadb.EphemeralClient()
    if full and llm is None:
        llm = load_llm("openai", "gpt-4o-mini")

    rows = []
    for exp in experiments:
        parents = [c for doc in docs for c in chunk_document(doc, tokenizer, spec=exp.spec)]
        if exp.spec.child_size:
            collection = build_index(make_children(parents, tokenizer, exp.spec), embedder, client, parents=parents)
            parent_map = {p["chunk_id"]: p for p in parents}
        else:
            collection, parent_map = build_index(parents, embedder, client), None

        def searcher(query, top_k, collection=collection, parent_map=parent_map):
            return retrieve(query, top_k, embedder, collection, parent_map)

        out = run_evaluation(exp.name, exp.note, k, retrieval_only=not full, use_judge=full, force=True, chunks=parents,
                             searcher=searcher, config=experiment_config(exp.spec, tokenizer.name, embedder.name),
                             results_dir=results_dir, golden_path=golden_path, llm=llm, trace_path=results_dir / "experiment_traces.jsonl")
        rows.append({"experiment": exp, "summary": out["summary"], "by_type": out["by_type"],
                     "diagnostics": diagnose(parents, questions, out["results"])})
        s = out["summary"]
        print(f"{exp.name:56} chunks={len(parents):3}  recall={s['recall']:.2f}  mrr={s['mrr']:.2f}  ndcg={s['ndcg']:.2f}"
              f"  hit={s['hit']:.2f}" + (f"  correct={s['correctness']:.2f}" if full and s["correctness"] is not None else ""), flush=True)
    if not only:
        (results_dir / (report_name or chosen.report)).write_text(render_report(rows, k, stack, results_dir), encoding="utf-8")
    return rows


# ---------------------------------------------------------------- report

def _f(value, digits=2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _score(row: dict) -> float:
    """Rank-aware score used to pick the winner: precision is not comparable across chunk sets, recall saturates."""
    s = row["summary"]
    return (s["mrr"] + s["ndcg"]) / 2


def pick_winner(rows: list[dict]) -> dict:
    return max(rows, key=lambda r: (_score(r), r["summary"]["recall"]))


def _names(ids: list, limit: int = 4) -> str:
    shown = [i if isinstance(i, str) else "/".join(i) for i in ids[:limit]]
    return ", ".join(shown) + (f" (+{len(ids) - limit} more)" if len(ids) > limit else "")


def cause_table(control: dict, winner: dict) -> list[str]:
    """Cause | symptom | in your baseline | with the winning strategy."""
    def row(cause, symptom, describe):
        return f"| {cause} | {symptom} | {describe(control['diagnostics'])} | {describe(winner['diagnostics'])} |"

    def count_of(key, unit):
        return lambda d: f"{len(d[key])} {unit}" + (f": {_names(d[key])}" if d[key] else "")

    def mixed(d):
        hubs = ", ".join(f"{cid} outranked the right chunk for {n} questions" for cid, n in d["hub_chunks"].items())
        return f"{len(d['mixed_topic_chunks'])} of {d['chunks']} chunks span 2+ top-level sections" + (f"; {hubs}" if hubs else "")

    def tail(d):
        miss = d["missed_in_mixed_chunk"]
        return "none missed" if not miss else "; ".join(f"{q}: relevant chunk {c} is a mixed chunk and was not retrieved" for q, c in miss)

    def splits(d):
        answers = f"{len(d['split_answers'])} single-fact answers split" + (f" ({_names(d['split_answers'])})" if d["split_answers"] else "")
        return f"{answers}; {len(d['mid_sentence_cuts'])} mid-sentence cuts"

    def no_context(d):
        return f"{len(d['context_free_chunks'])} of {d['chunks']} chunks do not say which section they belong to"

    return [
        "| Cause | Symptom | In your baseline | With the winner |",
        "|---|---|---|---|",
        row("Chunk too big, several topics (a hub chunk)", "One broad chunk outranks the right one", mixed),
        row("Several topics glued into a tail chunk", "The answer is a small part of the chunk, so its match is weak", tail),
        row("Answer split across chunk boundaries", "No single chunk holds all the key facts", splits),
        row("Orphan heading", "A heading ends one chunk and its text starts the next", count_of("orphan_headings", "chunks")),
        row("Table cut across chunks", "Rows are separated from the table's header row", count_of("split_tables", "cuts")),
        row("Chunk has no context", "A chunk does not say which document or section it came from", no_context),
    ]


def ladder_section(rows: list[dict]) -> list[str]:
    by_name = {r["experiment"].name: r for r in rows}
    steps = [by_name[n] for n in LADDER if n in by_name]
    if len(steps) < 2:
        return []
    lines = ["", "## What each added technique did (cumulative)", "",
             "| Step | Adds | MRR | nDCG | Recall | Hit | (MRR+nDCG)/2 | Change |", "|---|---|---|---|---|---|---|---|"]
    previous = None
    for r in steps:
        s, score = r["summary"], _score(r)
        change = "-" if previous is None else f"{score - previous:+.2f}"
        lines.append(f"| {r['experiment'].name} | {r['experiment'].note} | {_f(s['mrr'])} | {_f(s['ndcg'])} | {_f(s['recall'])} | "
                     f"{_f(s['hit'])} | {_f(score)} | {change} |")
        previous = score
    combined = by_name.get(f"p3o-{COMBINED.label}")
    child = next((r for r in rows if r["experiment"].spec.child_size), None)
    if combined and child:
        helps = _score(child) >= _score(combined) and child["summary"]["hit"] >= combined["summary"]["hit"]
        lines += ["", f"**Parent-child: {'keep' if helps else 'drop'}.** Searching 150-token children and answering with the parent "
                      f"section scores {_f(_score(child))} against {_f(_score(combined))} for the combined setup "
                      f"(Hit {_f(child['summary']['hit'])} against {_f(combined['summary']['hit'])})."]
    return lines


def embedding_section(rows: list[dict], results_dir: Path) -> list[str]:
    """The same chunking under bge-small and under OpenAI embeddings (bge results come from the bge stack's runs)."""
    lines = ["", "## Same chunking, different embeddings", "",
             "Sizes are counted in different tokens (bge vs cl100k), so a '500' is not exactly the same text; "
             "read this as the effect of the whole retrieval stack.", "",
             "| Chunking | bge-small MRR | OpenAI MRR | bge nDCG | OpenAI nDCG | bge Hit | OpenAI Hit |", "|---|---|---|---|---|---|---|"]
    found = False
    for r in rows:
        path = results_dir / f"p3-{r['experiment'].spec.label}" / "summary.json"
        if path.exists():
            b, o = json.loads(path.read_text(encoding="utf-8"))["overall"], r["summary"]
            lines.append(f"| {r['experiment'].spec.label} | {_f(b['mrr'])} | {_f(o['mrr'])} | {_f(b['ndcg'])} | {_f(o['ndcg'])} | "
                         f"{_f(b['hit'])} | {_f(o['hit'])} |")
            found = True
    return lines if found else []


def render_report(rows: list[dict], k: int, stack: str = "bge", results_dir: Path = RESULTS_DIR) -> str:
    control = next((r for r in rows if r["experiment"].spec == ChunkSpec("fixed", 500)), rows[0])
    winner = pick_winner(rows)
    best = {m: max(r["summary"][m] for r in rows) for m in METRICS}
    embed = STACKS[stack].embedder[1]

    def cell(r, m):
        value = _f(r["summary"][m])
        return f"**{value}**" if r["summary"][m] == best[m] else value

    lines = [
        f"# Phase 3: chunking experiments ({embed})",
        "",
        f"Golden set, {control['summary']['n_retrieval']} answerable questions, top-{k} retrieval, `{embed}` embeddings. "
        "No LLM is involved, so these numbers measure retrieval only. Bold = best in the column.",
        "",
        "**Reading the metrics.** Recall and precision depend on how many chunks count as relevant, which changes with "
        "the chunking, so compare MRR, nDCG and Hit across strategies. The winner is the best average of MRR and nDCG. "
        "With 25 questions, a difference of 0.04 is a single question.",
        "",
        "## Results",
        "",
        f"| Experiment | Chunks | Avg tokens | Recall@{k} | Precision@{k} | MRR | nDCG@{k} | Hit@{k} |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['experiment'].name} | {r['diagnostics']['chunks']} | {_f(r['diagnostics']['avg_tokens'], 0)} | "
                     + " | ".join(cell(r, m) for m in METRICS) + " |")
    lines += ["", f"**Winner: `{winner['experiment'].name}`** (MRR {_f(winner['summary']['mrr'])}, nDCG "
              f"{_f(winner['summary']['ndcg'])}, Recall@{k} {_f(winner['summary']['recall'])}) against the control "
              f"`{control['experiment'].name}` (MRR {_f(control['summary']['mrr'])}, nDCG {_f(control['summary']['ndcg'])}, "
              f"Recall@{k} {_f(control['summary']['recall'])}).", ""]
    lines += ladder_section(rows)
    lines += embedding_section(rows, results_dir) if stack != "bge" else []

    types = sorted({t for r in rows for t in r["by_type"] if t != "unanswerable"})
    lines += ["", "## MRR by question type", "", "| Experiment | " + " | ".join(types) + " |", "|---|" + "---|" * len(types)]
    for r in rows:
        lines.append(f"| {r['experiment'].name} | " + " | ".join(_f(r["by_type"][t]["mrr"]) for t in types) + " |")

    lines += ["", f"## Chunking causes: the control against `{winner['experiment'].name}`", ""]
    lines += cause_table(control, winner)

    lines += ["", "## What each strategy did to the chunks", "",
              "| Experiment | Mixed-topic chunks | Orphan headings | Cut tables | Mid-sentence cuts | Context-free chunks | Split answers |",
              "|---|---|---|---|---|---|---|"]
    for r in rows:
        d = r["diagnostics"]
        lines.append(f"| {r['experiment'].name} | {len(d['mixed_topic_chunks'])} | {len(d['orphan_headings'])} | "
                     f"{len(d['split_tables'])} | {len(d['mid_sentence_cuts'])} | {len(d['context_free_chunks'])} | "
                     f"{len(d['split_answers'])} |")

    if SKIPPED[stack]:
        lines += ["", "## Not run", ""] + [f"- `{name}`: {why}" for name, why in SKIPPED[stack]]
    lines += ["", "Every experiment is also a run in the Reports tab (version `exp`). Open a question there to see its "
              "retrieved chunks. Metrics come from heuristic relevance labels, so spot-check a few.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare chunking strategies on the golden set")
    parser.add_argument("--stack", choices=list(STACKS), default="bge")
    parser.add_argument("--only", help="comma-separated experiment names, e.g. p3o-fixed-500")
    parser.add_argument("--full", action="store_true", help="also answer (gpt-4o-mini) and judge; use with --only")
    parser.add_argument("-k", type=int, default=TOP_K)
    args = parser.parse_args()
    run_experiments(args.only.split(",") if args.only else None, args.k, stack=args.stack, full=args.full)
    print(f"\nreports are in {RESULTS_DIR}")
