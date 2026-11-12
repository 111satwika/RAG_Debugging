"""Phase 6, step 4: where the time of a query goes, and what sending fewer chunks or reranking fewer candidates would change.

    python -m eval.latency_breakdown                    # time every stage of the retrieval for 24 golden questions
    python -m eval.latency_breakdown --reranker-sweep   # time the reranker alone for different pool sizes and chunk lengths

The stages are timed by wrapping the real functions of the active version, so the numbers are what the pipeline pays:
query embedding (an OpenAI call), the vector search (Chroma), BM25, the cross-encoder reranker (a local model on the CPU), and the
model call that plans a hard question. The answer model call is timed in each trace (llm_call) and is not repeated here.
Written to eval/results/latency_breakdown.md.
"""
import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from eval.run_eval import GOLDEN_PATH, RESULTS_DIR
from src.config import RERANK_TOP, RERANKER, TOP_K

REPORT_PATH = RESULTS_DIR / "latency_breakdown.md"
SAMPLE = 24          # questions timed: every second standard question plus the cross-document ones


class Timers:
    """Adds up the time spent inside wrapped functions, per stage, for one query at a time."""

    def __init__(self):
        self.stages = defaultdict(float)
        self.calls = defaultdict(int)

    def wrap(self, stage: str, fn):
        def timed(*args, **kwargs):
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self.stages[stage] += (time.perf_counter() - start) * 1000
                self.calls[stage] += 1
        return timed

    def reset(self) -> dict:
        taken = {"ms": dict(self.stages), "calls": dict(self.calls)}
        self.stages.clear()
        self.calls.clear()
        return taken


def summarize(rows: list[dict]) -> dict:
    """Median per stage over queries (a stage a query never used counts as 0), plus how many queries used it."""
    stages = sorted({s for r in rows for s in r["ms"]})
    return {s: {"median_ms": statistics.median(r["ms"].get(s, 0.0) for r in rows),
                "mean_ms": statistics.mean(r["ms"].get(s, 0.0) for r in rows),
                "queries": sum(1 for r in rows if s in r["ms"]),
                "calls_per_query": statistics.mean(r["calls"].get(s, 0) for r in rows)} for s in stages}


def sample_questions(golden: list[dict]) -> list[dict]:
    answerable = [q for q in golden if q["type"] != "unanswerable"]
    cross = [q for q in answerable if q["type"] == "cross_doc"]
    plain = [q for q in answerable if q["type"] != "cross_doc"][::2]
    return (plain[: SAMPLE - len(cross)] + cross)


def measure(retriever, spec, questions: list[dict], k: int = TOP_K) -> list[dict]:
    """Time each stage of `retriever.search` for every question. The retriever's caches are cleared first so nothing is free."""
    from src.retrieve import retriever as module
    from src.retrieve.rerank import get_reranker

    timers = Timers()
    originals = (retriever.embedder.embed_queries, module.dense_search, retriever.bm25.search)
    reranker = retriever.reranker or get_reranker(spec.reranker)
    model = reranker._load()
    predict = model.predict
    retriever.embedder.embed_queries = timers.wrap("embed query", originals[0])
    module.dense_search = timers.wrap("vector search", originals[1])
    retriever.bm25.search = timers.wrap("bm25", originals[2])
    model.predict = timers.wrap("rerank", predict)
    rows = []
    try:
        for q in questions:
            retriever._vectors.clear()
            retriever._plans.clear()
            retriever.llm_ms.clear()
            timers.reset()
            start = time.perf_counter()
            hits = retriever.search(spec, q["question"], k)
            total = (time.perf_counter() - start) * 1000
            taken = timers.reset()
            if retriever.llm_ms:
                taken["ms"]["query plan (model call)"] = sum(retriever.llm_ms)
                taken["calls"]["query plan (model call)"] = len(retriever.llm_ms)
            rows.append({"id": q["id"], "type": q["type"], "total_ms": total, "hits": len(hits), **taken})
            print(f"{q['id']} {total / 1000:.1f} s  " + "  ".join(f"{s}={v / 1000:.1f}" for s, v in taken["ms"].items()), flush=True)
    finally:
        retriever.embedder.embed_queries, module.dense_search, retriever.bm25.search = originals
        model.predict = predict
    return rows


def render_report(rows: list[dict], sweep: list[dict] | None = None) -> str:
    plain = [r for r in rows if r["type"] != "cross_doc"]
    cross = [r for r in rows if r["type"] == "cross_doc"]
    lines = ["# Phase 6 step 4: where a query's time goes", "",
             f"Retrieval of the active version (hybrid, reranker `{RERANKER}` over the top {RERANK_TOP}), timed per stage for "
             f"{len(plain)} ordinary questions and {len(cross)} cross-document questions (which also make the query-plan model call).", ""]
    for name, group in (("Ordinary questions", plain), ("Cross-document questions", cross)):
        if not group:
            continue
        total = statistics.median(r["total_ms"] for r in group)
        lines += [f"## {name}: median retrieval {total / 1000:.1f} s", "",
                  "| Stage | Median ms | Share of the median total | Calls per query |", "|---|---|---|---|"]
        for stage, s in sorted(summarize(group).items(), key=lambda kv: -kv[1]["median_ms"]):
            lines.append(f"| {stage} | {s['median_ms']:.0f} | {s['median_ms'] / total:.0%} | {s['calls_per_query']:.1f} |")
        lines.append("")
    if sweep:
        lines += ["## The reranker alone: pool size and chunk length", "",
                  "| Chunks reranked | Max tokens per chunk | Median ms | Per chunk ms |", "|---|---|---|---|"]
        for r in sweep:
            lines.append(f"| {r['pool']} | {r['max_length']} | {r['median_ms']:.0f} | {r['median_ms'] / r['pool']:.0f} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def reranker_sweep(retriever, spec, questions: list[dict], pools=(3, 5, 10, 20), lengths=(512, 256)) -> list[dict]:
    """Time the cross-encoder alone on the real candidates (the hybrid top 20) for several pool sizes and max token lengths."""
    from src.retrieve.rerank import get_reranker

    reranker = retriever.reranker or get_reranker(spec.reranker)
    model = reranker._load()
    original_length = model.max_length
    out = []
    try:
        for length in lengths:
            model.max_length = length
            for pool in pools:
                times = []
                for q in questions[:8]:
                    candidates = retriever._candidates("hybrid", q["question"], pool, None)
                    start = time.perf_counter()
                    reranker.rerank(q["question"], candidates, TOP_K)
                    times.append((time.perf_counter() - start) * 1000)
                out.append({"pool": pool, "max_length": length, "median_ms": statistics.median(times)})
                print(f"pool {pool} length {length}: {out[-1]['median_ms']:.0f} ms", flush=True)
    finally:
        model.max_length = original_length
    return out


if __name__ == "__main__":
    from src.ingest.embed import load_embedder
    from src.retrieve.dense import get_collection
    from src.retrieve.rerank import get_reranker
    from src.retrieve.retriever import Retriever, configured_spec
    from src.generate.llm import load_llm

    parser = argparse.ArgumentParser(description="Latency breakdown of the retrieval")
    parser.add_argument("--reranker-sweep", action="store_true", help="also time the reranker for several pool sizes and chunk lengths")
    args = parser.parse_args()
    spec = configured_spec()
    if not spec.rerank_top:
        raise SystemExit("this measures the reranked retrieval (a version with RERANK_TOP > 0)")
    embedder = load_embedder()
    retriever = Retriever(embedder, get_collection(embedder), llm=load_llm())
    get_reranker(spec.reranker).warm_up()
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    questions = sample_questions(golden)
    rows = measure(retriever, spec, questions)
    sweep = reranker_sweep(retriever, spec, questions) if args.reranker_sweep else None
    REPORT_PATH.write_text(render_report(rows, sweep), encoding="utf-8")
    print(render_report(rows, sweep))
