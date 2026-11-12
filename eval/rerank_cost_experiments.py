"""Phase 6, step 4: what reranking fewer chunks, or shorter chunks, costs in retrieval quality.

    python -m eval.rerank_cost_experiments            # score the hybrid top 10 of every labelled question, then compare settings

The reranker is 95% of the retrieval time (about 1 s per chunk on this CPU, see eval/latency_breakdown.py). A cross-encoder scores
every (question, chunk) pair on its own, so the scores of the top 10 candidates, computed once per question and per maximum length
(512 tokens, or 256 which cuts a long chunk), also give every smaller pool: reranking the top 5 is the same as keeping the first 5
candidates and sorting them by those scores. Each setting is then measured on the 44 answerable questions with the relevance labels
of the saved `v3` run: recall, MRR and nDCG over the k chunks that would be sent to the answer model.

Plain hybrid retrieval is used for every question (no query transformation), so the comparison is about the reranker alone.
Written to eval/results/rerank_cost_experiments.md.
"""
import argparse
import json
import statistics
import time

from eval.metrics import retrieval_metrics
from eval.run_eval import GOLDEN_PATH, RESULTS_DIR, mean

REPORT_PATH = RESULTS_DIR / "rerank_cost_experiments.md"
RECORDS_PATH = RESULTS_DIR / "rerank_cost_records.json"
SOURCE_RUN = "v3"
POOLS = (10, 7, 5, 3)
KS = (5, 3)
LENGTHS = (512, 256)
POOL_MAX = max(POOLS)


def collect(retriever, reranker, questions: list[dict], lengths=LENGTHS, pool: int = POOL_MAX) -> list[dict]:
    """For each question: its hybrid top `pool` candidates and the cross-encoder score of each at every maximum length."""
    model = reranker._load()
    original = model.max_length
    records = []
    try:
        for q in questions:
            candidates = retriever._candidates("hybrid", q["question"], pool, None)
            rec = {"id": q["id"], "chunks": [c["chunk_id"] for c in candidates], "scores": {}, "ms": {}}
            for length in lengths:
                model.max_length = length
                start = time.perf_counter()
                scores = model.predict([(q["question"], c["text"]) for c in candidates], show_progress_bar=False)
                rec["ms"][str(length)] = (time.perf_counter() - start) * 1000
                rec["scores"][str(length)] = [float(s) for s in scores]
            records.append(rec)
            print(f"{q['id']} scored {len(candidates)} candidates", flush=True)
    finally:
        model.max_length = original
    return records


def rank(record: dict, pool: int, length: int) -> list[str]:
    """The chunk ids after reranking the first `pool` candidates with the scores of that maximum length (best first)."""
    pairs = list(zip(record["scores"][str(length)][:pool], record["chunks"][:pool]))
    return [cid for _, cid in sorted(pairs, key=lambda p: -p[0])]


def evaluate(records: list[dict], labels: dict, pool: int, k: int, length: int) -> dict:
    """Mean retrieval metrics of one setting over the questions that have labelled relevant chunks."""
    rows = []
    for r in records:
        relevant = set(labels.get(r["id"], {}).get("ids", []))
        if relevant:
            rows.append(retrieval_metrics(rank(r, pool, length)[:k], relevant, k))
    per_chunk_ms = statistics.median(r["ms"][str(length)] / len(r["chunks"]) for r in records)
    return {"pool": pool, "k": k, "length": length, "n": len(rows), **{m: mean(x[m] for x in rows) for m in ("recall", "mrr", "ndcg", "hit")},
            "rerank_s": pool * per_chunk_ms / 1000}


def render_report(records: list[dict], labels: dict) -> str:
    lines = ["# Phase 6 step 4: reranking fewer or shorter chunks", "",
             f"Hybrid top {POOL_MAX} candidates of {len(records)} questions, scored once by the cross-encoder at each maximum length; "
             "smaller pools are derived from the same scores. Quality is measured on the answerable questions with relevance labels "
             f"(the `{SOURCE_RUN}` labels), over the k chunks sent to the answer model. `rerank s` = the measured reranker time per query for that pool.", "",
             "| Candidates reranked | k sent to the model | Max tokens per chunk | Recall | MRR | nDCG | Hit | Rerank s per query |",
             "|---|---|---|---|---|---|---|---|"]
    for length in LENGTHS:
        for pool in POOLS:
            for k in KS:
                if k > pool:
                    continue
                s = evaluate(records, labels, pool, k, length)
                lines.append(f"| {pool} | {k} | {length} | {s['recall']:.3f} | {s['mrr']:.3f} | {s['ndcg']:.3f} | {s['hit']:.3f} | {s['rerank_s']:.1f} |")
    lines += ["", "Baseline (the current version): 10 candidates, k = 5, 512 tokens.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    from src.generate.llm import load_llm
    from src.ingest.embed import load_embedder
    from src.retrieve.dense import get_collection
    from src.retrieve.rerank import get_reranker
    from src.retrieve.retriever import Retriever, configured_spec

    parser = argparse.ArgumentParser(description="Reranker cost against quality")
    parser.add_argument("--reuse", action="store_true", help="report again from the saved scores (no model runs)")
    args = parser.parse_args()
    labels = json.loads((RESULTS_DIR / SOURCE_RUN / "relevance_labels.json").read_text(encoding="utf-8"))
    if args.reuse:
        records = json.loads(RECORDS_PATH.read_text(encoding="utf-8"))
    else:
        spec = configured_spec()
        embedder = load_embedder()
        retriever = Retriever(embedder, get_collection(embedder), llm=load_llm())
        reranker = get_reranker(spec.reranker)
        reranker.warm_up()
        golden = [q for q in json.loads(GOLDEN_PATH.read_text(encoding="utf-8")) if q["type"] != "unanswerable"]
        records = collect(retriever, reranker, golden)
        RECORDS_PATH.write_text(json.dumps(records), encoding="utf-8")
    report = render_report(records, labels)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
