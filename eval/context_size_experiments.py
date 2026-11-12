"""Phase 6, step 4: does sending fewer chunks to the answer model change the answers?

    python -m eval.context_size_experiments            # k = 5 and k = 3, both replaying the chunks the saved `v3` run retrieved

Retrieval is not repeated: each setting answers from the first k chunks of what `v3` retrieved for the question (a cross-document
question had 6 chunks, 3 per document in round-robin order, so k = 5 drops the last one and k = 3 keeps 2 of one document and 1 of
the other). The answer prompt and the judge are the same, so the difference is the number of chunks. Runs are saved as
p6-k<k>-on-v3 and listed in eval/results/experiment_log.md; the report is eval/results/context_size_experiments.md.
"""
import argparse

from eval.prompt_experiments import saved_retrieval
from eval.run_eval import RESULTS_DIR, load_index_chunks, run_evaluation
from src.config import snapshot
from src.generate.llm import load_llm

SOURCE_RUN = "v3"
KS = (5, 3)
REPORT_PATH = RESULTS_DIR / "context_size_experiments.md"


def run_experiments(ks=KS, source: str = SOURCE_RUN, results_dir=RESULTS_DIR, llm=None, judge_llm=None, chunks=None) -> list[dict]:
    retrieved = saved_retrieval(source, results_dir)
    chunks = chunks if chunks is not None else load_index_chunks()
    llm = llm or load_llm()
    rows = []
    for k in ks:
        out = run_evaluation(f"p6-k{k}-on-{source}", f"the answer model gets the first {k} chunks of what {source} retrieved", k, force=True,
                             chunks=chunks, searcher=lambda question, top_k: retrieved[question][:top_k],
                             config={**snapshot(), "version": "exp", "top_k": k}, llm=llm, judge_llm=judge_llm, results_dir=results_dir,
                             trace_path=results_dir / "experiment_traces.jsonl")
        rows.append({"k": k, "summary": out["summary"], "results": out["results"]})
        s = out["summary"]
        print(f"k={k} correct={s['correctness']:.3f} complete={s['completeness']:.3f} faithful={s['faithfulness']:.3f} "
              f"abstention={s['abstention_accuracy']:.3f} tokens={s['avg_input_tokens']:.0f} p50={s['latency_p50_s']:.1f}s", flush=True)
    return rows


def render_report(rows: list[dict], source: str = SOURCE_RUN) -> str:
    lines = [f"# Phase 6 step 4: how many chunks the answer model gets (replaying the `{source}` retrieval)", "",
             "| k | Correctness | Completeness | Faithfulness | Abstention accuracy | Input tokens | LLM call s (p50) |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        s = r["summary"]
        llm_s = sorted(x["llm_ms"] for x in r["results"] if x["llm_ms"])
        lines.append(f"| {r['k']} | {s['correctness']:.3f} | {s['completeness']:.3f} | {s['faithfulness']:.3f} | {s['abstention_accuracy']:.3f} | "
                     f"{s['avg_input_tokens']:.0f} | {llm_s[len(llm_s) // 2] / 1000:.1f} |")
    base = {x["id"]: x for x in rows[0]["results"]}
    for r in rows[1:]:
        worse = [x["id"] for x in r["results"] if x["judge"] and base[x["id"]]["judge"]
                 and (x["judge"]["correctness"] < base[x["id"]]["judge"]["correctness"]
                      or (x["judge"]["completeness"] or 0) < (base[x["id"]]["judge"]["completeness"] or 0))]
        better = [x["id"] for x in r["results"] if x["judge"] and base[x["id"]]["judge"]
                  and (x["judge"]["correctness"] > base[x["id"]]["judge"]["correctness"]
                       or (x["judge"]["completeness"] or 0) > (base[x["id"]]["judge"]["completeness"] or 0))]
        lines += ["", f"k = {r['k']} against k = {rows[0]['k']}: correctness or completeness lower on {', '.join(worse) or 'none'}; higher on {', '.join(better) or 'none'}."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    argparse.ArgumentParser(description="Answer quality for different numbers of chunks").parse_args()
    rows = run_experiments()
    REPORT_PATH.write_text(render_report(rows), encoding="utf-8")
    print(render_report(rows))
