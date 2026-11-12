"""Phase 5: compare answer prompts on the golden set, with the retrieval held fixed.

    python -m eval.prompt_experiments                       # v1, v2 and the drill (v2 without the grounding rules)
    python -m eval.prompt_experiments --only v2             # one prompt

Every prompt answers the same chunks: the ones a saved run retrieved (default: the `v2-extended` run), so the comparison
shows what the prompt does and nothing else (retrieval is not repeated and its latency is not part of the numbers).
Each prompt is saved as a run named p5-<prompt> with version "exp" and compared in eval/results/prompt_experiments.md.

  v1             the baseline: chunks pasted in, "answer using the context", plain text
  v2             grounding rules, sources labelled [S1]..., citations required, JSON answer (answer, sources, found)
  v2-ungrounded  the drill: v2 without the grounding rules, to see the hallucinations they prevent
"""
import argparse
import json
from pathlib import Path

from eval.run_eval import GOLDEN_PATH, RESULTS_DIR, load_index_chunks, read_jsonl, run_evaluation
from src.config import TOP_K, snapshot
from src.config import JUDGE_MODEL, LLM_MODEL
from src.generate.llm import load_llm

EXPERIMENT_VERSION = "exp"
REPORT_NAME = "prompt_experiments.md"
SOURCE_RUN = "v2-extended"
PROMPTS = {
    "v1": "Baseline: the chunks pasted in, answered in plain text",
    "v2": "Grounding rules, sources labelled [S1]..., citations required, JSON answer (answer, sources, found)",
    "v2-ungrounded": "Drill: v2 without the grounding rules (no 'only the sources', no 'not found')",
    "v2.1": "v2 with completeness rules: every part and detail, evidence from each document, conflicting values, no wrong attachment",
    "v2.2": "v2 plus three narrow rules: begin yes/no questions with Yes or No, list every item for what-changed questions, keep each fact with its own item",
    "v2.3": "v2 with 'keep it short' replaced by 'concise but every fact that answers', attribute each fact to its own item (name the requirement, e.g. FR-10), yes/no starts with Yes or No",
    "v2.4": "v2.3 with the yes/no rule refined: Yes only if it holds for everything asked; if only for part ('every', 'all'), answer No and say which part holds",
    "v2.5": "v2.4 with the attribution rule reworded: keep each fact with the item it describes, do not move a fact between items (no longer 'do not combine facts from different parts of a sentence')",
}
DEFAULT_PROMPTS = ("v1", "v2", "v2-ungrounded")      # the Phase 5 comparison; v2.1 to v2.5 are run with --only
CONTROL = "v1"


def saved_retrieval(run: str = SOURCE_RUN, results_dir: Path = RESULTS_DIR) -> dict[str, list[dict]]:
    """question text -> the chunks the model read in that run, rebuilt from its saved answers (chunk ids, order and text) and
    results (document, page, score). The answering pass is the source of truth: with validation it can differ from the retrieval
    pass (a corrective search merged other chunks), and an experiment on the prompt must give the model what it saw."""
    answers = read_jsonl(results_dir / run / "answers.jsonl")
    results = [json.loads(line) for line in (results_dir / run / "results.jsonl").read_text(encoding="utf-8").splitlines() if line]
    hits = {}
    for res in results:
        meta = {h["chunk_id"]: h for h in res["retrieved"]}
        rows = []
        for rank, c in enumerate(answers[res["id"]]["chunks"], start=1):
            known = meta.get(c["chunk_id"]) or {"chunk_id": c["chunk_id"], "doc_id": c["chunk_id"].rsplit("_c", 1)[0], "page": 1, "score": 0.0}
            rows.append({**known, "rank": rank, "doc_type": "", "text": c["text"]})
        hits[res["question"]] = rows
    return hits


def run_experiments(only: list[str] | None = None, k: int = TOP_K, results_dir: Path = RESULTS_DIR,
                    golden_path: Path = GOLDEN_PATH, llm=None, judge_llm=None, source_run: str = SOURCE_RUN,
                    chunks: list[dict] | None = None) -> list[dict]:
    prompts = [p for p in PROMPTS if (p in only if only else p in DEFAULT_PROMPTS)]
    if not prompts:
        raise ValueError(f"no prompt matches {only}")
    custom = source_run != SOURCE_RUN                    # another run's retrieval: keep the earlier runs and report
    retrieved = saved_retrieval(source_run, results_dir)
    chunks = chunks if chunks is not None else load_index_chunks()
    llm = llm or load_llm()

    def searcher(question, top_k):
        return retrieved[question][:top_k]

    rows = []
    for prompt in prompts:
        config = {**snapshot(), "version": EXPERIMENT_VERSION, "top_k": k, "prompt_version": prompt}
        out = run_evaluation(f"p5-{prompt}" + (f"-on-{source_run}" if custom else ""), PROMPTS[prompt], k, force=True,
                             chunks=chunks, searcher=searcher, config=config,
                             prompt_version=prompt, llm=llm, judge_llm=judge_llm, results_dir=results_dir,
                             golden_path=golden_path, trace_path=results_dir / "experiment_traces.jsonl")
        rows.append({"prompt": prompt, "summary": out["summary"], "by_tier": out["by_tier"], "results": out["results"]})
        s = out["summary"]
        print(f"p5-{prompt:15} correct={s['correctness']:.2f} faithful={s['faithfulness']:.2f} abstention={s['abstention_accuracy']:.2f}"
              f" refusal={s['refusal_rate_unanswerable']:.2f}" + (f" cites={s['citation_rate']:.2f}" if s["citation_rate"] is not None else ""),
              flush=True)
    if not only or custom:
        report = f"prompt_experiments_on_{source_run}.md" if custom else REPORT_NAME
        (results_dir / report).write_text(render_report(rows, source_run), encoding="utf-8")
    return rows


# ---------------------------------------------------------------- report

def _f(value, digits=2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


METRICS = (
    ("Correctness", "correctness"), ("Relevance", "relevance"), ("Completeness", "completeness"),
    ("Faithfulness", "faithfulness"), ("Abstention accuracy", "abstention_accuracy"),
    ("Refusal on unanswerable", "refusal_rate_unanswerable"), ("False abstention (answerable)", "false_abstention_rate"),
    ("Citation rate (answerable, answered)", "citation_rate"), ("Cites a relevant chunk", "cites_relevant_rate"),
    ("Answer format fully valid", "format_ok_rate"), ("Reply not valid JSON", "format_invalid_rate"),
    ("`found` agrees with the judge", "found_agreement"), ("Tokens in / query", "avg_input_tokens"),
    ("Tokens out / query", "avg_output_tokens"),
)


def render_report(rows: list[dict], source_run: str) -> str:
    names = [r["prompt"] for r in rows]
    lines = ["# Phase 5: answer prompts", "",
             f"Retrieval is fixed: every prompt answers the chunks the `{source_run}` run retrieved. 50 golden questions "
             f"(44 answerable, 6 unanswerable); answers {LLM_MODEL}, judge {JUDGE_MODEL}. One question is 0.02 of a score.", "",
             "## All prompts", "", "| Metric | " + " | ".join(f"`{n}`" for n in names) + " |", "|---|" + "---|" * len(names)]
    for label, key in METRICS:
        digits = 0 if key.startswith("avg_") else 2
        lines.append(f"| {label} | " + " | ".join(_f(r["summary"].get(key), digits) for r in rows) + " |")
    for tier in ("standard", "hard"):
        lines.append(f"| Correctness, {tier} questions | " + " | ".join(_f((r["by_tier"].get(tier) or {}).get("correctness")) for r in rows) + " |")
    lengths = [[len(res["answer"]) for res in r["results"] if res["answer"]] for r in rows]
    lines.append("| Average answer length (characters) | " + " | ".join(f"{sum(n) / max(len(n), 1):.0f}" for n in lengths) + " |")

    unanswerable = [res["id"] for res in rows[0]["results"] if res["type"] == "unanswerable"]
    lines += ["", "## The unanswerable questions (the drill)", "",
              "R = refused (said the documents do not contain it), A = answered from somewhere.", "",
              "| Question | " + " | ".join(f"`{n}`" for n in names) + " |", "|---|" + "---|" * len(names)]
    by_prompt = {r["prompt"]: {res["id"]: res for res in r["results"]} for r in rows}
    for qid in unanswerable:
        question = by_prompt[names[0]][qid]["question"]
        cells = []
        for n in names:
            res = by_prompt[n][qid]
            verdict = "-" if not res["judge"] else ("R" if res["judge"]["abstained"] else "A")
            cells.append(verdict)
        lines.append(f"| {qid}: {question} | " + " | ".join(cells) + " |")

    lines += ["", "## What each prompt is", ""] + [f"- `{n}`: {PROMPTS[n]}" for n in names]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5 prompt experiments")
    parser.add_argument("--only", help="comma-separated prompt names, e.g. v2,v2.1")
    parser.add_argument("--source", default=SOURCE_RUN, help="the saved run whose retrieved chunks every prompt answers")
    parser.add_argument("-k", type=int, default=TOP_K)
    args = parser.parse_args()
    run_experiments(args.only.split(",") if args.only else None, args.k, source_run=args.source)
    print(f"reports are in {RESULTS_DIR}")
