"""Compare answer prompts with repeats, on the exact chunks a saved run gave the model.

    python -m eval.prompt_repeats --source v4 --prompts v2,v2.3 --repeats 3

One run of 50 questions cannot show whether a prompt is better: the answer model gives different answers to the same prompt in
different runs (Phase 6 notes), so a gain or loss of one or two questions is not a result. Here every prompt answers every
question `repeats` times on identical chunks (the chunks the model read in the saved run), each answer is judged, and the report
shows the mean scores over all repeats and the questions where the prompts differ consistently. Retrieval is not run.
Written to eval/results/prompt_repeats_<source>.md; the raw verdicts to prompt_repeats_<source>.json.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eval.judge import judge_answer
from eval.run_eval import GOLDEN_PATH, RESULTS_DIR, mean, read_jsonl
from eval.rescore import collection_for
from src.config import JUDGE_MODEL, JUDGE_PROMPT_VERSION, JUDGE_PROVIDER
from src.generate.answer import parse_answer
from src.generate.generate import build_prompt, source_ids
from src.generate.llm import load_llm
from src.ingest.index import get_client

WORKERS = 6


def chunk_metadata(source: str) -> dict:
    """{chunk_id: {doc_id, page}} from the index the source run read, so the prompt here is the one the pipeline builds
    (the saved answers keep only chunk_id and text; filling in page 1 for every chunk made a different prompt)."""
    config = json.loads((RESULTS_DIR / source / "config.json").read_text(encoding="utf-8"))["config"]
    data = get_client().get_collection(collection_for(config)).get(include=["metadatas"])
    return {i: {"doc_id": m["doc_id"], "page": m.get("page", "?")} for i, m in zip(data["ids"], data["metadatas"])}


def answer_once(llm, judge, question: dict, chunks: list[dict], prompt: str, meta: dict | None = None) -> dict:
    """One answer to one question with one prompt, and the judge's verdict on it (the median of JUDGE_VOTES calls)."""
    meta = meta or {}
    rows = [{**c, **meta.get(c["chunk_id"], {"doc_id": c["chunk_id"].rsplit("_c", 1)[0], "page": "?"})} for c in chunks]
    parsed = parse_answer(llm.complete(build_prompt(question["question"], rows, prompt), json_mode=True)["text"], source_ids(rows))
    verdict = judge_answer(judge, question, parsed.text, chunks, found=parsed.found)
    return {"answer": parsed.text, "found": parsed.found, "status": parsed.status,
            **{m: verdict[m] for m in ("correctness", "completeness", "faithfulness", "relevance")}, "abstained": verdict["abstained"]}


def run_repeats(llm, judge, questions: list[dict], answers: dict, prompts: list[str], repeats: int,
                meta: dict | None = None) -> dict:
    """{prompt: {question id: [verdict, ...]}}: every prompt answers every question `repeats` times."""
    jobs = [(p, q) for q in questions if answers.get(q["id"], {}).get("chunks") for p in prompts for _ in range(repeats)]
    with ThreadPoolExecutor(WORKERS) as pool:
        done = list(pool.map(lambda job: answer_once(llm, judge, job[1], answers[job[1]["id"]]["chunks"], job[0], meta), jobs))
    out = {p: {} for p in prompts}
    for (p, q), verdict in zip(jobs, done):
        out[p].setdefault(q["id"], []).append(verdict)
    return out


def score(runs: dict, ids: list[str] | None = None) -> dict:
    """Mean of each metric over every repeat of every question (completeness is not defined for unanswerable questions)."""
    rows = [v for qid, vs in runs.items() if ids is None or qid in ids for v in vs]
    return {m: mean(r[m] for r in rows) for m in ("correctness", "completeness", "faithfulness", "relevance")} | {
        "fully_correct": mean(float(r["correctness"] == 1.0) for r in rows),
        "fully_complete": mean(float(r["completeness"] == 1.0) for r in rows if r["completeness"] is not None),
        "false_abstention": mean(float(r["abstained"]) for r in rows),
        "n": len(rows), "avg_chars": mean(len(r["answer"]) for r in rows)}


def consistent_differences(results: dict, base: str, other: str, margin: int = 2) -> dict:
    """Questions where `other` is correct and complete in at least `margin` more repeats than `base` (better) or fewer (worse)."""
    def good(runs):
        return sum(r["correctness"] == 1.0 and r["completeness"] in (None, 1.0) for r in runs)

    better, worse = [], []
    for qid, base_runs in results[base].items():
        a, b = good(base_runs), good(results[other][qid])
        if b - a >= margin:
            better.append((qid, a, b))
        elif a - b >= margin:
            worse.append((qid, a, b))
    return {"better": better, "worse": worse}


def render_report(results: dict, source: str, repeats: int) -> str:
    prompts = list(results)
    lines = [f"# Prompt repeats on the chunks of `{source}`", "",
             f"Every prompt answered every question {repeats} times on the same chunks; each answer was judged ({JUDGE_MODEL}, judge prompt {JUDGE_PROMPT_VERSION}). "
             "`fully correct` / `fully complete` = the share of answers with correctness 1.0 / completeness 1.0.", "",
             "| Prompt | Answers | Correct | Complete | Faithful | Fully correct | Fully complete | Answer chars |", "|---|---|---|---|---|---|---|---|"]
    for p in prompts:
        s = score(results[p])
        lines.append(f"| `{p}` | {s['n']} | {s['correctness']:.3f} | {s['completeness']:.3f} | {s['faithfulness']:.3f} | "
                     f"{s['fully_correct']:.3f} | {s['fully_complete']:.3f} | {s['avg_chars']:.0f} |")
    for other in prompts[1:]:
        d = consistent_differences(results, prompts[0], other)
        lines += ["", f"`{other}` against `{prompts[0]}`, questions that differ in at least 2 of {repeats} repeats (good = correct and complete):",
                  f"- better: {', '.join(f'{q} ({a} to {b} of {repeats})' for q, a, b in d['better']) or 'none'}",
                  f"- worse: {', '.join(f'{q} ({a} to {b} of {repeats})' for q, a, b in d['worse']) or 'none'}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt comparison with repeats on fixed chunks")
    parser.add_argument("--source", default="v4")
    parser.add_argument("--prompts", default="v2,v2.3")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--golden", default=str(GOLDEN_PATH), help="the question file the source run answered (default: eval/golden_set.json)")
    parser.add_argument("--tag", default="", help="added to the output file names, e.g. holdout")
    parser.add_argument("--ids", help="only these question ids, e.g. q04,q25")
    args = parser.parse_args()
    questions = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    if args.ids:
        questions = [q for q in questions if q["id"] in args.ids.split(",")]
    answers = read_jsonl(RESULTS_DIR / args.source / "answers.jsonl")
    results = run_repeats(load_llm(), load_llm(JUDGE_PROVIDER, JUDGE_MODEL), questions, answers, args.prompts.split(","), args.repeats,
                          chunk_metadata(args.source))
    stem = f"prompt_repeats_{args.source}" + (f"_{args.tag}" if args.tag else "")
    (RESULTS_DIR / f"{stem}.json").write_text(json.dumps(results), encoding="utf-8")
    report = render_report(results, args.source, args.repeats)
    (RESULTS_DIR / f"{stem}.md").write_text(report, encoding="utf-8")
    print(report)
