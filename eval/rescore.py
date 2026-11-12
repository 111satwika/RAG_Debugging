"""Score the saved answers of an earlier run again, with the current golden set and judge prompt. Nothing is re-answered.

    python -m eval.rescore --source v3-judge-v1 --name v3-rescored

Retrieval is not redone: the chunks each question retrieved are read from the saved run, so retrieval metrics stay the
same and any change in the generation scores comes from the new reference answers alone.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

from eval.run_eval import (GOLDEN_PATH, RESULTS_DIR, append_jsonl, attach_failures, build_results, judge_answers,
                           load_index_chunks, log_experiment, read_jsonl, render_report, summarize, summarize_by_tier,
                           summarize_by_type)
from src.config import JUDGE_MODEL, JUDGE_PROMPT_VERSION, JUDGE_PROVIDER
from src.generate.llm import load_llm


def collection_for(config: dict) -> str:
    """The index a run read from: the fixed 500-token chunks (v1) or the structure-aware ones (v2 and later)."""
    return "rag_chunks_v2" if config.get("chunk_strategy", "fixed") == "structure" else "rag_chunks"


def saved_search(results: dict):
    """A search function that answers from the saved retrieval instead of the index."""
    by_question = {r["question"]: r["retrieved"] for r in results.values()}
    return lambda question, k: by_question[question][:k]


def verdict_changes(old: dict, new: dict) -> list[dict]:
    """Per-question change in correctness and completeness between two runs' results (keyed by id)."""
    changes = []
    for qid, r in new.items():
        before, after = old.get(qid, {}).get("judge"), r["judge"]
        if not before or not after:
            continue
        diff = {m: (before[m], after[m]) for m in ("correctness", "completeness", "faithfulness") if before[m] != after[m]}
        if diff:
            changes.append({"id": qid, **diff})
    return changes


def rescore(source: str, name: str, note: str = "", judge_llm=None, results_dir: Path = RESULTS_DIR,
            golden_path: Path = GOLDEN_PATH, chunks: list[dict] | None = None, force: bool = False,
            rejudge_all: bool = False) -> dict:
    src_dir = results_dir / source
    old_meta = json.loads((src_dir / "config.json").read_text(encoding="utf-8"))
    old_results = read_jsonl(src_dir / "results.jsonl")
    answers = read_jsonl(src_dir / "answers.jsonl")
    labels = json.loads((src_dir / "relevance_labels.json").read_text(encoding="utf-8"))
    k = old_meta["k"]

    golden = {q["id"]: q for q in json.loads(golden_path.read_text(encoding="utf-8"))}
    questions = [golden[qid] for qid in old_results]

    run_dir = results_dir / name
    if force and (run_dir / "judgments.jsonl").exists():
        (run_dir / "judgments.jsonl").unlink()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "answers.jsonl").write_text((src_dir / "answers.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "relevance_labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")

    judge_llm = judge_llm or load_llm(JUDGE_PROVIDER, JUDGE_MODEL)
    # A question without a split has the same judge prompt text as before, so its old verdict is kept: only the
    # questions whose reference changed are judged again, which keeps judge noise out of the comparison.
    judged_path = run_dir / "judgments.jsonl"
    if not judged_path.exists() and not rejudge_all:
        unchanged = {q["id"] for q in questions if not q.get("required_answer")}
        for qid, verdict in read_jsonl(src_dir / "judgments.jsonl").items():
            if qid in unchanged:
                append_jsonl(judged_path, verdict)
    judge_answers(questions, answers, run_dir, judge_llm)
    judgments = read_jsonl(run_dir / "judgments.jsonl")

    hits = {qid: r["retrieved"] for qid, r in old_results.items()}
    results = build_results(questions, hits, labels, answers, judgments, k)
    chunks = chunks if chunks is not None else load_index_chunks(collection_for(old_meta["config"]))
    attach_failures(questions, results, chunks, saved_search(old_results))

    meta = {**old_meta, "name": name, "created": datetime.now().strftime("%Y-%m-%d %H:%M"), "rescored_from": source,
            "note": note or f"saved answers of '{source}' scored again with the split golden answers (judge prompt {JUDGE_PROMPT_VERSION})",
            "judge": f"{JUDGE_PROVIDER}:{JUDGE_MODEL} (prompt {JUDGE_PROMPT_VERSION})"}
    (run_dir / "config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    summary = summarize(results, k)
    by_type, by_tier = summarize_by_type(results, k), summarize_by_tier(results, k)
    with open(run_dir / "results.jsonl", "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (run_dir / "summary.json").write_text(json.dumps({"overall": summary, "by_type": by_type, "by_tier": by_tier}, indent=2), encoding="utf-8")
    report = render_report(meta, results, summary, by_type, by_tier)
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    log_experiment(meta, summary, results_dir)
    changes = verdict_changes(old_results, {r["id"]: r for r in results})
    return {"run_dir": run_dir, "summary": summary, "results": results, "changes": changes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score a saved run's answers again with the current golden set and judge")
    parser.add_argument("--source", required=True, help="the run whose saved answers are scored again")
    parser.add_argument("--name", required=True, help="name of the new run")
    parser.add_argument("--note", default="")
    parser.add_argument("--force", action="store_true", help="discard saved judgments of the new run and judge again")
    parser.add_argument("--rejudge-all", action="store_true",
                        help="judge every question with the current judge prompt instead of keeping old verdicts of unsplit ones")
    args = parser.parse_args()
    out = rescore(args.source, args.name, args.note, force=args.force, rejudge_all=args.rejudge_all)
    old = json.loads((RESULTS_DIR / args.source / "summary.json").read_text(encoding="utf-8"))["overall"]
    new = out["summary"]
    print(f"\n{'metric':<14}{'before':>8}{'after':>8}")
    for m in ("correctness", "relevance", "completeness", "faithfulness", "abstention_accuracy"):
        print(f"{m:<14}{old[m]:>8.3f}{new[m]:>8.3f}")
    print("\nchanged verdicts:")
    for c in out["changes"]:
        print(" ", c)
    print(f"\nsaved to {out['run_dir']}")
