"""Phase 2 evaluation harness: retrieval metrics, generation metrics and a failure report.

    python -m eval.run_eval --name baseline --note "naive 500-token chunks"
    python -m eval.run_eval --name baseline --retrieval-only     # seconds, no LLM at all
    python -m eval.run_eval --name smoke --ids q11,q12,q28       # a few questions

Re-running the same --name resumes: saved answers and judgments are not redone.
Everything for a run is written to eval/results/<name>/.
"""
import argparse
import json
import math
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from eval.failures import classify
from eval.judge import judge_answer
from eval.metrics import (RELEVANCE_THRESHOLD, citation_check, citation_summary, context_precision, keyword_check, label_relevant,
                          retrieval_metrics)
from src.config import (COLLECTION_NAME, JUDGE_MODEL, JUDGE_WORKERS, JUDGE_PROMPT_VERSION, JUDGE_PROVIDER, TOP_K, snapshot,
                        with_legacy_defaults)
from src.generate.llm import api_key_available, load_llm
from src.ingest.index import get_client, load_parents
from src.observability.logger import get_logger
from src.observability.tracer import TRACES_PATH
from src.pipeline import run_query
from src.retrieve.dense import retrieve
from src.retrieve.retriever import configured_search

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_PATH = EVAL_DIR / "golden_set.json"
RESULTS_DIR = EVAL_DIR / "results"
DEEP_K = 20          # wider search used to tell a retrieval miss that was only ranked low from one not found at all
RETRIEVAL_KEYS = ("recall", "precision", "context_precision", "mrr", "ndcg", "hit")
GENERATION_KEYS = ("correctness", "relevance", "completeness", "faithfulness")

log = get_logger("eval")


def read_jsonl(path: Path) -> dict:
    """Records keyed by id; a later line replaces an earlier one."""
    records = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                records[rec["id"]] = rec
    return records


def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_index_chunks(collection_name: str = COLLECTION_NAME) -> list[dict]:
    """The chunks the model reads. With parent-child retrieval those are the parents, not the indexed children."""
    collection = get_client().get_collection(collection_name)
    if collection.metadata.get("parent_child"):
        return [{"chunk_id": p["chunk_id"], "doc_id": p["doc_id"], "text": p["text"], "header": p.get("header", "")}
                for p in load_parents(collection_name).values()]
    data = collection.get(include=["documents", "metadatas"])
    return [{"chunk_id": i, "doc_id": m["doc_id"], "text": t, "header": m.get("header", "")}
            for i, t, m in zip(data["ids"], data["documents"], data["metadatas"])]


def mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def percentile(values: list[float], pct: float) -> float | None:
    values = sorted(v for v in values if v is not None)
    return values[max(0, math.ceil(pct / 100 * len(values)) - 1)] if values else None


# ---------------------------------------------------------------- stages

def generate_answers(questions: list[dict], run_dir: Path, llm, k: int, trace_path=TRACES_PATH,
                     search_fn=None, config: dict | None = None, prompt_version: str | None = None) -> None:
    path = run_dir / "answers.jsonl"
    done = read_jsonl(path)
    for i, q in enumerate(questions, start=1):
        if q["id"] in done and not done[q["id"]].get("error"):
            continue
        try:
            out = run_query(q["question"], k, llm, trace_path=trace_path, search_fn=search_fn, config=config,
                            **({"version": prompt_version} if prompt_version else {}))
            spans = out["trace"]["spans"]
            llm_span = spans.get("llm_call", {})            # absent when the validation layer blocked the query
            structured = "answer_status" in out          # a cited prompt: the answer came as JSON with sources
            rec = {
                "id": q["id"], "trace_id": out["trace_id"], "answer": out["answer"], "error": None,
                "chunks": [{"chunk_id": c["chunk_id"], "text": c["text"], **({"source_id": f"S{n}"} if structured else {})}
                           for n, c in enumerate(out["chunks"], start=1)],
                "total_ms": out["trace"]["total_ms"], "llm_ms": llm_span.get("ms", 0),
                "input_tokens": llm_span.get("input_tokens", 0),
                "output_tokens": llm_span.get("output_tokens", 0),
            }
            if "validation" in out:
                v = out["validation"]
                rec["validation"] = {"grade": v["grade"], "first_grade": v["first_grade"], "action": v["action"],
                                     "best_score": v["best_score"], "reason": v["reason"],
                                     "retried": any("search_ms" in r for r in v["retries"])}
            if structured:
                rec.update(status=out["answer_status"], found=out["found"], cited=[s["chunk_id"] for s in out["sources"]],
                           issues=llm_span.get("structured", {}).get("issues", []))
        except Exception as exc:  # one bad question must not stop the run
            log.exception("question %s failed", q["id"])
            rec = {"id": q["id"], "trace_id": None, "answer": None, "chunks": [],
                   "error": f"{type(exc).__name__}: {exc}"}
        append_jsonl(path, rec)
        print(f"[{i}/{len(questions)}] {q['id']} answered in {(rec.get('total_ms') or 0) / 1000:.0f}s"
              + (f"  ERROR {rec['error']}" if rec["error"] else ""), flush=True)


def judge_answers(questions: list[dict], answers: dict, run_dir: Path, judge) -> None:
    path = run_dir / "judgments.jsonl"
    done = read_jsonl(path)
    todo = [q for q in questions if answers.get(q["id"], {}).get("answer") is not None and q["id"] not in done]
    lock, finished = threading.Lock(), [0]

    def judge_one(q: dict) -> None:
        a = answers[q["id"]]
        try:
            verdict = judge_answer(judge, q, a["answer"], a["chunks"], found=a.get("found"))
        except Exception as exc:
            log.exception("judging %s failed", q["id"])
            print(f"judge failed on {q['id']}: {exc}", flush=True)
            return
        with lock:                                  # one writer at a time; finished verdicts are kept if the run stops
            append_jsonl(path, {"id": q["id"], **verdict})
            finished[0] += 1
            print(f"[judge {finished[0]}/{len(todo)}] {q['id']}", flush=True)

    # Each answer takes JUDGE_VOTES x 2-3 judge calls one after another; answers are judged in parallel (JUDGE_WORKERS)
    with ThreadPoolExecutor(max(1, JUDGE_WORKERS)) as pool:
        list(pool.map(judge_one, todo))


# ---------------------------------------------------------------- results

def is_failure(res: dict) -> bool:
    if res["error"]:
        return True
    judge, keyword = res["judge"], res["keyword"]
    if judge:
        if judge["abstained"] != (res["type"] == "unanswerable"):
            return True
        if res["type"] == "unanswerable":
            return judge["faithfulness"] < 1
        completeness = judge["completeness"]
        return judge["correctness"] < 1 or judge["faithfulness"] < 1 or (completeness is not None and completeness < 1)
    if keyword:
        return ((keyword["include_rate"] is not None and keyword["include_rate"] < 1) or bool(keyword["forbidden"])
                or not keyword.get("verdict_ok", True))
    return False


def _with_given_chunks(metrics: dict | None, answer: dict, relevant: set[str]) -> dict | None:
    """Context precision is about what the model read: the chunks of the answer (after the validation's corrective search
    and v10's refinement), not the search results (`ranked`), which can hold more (v10: 3.4 searched, 2.1 read)."""
    given = [c["chunk_id"] for c in answer.get("chunks") or []]
    if metrics is not None and given:
        metrics["context_precision"] = context_precision(given, relevant)
    return metrics


def build_results(questions, hits, labels, answers, judgments, k) -> list[dict]:
    results = []
    for q in questions:
        a = answers.get(q["id"], {})
        ranked = [h["chunk_id"] for h in hits[q["id"]]]
        label = labels[q["id"]]
        results.append({
            "id": q["id"], "type": q["type"], "tier": q.get("tier", "standard"), "question": q["question"],
            "expected": q["expected_answer"],
            "answer": a.get("answer"), "error": a.get("error"), "trace_id": a.get("trace_id"),
            "retrieved": hits[q["id"]], "relevant_ids": label["ids"], "weak_labels": label["weak"],
            "retrieval": _with_given_chunks(retrieval_metrics(ranked, set(label["ids"]), k), a, set(label["ids"])),
            "keyword": keyword_check(a.get("answer"), q, a.get("found")),
            "judge": judgments.get(q["id"]),
            "citation": citation_check(a, set(label["ids"])),
            "total_ms": a.get("total_ms"), "llm_ms": a.get("llm_ms"),
            "input_tokens": a.get("input_tokens"), "output_tokens": a.get("output_tokens"),
            "failure": None,
        })
    return results


def attach_failures(questions: list[dict], results: list[dict], chunks: list[dict], search=retrieve) -> None:
    by_id = {q["id"]: q for q in questions}
    for res in results:
        if not is_failure(res):
            continue
        if res["error"]:
            res["failure"] = {"stage": "generation", "note": res["error"]}
            continue
        q = by_id[res["id"]]
        deep_ids = [h["chunk_id"] for h in search(q["question"], DEEP_K)]
        stage, note = classify(q, res, chunks, deep_ids)
        res["failure"] = {"stage": stage, "note": note}


def summarize(results: list[dict], k: int) -> dict:
    judged = [r for r in results if r["judge"]]
    s = {"n": len(results), "k": k, "n_judged": len(judged),
         "n_retrieval": sum(1 for r in results if r["retrieval"]),
         "n_failed": sum(1 for r in results if r["failure"])}
    for m in RETRIEVAL_KEYS:
        s[m] = mean(r["retrieval"].get(m) for r in results if r["retrieval"])      # runs made before a metric existed lack it
    for m in GENERATION_KEYS:
        s[m] = mean(r["judge"][m] for r in judged)
    s["keyword_include"] = mean(r["keyword"]["include_rate"] for r in results if r["keyword"])
    s["abstention_accuracy"] = mean(float(r["judge"]["abstained"] == (r["type"] == "unanswerable")) for r in judged)
    s["refusal_rate_unanswerable"] = mean(float(r["judge"]["abstained"]) for r in judged if r["type"] == "unanswerable")
    s["false_abstention_rate"] = mean(float(r["judge"]["abstained"]) for r in judged if r["type"] != "unanswerable")
    latencies = [r["total_ms"] for r in results if r["total_ms"]]
    s["latency_p50_s"] = percentile(latencies, 50) / 1000 if latencies else None
    s["latency_p95_s"] = percentile(latencies, 95) / 1000 if latencies else None
    s["avg_input_tokens"] = mean(r["input_tokens"] for r in results)
    s["avg_output_tokens"] = mean(r["output_tokens"] for r in results)
    s.update(citation_summary(results))
    return s


def summarize_by_type(results: list[dict], k: int) -> dict:
    types = sorted({r["type"] for r in results})
    return {t: summarize([r for r in results if r["type"] == t], k) for t in types}


def summarize_by_tier(results: list[dict], k: int) -> dict:
    """Standard against hard questions (a question with no `tier` is standard). Empty if there is only one tier."""
    tiers = sorted({r["tier"] for r in results}, key=lambda t: (t != "standard", t))
    return {t: summarize([r for r in results if r["tier"] == t], k) for t in tiers} if len(tiers) > 1 else {}


# ---------------------------------------------------------------- report

def f(value, digits=2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def judge_vs_keyword(results: list[dict]) -> tuple[float | None, list[str]]:
    """How often the cheap keyword check and the LLM judge agree on 'fully correct and complete'."""
    pairs = [(r["id"], r["keyword"], r["judge"]) for r in results if r["keyword"] and r["judge"]]
    disagree = []
    for rid, kw, jd in pairs:
        keyword_ok = (kw["include_rate"] in (None, 1.0)) and not kw["forbidden"] and kw.get("verdict_ok", True)
        judge_ok = jd["correctness"] == 1.0 and jd["completeness"] in (None, 1.0)
        if keyword_ok != judge_ok:
            disagree.append(rid)
    return ((len(pairs) - len(disagree)) / len(pairs) if pairs else None), disagree


def render_report(meta: dict, results: list[dict], summary: dict, by_type: dict, by_tier: dict | None = None) -> str:
    k = summary["k"]
    s = summary
    lines = [
        f"# Eval report: {meta['name']}",
        f"- date: {meta['created']} | questions: {s['n']} | note: {meta['note'] or '-'}",
        f"- config: " + ", ".join(f"{a}={b}" for a, b in meta["config"].items()),
        f"- judge: {meta['judge'] if s['n_judged'] else 'not judged yet (set OPENAI_API_KEY in .env and re-run this command)'}",
        "", "## Retrieval metrics (answerable questions, chunk level)",
        f"Computed on {s['n_retrieval']} questions that have labelled relevant chunks.", "",
        "| Metric | Value |", "|---|---|",
        f"| Recall@{k} | {f(s['recall'])} |", f"| Precision@{k} | {f(s['precision'])} |",
        f"| Context precision (relevant / chunks given to the model) | {f(s.get('context_precision'))} |",
        f"| MRR | {f(s['mrr'])} |", f"| nDCG@{k} | {f(s['ndcg'])} |", f"| Hit@{k} | {f(s['hit'])} |",
        "", f"## Generation metrics (judged: {s['n_judged']} of {s['n']})", "",
        "| Metric | Value |", "|---|---|",
        f"| Correctness | {f(s['correctness'])} |", f"| Relevance | {f(s['relevance'])} |",
        f"| Completeness | {f(s['completeness'])} |", f"| Faithfulness | {f(s['faithfulness'])} |",
        f"| Abstention accuracy | {f(s['abstention_accuracy'])} |",
        f"| Refusal rate on unanswerable | {f(s['refusal_rate_unanswerable'])} |",
        f"| False abstention rate (answerable) | {f(s['false_abstention_rate'])} |",
        f"| Keyword check (share of key facts in answer) | {f(s['keyword_include'])} |",
        "", "## Cost and speed", "",
        f"- latency p50 {f(s['latency_p50_s'], 1)} s, p95 {f(s['latency_p95_s'], 1)} s per query",
        f"- tokens per query: {f(s['avg_input_tokens'], 0)} in, {f(s['avg_output_tokens'], 0)} out",
        "", "## By question type", "",
        f"| Type | n | Recall@{k} | Prec@{k} | MRR | nDCG@{k} | Correct | Relevant | Complete | Faithful | Abstain acc |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for t, ts in by_type.items():
        lines.append(f"| {t} | {ts['n']} | {f(ts['recall'])} | {f(ts['precision'])} | {f(ts['mrr'])} | {f(ts['ndcg'])} | "
                     f"{f(ts['correctness'])} | {f(ts['relevance'])} | {f(ts['completeness'])} | "
                     f"{f(ts['faithfulness'])} | {f(ts['abstention_accuracy'])} |")
    if by_tier:
        lines += ["", "## By tier (standard = the original questions, hard = the added harder ones)", "",
                  f"| Tier | n | Recall@{k} | Prec@{k} | MRR | nDCG@{k} | Correct | Complete | Faithful | Abstain acc |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for t, ts in by_tier.items():
            lines.append(f"| {t} | {ts['n']} | {f(ts['recall'])} | {f(ts['precision'])} | {f(ts['mrr'])} | {f(ts['ndcg'])} | "
                         f"{f(ts['correctness'])} | {f(ts['completeness'])} | {f(ts['faithfulness'])} | "
                         f"{f(ts['abstention_accuracy'])} |")

    failures = [r for r in results if r["failure"]]
    lines += ["", f"## Failure report ({len(failures)} of {s['n']} questions)", ""]
    if failures:
        stages = {}
        for r in failures:
            stages[r["failure"]["stage"]] = stages.get(r["failure"]["stage"], 0) + 1
        lines.append("By stage: " + ", ".join(f"{a}={b}" for a, b in sorted(stages.items())))
        lines += ["", "| Question | Type | Stage | Why | Trace |", "|---|---|---|---|---|"]
        for r in failures:
            note = r["failure"]["note"].replace("|", "/")
            lines.append(f"| {r['id']} | {r['type']} | {r['failure']['stage']} | {note} | {r['trace_id'] or '-'} |")
        lines.append("\nOpen a trace with: `python -m scripts.show_trace <trace_id>`. Stages are heuristics; confirm in the trace.")
    else:
        lines.append("None found" + ("" if s["n_judged"] else " (without the judge only the keyword check can detect failures)"))

    agreement, disagree = judge_vs_keyword(results)
    lines += ["", "## Judge check", ""]
    if agreement is None:
        lines.append("Needs judged answers.")
    else:
        lines.append(f"The keyword check and the judge agree on {agreement:.0%} of answerable questions. "
                     f"Disagreements to review by hand: {', '.join(disagree) or 'none'}.")

    lines += ["", "## Relevance labels (review these)", "",
              "A chunk is relevant if it comes from a source document and holds at least "
              f"{RELEVANCE_THRESHOLD:.0%} of the question's must_include facts. 'weak' means no chunk met that, "
              "so the best-covering chunks were used.", "", "| Question | Relevant chunks | Weak | In top-k |", "|---|---|---|---|"]
    for r in results:
        if r["type"] == "unanswerable":
            continue
        retrieved = {h["chunk_id"] for h in r["retrieved"]}
        found = len(set(r["relevant_ids"]) & retrieved)
        lines.append(f"| {r['id']} | {len(r['relevant_ids'])} | {'yes' if r['weak_labels'] else ''} | {found} |")
    return "\n".join(lines) + "\n"


def log_experiment(meta: dict, summary: dict, results_dir: Path) -> None:
    path = results_dir / "experiment_log.md"
    k = summary["k"]
    if not path.exists():
        path.write_text(
            f"# Experiment log\n\n| Run | Change | Recall@{k} | Precision@{k} | MRR | nDCG@{k} | Correctness | Relevance | "
            "Completeness | Faithfulness | Abstention acc. | p95 latency (s) | Keep? |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n", encoding="utf-8")
    row = [meta["name"], meta["note"] or "-", f(summary["recall"]), f(summary["precision"]), f(summary["mrr"]),
           f(summary["ndcg"]), f(summary["correctness"]), f(summary["relevance"]), f(summary["completeness"]),
           f(summary["faithfulness"]), f(summary["abstention_accuracy"]), f(summary["latency_p95_s"], 1), ""]
    # Re-running the same name (resume) replaces its row instead of adding a duplicate.
    kept = [line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith(f"| {meta['name']} |")]
    kept.append("| " + " | ".join(row) + " |")
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- main

def run_evaluation(name: str, note: str = "", k: int = TOP_K, ids: list[str] | None = None,
                   limit: int | None = None, retrieval_only: bool = False, use_judge: bool = True,
                   force: bool = False, llm=None, judge_llm=None, results_dir: Path = RESULTS_DIR,
                   golden_path: Path = GOLDEN_PATH, trace_path=TRACES_PATH,
                   chunks: list[dict] | None = None, searcher=None, config: dict | None = None,
                   prompt_version: str | None = None) -> dict:
    """Run the golden set. `chunks`, `searcher` and `config` let an experiment evaluate an index other than
    the main one (see eval/chunking_experiments.py); by default the main index and settings are used.
    `prompt_version` answers with another prompt than the active version's (see eval/prompt_experiments.py)."""
    questions = json.loads(golden_path.read_text(encoding="utf-8"))
    if ids:
        questions = [q for q in questions if q["id"] in ids]
    if limit:
        questions = questions[:limit]
    if not questions:
        raise ValueError("no questions selected")

    config = config or {**snapshot(), "top_k": k}
    search = searcher or configured_search() or retrieve       # the active version's retrieval unless an experiment gives its own
    run_dir = results_dir / name
    if force and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.json"
    previous = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else None
    if previous and with_legacy_defaults(previous["config"]) != config:
        raise RuntimeError(f"Run '{name}' was made with different settings. Use a new --name, or --force to start over.")
    judge_name = f"{JUDGE_PROVIDER}:{JUDGE_MODEL} (prompt {JUDGE_PROMPT_VERSION})"
    meta = {"name": name, "note": note or (previous or {}).get("note", ""),
            "created": (previous or {}).get("created") or datetime.now().strftime("%Y-%m-%d %H:%M"),
            "k": k, "config": config, "judge": judge_name, "relevance_threshold": RELEVANCE_THRESHOLD}
    config_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    chunks = chunks if chunks is not None else load_index_chunks()
    labels = {q["id"]: label_relevant(q, chunks) for q in questions}
    (run_dir / "relevance_labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    hits = {q["id"]: [{key: h[key] for key in ("chunk_id", "doc_id", "page", "score")}
                      for h in search(q["question"], k)] for q in questions}

    if not retrieval_only:
        generate_answers(questions, run_dir, llm or load_llm(), k, trace_path,
                         search_fn=searcher, config=config if searcher else None, prompt_version=prompt_version)
    answers = read_jsonl(run_dir / "answers.jsonl")

    if use_judge and any(a.get("answer") is not None for a in answers.values()):
        if judge_llm is None and api_key_available(JUDGE_PROVIDER):
            judge_llm = load_llm(JUDGE_PROVIDER, JUDGE_MODEL)
        if judge_llm is not None:
            judge_answers(questions, answers, run_dir, judge_llm)
        else:
            print(f"No {JUDGE_PROVIDER} key found: skipping the judge. Add OPENAI_API_KEY to .env and re-run.", flush=True)
    judgments = read_jsonl(run_dir / "judgments.jsonl")

    results = build_results(questions, hits, labels, answers, judgments, k)
    attach_failures(questions, results, chunks, search)
    summary = summarize(results, k)
    by_type = summarize_by_type(results, k)
    by_tier = summarize_by_tier(results, k)

    with open(run_dir / "results.jsonl", "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (run_dir / "summary.json").write_text(json.dumps({"overall": summary, "by_type": by_type, "by_tier": by_tier}, indent=2), encoding="utf-8")
    report = render_report(meta, results, summary, by_type, by_tier)
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    log_experiment(meta, summary, results_dir)
    return {"run_dir": run_dir, "summary": summary, "by_type": by_type, "by_tier": by_tier, "results": results, "report": report}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the golden-set evaluation")
    parser.add_argument("--name", default=datetime.now().strftime("run_%Y%m%d_%H%M"))
    parser.add_argument("--note", default="", help="what changed in this run (goes in the experiment log)")
    parser.add_argument("-k", type=int, default=TOP_K)
    parser.add_argument("--ids", help="comma-separated golden ids, e.g. q11,q12")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retrieval-only", action="store_true", help="skip answer generation and judging")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--force", action="store_true", help="delete the run folder and start over")
    parser.add_argument("--golden", default=str(GOLDEN_PATH), help="the question file (default: eval/golden_set.json), e.g. eval/holdout_set.json")
    args = parser.parse_args()

    out = run_evaluation(args.name, args.note, args.k, args.ids.split(",") if args.ids else None, args.limit,
                         args.retrieval_only, not args.no_judge, args.force, golden_path=Path(args.golden))
    print("\n" + out["report"])
    print(f"saved to {out['run_dir']}")
