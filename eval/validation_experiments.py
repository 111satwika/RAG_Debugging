"""Phase 6: the CRAG-style relevance grade and the corrective search, measured on the golden set and extra traps.

    python -m eval.validation_experiments            # run the corrective search for every weakly scored question, then sweep
    python -m eval.validation_experiments --reuse    # report again from the saved records (no model calls)

The 50 golden questions replay the chunks the saved `v3` run retrieved. The 7 extra traps of the Phase 5 drill (questions the
documents cannot answer) are retrieved fresh with the active version. Every question that scores below the highest cutoff tried
gets the real corrective search once (the rewrite and a fresh reranked search), so the cutoffs can be swept offline: for a pair
of cutoffs a question is graded from its best reranker score, a question that is not "correct" takes the better of its first and
corrected scores (the merged chunks keep the best of both), and it is blocked when it is still "incorrect".

A block is a *false abstention* when it stops an answerable question that the saved `v3` run answered fully correctly
(correctness 1.0). Written to eval/results/validation_experiments.md and eval/results/validation_records.json.
"""
import argparse
import json
import time
from pathlib import Path

from eval.prompt_drill import EXTRA_TRAPS
from eval.prompt_experiments import saved_retrieval
from eval.run_eval import GOLDEN_PATH, RESULTS_DIR
from src.config import TOP_K
from src.generate.llm import load_llm
from src.retrieve.rewrite import rewrite_query
from src.validate.gate import is_new_query
from src.validate.threshold import best_score, grade_context

RECORDS_PATH = RESULTS_DIR / "validation_records.json"
REPORT_PATH = RESULTS_DIR / "validation_experiments.md"
SOURCE_RUN = "v3"
HIGHS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
LOWS = (0.0, 0.0005, 0.005)


def collect(llm, search, results_dir: Path = RESULTS_DIR, golden_path: Path = GOLDEN_PATH, source_run: str = SOURCE_RUN,
            k: int = TOP_K, retry_below: float = max(HIGHS)) -> list[dict]:
    """One record per question: best score, answerable or not, how the saved run did, and the corrective search when the
    score is under `retry_below` (the rewritten query, its best score, whether it found a labelled evidence chunk)."""
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    saved = saved_retrieval(source_run, results_dir)
    results = {json.loads(l)["id"]: json.loads(l)
               for l in (results_dir / source_run / "results.jsonl").read_text(encoding="utf-8").splitlines() if l}
    labels = json.loads((results_dir / source_run / "relevance_labels.json").read_text(encoding="utf-8"))
    items = [(q["id"], q["question"], q["type"] != "unanswerable", saved[q["question"]],
              (results[q["id"]]["judge"] or {}).get("correctness")) for q in golden]
    items += [(f"x{i:02d}", question, False, search(question, k), None) for i, (question, _) in enumerate(EXTRA_TRAPS, start=1)]
    records = []
    for qid, question, answerable, hits, correctness in items:
        relevant = set(labels.get(qid, {}).get("ids", []))
        rec = {"id": qid, "question": question, "answerable": answerable, "top": best_score(hits),
               "chunks": [h["chunk_id"] for h in hits], "v3_correctness": correctness,
               "first_had_relevant": bool(relevant & {h["chunk_id"] for h in hits}) if relevant else None}
        if rec["top"] < retry_below:
            start = time.perf_counter()
            query = rewrite_query(question, llm)
            rewrite_ms = round((time.perf_counter() - start) * 1000)
            if not is_new_query(query, question):
                rec["retry"] = {"query": query, "skipped": True, "rewrite_ms": rewrite_ms}
            else:
                start = time.perf_counter()
                extra = search(query, k)
                rec["retry"] = {"query": query, "rewrite_ms": rewrite_ms, "search_ms": round((time.perf_counter() - start) * 1000),
                                "top": best_score(extra), "chunks": [h["chunk_id"] for h in extra],
                                "has_relevant": bool(relevant & {h["chunk_id"] for h in extra}) if relevant else None}
        records.append(rec)
        print(f"{qid} top={rec['top']:.3f}" + (f" -> retry top={rec['retry']['top']:.3f}" if rec.get("retry") and "top" in rec["retry"] else ""),
              flush=True)
    return records


def simulate(records: list[dict], high: float, low: float, retry: bool = True) -> dict:
    """What a pair of cutoffs would have done. Without `retry`, a question is blocked when its first score is under `low`."""
    out = {"high": high, "low": low, "retried": 0, "retry_s": 0.0, "rescued": 0, "answerable": 0, "answerable_blocked": 0,
           "false_abstentions": 0, "unanswerable": 0, "unanswerable_blocked": 0}
    for r in records:
        top = r["top"]
        first = grade_context([{"score": top}], high, low)
        again = r.get("retry")
        if retry and first != "correct" and again:
            out["retried"] += 1
            out["retry_s"] += ((again.get("rewrite_ms") or 0) + (again.get("search_ms") or 0)) / 1000
            if "top" in again:
                top = max(top, again["top"])           # the merged chunks keep the best of both searches
        final = grade_context([{"score": top}], high, low)
        blocked = final == "incorrect"
        out["rescued"] += first == "incorrect" and not blocked
        if r["answerable"]:
            out["answerable"] += 1
            out["answerable_blocked"] += blocked
            out["false_abstentions"] += blocked and r["v3_correctness"] == 1.0
        else:
            out["unanswerable"] += 1
            out["unanswerable_blocked"] += blocked
    return out


def retry_summary(records: list[dict]) -> dict:
    """Did the corrective search help where the evidence was missing, and did it change the best score?"""
    retried = [r for r in records if r.get("retry") and "top" in r["retry"]]
    answerable = [r for r in retried if r["answerable"]]
    lost = [r for r in answerable if r["first_had_relevant"] is False]
    return {
        "retried": len(retried), "skipped": sum(1 for r in records if r.get("retry", {}).get("skipped")),
        "found_missing_evidence": [r["id"] for r in lost if r["retry"]["has_relevant"]],
        "first_search_missed_evidence": [r["id"] for r in lost],
        "lost_evidence": [r["id"] for r in answerable if r["first_had_relevant"] and r["retry"]["has_relevant"] is False],
        "score_raised": sum(r["retry"]["top"] > r["top"] for r in retried),
        "median_retry_s": sorted((r["retry"]["rewrite_ms"] + r["retry"]["search_ms"]) / 1000 for r in retried)[len(retried) // 2] if retried else None,
    }


def render_report(records: list[dict]) -> str:
    s = retry_summary(records)
    answerable = sorted(r["top"] for r in records if r["answerable"])
    unanswerable = sorted(r["top"] for r in records if not r["answerable"])
    lines = [
        "# Phase 6: CRAG-style relevance grade + corrective search", "",
        f"{len(records)} questions: {len(answerable)} answerable, {len(unanswerable)} unanswerable (6 golden + 7 extra traps). "
        f"Chunks: the saved `{SOURCE_RUN}` run (golden) or the active retrieval (traps). The corrective search is the real one: "
        "a gpt-4o-mini rewrite and a fresh reranked search; the merged chunks keep the best score of both.", "",
        "## Best reranker score", "", "| Group | n | min | median | max |", "|---|---|---|---|---|",
        f"| answerable | {len(answerable)} | {answerable[0]:.3f} | {answerable[len(answerable) // 2]:.3f} | {answerable[-1]:.3f} |",
        f"| unanswerable | {len(unanswerable)} | {unanswerable[0]:.3f} | {unanswerable[len(unanswerable) // 2]:.3f} | {unanswerable[-1]:.3f} |",
        "", "## What the corrective search did", "",
        f"- searched again: {s['retried']} questions (rewrite repeated the question, so skipped: {s['skipped']}); "
        f"median extra time {s['median_retry_s']:.1f} s per retried question" if s["median_retry_s"] is not None else "- no corrective search was run",
        f"- answerable questions whose first search missed the labelled evidence chunk: {', '.join(s['first_search_missed_evidence']) or 'none'}; "
        f"the corrective search found it for: {', '.join(s['found_missing_evidence']) or 'none'}",
        f"- answerable questions where the corrective search lost an evidence chunk the first search had: {', '.join(s['lost_evidence']) or 'none'} "
        "(the merge keeps the first search's chunks, so this is about the retry's own list)",
        f"- the best score went up for {s['score_raised']} of the {s['retried']} searched questions", "",
        "## What each pair of cutoffs would have done", "",
        "`retried` = questions that get a corrective search; `extra s/query` = the average added time over all 57 questions; "
        "`rescued` = first graded incorrect, not blocked after the search.", "",
        "| high | low | retried | extra s/query | rescued | answerable blocked | false abstentions | unanswerable blocked |",
        "|---|---|---|---|---|---|---|---|"]
    for high in HIGHS:
        for low in LOWS:
            for use in (False, True):
                t = simulate(records, high, low, retry=use)
                lines.append(f"| {high} | {low} | {t['retried'] if use else '-'} (search {'on' if use else 'off'}) | "
                             f"{t['retry_s'] / len(records):.1f} | {t['rescued'] if use else '-'} | {t['answerable_blocked']} | "
                             f"{t['false_abstentions']} | {t['unanswerable_blocked']} of {t['unanswerable']} |")
    lines += ["", "False abstention = an answerable question blocked that the saved run had answered with correctness 1.0.", "",
              "## Questions that were searched again", "",
              "| Question | Answerable | First score | New query | New score | Evidence in first / new search |", "|---|---|---|---|---|---|"]
    for r in records:
        t = r.get("retry")
        if not t:
            continue
        ev = "-" if r["first_had_relevant"] is None else f"{'yes' if r['first_had_relevant'] else 'no'} / {'yes' if t.get('has_relevant') else 'no'}"
        lines.append(f"| {r['id']} | {'yes' if r['answerable'] else 'no'} | {r['top']:.3f} | "
                     + (f"(rewrite repeated the question) | - | {ev} |" if t.get("skipped") else f"{t['query']} | {t['top']:.3f} | {ev} |"))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    from src.retrieve.retriever import configured_search

    parser = argparse.ArgumentParser(description="Phase 6: relevance grade and corrective search")
    parser.add_argument("--reuse", action="store_true", help="skip the model calls and report from the saved records")
    args = parser.parse_args()
    if args.reuse:
        records = json.loads(RECORDS_PATH.read_text(encoding="utf-8"))
    else:
        search = configured_search()
        if search is None:
            raise SystemExit("this experiment needs the reranked retrieval (a version with RERANK_TOP > 0)")
        records = collect(load_llm(), search)
        RECORDS_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")
    report = render_report(records)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
