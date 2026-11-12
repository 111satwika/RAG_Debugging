"""Check the judge on saved answers whose right scores are known, before trusting it on a full run.

    python -m eval.judge_calibration

Each case is a saved answer and the correctness a careful reader gives it. The chunks are read again from the current index
(same chunk ids), so a fix to the documents (such as the repaired Table 2) is part of the check. About 66 judge calls.
"""
import json
from pathlib import Path

from eval.judge import judge_answer
from eval.run_eval import GOLDEN_PATH, RESULTS_DIR, read_jsonl
from src.config import JUDGE_MODEL, JUDGE_PROMPT_VERSION, JUDGE_PROVIDER
from src.generate.llm import load_llm
from src.ingest.index import get_client

HOLDOUT, HOLDOUT_V1 = GOLDEN_PATH.parent / "holdout_set.json", GOLDEN_PATH.parent / "holdout_set_v1.json"

# (question file, saved run, question id, prompt in a prompt_repeats file or None for the run's own answer, right correctness,
#  right faithfulness or None when it is debatable, why)
CASES = [
    (GOLDEN_PATH, "v6", "q19", None, 1.0, 1.0, "41.8, the Table 2 value"),
    (HOLDOUT_V1, "holdout18-v5", "h08", None, 1.0, 1.0, "uses 41.0, which the reference accepts"),
    (HOLDOUT_V1, "holdout-v6", "h03", None, 0.0, None, "says No; the answer is Yes"),
    (HOLDOUT_V1, "holdout-v6", "h17", None, 0.0, 1.0, "'not found' where the answer is 'not mentioned'"),
    (HOLDOUT_V1, "holdout-v6", "h26", None, 1.0, 1.0, "gives translation, misses constituency parsing: right but incomplete"),
    (GOLDEN_PATH, "v6", "q01", None, 1.0, 1.0, "plain correct answer"),
    (HOLDOUT, "fresh-v4", "n02", "v2", 1.0, 1.0, "correct, with a true extra detail"),
    (HOLDOUT, "fresh-v4", "n16", "v2.9", 0.5, 0.5, "both projects, plus a wrong third one"),
    (HOLDOUT, "fresh-v4", "n23", None, 1.0, 1.0, "correct refusal of an unanswerable question"),
    (GOLDEN_PATH, "v6", "q34", "v2.9", 1.0, 1.0, "36 million vs 4.5 million, same facts as the reference"),
    (GOLDEN_PATH, "v6", "q40", "v2", 0.0, 1.0, "says Yes, but its sentence is copied from the table"),
    # v7 answers that are correct but whose claims are read from a table or rephrased (no verbatim quote)
    (GOLDEN_PATH, "v7", "q08", None, 1.0, 1.0, "18 days, carried forward up to 30 (table)"),
    (GOLDEN_PATH, "v7", "q16", None, 1.0, 1.0, "the five base-model dimensions, from two sections"),
    (GOLDEN_PATH, "v7", "q38", None, 1.0, 1.0, "the four leave types with 'No' under Carry Forward"),
    (GOLDEN_PATH, "v7", "q44", None, 1.0, 1.0, "faster at the median, slower at p95 (a comparison)"),
    (GOLDEN_PATH, "v7", "q29", None, 0.0, None, "names DevPilot AI; the project is IntelliFusion"),
    # v2.10 answer: names one of the two projects the reference accepts ("naming either project is fully correct")
    (GOLDEN_PATH, "v7", "q25", "v2.10", 1.0, 1.0, "Yes, names AI Visual Placemat (either project is accepted)"),
]


def load_case(questions: Path, run: str, qid: str, prompt: str | None) -> tuple[dict, str, bool | None, list[dict]]:
    question = {q["id"]: q for q in json.loads(questions.read_text(encoding="utf-8"))}[qid]
    saved = read_jsonl(RESULTS_DIR / run / "answers.jsonl")[qid]
    answer, found = saved["answer"], saved.get("found")
    if prompt:
        files = sorted(RESULTS_DIR.glob(f"prompt_repeats_{run}_*.json"))        # the comparison that answered with `prompt`
        runs = next(r for r in (json.loads(f.read_text(encoding="utf-8")) for f in files) if prompt in r)
        repeat = runs[prompt][qid][0]
        answer, found = repeat["answer"], repeat["found"]
    current = dict(zip(*[get_client().get_collection("rag_chunks_v2").get(ids=[c["chunk_id"] for c in saved["chunks"]],
                                                                       include=["documents"])[k] for k in ("ids", "documents")]))
    chunks = [{**c, "text": current.get(c["chunk_id"], c["text"])} for c in saved["chunks"]]
    return question, answer, found, chunks


def main() -> None:
    judge = load_llm(JUDGE_PROVIDER, JUDGE_MODEL)
    right = right_faith = 0
    print(f"judge {JUDGE_MODEL}, prompt {JUDGE_PROMPT_VERSION}\n")
    print("| Case | Expected | Judge correctness | Votes | OK | Completeness | Expected faithful | Judge faithful | OK | What the answer does |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for questions, run, qid, prompt, expected, faithful, why in CASES:
        question, answer, found, chunks = load_case(questions, run, qid, prompt)
        v = judge_answer(judge, question, answer, chunks, found=found)
        ok = v["correctness"] == expected
        faith_ok = faithful is None or v["faithfulness"] == faithful
        right += ok
        right_faith += faith_ok
        votes = [x["correctness"] for x in v.get("votes", [v])]
        print(f"| {qid}{' ' + prompt if prompt else ''} | {expected} | {v['correctness']} | {votes} | {'yes' if ok else 'NO'} | "
              f"{v['completeness']} | "
              f"{'-' if faithful is None else faithful} | {v['faithfulness']} | {'yes' if faith_ok else 'NO'} | {why} |")
    print(f"\n{right} of {len(CASES)} correctness scores match, {right_faith} of {len(CASES)} faithfulness scores")


if __name__ == "__main__":
    main()
