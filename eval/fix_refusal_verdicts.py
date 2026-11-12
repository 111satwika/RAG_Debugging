"""Correct saved verdicts judged before the judge counted the pipeline's own refusal text as a refusal. No model calls.

    python -m eval.fix_refusal_verdicts eval/results/prompt_repeats_fresh-v4_final.json --golden eval/holdout_set.json

The split judge (v3) marked "Information not found in the provided documents." as not abstained, so every correct refusal
of an unanswerable question scored 0 (n23-n25). eval/judge.py now sets abstained for that exact text, and parse_verdict then
gives a correct refusal full scores. This applies the same deterministic rule to a saved prompt_repeats file, in place
(the original is kept next to it as *.before_refusal_fix.json).
"""
import argparse
import json
import shutil
from pathlib import Path

from src.generate.answer import NOT_FOUND


def fix(runs: dict, types: dict) -> int:
    """Apply the rule to {prompt: {question id: [verdict, ...]}}. Returns how many verdicts changed."""
    changed = 0
    for by_question in runs.values():
        for qid, verdicts in by_question.items():
            for v in verdicts:
                if v["answer"].strip() != NOT_FOUND or v["abstained"] and types[qid] != "unanswerable":
                    continue
                before = dict(v)
                v["abstained"] = True
                if types[qid] == "unanswerable":
                    v.update(correctness=1.0, relevance=1.0, faithfulness=1.0, completeness=None)
                changed += v != before
    return changed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", help="a prompt_repeats_*.json file")
    parser.add_argument("--golden", required=True, help="the question file that run answered")
    args = parser.parse_args()
    path = Path(args.path)
    types = {q["id"]: q["type"] for q in json.loads(Path(args.golden).read_text(encoding="utf-8"))}
    runs = json.loads(path.read_text(encoding="utf-8"))
    backup = path.with_suffix(".before_refusal_fix.json")
    if not backup.exists():
        shutil.copy(path, backup)
    print(f"{fix(runs, types)} verdicts corrected in {path.name}")
    path.write_text(json.dumps(runs), encoding="utf-8")
