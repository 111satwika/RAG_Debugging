"""Phase 5 final check: answer prompt v2 against v2.9, on the golden set and the fresh held-out set (n01-n25).

    python -m eval.phase5_final

1. Answers the held-out set once with the v4 pipeline (prompt v2) to get the chunks each question retrieves (run
   `fresh-v4`; skipped when it is already complete).
2. Answers every question with both prompts on those exact chunks: 5 repeats on the held-out set (the set that decides),
   3 on the golden set (the regression check, on the chunks of the `v6` run). Retrieval is the same for v4 to v7.
3. Prints both sets side by side and which prompt to use. Nothing in src/config.py is changed.

The held-out questions must never be used to write or tune a prompt: they only decide.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

from eval.prompt_repeats import score
from eval.run_eval import GOLDEN_PATH, RESULTS_DIR, read_jsonl

HOLDOUT_PATH = GOLDEN_PATH.parent / "holdout_set.json"
HOLDOUT_SOURCE, GOLDEN_SOURCE = "fresh-v4", "v6"
BASE, CANDIDATE = "v2", "v2.9"
TAG = "final"
MARGIN = 0.02        # about one question in 50: smaller differences are noise


def ids_of(path: Path) -> set[str]:
    return {q["id"] for q in json.loads(path.read_text(encoding="utf-8"))}


def check_source(source: str, questions: Path) -> bool:
    """True when `source` answered exactly the questions in `questions` (so an old question file cannot slip in)."""
    answers = read_jsonl(RESULTS_DIR / source / "answers.jsonl")
    if not answers:
        return False
    if set(answers) != ids_of(questions):
        sys.exit(f"run '{source}' answered other questions than {questions.name}: delete eval/results/{source} and run again")
    return all(a.get("chunks") for a in answers.values())


def run(*args: str, env: dict | None = None) -> None:
    print("\n$", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-m", *args], check=True, env={**os.environ, "PYTHONIOENCODING": "utf-8", **(env or {})})


def repeats_file(source: str) -> Path:
    return RESULTS_DIR / f"prompt_repeats_{source}_{TAG}.json"


def main() -> None:
    from src.generate.llm import load_llm      # fail early, before any run, when the API cannot be reached
    try:
        load_llm().complete("Say ok")
    except Exception as exc:
        sys.exit(f"the answer model cannot be reached: {exc}")

    if not check_source(HOLDOUT_SOURCE, HOLDOUT_PATH):
        run("eval.run_eval", "--golden", str(HOLDOUT_PATH), "--name", HOLDOUT_SOURCE,
            "--note", "fresh held-out set n01-n25, v4 pipeline (prompt v2): source chunks for the phase 5 final check",
            env={"RAG_PIPELINE_VERSION": "v4"})
        if not check_source(HOLDOUT_SOURCE, HOLDOUT_PATH):
            sys.exit(f"run '{HOLDOUT_SOURCE}' is incomplete: run this script again (finished questions are kept)")
    if not check_source(GOLDEN_SOURCE, GOLDEN_PATH):
        sys.exit(f"run '{GOLDEN_SOURCE}' does not hold answers for every golden question")

    for source, questions, repeats in ((HOLDOUT_SOURCE, HOLDOUT_PATH, 5), (GOLDEN_SOURCE, GOLDEN_PATH, 3)):
        if not repeats_file(source).exists():
            run("eval.prompt_repeats", "--source", source, "--golden", str(questions), "--prompts", f"{BASE},{CANDIDATE}",
                "--repeats", str(repeats), "--tag", TAG)

    scores = {name: {p: score(runs) for p, runs in json.loads(repeats_file(src).read_text(encoding="utf-8")).items()}
              for name, src in (("golden", GOLDEN_SOURCE), ("holdout", HOLDOUT_SOURCE))}
    print("\n| Prompt | Golden correct | Holdout correct | Gap | Golden complete | Holdout complete | Golden faithful | Holdout faithful |")
    print("|---|---|---|---|---|---|---|---|")
    for p in (BASE, CANDIDATE):
        g, h = scores["golden"][p], scores["holdout"][p]
        print(f"| `{p}` | {g['correctness']:.3f} | {h['correctness']:.3f} | {abs(g['correctness'] - h['correctness']):.3f} | "
              f"{g['completeness']:.3f} | {h['completeness']:.3f} | {g['faithfulness']:.3f} | {h['faithfulness']:.3f} |")

    g_base, g_cand = scores["golden"][BASE]["correctness"], scores["golden"][CANDIDATE]["correctness"]
    h_base, h_cand = scores["holdout"][BASE]["correctness"], scores["holdout"][CANDIDATE]["correctness"]
    if g_cand >= g_base - MARGIN and h_cand >= h_base - MARGIN and (g_cand + h_cand) > (g_base + h_base):
        print(f"\nUse {CANDIDATE}: set PIPELINE_VERSION to \"v7\" in src/config.py, then run_eval both sets with v7.")
    else:
        print(f"\nUse {BASE}: set PIPELINE_VERSION to \"v4\" in src/config.py ({CANDIDATE} is not better on both sets).")
    print("Per-question differences: eval/results/prompt_repeats_*_final.md")


if __name__ == "__main__":
    main()
