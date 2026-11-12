"""Phase 5 drill: remove the grounding rules and watch hallucinations appear on questions the documents cannot answer.

    python -m eval.prompt_drill                 # v1, v2-ungrounded and v2 on 13 unanswerable questions, 2 runs each

The 6 unanswerable golden questions are too few to see an effect, so this adds 7 more traps that tempt a model to answer from
its own knowledge (general knowledge, a famous model, a company's CEO, a guess about how QuickLeave is built). Each is checked
to be absent from the documents. Every prompt gets the same retrieved chunks (plain hybrid search, top 5). A reply counts as
a hallucination when the judge says it did not refuse and its faithfulness is below 1 (it states facts the sources do not
contain). The report is eval/results/prompt_drill.md.
"""
import argparse
import json
import re
from pathlib import Path

from eval.judge import judge_answer
from eval.run_eval import GOLDEN_PATH, RESULTS_DIR
from src.config import TOP_K
from src.generate.answer import parse_answer
from src.generate.generate import build_prompt, is_cited, source_ids
from src.generate.llm import load_llm

REPORT_PATH = RESULTS_DIR / "prompt_drill.md"
PROMPTS = ("v1", "v2-ungrounded", "v2")
TRIALS = 2
NORMALIZED = Path(__file__).resolve().parents[1] / "data" / "processed" / "normalized"

# (question, words that would show the documents DO answer it; the drill checks none of them appears)
EXTRA_TRAPS = [
    ("What is the capital of France?", ["france", "paris"]),
    ("How many layers does BERT-base have?", ["bert"]),
    ("Who is the CEO of Cprime Inc?", ["ceo", "chief executive"]),
    ("What programming language is QuickLeave written in?", ["written in", "implemented in", "built with"]),
    ("Does QuickLeave support the Hindi language?", ["hindi", "multilingual", "localization"]),
    ("What is the average salary of an AI engineer in India?", ["average salary", "lpa"]),
    ("Which version of Python does the candidate use?", ["python 3", "python3", "python 2"]),
]


def traps(golden_path: Path = GOLDEN_PATH) -> list[dict]:
    golden = [q for q in json.loads(golden_path.read_text(encoding="utf-8")) if q["type"] == "unanswerable"]
    extra = [{"id": f"x{i:02d}", "question": q, "expected_answer": "ABSTAIN: the documents do not contain this.",
              "type": "unanswerable", "must_include": [], "source_docs": [], "source_section": None}
             for i, (q, _) in enumerate(EXTRA_TRAPS, start=1)]
    return golden + extra


def check_traps_are_absent(documents: dict[str, str]) -> list[str]:
    """Words of EXTRA_TRAPS that do appear in a document (a trap that the documents can answer is not a trap)."""
    text = "\n".join(documents.values()).lower()
    return [f"'{w}' (for: {q})" for q, words in EXTRA_TRAPS for w in words if re.search(rf"\b{re.escape(w)}\b", text)]


def ask(llm, prompt_version: str, question: str, chunks: list[dict]) -> str:
    """The answer text one prompt produces (a cited prompt's JSON is parsed, as in the pipeline)."""
    prompt = build_prompt(question, chunks, prompt_version)
    if not is_cited(prompt_version):
        return llm.complete(prompt)["text"]
    return parse_answer(llm.complete(prompt, json_mode=True)["text"], source_ids(chunks)).text


def run_drill(retrieve, llm, judge_llm, questions: list[dict] | None = None, prompts=PROMPTS, trials: int = TRIALS, k: int = TOP_K) -> dict:
    """{prompt: {question id: [{answer, refused, faithfulness}, ...]}}"""
    questions = questions or traps()
    out = {p: {} for p in prompts}
    for q in questions:
        hits = retrieve(q["question"], k)
        chunks = [{"chunk_id": h["chunk_id"], "doc_id": h["doc_id"], "page": h["page"], "text": h["text"],
                   "source_id": f"S{n}"} for n, h in enumerate(hits, start=1)]
        for prompt in prompts:
            runs = []
            for _ in range(trials):
                answer = ask(llm, prompt, q["question"], chunks)
                verdict = judge_answer(judge_llm, q, answer, chunks if is_cited(prompt) else
                                       [{k_: v for k_, v in c.items() if k_ != "source_id"} for c in chunks])
                runs.append({"answer": answer, "refused": verdict["abstained"], "faithfulness": verdict["faithfulness"]})
            out[prompt][q["id"]] = runs
        print(f"{q['id']} done", flush=True)
    return out


def hallucinated(run: dict) -> bool:
    return not run["refused"] and run["faithfulness"] < 1


def render_report(drill: dict, questions: list[dict]) -> str:
    prompts = list(drill)
    n = sum(len(runs) for runs in drill[prompts[0]].values())
    lines = ["# Phase 5 drill: what the grounding rules prevent", "",
             f"{len(questions)} questions the documents cannot answer ({sum(q['id'].startswith('q') for q in questions)} from the golden set, "
             f"{sum(q['id'].startswith('x') for q in questions)} extra traps), {n // len(drill[prompts[0]])} runs of each. The retrieved chunks are the "
             "same for every prompt. gpt-4o-mini answers and judges.", "",
             "| Prompt | Refused | Answered | Hallucinated (answered and not faithful) |", "|---|---|---|---|"]
    for p in prompts:
        runs = [r for rs in drill[p].values() for r in rs]
        refused = sum(r["refused"] for r in runs)
        lines.append(f"| `{p}` | {refused} of {len(runs)} ({refused / len(runs):.0%}) | {len(runs) - refused} | "
                     f"{sum(hallucinated(r) for r in runs)} |")
    lines += ["", "## Per question (R = refused in every run, A = answered in every run, M = mixed)", "",
              "| Question | " + " | ".join(f"`{p}`" for p in prompts) + " |", "|---|" + "---|" * len(prompts)]
    for q in questions:
        cells = []
        for p in prompts:
            flags = [r["refused"] for r in drill[p][q["id"]]]
            cells.append("R" if all(flags) else "A" if not any(flags) else "M")
        lines.append(f"| {q['id']}: {q['question']} | " + " | ".join(cells) + " |")
    shown = [(p, q, r) for p in prompts for q in questions for r in drill[p][q["id"]][:1] if hallucinated(r)][:6]
    if shown:
        lines += ["", "## Examples of answers that were not refused", ""]
        for p, q, r in shown:
            lines.append(f"- `{p}` on {q['id']} ({q['question']}): {r['answer'].strip()[:300]}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    from src.ingest.embed import load_embedder
    from src.retrieve.dense import get_collection
    from src.retrieve.retriever import RetrievalSpec, Retriever

    parser = argparse.ArgumentParser(description="Phase 5 grounding drill")
    parser.add_argument("--prompts", help="comma-separated prompt versions (default: v1, v2-ungrounded, v2), e.g. v2,v2.1")
    args = parser.parse_args()
    chosen = tuple(args.prompts.split(",")) if args.prompts else PROMPTS
    report_path = REPORT_PATH if chosen == PROMPTS else RESULTS_DIR / f"prompt_drill_{'_'.join(chosen)}.md"
    present = check_traps_are_absent({p.stem: p.read_text(encoding="utf-8") for p in NORMALIZED.glob("*.md")})
    if present:
        raise SystemExit("these traps are answerable from the documents, fix EXTRA_TRAPS: " + "; ".join(present))
    embedder = load_embedder()
    retriever = Retriever(embedder, get_collection(embedder))
    spec = RetrievalSpec("hybrid")
    questions = traps()
    drill = run_drill(lambda question, k: retriever.search(spec, question, k), load_llm(), load_llm(), questions, prompts=chosen)
    report_path.write_text(render_report(drill, questions), encoding="utf-8")
    print(render_report(drill, questions))
