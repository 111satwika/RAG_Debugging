"""Phase 4: measure each retrieval technique on the golden set, using the main (v2) index.

    python -m eval.retrieval_experiments                      # every technique, retrieval only (a few minutes)
    python -m eval.retrieval_experiments --set rerankers      # compare open reranking models (retrieve 10, keep 5)
    python -m eval.retrieval_experiments --only p4-hybrid-rerank20 --full    # also answer and judge one of them

Every technique is a RetrievalSpec (src/retrieve/retriever.py). Each is saved as a run named p4-<label> with version
"exp" and compared with plain dense search, the v2 retrieval. The index is not changed. A fresh Retriever is made for
each technique, so its latency includes embedding the question and any model call (rewrite, route) it makes.

route-oracle reads the answer key (the golden set's source documents): it is the ceiling for what a perfect filter
could give, not something the app can do. route-llm asks the model which documents the question needs.
"""
import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from eval.run_eval import GOLDEN_PATH, RESULTS_DIR, load_index_chunks, run_evaluation
from src.config import TOP_K, snapshot
from src.generate.llm import load_llm
from src.ingest.embed import load_embedder
from src.retrieve.dense import get_collection
from src.retrieve.filters import oracle_routes
from src.retrieve.rerank import RERANKERS, get_reranker
from src.retrieve.retriever import RetrievalSpec, Retriever

EXPERIMENT_VERSION = "exp"
REPORT_NAME = "retrieval_experiments.md"
RERANKER_REPORT_NAME = "retrieval_rerankers.md"
CONTROL = "p4-dense"
TYPES = ("factual", "exact_term", "table", "multi_section", "cross_doc")


@dataclass(frozen=True)
class Experiment:
    spec: RetrievalSpec
    note: str

    @property
    def name(self) -> str:
        return f"p4-{self.spec.label}"


EXPERIMENTS = [
    Experiment(RetrievalSpec("dense"), "Control: dense search only (the v2 retrieval)"),
    Experiment(RetrievalSpec("dense", route="oracle"), "1a. Metadata filter, answer-key documents (a ceiling, not a real filter)"),
    Experiment(RetrievalSpec("dense", route="llm"), "1b. Metadata filter: the model picks the documents; several documents get a share each"),
    Experiment(RetrievalSpec("bm25"), "2. BM25 keyword search alone"),
    Experiment(RetrievalSpec("hybrid"), "3. Hybrid: dense + BM25 merged with RRF"),
    Experiment(RetrievalSpec("dense", rerank_top=20), "4a. Dense top 20, cross-encoder rerank (MiniLM), keep 5"),
    Experiment(RetrievalSpec("hybrid", rerank_top=20), "4b. Hybrid top 20, cross-encoder rerank (MiniLM), keep 5"),
    Experiment(RetrievalSpec("dense", rewrite=True), "5. Query rewriting on dense search"),
    Experiment(RetrievalSpec("hybrid", route="llm"), "6a. Hybrid + model-chosen documents"),
    Experiment(RetrievalSpec("hybrid", route="llm", rerank_top=20), "6b. Hybrid + documents + rerank"),
    Experiment(RetrievalSpec("hybrid", route="llm", rerank_top=20, rewrite=True), "6c. Everything: + query rewriting"),
]

# Retrieve 10, rerank, keep 5, with open rerankers that run locally (nothing is sent to a third party).
RERANKER_EXPERIMENTS = [
    Experiment(RetrievalSpec("dense"), "Control: dense search only (the v2 retrieval)"),
    Experiment(RetrievalSpec("hybrid"), "Control: hybrid, no reranking (the Phase 4 pick)"),
    Experiment(RetrievalSpec("hybrid", rerank_top=20), "Hybrid top 20, MiniLM reranker, keep 5"),
    Experiment(RetrievalSpec("hybrid", rerank_top=20, reranker="bge-base"), "Hybrid top 20, bge-reranker-base, keep 5"),
    *[Experiment(RetrievalSpec(mode, rerank_top=10, reranker=key), f"{mode} top 10, rerank with {RERANKERS[key]}, keep 5")
      for key in ("minilm", "bge-base", "bge-v2-m3") for mode in ("dense", "hybrid")],
]
# Query transformation (rewriting, expansion, multi-query), on every question or only on the hard ones the gate selects.
TRANSFORM_EXPERIMENTS = [
    Experiment(RetrievalSpec("hybrid"), "Control: hybrid, no reranker, no transformation"),
    Experiment(RetrievalSpec("hybrid", transform="always"), "Transform every question, no reranker"),
    Experiment(RetrievalSpec("hybrid", transform="auto"), "Transform only the questions the gate selects, no reranker"),
    Experiment(RetrievalSpec("hybrid", rerank_top=10, reranker="bge-base"), "Control: hybrid + bge-reranker-base (v2-extended without transformation)"),
    Experiment(RetrievalSpec("hybrid", rerank_top=10, reranker="bge-base", transform="auto"),
               "v2-extended + gated transformation"),
]
TRANSFORM_REPORT_NAME = "retrieval_transform.md"
SETS = {"main": (EXPERIMENTS, REPORT_NAME, "Phase 4: retrieval techniques"),
        "rerankers": (RERANKER_EXPERIMENTS, RERANKER_REPORT_NAME, "Phase 4: reranking models"),
        "transform": (TRANSFORM_EXPERIMENTS, TRANSFORM_REPORT_NAME, "Phase 4: query transformation for hard questions")}


def experiment_config(spec: RetrievalSpec, k: int, reranker: str) -> dict:
    return {**snapshot(), "version": EXPERIMENT_VERSION, "top_k": k, "retrieval": spec.label,
            "retrieval_mode": spec.mode, "rerank_top": spec.rerank_top, "reranker": reranker if spec.rerank_top else None,
            "query_transform": spec.transform}


def run_experiments(only: list[str] | None = None, k: int = TOP_K, results_dir: Path = RESULTS_DIR,
                    golden_path: Path = GOLDEN_PATH, experiments: list[Experiment] | None = None, embedder=None,
                    collection=None, llm=None, reranker=None, full: bool = False, answer_llm=None,
                    set_name: str = "main", suffix: str = "") -> list[dict]:
    """`reranker` replaces the model of every experiment (used by tests); by default each names its own.
    `suffix` is added to every run name (for example "-ext"), so a re-run on another golden set keeps the earlier runs."""
    default_experiments, report_name, title = SETS[set_name]
    experiments = [e for e in (experiments or default_experiments) if not only or e.name in only]
    if not experiments:
        raise ValueError(f"no experiment matches {only}")
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    embedder = embedder or load_embedder()
    collection = collection or get_collection(embedder)
    llm = llm or load_llm("openai", "gpt-4o-mini")
    for key in {e.spec.reranker for e in experiments if e.spec.rerank_top}:
        (reranker or get_reranker(key)).rerank("warm up", [{"text": "warm up", "score": 0.0}], 1)    # keep model loads out of the timings
    chunks = load_index_chunks()
    if full and answer_llm is None:
        answer_llm = load_llm("openai", "gpt-4o-mini")

    rows, control = [], None
    for exp in experiments:
        retriever = Retriever(embedder, collection, llm=llm, reranker=reranker)
        retriever.oracle = oracle_routes(golden, retriever.catalog)
        search, timings, cache = retriever.searcher(exp.spec), {}, {}

        def timed(question, top_k, search=search, timings=timings, cache=cache):
            # The evaluation searches each question twice (metrics, then the answer); the second call reuses the first.
            if (question, top_k) not in cache:
                started = time.perf_counter()
                cache[(question, top_k)] = search(question, top_k)
                if top_k == k and question not in timings:         # the evaluation's deeper searches are not part of a query
                    timings[question] = (time.perf_counter() - started) * 1000
            return cache[(question, top_k)]

        model = (reranker or get_reranker(exp.spec.reranker)).name
        out = run_evaluation(exp.name + suffix, exp.note, k, retrieval_only=not full, use_judge=full, force=True, chunks=chunks,
                             searcher=timed, config=experiment_config(exp.spec, k, model), results_dir=results_dir,
                             golden_path=golden_path, llm=answer_llm, trace_path=results_dir / "experiment_traces.jsonl")
        mrr = {r["id"]: r["retrieval"]["mrr"] for r in out["results"] if r.get("retrieval")}
        row = {"experiment": exp, "summary": out["summary"], "by_type": out["by_type"], "mrr_by_question": mrr,
               "ms": sum(timings.values()) / max(len(timings), 1), "llm_ms": sum(retriever.llm_ms) / max(len(retriever.llm_ms), 1),
               "calls": len(retriever.llm_ms) / max(len(timings), 1)}      # model calls per query, measured
        if exp.name == CONTROL:
            control = row
        rows.append(row)
        s = out["summary"]
        print(f"{exp.name + suffix:44} recall={s['recall']:.2f}  mrr={s['mrr']:.2f}  ndcg={s['ndcg']:.2f}  hit={s['hit']:.2f}"
              f"  {row['ms']:6.0f} ms/query" + (f"  correct={s['correctness']:.2f}" if full and s["correctness"] is not None else ""), flush=True)
    if not only:
        (results_dir / report_name.replace(".md", f"{suffix}.md")).write_text(render_report(rows, k, title), encoding="utf-8")
    return rows


# ---------------------------------------------------------------- report

def _f(value, digits=2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def score(row: dict) -> float:
    """Rank-aware score used for comparing: the mean of MRR and nDCG (Recall@k saturates, precision cannot rise much)."""
    s = row["summary"]
    return (s["mrr"] + s["ndcg"]) / 2


def question_changes(row: dict, control: dict) -> str:
    """How many questions ranked their first relevant chunk better / worse than the control."""
    better = sum(1 for q, v in row["mrr_by_question"].items() if v > control["mrr_by_question"].get(q, 0))
    worse = sum(1 for q, v in row["mrr_by_question"].items() if v < control["mrr_by_question"].get(q, 0))
    return f"+{better} / -{worse}"


def render_report(rows: list[dict], k: int, title: str = "Phase 4: retrieval techniques") -> str:
    control = next((r for r in rows if r["experiment"].name == CONTROL), rows[0])
    models = sorted({r["experiment"].spec.reranker for r in rows if r["experiment"].spec.rerank_top})
    lines = [
        f"# {title}",
        "",
        f"Index: the main v2 index (`{snapshot()['embed_model']}`, structure-aware chunks). Top-{k}, retrieval only, "
        "25 answerable golden questions. One question is 0.04, so differences of about that size are noise.",
        "Rerankers (all open models, run locally): " + (", ".join(f"`{m}` = {RERANKERS[m]}" for m in models) or "none")
        + ". Rewriting and routing use the same OpenAI model as the app (gpt-4o-mini).",
        "`route-oracle` reads the answer key: it is a ceiling for what a perfect filter could give, not a usable technique.",
        "",
        "## All techniques",
        "",
        "| Technique | Recall@k | MRR | nDCG | Hit@k | Score* | vs control** | Model calls/query | ms/query |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        s, spec = row["summary"], row["experiment"].spec
        delta = score(row) - score(control)
        lines.append(f"| {row['experiment'].name.removeprefix('p4-')} | {_f(s['recall'])} | {_f(s['mrr'])} | {_f(s['ndcg'])} | "
                     f"{_f(s['hit'])} | {_f(score(row), 3)} | {question_changes(row, control)} ({delta:+.3f}) | "
                     f"{row['calls'] if spec.transform != 'none' else spec.llm_calls:.2f} | {row['ms']:.0f} |")
    lines += ["", "\\* mean of MRR and nDCG.  \\*\\* questions whose first relevant chunk moved up / down against plain dense, "
              "and the change in the score.", ""]
    for metric, title in (("mrr", "MRR by question type"), ("recall", "Recall@k by question type")):
        lines += [f"## {title}", "", "| Technique | " + " | ".join(TYPES) + " |", "|---|" + "---|" * len(TYPES)]
        for row in rows:
            cells = [_f((row["by_type"].get(t) or {}).get(metric)) for t in TYPES]
            lines.append(f"| {row['experiment'].name.removeprefix('p4-')} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## What each technique is", ""]
    lines += [f"- `{r['experiment'].name.removeprefix('p4-')}`: {r['experiment'].note}" for r in rows]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4 retrieval experiments")
    parser.add_argument("--set", choices=list(SETS), default="main", help="which group of experiments to run")
    parser.add_argument("--only", help="comma-separated experiment names, e.g. p4-hybrid,p4-bm25")
    parser.add_argument("-k", type=int, default=TOP_K)
    parser.add_argument("--full", action="store_true", help="also answer with gpt-4o-mini and judge (costs API calls)")
    parser.add_argument("--suffix", default="", help="added to every run name, e.g. -ext, to keep earlier runs")
    args = parser.parse_args()
    run_experiments(args.only.split(",") if args.only else None, args.k, full=args.full, set_name=args.set,
                    suffix=args.suffix)
    print(f"reports are in {RESULTS_DIR}")
