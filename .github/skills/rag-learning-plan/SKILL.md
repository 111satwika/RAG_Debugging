---
name: rag-learning-plan
description: Guide the user step by step through building a production-style RAG system (resumes, spec docs, research papers) to learn prompting, chunking, retrieval, evaluation, debugging and optimization. Use when working on the user's RAG learning project or when they ask which phase or step comes next.
---

# RAG Learning Plan — Production-Style, Step by Step

## Goal

Build a RAG system over **resumes, spec documents and research papers** to deeply understand:
- How the **prompt** is assembled and what the LLM actually sees
- How **chunking** and **retrieval** work, and how they fail
- How to **evaluate, debug and optimize** like a production engineer
- What to do when the system gives **wrong answers**

## How to guide the user (instructions for the assistant)

- Act as a **mentor**, not just a code generator. Explain *why* before *how*.
- Work **one phase at a time**. Do not jump ahead unless the user asks.
- Before writing code for a phase, state: the goal, what will be built, and the "done when" check.
- Keep code **simple, readable, plain Python**. No LangChain / LlamaIndex until Phase 7 (and only if the user wants it).
- **Every change must be measured** against the golden set. Change **one thing at a time**.
- When an answer is wrong, follow the **Debugging Checklist** below before changing anything.
- Use short, clear explanations. Prefer small working steps over large code dumps.

## Ground rules

1. **No framework at first** — write each step yourself to see how it works.
2. **Trace and log everything from Phase 1** — every query gets a trace with chunks, scores, the final prompt, timings and config. Every new stage added later gets its own span.
3. **Measure before optimizing** — every improvement must show in a metric.
4. **One change at a time** — otherwise you can't tell what helped.
5. **Resumes contain personal data** — use your own or synthetic resumes only.

## Current state (22 October 2026)

**Live version: `v12`** (`PIPELINE_VERSION` in `src/config.py`; the app on `127.0.0.1:8765` runs it). Every earlier profile is kept unchanged in `PROFILES`, so any version can be run or restored.

| Version | = previous + | Phase | Golden correct / MRR / nDCG | Holdout correct / MRR |
|---|---|---|---|---|
| `v1` | baseline: bge-small, fixed 500, Qwen 7B local | 1 | 0.91 / 0.69 / 0.73 | - |
| `v2` | OpenAI embeddings, structure-aware chunks + headers + tables + overlap, gpt-4o-mini | 3 | - | - |
| `v2-extended` | hybrid (dense + BM25, RRF) + bge-reranker + gated query transform | 4 | 0.92 / 0.91 / 0.92 | - |
| `v3` | cited, grounded JSON prompt `v2` | 5 | 0.97 / 0.91 / 0.91 | - |
| `v4` | CRAG-style validation (score grade + corrective search), 7 candidates reranked | 6 | 0.96 / 0.90 / 0.90 | - |
| `v5`, `v6` | prompts `v2.4`, `v2.5` (tuned on golden, did not generalize) | 6 | 0.99 (old judge) | 0.89 (old holdout) |
| `v7` | prompt `v2.9` | 5 (revisited) | 0.97 / 0.92 / 0.92 | 0.96 / 0.97 |
| `v8` | prompt `v2.11` (every fact + item named; attribution checks) | 5 | 1.00 / 0.91 / 0.91 | 0.96 / 0.97 |
| `v9` | precision cut-off after reranking (2% of best score) | 4 | 0.97 / 0.91 / 0.90 | 0.96 / 0.97 |
| `v10` | CRAG knowledge refinement (LLM filter drops unhelpful chunks) | 6 | 0.99 / 0.90 / 0.89 | 1.00 / 0.97 |
| `v11` | table chunks headed by their caption (own index) + asked-about document first | 3 / 4 | 1.00 / 0.95 / 0.93 | 0.96 / 0.97 |
| **`v12`** | refinement grader sees the kept passages (fixes q37) | 6 | **0.99 / 0.95 / 0.93** | **0.96 / 0.97** |

Scores from `v7` on are judged with the final judge (judge `v4`, Phase 2), on the 50-question golden set and the fresh 25-question holdout (`n01`-`n25`); earlier rows used older judges and sets and are not directly comparable. Context precision (relevant chunks / chunks the model read): about 0.27 up to `v8`, 0.54 with `v9`, 0.75 from `v10`.

**The app (Phase 7):** streaming answers over Server-Sent Events with TTFT, the CRAG grade of every query and of every retrieved chunk, sources listed by chunk name, the live version in the chat header, and a Reports page with context precision.

**Open weaknesses (shared by every recent version):** answers that stop before naming every item (`q04`, `n13`), reading a table reliably (`n12` about 4 of 5, `q35` and the paper's 41.0 / 41.8 conflict), golden table questions `q18`-`q20` ranked below the Results text, and the word "user" in a question pulling the QuickLeave spec instead of the resume.

**Budget note:** the OpenAI credit is nearly used up. Estimate the cost before any paid run, prefer free analysis of saved runs, and ask before spending. Judge with gpt-4o-mini; do not use gpt-4o unless the user asks.

## Suggested stack (free-friendly)

| Part | Tool |
|---|---|
| Parsing | PyMuPDF / pymupdf4llm (PDF → text / markdown) |
| Embeddings | sentence-transformers (`BAAI/bge-small-en-v1.5`), local |
| Vector DB | Chroma first → Qdrant (Docker) in Phase 7 |
| Keyword search | `rank_bm25` |
| Reranker | `BAAI/bge-reranker-base` |
| LLM | Any API (Groq, OpenAI, Anthropic) or Ollama locally |
| Evaluation | Own metrics + RAGAS |
| Tracing | Langfuse |
| API / deploy | FastAPI + Docker Compose |

## Suggested project structure

```
rag-project/
├── data/
│   ├── raw/              # resumes/, specs/, papers/
│   └── processed/        # parsed text + chunks (JSON)
├── eval/
│   ├── golden_set.json   # questions, expected answers, source docs
│   └── results/          # eval run outputs (one file per run)
├── src/
│   ├── ingest/           # parse.py, chunk.py, embed.py, index.py
│   ├── retrieve/         # dense.py, bm25.py, hybrid.py, rerank.py, filters.py
│   ├── generate/         # prompts/ (versioned), generate.py
│   ├── validate/         # threshold.py, citations.py, faithfulness.py
│   ├── pipeline.py       # end-to-end query flow
│   └── config.py         # chunk size, top-k, models, thresholds
├── src/observability/    # tracer.py (trace + spans), logger.py — from Phase 1
├── logs/                 # traces.jsonl, ingest.jsonl (git-ignored)
├── scripts/              # show_trace.py
├── api/                  # FastAPI backend for the UI: chat, traces, logs, reports (grows in Phase 7)
├── ui/                   # React + TypeScript web UI (Chat, Traces, Logs, Reports)
├── tests/
├── config.yaml
└── docker-compose.yml    # Phase 7
```

---

## Phase 0 — Data + golden test set (2–3 days)

**Goal:** Have data and a way to measure before building anything.

**Build:**
- Collect **1 resume, 1 spec doc, 1 research paper** into `data/raw/` (resumes/, specs/, papers/).
  - Pick documents with variety: sections, at least one table, and some exact terms (names, IDs, spec numbers).
  - Prefer a research paper of 8+ pages so chunking choices actually matter.
- Create `eval/golden_set.json` with **~25–30 questions** (about 7–8 per document, 2+ cross-doc, 3 unanswerable). Each entry:
  ```json
  {
    "id": "q09",
    "question": "How many days of Sick Leave does QuickLeave allow per year?",
    "expected_answer": "12 days per year (increased from 10 days in version 1.2).",
    "must_include": ["12"],
    "must_not_include": ["10 days per year"],
    "source_docs": ["quickleave_spec_v1.2.pdf"],
    "source_section": "5. Leave Policy; 10. Revision History",
    "type": "table",
    "notes": "Version trap: Revision History mentions the old value of 10 days."
  }
  ```
- Required fields: `id`, `question`, `expected_answer`, `source_docs`, `source_section`, `type`.
- Optional fields:
  - `must_include` — key facts the answer must contain (for simple keyword checks in Phase 2, before LLM-as-judge).
  - `must_not_include` — stale or wrong values the answer must NOT contain (catches version and conflict errors).
  - `notes` — explains any trap the question is testing.
- For unanswerable questions: `expected_answer` starts with `ABSTAIN:`, `source_docs` is `[]`, `source_section` is `null`.
- Include types: `factual`, `multi_section` (answer needs two sections of the same doc), `cross_doc` (e.g., "Does the candidate's experience match the spec's requirements?"), `exact_term` (names, IDs, spec numbers), `table`, `unanswerable`.
- Include **traps**: a version trap (old value still in the doc), a conflict trap (doc contradicts itself), a false-abstention trap (looks unanswerable but is answered), and a parametric-knowledge trap (the LLM knows it, but the docs don't).

**Current data set (built):**
- `data/raw/resumes/Satwika_Agentic_AI_Engineer_Resume.pdf` — 7 questions (q01–q07)
- `data/raw/specs/quickleave_spec_v1.2.pdf` — 8 questions (q08–q15)
- `data/raw/papers/attention_is_all_you_need.pdf` — 8 questions (q16–q23)
- Cross-doc (spec + resume) — 2 questions (q24–q25)
- Unanswerable — 3 questions (q26–q28)
- **Total: 28 questions**

**Why first:** Without a test set you can't debug or prove improvement.

**Done when:** golden set has ~25–30 questions covering all types and traps, including at least 3 unanswerable. ✅ Done (28 questions). Extended to 50 in Phase 4: 22 added questions carry `"tier": "hard"` (the original 28 are the standard tier). Results report both tiers, and `summary.json` has a `by_tier` section.

**Note on small data:** With only 3 documents, retrieval is easier than in real systems, so focus on **chunk-level** retrieval (is the right *section* retrieved?) rather than just doc-level. Record `source_section` in the golden set too. Add more documents later to see retrieval get harder.

**Update (22 Oct): reference fixes and a fresh held-out set.**
- **Golden references corrected** (they had marked right answers wrong): `q19` asked for the location of the value ("abstract and Table 2"), which moved to `optional_details`; `q25` is now worded as a real alternative ("AI Visual Placemat **or** AI ATP Toolkit; either one is enough"), because the judge read "A and B; either is fine" as two required facts; `q24` also accepts "AD SSO" (the skills chunk says "Azure (AD SSO ...)", which the key phrase "Azure AD" did not match).
- **The 41-question held-out set was retired** (`eval/holdout_set_v1.json`): `h01`-`h18` had been looked at and the prompt rules of `v2.6`-`v2.9` were written from `h03` and `h17`, so it was no longer unseen.
- **New held-out set** (`eval/holdout_set.json`, `n01`-`n25`, written before any run): 8 factual, 4 table, 4 multi-section, 3 exact-term, 3 cross-document, 3 unanswerable; 12 easy and 13 hard; 4 comparison questions ("every", "all", "more than twice", "highest") and 4 "does the resume mention X" questions (1 yes, 3 no). Every key fact was checked to sit in a chunk of its source document, every "not mentioned" term to be absent from the resume, and every unanswerable question to have no answer in the documents. Rule: never write or tune a prompt from these questions; they only decide.

---

## Phase 1 — Naive RAG from scratch (1 week)

**Goal:** Understand the full pipeline end to end.

**Build:**
1. Parse PDFs → text
2. Fixed-size chunking (~500 tokens, no overlap)
3. Embed chunks → store in Chroma with metadata (`doc_id`, `doc_type`, `page`)
4. Embed query → top-k (k=5) search
5. Build prompt with retrieved chunks → call LLM → answer

**Tracing and logging (from day one):**

Build a small tracer yourself first (no external tool yet), so you understand what a trace is.

- **Python `logging`** with levels (DEBUG for chunk contents, INFO for steps, ERROR for failures). No `print()` in pipeline code.
- **One trace per query**, with a unique `trace_id`, saved as one JSON line to `logs/traces.jsonl`.
- **One span per stage**, each with start time, duration (ms), inputs and outputs: `embed_query`, `retrieve`, `build_prompt`, `llm_call`.
- **Ingestion log** in `logs/ingest.jsonl`: per file → pages parsed, characters extracted, number of chunks, errors. (Catches parsing failures early.)

**Each query trace records:**
```json
{
  "trace_id": "a1b2c3",
  "timestamp": "2026-10-18T21:00:00",
  "config": {"chunk_size": 500, "top_k": 5, "embed_model": "bge-small-en-v1.5", "llm": "…", "prompt_version": "v1"},
  "query": "What is the attention mechanism in paper X?",
  "spans": {
    "embed_query": {"ms": 12},
    "retrieve": {"ms": 30, "results": [{"chunk_id": "paper_x_c14", "doc_id": "paper_x", "page": 4, "score": 0.82, "preview": "first 150 chars…"}]},
    "build_prompt": {"ms": 1, "prompt_tokens": 1450, "final_prompt": "…exact prompt…"},
    "llm_call": {"ms": 1800, "input_tokens": 1450, "output_tokens": 120, "temperature": 0}
  },
  "answer": "…",
  "total_ms": 1843,
  "error": null
}
```

- Store the **config** in every trace, so you can always tell which settings produced which answer.
- Add a tiny script `scripts/show_trace.py <trace_id>` that pretty-prints a trace: chunks with scores, then the final prompt, then the answer.

**You learn:** how the prompt is assembled, what the model actually sees, where time is spent, and how to read a trace when debugging.

**Done when:** you can answer questions, open any query's trace, and explain every chunk, score and line of the prompt.

---

## Phase 2 — Evaluation + debugging harness (1 week)

**Goal:** Measure quality and diagnose failures systematically.

**Build:**
- `eval/run_eval.py` — runs all golden-set questions and records two groups of metrics.

  **A. Retrieval metrics** (no LLM needed; computed from the `retrieve` span of each trace):
  - **Recall@k** — what share of the relevant chunks appear in the top-k?
  - **Precision@k** — what share of the top-k chunks are relevant? (shows how much noise reaches the prompt)
  - **MRR** — how high is the first relevant chunk? (1 / rank, averaged)
  - **nDCG@k** — are the relevant chunks ranked near the top? (rewards a correct chunk at rank 1 more than at rank 5)
  - **What counts as relevant:** judge at **chunk level**, not just doc level (with 3 documents, doc-level is too easy). A chunk is relevant if it comes from one of the question's `source_docs` and contains the answer or matches `source_section`. Skip `unanswerable` questions for these metrics, since they have no relevant chunk.

  **B. Generation metrics** (scored per answer, 0–1 or pass/fail; LLM-as-judge, calibrated on a few by hand):
  - **Correctness** — does the answer match the expected answer? Use `must_include` / `must_not_include` as a cheap keyword check first, then the LLM judge.
  - **Relevance** — does the answer address the question that was asked?
  - **Completeness** — does it cover *all* the key facts, not just part of the answer? (e.g. `q11` gives the meaning but drops HTTP 409)
  - **Faithfulness / groundedness** — is every claim supported by the retrieved chunks, with nothing taken from the model's own knowledge? (judged against the chunks in the trace)
  - **Abstention accuracy** — did it refuse unanswerable questions, and answer the answerable ones? Also track **false abstentions**.
- Report metrics **overall and by question type** (`factual`, `table`, `exact_term`, …).
- Every eval question produces a normal trace, and the eval result stores its `trace_id`, so any failure can be opened with `show_trace.py`.
- A **failure report**: for each wrong answer, label the failing stage:
  `indexing | chunking | filter | search | ranking | generation | data_quality`
  - Retrieval metrics point to `chunking | filter | search | ranking`; generation metrics point to `generation`. If retrieval is fine but faithfulness or correctness is low, the problem is in the prompt or model, not search.

**You learn:** the debugging flow, applied to your own system, and how to tell a retrieval problem from a generation problem.

**Done when:** one command runs the eval and prints the retrieval metrics (Recall@k, Precision@k, MRR, nDCG@k) and the generation metrics (correctness, relevance, completeness, faithfulness, abstention accuracy) plus a failure report. Save this as the **baseline**.

### Update (22 Oct): the judge, rebuilt until it is trustworthy

The prompt work of Phase 5 kept going in circles because the judge was wrong as often as the answers. The judge (gpt-4o-mini, the same model as the answers) was rebuilt in steps; each step was checked on answers whose right score was known, with `python -m eval.judge_calibration` (17 cases: right answers, wrong Yes/No, a missing fact, an extra wrong item, a refusal, a calculation, the paper's own 41.0 / 41.8 conflict).

| Judge prompt | How it scores | Calibration |
|---|---|---|
| `v2` | one call, the model gives the four scores | missed q19, h08, h12 |
| `v3` | two calls: correctness/completeness without the context, faithfulness without the reference (the context had made it overrule the reference) | still wrong on wording (n02) and refusals |
| **`v4`** (current) | the judge only **lists** the reference's key facts (present / missing / contradicted) and the answer's claims with an **exact quote** from the context; **code** computes the scores | **17 of 17 correctness, 17 of 17 faithfulness** |

Rules that live in code (`eval/judge.py`), each added after a calibration case failed:
- **Correctness = is anything wrong; completeness = how much of the reference is covered.** An incomplete but right answer scores correctness 1, completeness 0.5 (`score_facts`). A Yes/No that is the opposite of the question's `expected_verdict` scores 0.
- **Faithfulness from verified quotes** (`quote_in_context`, `score_claims`): a claim is supported when its quote is really in the context (pipes and spacing ignored), when it is a calculation from numbers the answer states (`calculated_from_answer`: "larger by 31.5 million" = 36 - 4.5), when it is a comparison, or, only for a negative claim, by absence. A claim with no quote gets a second chance: one call asks whether the context states or implies it (`entailed`).
- **Refusals:** the answer model's `found: false` (or the exact "not found" text) is a refusal; a correct refusal of an unanswerable question scores 1, a refusal of an answerable one scores 0.
- **Votes:** each answer is judged 3 times and the median counts (`JUDGE_VOTES`); answers are judged in parallel (`JUDGE_WORKERS`, same cost, about 3 times faster).

New metric: **context precision** = relevant chunks / chunks the model actually read (after the validation's corrective search and the refinement filter). Precision@k always divides by k, so it cannot show a version that sends fewer chunks (`v9`: Precision@5 0.27 against context precision 0.55). Shown in the reports and the Reports page.

New tools: `eval/phase5_final.py` (compare two prompts on golden and holdout with repeats, prints which to keep), `eval/judge_calibration.py`, `eval/fix_refusal_verdicts.py`; `eval/prompt_repeats.py` now uses the configured judge and the real document and page labels of each chunk (it used to label every chunk "page 1", so its prompt was not the pipeline's prompt). The failure classifier labels a "does the resume mention X" question as generation, not retrieval (its key phrase "not mention" can never be in a chunk).

**Lessons:**
- **Compare runs only when they were judged by the same judge in the same pass.** Two identical answers (`q34`, `q37`, same chunks, same text) were scored pass in one run and fail in another; a failure count from two separately judged runs differs by 1 or 2 for reasons that have nothing to do with the system. `eval.rescore` re-judges saved answers in place (answers unchanged); keep the old verdicts (`judgments.before_rejudge.jsonl`).
- **A re-judge can expose a hidden regression.** Re-judging `v10` turned its `q37` pass into a fail; the cause was real (the refinement filter dropped the revision-history chunk) and had been hidden by a lenient first verdict. At the user's request the `v10` reports were restored to their original verdicts (rebuilt from the backup and checked equal to the backed-up summary); `v12` fixes the cause.
- **Repeats beat single runs.** For a prompt change, answer every question 3 to 5 times on identical chunks (`eval.prompt_repeats`) before trusting a difference of one or two questions.

---

## Phase 3 — Chunking experiments (1 week)

**Goal:** Learn how chunking affects retrieval.

**Experiments (one at a time, measure each):**
1. Fixed-size 500 vs 300 tokens
2. Add overlap (10–20%)
3. Recursive splitting (paragraph → sentence)
4. **Structure-aware chunking:**
   - Resumes → by section (Experience, Skills, Education, Projects)
   - Papers → by heading (Abstract, Method, Results…)
   - Specs → by numbered section
5. **Contextual chunk headers** — prepend `doc title > section` to each chunk
6. Keep **tables intact** as one chunk

**Drill:** Find a question that fails because the answer is split across chunks. Fix it.

**Done when:** you can show which strategy wins on recall@k and explain why.

**How to run it:**
- `python -m eval.chunking_experiments` (bge-small stack) and `python -m eval.chunking_experiments --stack openai` (OpenAI embeddings) — chunk the documents each way, score the golden set on a temporary index (the main index is not touched), save every experiment as a run with version `exp`, and write `eval/results/chunking_experiments.md` and `chunking_experiments_openai.md`. Retrieval only, about a minute. Add `--only <name> --full` to also answer with gpt-4o-mini and judge one experiment.
- `python -m eval.chunking_drill` — the drill, using the local model.
- Strategies live in `src/ingest/chunkers.py`: `fixed`, `recursive`, `structure`, plus overlap, headers, intact tables, small-chunk merging (`min_tokens`) and parent-child (`child_size`).
- Each pipeline version is a full settings profile in `src/config.py` (`PROFILES`). To make a new version: add a profile, set `PIPELINE_VERSION`, run `python -m src.ingest.index` (each version gets its own index, so v1 stays on disk; a profile with `INDEX_FROM` reuses another version's index), then `python -m eval.run_eval --name <name>`. `RAG_PIPELINE_VERSION=v1` switches back; the tests use it. Versions so far: `v1` (baseline), `v2` (this phase), `v2-extended` (Phase 4: hybrid retrieval, reranker, gated query transformation), `v3` (Phase 5: `v2-extended` plus the cited prompt; the default).

**v2 is the Phase 3 pipeline:** OpenAI `text-embedding-3-small`, gpt-4o-mini, and structure-aware chunks that vary in size:
- One chunk per section (resume sections, paper headings, spec numbered sections).
- A section over ~500 tokens is split at paragraph, then sentence, boundaries, with 10% (50 tokens) of overlap inside the same section only.
- Every piece starts with the header `doc title > section`; the 500 budget includes it.
- Tables stay whole, up to the embedding model's input limit (a larger one is cut by rows with the header row repeated).
- A piece under ~50 tokens is merged with a neighbour from the same section; a tiny sub-section joins its parent or a sibling, never another top-level section.
- Parent-child was measured and left out.

**Chunking causes (the reference for this phase).** The "In your baseline" column is measured on the v1 chunks by `eval/chunk_diagnostics.py`.

| Cause | Symptom | In your baseline |
|---|---|---|
| Chunk too big, several topics (a hub chunk) | One broad chunk outranks the right one | 7 of 38 chunks span 2+ top-level sections; `spec_c0` outranked the right chunk for 7 questions |
| Several topics glued into a tail chunk | The answer is a small part of the chunk, so its match is weak | `q07`: its chunk is a mixed tail chunk and was not retrieved |
| Answer split across chunk boundaries | No single chunk holds all the key facts | 0 single-fact answers split at 500 tokens, but 13 mid-sentence cuts; at 300 tokens `q03` (the number `0.831` is cut in two) and `q21` split |
| Orphan heading | A heading ends one chunk and its text starts the next | 1 (`spec_c3`) |
| Table cut across chunks | Rows are separated from the table's header row | 2 (the spec's API table, a paper table) |
| Chunk has no context | A chunk does not say which document or section it came from | 14 of 38 chunks |

**Results (built).** 25 answerable golden questions, top-5. Differences of 0.04 are one question, so treat them as hints.

*bge-small stack, retrieval only* (`eval/results/chunking_experiments.md`): no strategy won everywhere. Fixed 500 with 10% overlap was best on Recall@5 (0.96) and Hit@5 (1.00); structure-aware + headers was best on MRR (0.77 against 0.69) and removed the structural defects (mixed chunks 7 to 1, orphan headings 1 to 0, mid-sentence cuts 13 to 0, context-free chunks 14 to 0) and fixed `q07`, but created new misses (`q01`, `q25`). 20% overlap was worse than 10%; smaller chunks and tables-intact did not help retrieval.

*OpenAI stack, retrieval only* (`chunking_experiments_openai.md`): the embeddings alone lift fixed 500 to Recall 0.97, MRR 0.79, Hit 1.00. On top of them, contextual headers are the big gain (structure 0.68 MRR, structure + headers 0.86). Tables-intact, overlap, the 50-token merge (58 to 56 chunks) and parent-child changed nothing measurable (Recall 0.94, MRR 0.86 for every step), because almost every section is under 500 tokens. Recursive splitting alone was worse than fixed 500 (MRR 0.70, 0.75 with overlap). Semantic chunking and fixed 800 were tried and dropped.

*Full evaluation (answers by gpt-4o-mini, judged by gpt-4o-mini):*

| Run | Recall@5 | MRR | nDCG | Hit | Correct | Complete | Faithful | Abstention | p95 |
|---|---|---|---|---|---|---|---|---|---|
| v1 baseline (bge, fixed 500, Qwen) | 0.91 | 0.69 | 0.73 | 0.96 | 0.91 | 0.88 | 0.96 | 0.93 | 184 s |
| OpenAI stack, fixed 500 | 0.97 | 0.79 | 0.82 | 1.00 | 0.93 | 0.96 | 0.95 | 0.96 | 4 s |
| **v2** (OpenAI, structure 500 + overlap + headers + tables + min 50) | 0.94 | 0.86 | 0.85 | 1.00 | 0.86 | 0.88 | 0.96 | 0.89 | 3 s |

- **v2 ranks better but answers worse than the OpenAI fixed-500 control.** The cause is cross-document questions (`q24`, `q25`): every spec chunk now starts with `QuickLeave - Product Specification > ...`, so a question that names QuickLeave is closest to all 11 spec chunks, which fill ranks 1 to 11 and push the resume chunk with the other half of the answer to rank 14. Top-8 and top-10 did not help, and removing the document title from the header made everything worse (MRR 0.76). It needs a retrieval-side fix (Phase 4: hybrid search, a per-document quota, multi-query).
- `q28` (GPT-3) now fails because gpt-4o-mini adds "widely known" facts that are not in the context. That needs the grounding rule in Phase 5.
- v2 sends about half the context (1,087 tokens against about 2,000 per query).
- Caveats: the judge and the generator are the same model; the embedding model, the tokenizer and the LLM all changed at once between v1 and v2, so the OpenAI-stack control is the fair comparison for the chunking.

**Drill:** with 300-token chunks `q03` fails because `0.831` is cut into `0.83` and `1`, and the model answers 0.83. Overlap, recursive splitting and structure-aware chunking each keep the number whole and give 0.831.

### Update (22 Oct): two table fixes in the paper

- **Table repair in normalization** (`src/ingest/normalize.py`, data version `v3.1` in `data/processed/versions/versions.md`). The PDF parser broke Table 2 (BLEU scores): the header came out as fragments (`||BL|EU|Training C|ost (FLOPs)|`, then `|Model|||||` and `||EN-DE|EN-FR|...|` as fake data rows), underscores sat inside numbers (`1_.0·_10^20`) and the Transformer rows split one value over two cells, shifting it into the next column. Nothing said which column was English-to-French, so both the answer model and the judge misread it. Fixes: general rules for underscores in numbers and split cells, and an explicit, documented header fix for this one table (`TABLE_FIXES`). Only chunks `c19` (Table 2) and `c23` (Table 3) changed; same chunk ids; the spec and resume came out byte for byte the same; the index `rag_chunks_v2` was rebuilt (cost: well under a cent).
- **Table chunks headed by their caption** (`table_captions` in `ChunkSpec`, `src/ingest/chunkers.py`, profile key `CHUNK_TABLE_CAPTIONS`; used from `v11`). The parser puts a table under the section that precedes it on the page: Table 2 was headed "5 Training > 5.4 Regularization" and Table 1 "3.4 Embeddings and Softmax". A chunk with table rows is now headed by the caption before its first row, in the chunk itself (Tables 1, 2) or carried over from the end of the previous chunk (Tables 3, 4). Because it changes chunk headers, it got its own index (`rag_chunks_v11`, `INDEX_FROM = "v11"`), so the live versions' index was not touched. Effect: `q16` moved to rank 1 (Table 3); `q18`-`q20` did not move (the reranker scores the Results text and the table almost equally, 0.9999 against 0.996).
- **Lesson:** a shared index must not be rebuilt for an experiment; give the experiment its own `INDEX_FROM`.

---

## Phase 4 — Retrieval improvements (1–2 weeks)

**Goal:** Maximise retrieval recall and ranking quality.

**Add one at a time, measure each:**
1. **Metadata filters** — `doc_type` (resume / spec / paper), doc name
2. **BM25 keyword search** alone — compare with dense
3. **Hybrid search** — dense + BM25 merged with **Reciprocal Rank Fusion (RRF)**
4. **Over-retrieve + rerank** — top 20–30 → cross-encoder reranker → top 5 (tried with top 20 and top 10, and with three open rerankers)
5. **Query rewriting** — LLM turns vague queries into clear ones
6. **HyDE** — embed a hypothetical answer instead of the question *(skipped by choice)*

**Drill:** Query exact terms (a candidate's name, a spec ID). Watch dense search miss them; confirm hybrid fixes it.

**Done when:** you have a table of recall@k / MRR for each technique and know which to keep (consider latency too).

**How to run it:**
- `python -m eval.retrieval_experiments` — every technique on the main index, retrieval only (about 10 minutes; the reranker is slow on a CPU). Each is saved as a run named `p4-<technique>` with version `exp`, and the comparison is written to `eval/results/retrieval_experiments.md`. `--only p4-hybrid --full` also answers and judges it. Re-running a technique replaces its saved run, so a retrieval-only run overwrites an earlier `--full` run of the same name.
- `python -m eval.retrieval_experiments --set rerankers` — retrieve 10, rerank, keep 5, with open rerankers that run locally (`eval/results/retrieval_rerankers.md`, about 40 minutes on a CPU; the first run downloads about 3 GB of models).
- `python -m eval.retrieval_drill` — the exact-term drill (`eval/results/retrieval_drill.md`).
- Code: `src/retrieve/` — `bm25.py` (keyword search with a stopword list), `hybrid.py` (RRF and the per-document quota), `filters.py` (document routing), `rerank.py` (the cross-encoders: `minilm`, `bge-base`, `bge-v2-m3`), `rewrite.py`, and `retriever.py` (a `RetrievalSpec` picks the combination).

**Results (built).** v2 index, top-5, 25 answerable questions; one question is 0.04. *Score* is the mean of MRR and nDCG.

| Technique | Recall@5 | MRR | nDCG | Hit | Score | Model calls | ms/query |
|---|---|---|---|---|---|---|---|
| dense (v2 control) | 0.94 | 0.86 | 0.85 | 1.00 | 0.853 | 0 | 340 |
| filter, answer-key documents (a ceiling) | 0.94 | 0.86 | 0.85 | 1.00 | 0.853 | 0 | 290 |
| filter, model picks the documents | 0.94 | 0.86 | 0.85 | 1.00 | 0.853 | 1 | 1,300 |
| BM25 alone | 0.92 | 0.84 | 0.84 | 0.96 | 0.840 | 0 | 0 |
| **hybrid (dense + BM25, RRF)** | **0.96** | **0.89** | **0.88** | **1.00** | **0.885** | 0 | 360 |
| dense + rerank top 20 (MiniLM) | 0.95 | 0.88 | 0.87 | 1.00 | 0.874 | 0 | 3,600 |
| hybrid + rerank top 20 (MiniLM) | 0.95 | 0.88 | 0.87 | 1.00 | 0.874 | 0 | 3,200 |
| hybrid + rerank top 10 (`bge-reranker-base`) | 0.98 | 0.88 | 0.90 | 1.00 | 0.889 | 0 | 7,300 |
| hybrid + rerank top 10 (`bge-reranker-v2-m3`) | 0.98 | 0.87 | 0.89 | 1.00 | 0.876 | 0 | 25,800 |
| dense + query rewriting | 0.94 | 0.85 | 0.85 | 1.00 | 0.852 | 1 | 1,200 |
| hybrid + model-picked documents | 0.96 | 0.89 | 0.89 | 1.00 | 0.891 | 1 | 1,300 |
| everything (hybrid + documents + rerank + rewrite) | 0.97 | 0.89 | 0.89 | 1.00 | 0.892 | 2 | 4,200 |

*Full evaluation (answers and judge by gpt-4o-mini):*

| Run | Recall@5 | MRR | nDCG | Correct | Complete | Abstention | Cross-doc correct |
|---|---|---|---|---|---|---|---|
| v2 | 0.94 | 0.86 | 0.85 | 0.86 | 0.88 | 0.89 | 0.00 |
| `p4-hybrid` | 0.96 | 0.89 | 0.88 | 0.95 | 0.92 | 0.96 | 1.00 |
| `p4-hybrid-route-llm` | 0.96 | 0.89 | 0.89 | 0.95 | 0.96 | 0.96 | 1.00 |
| `p4-hybrid-rerank10-bge-base` | 0.98 | 0.88 | 0.90 | 0.91 | 0.92 | 0.93 | 0.50 |

(`p4-hybrid` completeness was 0.96 in its first run and 0.92 when I re-ran it: the judge is not fully repeatable, and 0.04 is one question.)

- **Keep hybrid; drop the rest.** Hybrid is the only technique that helps and costs almost nothing (+20 ms, no model call). It moved `q14` and `q15` from rank 2 to rank 1 and, in the full evaluation, fixed `q01`, `q24` and `q25`. BM25 finds "Azure AD" and "audit log" in the resume even though the question is full of spec words.
- **Reranking (MiniLM, top 20)** raised the score by 0.02 (within noise) but hurt multi-section questions (MRR 0.75 to 0.61) and added about 3 s per query on a CPU. **Query rewriting** alone did nothing (0.852 against 0.853) and hurt cross-document ranking. **Document filtering** did nothing, even with the answer key: the problem was never searching the wrong document, it was ranking a weak chunk of the right one. The model-picked filter adds a model call (about 1 s) for the same score.
- **Better open rerankers, retrieve 10 and keep 5** (`retrieval_rerankers.md`; OpenAI has no reranking model, so these run locally and no text leaves the machine). Hybrid first gives the 10 candidates. `bge-reranker-base` gave the best ranking score of the three (nDCG 0.90, Recall 0.98) and `bge-reranker-v2-m3`, the larger one, was no better (MRR 0.87), at 25 s per query. Neither beat plain hybrid on MRR (0.89), and retrieving 10 instead of 20 made no difference for MiniLM (identical results).
- **Why the reranker is not kept:** in the full evaluation `hybrid + bge-reranker-base` scored correctness 0.91 against 0.95 for plain hybrid, because it broke `q25`. Its retrieval metrics for `q25` look perfect (Recall 1.00), but its top 5 held only spec chunks, so the model had no resume chunk and answered "the context does not contain information about the candidate". The relevance label for `q25` names only the spec chunks, so it cannot see this. The reranker prefers chunks that closely match the question's wording (the spec), and cross-document questions need a chunk from each document. The cost is 7 s per query on this CPU (about 0.1 s on a GPU), against 0.3 s for hybrid.
- **Table caveat:** `bge-reranker` reads at most 512 of its own tokens, while a chunk can be 500 `cl100k` tokens, and numeric tables use more tokens in the reranker's tokenizer, so long tables are probably truncated before scoring. All rerankers still rank `q16`'s table lower than hybrid does, but I did not test truncation directly.
- **BM25 needed a stopword list.** Without it, `q01` ("What is the candidate's current job title and company?") matched unrelated chunks on filler words, and hybrid lost that question entirely (its MRR went from 1.00 to 0.00) and gained nothing overall. With the list, hybrid improves instead (MRR 0.86 to 0.89).
- **Drill:** with the OpenAI embeddings and headers, dense already finds 6 of 7 exact terms (name, spec ID, error code, certification, parameter name, requirement ID); it misses `0.831`. BM25 finds all 7, and hybrid ranks `0.831` third. The drill is milder than the plan expects because v2's dense search is strong.
- **Caveats:** only three questions changed in the full evaluation, so treat the +0.09 correctness as encouraging, not proven. The judge and the generator are the same model. `q25`'s relevance label (its only key fact is "audit log", which the spec chunks satisfy) never marks the resume chunk as relevant, so its recall is understated; `q28` (an unanswerable question) is still answered with outside facts, which is a Phase 5 grounding problem.

**Harder golden questions (50-question re-run).** The 28 questions were too easy to separate the rerankers: hybrid already ranked the right chunk first on 20 of 25. I added 22 questions (`q29`–`q50`, tier `hard`) written from the documents: paraphrases that share few words with the answer, version and conflict traps (the paper says 41.0 BLEU in the text and 41.8 in Table 2), negation and comparison over the leave table, two- and three-chunk answers, cross-document questions, and three near-miss unanswerable questions. Every key fact was checked to appear in its source. Runs are `v2-50q` (the v2 dense pipeline on all 50) and `p4-*-ext`.

| v2 pipeline (dense), n | Recall@5 | MRR | nDCG | Hit | Correct | Complete | Abstention |
|---|---|---|---|---|---|---|---|
| standard, 28 | 0.94 | 0.86 | 0.85 | 1.00 | 0.88 | 0.88 | 0.89 |
| hard, 22 | 0.85 | 0.88 | 0.82 | 0.95 | 0.73 | 0.76 | 0.82 |

The standard tier reproduces the earlier v2 run (retrieval numbers identical), so the setup did not drift. The hard tier is harder for recall and answers, not for MRR: most of its questions still find a relevant chunk at rank 1, but the multi-chunk and cross-document ones miss part of the evidence. Its worst types are cross-document (Recall 0.50, correctness 0.00) and multi-section (Recall 0.79).

| Technique, retrieval only (R / MRR / nDCG) | All 50 | Standard 28 | Hard 22 | ms/query |
|---|---|---|---|---|
| dense | 0.90 / 0.87 / 0.84 | 0.94 / 0.86 / 0.85 | 0.85 / 0.88 / 0.82 | 360 |
| hybrid | 0.95 / 0.90 / 0.88 | 0.96 / 0.89 / 0.88 | 0.93 / 0.92 / 0.88 | 370 |
| hybrid + MiniLM rerank, top 10 | 0.96 / 0.93 / 0.91 | 0.95 / 0.88 / 0.87 | 0.97 / **1.00** / **0.96** | 1,800 |
| hybrid + `bge-reranker-base`, top 10 | **0.98** / 0.92 / **0.92** | 0.98 / 0.88 / 0.90 | 0.97 / 0.97 / 0.96 | 10,600 |

| Full evaluation, correctness / completeness / abstention | All 50 | Standard 28 | Hard 22 | Cross-doc (4) |
|---|---|---|---|---|
| v2 (dense) | 0.81 / 0.83 / 0.86 | 0.88 / 0.88 / 0.89 | 0.73 / 0.76 / 0.82 | 0.00 |
| hybrid | 0.88 / 0.91 / 0.92 | **0.95** / 0.94 / 0.96 | 0.80 / 0.87 / 0.86 | **0.75** |
| hybrid + MiniLM | 0.88 / 0.89 / 0.94 | 0.86 / 0.88 / 0.89 | **0.91** / 0.89 / 1.00 | 0.38 |
| hybrid + `bge-reranker-base` | 0.90 / 0.89 / 0.96 | 0.91 / 0.90 / 0.93 | 0.89 / 0.87 / 1.00 | 0.63 |

- **A reranker does raise MRR once the questions are harder**, as expected. On the hard tier hybrid's MRR goes 0.92 to 1.00 and nDCG 0.88 to 0.96 (three questions, `q29`, `q31` and `q44`, move to rank 1), while on the standard tier it is flat or slightly lower. On the 28 easy questions there was nothing to fix.
- **The ranking gain does not become a reliable answer gain.** Overall correctness is 0.88 for hybrid and for hybrid + MiniLM and 0.90 for hybrid + `bge-reranker-base` (one question). Rerankers help the hard tier (0.80 to 0.91 and 0.89) but lose on the standard tier (0.95 to 0.86 and 0.91), and they hurt the cross-document questions (0.75 to 0.38 and 0.63): they favour the chunk that matches the question's words, so the second document's chunk drops out. Most of the hard-tier correctness gain comes from two of the three near-miss unanswerable questions (`q48`, `q50`) and from `q47`, which flip with small changes in the retrieved context, so on 22 questions treat it as a hint.
- **Cost:** the reranker adds 1.5 s (MiniLM) or 10 s (`bge-reranker-base`) per query on this CPU.
- **Top 20 instead of top 10 for the reranker (50 questions, hybrid pool):** no meaningful change. MiniLM retrieval is identical (Recall 0.96, MRR 0.93, nDCG 0.91; correctness 0.87 against 0.88, one question, `q34`). `bge-reranker-base` went from MRR 0.92 / nDCG 0.92 to 0.91 / 0.915, correctness stayed 0.90, abstention rose 0.96 to 0.98 and cross-document correctness 0.63 to 0.75 (`q25`, one question). The pool does not matter because hybrid's Recall@10 and Recall@20 are both 0.98 on the standard questions: every chunk the reranker can use is already in the top 10, and `q23`'s missing chunks are in neither. The cost is higher: retrieval takes 3.2 s (MiniLM) and 15.2 s (`bge-reranker-base`) per query against 1.8 s and 10.6 s. The `latency_p50_s` of the top-20 runs is not comparable (the runner now reuses the search for the answer, so it leaves the reranking time out); use `ms/query`.
- **Decision:** hybrid stays the pick. A reranker is worth revisiting with a GPU or a larger corpus, and only together with a per-document quota applied after reranking so cross-document questions keep a chunk from each document. That was not tested.
- **Question-quality notes:** `q44` ("the fine-tuned model") does not name the project, so dense search retrieves paper chunks; it is a fair test of a vague query, but ambiguous. The judge scored `q48` (a correct "not in the spec" answer that also lists the real leave types) as not abstaining, which is a strict reading of the abstention rule, the same as for `q27`. The labels are still heuristic.

**v2-extended is the Phase 4 pipeline (the retrieval part of the app's current version, v3).** It is v2 with a better retrieval step and nothing else changed: the same chunks, the same index (`rag_chunks_v2` is shared, so nothing is rebuilt), the same OpenAI embeddings and gpt-4o-mini. Each query (1) searches dense and BM25 (with a stopword list), (2) merges the two rankings with Reciprocal Rank Fusion and keeps the 10 best, (3) reranks those 10 with the open cross-encoder `BAAI/bge-reranker-base` (it runs locally, nothing leaves the machine) and (4) keeps the best 5, and (5) for the hard questions only (the ones that name two documents), first plans the search with a model call: query rewriting, expansion and multi-query (see below). Document filtering and query rewriting on every question were measured and left out. In the profile, `RERANK_TOP = 0` gives hybrid search without the reranker (the 10 s per query then drops to 0.3 s), `RERANKER` picks `minilm` (much faster), `bge-base` or `bge-v2-m3`, and `QUERY_TRANSFORM = "none"` turns the transformation off (`"always"` applies it to every question, which only costs time). The settings are recorded in every trace and run (`retrieval_mode`, `rerank_top`, `reranker`, `query_transform`), and `RAG_PIPELINE_VERSION=v2` or `v1` switches back. The API loads the index and the reranker in the background at start-up, so the first question does not wait for the model.

| Run (all on the same index and answer model unless noted) | n | Recall@5 | Precision@5 | MRR | nDCG | Correct | Complete | Abstention |
|---|---|---|---|---|---|---|---|---|
| v1 baseline (bge, fixed 500, Qwen) | 28 | 0.91 | 0.22 | 0.69 | 0.73 | 0.91 | 0.88 | 0.93 |
| v2 (dense) | 28 | 0.94 | 0.24 | 0.86 | 0.85 | 0.86 | 0.88 | 0.89 |
| `v2-50q` (v2, dense) | 50 | 0.90 | 0.24 | 0.87 | 0.84 | 0.81 | 0.83 | 0.86 |
| `v2-extended-hybrid` (hybrid only, no reranker) | 50 | 0.95 | 0.25 | 0.90 | 0.88 | 0.86 | 0.89 | 0.92 |
| `v2-extended-rerank` (hybrid + reranker, no transformation) | 50 | 0.98 | 0.27 | 0.92 | 0.92 | 0.88 | 0.88 | 0.96 |
| **`v2-extended` (hybrid + reranker + gated query transformation)** | 50 | **0.99** | **0.27** | 0.91 | **0.92** | **0.92** | **0.92** | **0.98** |

Precision@5 is capped by the questions: 30 of the 44 answerable questions have one relevant chunk, so a search that is perfect on every question still scores only about 0.28 (0.26 to 0.27 for the earlier 28-question runs); `v2-extended` reaches about 95% of that (0.27 against 0.28). The rest is the cost of sending 5 chunks when the answer needs one or two (about 1.3 of the 5 are relevant). Sending fewer chunks (k=3 raises the cap to about 0.37) or cutting weak chunks after reranking (Phase 5) would raise it; neither is tested.

By tier, `v2-extended` against `v2-50q` (dense): standard 28 Recall 0.98 against 0.94, nDCG 0.88 against 0.85, correctness 0.95 against 0.88; hard 22 Recall 1.00 against 0.85, MRR 0.97 against 0.88, nDCG 0.98 against 0.82, correctness 0.89 against 0.73; cross-document correctness 1.00 against 0.00. Against hybrid alone (`v2-extended-hybrid`) the reranker raised ranking (nDCG 0.88 to 0.92) and abstention (0.92 to 0.96) and cost about 11 s per query, but correctness changed by one question; the query transformation then fixed the cross-document questions the reranker could not (0.63 to 1.00). The retrieval numbers of the reranker-only run match the `p4-hybrid-rerank10-bge-base-ext` experiment exactly. The generator and judge vary by one or two questions from run to run (the same retrieval scored correctness 0.88 to 0.90).

All `v2-extended` metrics (run `v2-extended`, the Reports tab; retrieval is scored on the 44 answerable questions, answers on all 50):

| Group | Metric | Value |
|---|---|---|
| Retrieval | Recall@5 | 0.99 |
| | Precision@5 | 0.27 (about 1.4 of the 5 chunks are relevant; the cap is about 0.28) |
| | MRR | 0.91 |
| | nDCG@5 | 0.92 |
| | Hit@5 | 1.00 |
| Answers | Correctness | 0.92 |
| | Relevance | 1.00 |
| | Completeness | 0.92 |
| | Faithfulness | 0.97 |
| | Abstention accuracy | 0.98 |
| | Refusal on unanswerable | 0.83 (5 of the 6 unanswerable questions were refused) |
| | False abstention (answerable) | 0.00 |
| Speed and size | Latency p50 / p95 | 9.1 s / 11.5 s (most of it is the reranker on this CPU) |
| | Tokens per query | 1,170 in, 49 out (plus about 580 in / 90 out for the 8% of questions that are transformed) |

### Query transformation for the hard questions only

The user's rule: transform a question only when it is hard, because every transformation costs a model call and time. A simple question must take exactly the path it took before.

**Which questions are hard.** The gate is a free rule with no model call (`src/retrieve/transform.py`): the question names two or more documents ("QuickLeave requires ... Does the candidate ..."). Document names come from the corpus: title words that no other document uses (`quickleave`, `satwika`) and words that name a kind of document (`candidate`, `specification`). It fires on exactly four of the 50 questions (`q24`, `q25`, `q46`, `q47`), and on none of the other 46. I tried two other gates on the golden set and dropped them: "a question word appears nowhere in the documents" fires on two thirds of the questions (mostly word endings such as "carried"), and "the reranker is not confident" never fires on a weak question, because a question that finds only half of its evidence still has a confident top result (`q46` scored 0.64). The gate was designed after seeing these four failures, so the numbers below are optimistic for new cross-document questions.

**What happens to a gated question.** One model call (gpt-4o-mini, JSON) plans the search with the three techniques: a rewrite (a clear, self-contained query in the documents' own words), expansion (up to 3 acronyms or synonyms such as "role-based access control" to "RBAC", added to each query) and multi-query (one query per document the question needs, each with the document id from the catalog). Each sub-query is searched inside its own document (a metadata filter, so the document's name never has to appear in the query text), reranked on a small pool against its own query (the reranker reads the same number of chunks as a normal search), and the results are merged round-robin so every document gets 3 places (6 chunks for two documents, one more than a normal answer; this was 3 + 2 of 5 until a replay of the 9 recorded plans showed the evidence chunk third in its document's list in the two plans that failed, and 3 places each fixed both: 7 of 9 plans became 9 of 9. A larger candidate pool did not help, and also searching each document with the question itself changed nothing). If the model gives no per-document queries but the question names two documents, each named document is searched with the rewrite. A single-need plan (used with `QUERY_TRANSFORM = "always"`) searches the question and its rewrite, merges them and reranks by the rewrite. If the model fails or returns something unusable, the question is searched as it is. The trace records why the question was transformed and the plan (`retrieve` span, `transform`).

**Why `q46` failed.** The question says "role-based access control"; the resume says only "four-tier RBAC", buried at the end of a 1,070-character chunk, so BM25 shared one word with it ("built") and dense search ranked it 14th. The question's first sentence is about the spec, so spec chunks filled the list. A reranker cannot recover a chunk that is not among its 10 candidates. Dropping the spec sentence put it at rank 8 and using the acronym brought BM25 to rank 6.

**Results (50 questions, retrieval only, then the full evaluation).**

| Retrieval (R / MRR / nDCG) | All 50 | Hard 22 | Model calls per query | ms per query |
|---|---|---|---|---|
| hybrid | 0.95 / 0.90 / 0.88 | 0.93 / 0.92 / 0.88 | 0 | 370 |
| hybrid + transform on every question | 0.96 / 0.89 / 0.88 | 0.96 / 0.93 / 0.90 | 1.00 | 2,100 |
| hybrid + gated transform | 0.95 / 0.89 / 0.88 | 0.93 / 0.92 / 0.88 | 0.08 | 490 |
| hybrid + bge-reranker | 0.98 / 0.92 / 0.92 | 0.97 / 0.97 / 0.96 | 0 | 10,600 |
| hybrid + bge-reranker + gated transform | 0.98 / 0.91 / 0.92 | 1.00 / 0.97 / 0.98 | 0.08 | 9,100 |

- **Transforming every question costs a lot and gives nothing extra**: 2.1 s per query against 0.37 s, and the same ranking scores. The gate keeps the benefit and spends 0.08 model calls per query.
- **Exactly the four gated questions changed.** The other 46 return identical top-5 lists, and their answers' costs did not change. The 4 plans cost about 580 tokens in and 90 out each (2 300 in / 400 out in a run where the answers used 58 500 in / 2 400 out) and 1.3 to 2.8 s each; the transformed questions were not slower overall (7.3 s against 8.5 s), because each sub-query reranks a smaller, filtered pool.
- **Before the change `q25` and `q46` had no resume chunk in the top 5; afterwards all four cross-document questions reach both documents.** Cross-document correctness went from 0.63 to 1.00 and overall correctness from 0.88 to 0.92 (`q25` 0 to 1, `q46` 0.5 to 1; a third question, `q35`, moved 0 to 0.5 with identical retrieval, which is generator noise). Abstention is 0.98 and false abstention 0.00.
- **The aggregate retrieval scores hardly move** (MRR 0.92 to 0.91): four questions are 8% of the set, and `q25`'s relevance label lists only spec chunks, so a resume chunk that holds the answer counts as a miss (its recall falls from 1.0 to 0.5 while its answer becomes correct). The document-coverage check and the answers are the fair measure here.
- **Caveats:** only four questions are gated, so the +0.04 correctness is two questions' worth and the judge and the generator are the same model. The model's plans vary a little between runs (Recall 0.98 in the experiment table above, 0.99 in the full evaluation, which asked the model again). Questions that need two parts of the same document (such as `q23`) and vague questions (`q44`) are not gated; no cheap rule I tried separated them from the simple questions that also have two clauses.

### Update (22 Oct): precision and ranking (`v9`, `v11`)

**Precision (`v9` = `v8` + a cut-off after reranking).** Most golden questions (31 of 44) have one relevant chunk, so always sending 5 chunks caps Precision@5 near 0.29; `v8` was at 0.28, at the ceiling. The only lever is to send fewer chunks when fewer are needed. `cut_low_scores` (`src/retrieve/retriever.py`, `RERANK_MIN_RATIO = 0.02`) drops reranked chunks scoring under 2% of the best one, with two guards found in the simulation:
- a per-document search (multi-query) is cut within its own list: scores reranked against different sub-queries are not comparable;
- a cross-document question keeps the best 2 chunks of each named document: the reranker scores real evidence very low there (`q24`: the Azure AD SSO chunk had 2.2% of the top score).

How it was chosen: simulated for free on saved runs, with a stronger check than relevance labels: would the cut drop a chunk the answer **cited**? 2% dropped none that a correct answer needed; 5% dropped `q37`'s evidence, 10% `q16`'s. Result: context precision 0.28 to 0.55 (golden), 0.25 to 0.51 (holdout), 17-22% fewer input tokens, answers unchanged on the holdout; `q38` improved (three distracting chunks removed). Two bugs found on the way: my first simulation capped every list at 5 and so "lost" the 6th chunk of cross-document questions, and the report's Precision@5 could not show the gain (see context precision, Phase 2).

**Ranking (`v11`).** `v10`'s golden MRR / nDCG (0.90 / 0.89) were lower than the holdout's (0.97 / 0.94) for every version since `v7`: the ranking code had not changed, the golden set simply has more hard ranking cases. The 8 golden misses had three causes: table chunks headed by the wrong section (`q16`, `q18`-`q20`; fix in Phase 3), cross-document questions whose interleaved list started with the spec although the question asks about the candidate (`q46`, `q47`), and a generic resume skills chunk ranked first (`q24`, `q29`; `q24` was a label problem, fixed in Phase 0). Fix: `asked_first` (`ASKED_FIRST`) lists the document named in the question's last sentence first ("Does the **candidate** have ..." asks about the resume). Simulated first with a retrieval-only run (a few cents): golden MRR 0.917 to 0.936, holdout unchanged. Full run: golden MRR 0.900 to 0.951, nDCG 0.894 to 0.928 (`q16`, `q24`, `q46`, `q47` to rank 1), holdout unchanged at 0.974 / 0.944, answers the same within noise. Not fixed: `q18`-`q20`. Not tried: down-weighting the skills chunk (it would tune to `q29`).

**Wording matters more than ranking for the resume.** "What is User's current job?" puts QuickLeave chunks first ("Users and Roles"); "the candidate's" or "Satwika's" puts the right resume chunk first. Even then resume chunks score low (best 0.02 with the name), because the reranker scores short tables and bullet points conservatively, so their CRAG grade is often "partial" while the retrieval is right.

---

## Phase 5 — Prompting + generation (1 week)

**Goal:** Make answers grounded, cited and honest.

**Build:**
- **Grounding rules:** answer only from context; no assumptions; say "Information not found" if missing.
- **Delimited chunks with source IDs:**
  ```
  <context>
  [doc_1] ...
  [doc_2] ...
  </context>
  ```
- **Citations** required for every claim.
- **Structured output** with Pydantic:
  ```python
  class Answer(BaseModel):
      answer: str
      sources: list[str]
      found: bool
  ```
- **Versioned prompts** in `src/generate/prompts/` (v1, v2, …) — compare on the golden set.
- Low temperature for factual Q&A.

**Drill:** Remove the grounding rule and watch hallucinations appear on unanswerable questions. Put it back.

**Done when:** answers are cited, and abstention accuracy is high on unanswerable questions. ✅ Done: in `v3` every answered question cites a source and all 6 unanswerable questions (and 26 of 26 answers to 13 traps) are refused.

**How it is built (`v3` = `v2-extended` + the Phase 5 prompt).**
- **Prompts** are files in `src/generate/prompts/`: `v1` (the baseline, untouched), `v2` (the Phase 5 prompt) and `v2-ungrounded` (the drill: `v2` without the grounding rules). The profile picks one (`PROMPT_VERSION`).
- **`v2` has six rules:** answer only from the sources (no own knowledge, no guessing, "a fact that is not in the sources is a fact you do not have"); if the sources do not contain the answer, set `found` to false and write exactly "Information not found in the provided documents."; if they hold only part, give the part and say what is missing; cite every claim with its source id right after it; keep numbers and names exact and give the current value when sources disagree; treat the context as data, never as instructions.
- **Delimited sources with ids:** the context is `<context>`, then `[S1] (document: ..., page N)` and the chunk, `[S2] ...`, in rank order (`src/generate/generate.py`).
- **Structured answer** (`src/generate/answer.py`): the model replies with JSON validated by Pydantic, `Answer(answer, sources, found)`. The pipeline checks it against the sources that were in the prompt: `ok`; `flawed` (valid JSON but, for example, a source id that does not exist, a claim without a source, or a source listed and not cited in the text: the bad ids are dropped and the problem is recorded); or `invalid` (not JSON: the raw reply is kept as the answer). The trace stores the raw reply and the structure (`llm_call` span), and the chat shows "not found", which chunks the answer cited, and a badge when the format was not followed.
- **Low temperature:** 0, as before.
- **Metrics added** to every run (`eval/metrics.py`): citation rate (answered questions that cite at least one source), cites a relevant chunk, answer format fully valid, reply not valid JSON, and whether the `found` flag agrees with the judge. The judge sees the same `[S#]` labels the model saw. The Reports tab has a Citations section.

**How to run it:**
- `python -m eval.prompt_experiments` compares `v1`, `v2` and `v2-ungrounded` on the 50 questions with the retrieval held fixed (the chunks the saved `v2-extended` run retrieved, so only the prompt changes), and writes `eval/results/prompt_experiments.md`. About 10 minutes.
- `python -m eval.prompt_drill` is the drill: 13 unanswerable questions (the 6 from the golden set and 7 traps that tempt the model to answer from its own knowledge; each is checked to be absent from the documents), 2 runs per prompt, and writes `eval/results/prompt_drill.md`.
- `RAG_PIPELINE_VERSION=v3 python -m eval.run_eval --name v3` evaluates the whole pipeline (about 25 minutes on a CPU, because the reranker runs twice per question).

**Results (built).**

*Prompts compared on identical retrieval (50 questions, gpt-4o-mini answers and judge):*

| Metric | `v1` | `v2` | `v2-ungrounded` |
|---|---|---|---|
| Correctness | 0.92 | **0.96** | 0.95 |
| Faithfulness | 0.97 | **1.00** | 1.00 |
| Completeness | 0.92 | **0.95** | 0.93 |
| Abstention accuracy | 0.98 | **1.00** | 1.00 |
| Refusal on the 6 unanswerable | 0.83 | **1.00** | 1.00 |
| False abstention | 0.00 | 0.00 | 0.00 |
| Citation rate | - | 1.00 | 1.00 |
| Cites a relevant chunk | - | 0.98 | 0.95 |
| Answer format fully valid | - | 0.96 | 0.82 |
| Input tokens per query | 1,170 | 1,562 | 1,414 |

*The drill: what the grounding rules prevent (13 unanswerable questions, 2 runs each, same chunks):*

| Prompt | Refused | Hallucinated (answered and not faithful) |
|---|---|---|
| `v1` (baseline) | 16 of 26 (62%) | 8 |
| `v2-ungrounded` | 17 of 26 (65%) | 7 |
| **`v2`** | **26 of 26 (100%)** | **0** |

Without the rule the model answers from its own knowledge about a third of the time: "The capital of France is Paris", "BERT-base has 12 layers", a salary range for an AI engineer in India, and, for `q28`, "the documents do not say, however it is widely known that GPT-3 has 175 billion parameters". With the rule it refused every one.

*The whole pipeline (`v3`, run on all 50 questions) against the earlier versions:*

| Run | Recall@5 | MRR | nDCG | Correct | Complete | Faithful | Abstention | Refusal on unanswerable | Latency p50 |
|---|---|---|---|---|---|---|---|---|---|
| `v2` dense (`v2-50q`) | 0.90 | 0.87 | 0.84 | 0.81 | 0.83 | 0.98 | 0.86 | 0.67 | 1.5 s |
| `v2-extended` | 0.99 | 0.91 | 0.92 | 0.92 | 0.92 | 0.97 | 0.98 | 0.83 | 9.1 s |
| **`v3`** | 0.97 | 0.91 | 0.91 | **0.97** | **0.98** | **1.00** | **1.00** | **1.00** | 11.6 s |

`v3` also has citation rate 1.00 (every answered question cites a source), cites a relevant chunk 0.96, answer format fully valid 0.98 (one answer, `q05`, lists two sources without citing them in the text), reply not valid JSON 0.00, the `found` flag agreeing with the judge 1.00, false abstention 0.00, and 1,566 input and 47 output tokens per query. By tier: standard correctness 0.95, hard 0.98; cross-document correctness 0.75.

**What I found.**
- **The grounding rule does the work.** Adding sources, citations and JSON without it (`v2-ungrounded`) gave no protection on the drill (7 of 26 hallucinated, against 8 for the baseline), and the rules take it to 0 of 26. On the 6 unanswerable golden questions alone the effect is hard to see: the baseline answered 1 (`q28`) and the ungrounded prompt none, so 6 questions cannot show what the rule prevents. That is why the drill has 13.
- **I found two problems in my own experiment.** My first drill run showed `...` as the answer for two questions: the model copied the placeholder from the JSON example (`"answer": "..."`) when it had nothing to say, and the judge read it as a refusal or an answer by chance. A descriptive placeholder (`<your answer, with [S#] citations>`) fixed it, and a test now guards it. The second problem was that retrieval changed between two `v3` runs: the model plans the search for cross-document questions differently on every run, and in the first `v3` run the wording of two plans left the evidence chunk outside the answer (cross-document correctness fell from 1.00 to 0.63). Replaying the 9 recorded plans showed the evidence third in its document's list, so each document now gets 3 places instead of 3 and 2 (9 of 9 plans work, against 7 of 9); that run is kept as `v3-before-slots`.
- **The rules cost context:** about 390 more input tokens per query (1,170 to 1,562, +33%) for the rules and the source labels. Output tokens do not change.
- **The four generation failures in `v3` (`q01`, `q04`, `q19`, `q25`) are all incomplete answers, not wrong ones** (correct, faithfulness 1.0), and in every one the missing detail was in the chunks the model read: the employment timeline (`q01`), the LLM-judge fallback (`q04`), the conflicting 41.0 next to 41.8 (`q19`) and the spec's FR-10 with the AI ATP Toolkit (`q25`). The rule "keep the answer short and exact" shortened the answers a lot (average 219 characters with the baseline prompt, 117 with `v2`), and the rule about sources that disagree was not followed for `q19`. `q25` also attached a feature to the wrong item: its resume bullet lists GPT-4 call instrumentation, SSO, RBAC and audit logging, and the answer says the GPT-4 calls were instrumented "with audit logging". So the prompt was part of the problem. I wrote `v2.1` (`src/generate/prompts/v2.1.txt`): answer every part and give every detail, use evidence from each document for a question that combines documents, report each value when sources disagree, attach a fact only to its own item, "complete and exact is better than short". On the same chunks (`python -m eval.prompt_experiments --source v3 --only v2,v2.1`) it fixed `q04` (completeness 0.5 to 1.0) but not `q01` or `q19`, hurt `q24` (completeness 1.0 to 0.5) and `q44` (correctness 1.0 to 0.5), and overall changed nothing: correctness 0.95 and 0.95, completeness 0.93 and 0.93, faithfulness 0.97 and 0.98, with 11% more input tokens, answers 22% longer (116 to 142 characters) and a drop in fully valid format from 0.98 to 0.92. It was not adopted. Two of the four are not prompt problems: `q01` asks only for the current title and company, and its reference answer adds the employment timeline; `q19`'s reference adds a section-6.1 discrepancy the question does not ask about. The judge scores a missing reference fact as incomplete. `q46` is a retrieval miss (the RBAC chunk was not in that run's plan), and the model reasons from the skills list.
- **Splitting the golden answers (`required_answer` and `optional_details`) barely changed the scores.** The reference of 26 answerable questions mixed what the question asks with extra facts it does not ask for (IDs, dates, histories, step counts). I split those 26 (the rule was written before editing: required = the part of the fact that answers what the question asks; optional = everything else; when in doubt it stays required, so `q04` and `q34` are unchanged). The judge (prompt `v2`) grades against the required part and shows the optional details as "do not require, do not penalise". `python -m eval.rescore --source v3-judge-v1 --name v3-rescored` scored the saved answers of the first `v3` run again, judging only the 26 changed questions and keeping the other 24 verdicts, so retrieval and judge noise stay out of the comparison. Result: correctness 0.96 to 0.97, completeness 0.943 to 0.955, one verdict changed (`q01`, 0.5 to 1.0). `q04` stays incomplete by design, `q25` misattributes the logging, `q46` is a retrieval miss, and `q19` stays 0.5 because the judge reads the conflicting 41.0 chunk as a contradiction of the correct 41.8 answer. So the failures were mostly real, not an artifact of the reference answers. A rise from rescoring is a change in the scoring and not an improvement of the system; the other versions were not rescored because `v3` showed no real change. **Judging all 50 with prompt v2** (`--rejudge-all`, every verdict `judge_prompt: v2`) gave correctness 0.97, completeness 0.977, faithfulness 1.00: three verdicts moved against the first judging (`q01`, `q04`, `q19`). `q04` has no split, so its prompt text is the same as before; its change from 0.5 to 1.0 is judge noise, not the golden fix, and `q19` is split-related only in part (the same answer scored 0.5 in `v3-rescored`). So about 0.02 of completeness is judge noise on a single re-judge. **This run is now the current `v3`** (`eval/results/v3`); the first judging is kept as `v3-judge-v1` (correctness 0.96, completeness 0.94). Remaining failures: `q19` (correctness 0.5), `q25` and `q46`.
- **Going back to 3 + 2 places per document did not help** (run made and deleted, not kept). With 3 + 2 the retrieval changed on exactly the four gated questions: `q46` was fixed (the RBAC chunk was found, correct 1.0) but `q47` broke (the skills chunk that names REST API design was dropped, so the answer was "not found", a false abstention), and correctness went 0.97 to 0.96. Neither setting wins; which of the two questions is missed depends on how the model words the plan, so `v3` keeps 3 each.
- **Citations are checked only for form:** the checks confirm that every cited id exists, that a source is cited in the text, and that a relevant chunk is among the cited ones; they do not verify that each claim is supported by the chunk it cites. That is the validation layer of Phase 6.
- **Caveats:** the judge and the answer model are the same model; one question is 0.02 of a score. `v3` has Recall 0.97 against 0.99 for `v2-extended` although the retrieval code is the same: only the four gated questions can differ (the other 46 retrieve identical chunks), and two of them did. `q46` missed the RBAC chunk because that run's plan was worded unluckily, and `q25` found the resume's two audit-logging chunks but not `spec_c6`, the spec chunk its label lists (the label rewards only spec chunks). Recall over all 6 chunks the model read is the same 0.97, so the sixth chunk is not the explanation. Replaying 12 recorded plans and 24 fresh ones, the current setting (3 places per document, reranked) finds evidence from both documents in 35 of 36 plans, so about 1 in 36 plans misses, and the difference between two runs is that kind of luck on 4 questions. A setting that skips the reranker inside each document scored 12 of 12 on the recorded plans but only 21 of 24 on fresh ones (the recorded plans were the ones used to find the problem), so it was not adopted.

### Update (22 Oct): prompts `v2.6`-`v2.12` and the answer format

**Why the earlier prompts were set aside:** with the gpt-4o judge, `v2` scored as well as `v2.5` or better on both sets (golden 0.967 against 0.960; old holdout 0.889 against 0.883, 5 repeats). The rules of `v2.1`-`v2.5` had each been written to fix one golden question.

| Prompt | Idea | Outcome |
|---|---|---|
| `v2.6`-`v2.8` | an `evidence` list written before every answer | fixed `h03` (0/5 to 5/5) but dropped facts everywhere else (golden 0.90): the model extracts evidence lossily, then says the rest "is not specified" |
| `v2.9` (`v7`) | `v2` + a `checks` scratchpad only for every/all/both/at-least questions + "No, the resume does not mention X" for yes/no questions + plain-text answers + a citation per sentence | beat `v2` on both sets (golden 0.930 vs 0.907, fresh holdout 0.904 vs 0.860, 3 and 5 repeats); a smoke test showed the checks list is often left empty |
| `v2.10` | "be concise, but give every fact with its numbers and name its item" instead of "keep it short", + attribution checks for which-item questions | fixed `n07` (94% vs 100% in one sentence) and `n16` (an extra wrong project); listed only some rows of `q38` |
| **`v2.11`** (`v8`-`v12`) | `v2.10` with "copy every sentence or table row that answers it" | holdout 0.920 to 0.968, golden 0.973 to 0.990 against `v2.9` (5 and 3 repeats); `q38` still unstable |
| `v2.12` | superlatives (highest, lowest, ...) added to the checks | tried on `n12` with the real model (2 of 3, as before) and left out: a prompt change needs a full eval |

**Answer format fixes** (`src/generate/answer.py`): the reply's scratch fields (`checks`, `evidence`) accept any shape, because a `checks` list of lists made the whole reply invalid and the raw JSON was shown as the answer (`q34` in the `v9` run; it could hit the live app); a reply whose answer is itself JSON is flagged.

**Still open (no prompt fixed them):** `q04` (leaves out the LLM-judge fallback in every version), list answers that stop early (`n13`), and table reading (`n12` about 4 of 5, `q35`). The next lever is probably the model, not another rule.

---

## Phase 6 — Validation layer (1 week)

**Goal:** Catch bad answers before the user sees them.

**Build:**
1. **Relevance threshold** — if the best reranker score < cutoff, don't call the LLM; return a fallback. Tune the cutoff on the golden set.
2. **Context sufficiency check** (optional) — small LLM call: "Does this context answer the question? Yes/No."
3. **Citation check** — every cited ID must exist in retrieved chunks.
4. **Faithfulness check** — LLM-as-judge or RAGAS: is each claim supported?
5. **Corrective retry** — if checks fail: rewrite query → retrieve again (max 1–2 retries) → else abstain.

**Done when:** the system refuses correctly on unanswerable questions without refusing answerable ones (track false abstentions).

### Phase 6 results (version `v4` = `v3` + the validation layer + 7 reranked candidates instead of 10; `v5` = `v4` + the `v2.4` answer prompt; `v6` = `v5` + the `v2.5` answer prompt; `v6` was the default until 22 October, see the update at the end of this phase)

**`v4` against `v3` on the 50 golden questions** (real end-to-end runs, judge prompt v2; `v3` is the run that was re-judged with prompt v2):

| | Recall@5 | MRR | nDCG | Correct | Complete | Faithful | Abstention | Refusal on unanswerable | False abstention | Latency p50 | Latency p95 | Input tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `v3` | 0.97 | 0.91 | 0.91 | 0.97 | 0.98 | 1.00 | 1.00 | 1.00 | 0.00 | 11.6 s | 14.5 s | 1,566 |
| **`v4`** | 0.97 | 0.90 | 0.90 | 0.96 | 0.94 | 0.98 | 1.00 | 1.00 | 0.00 | **8.3 s** | 14.1 s | 1,527 |

- **What Phase 6's "done when" asks:** the system refuses correctly on unanswerable questions (refusal 1.00 on all 6, one of them, `q26`, stopped by the validation layer before the model) without refusing answerable ones (false abstention 0.00, abstention accuracy 1.00). Met, on this set.
- **The median query is 3.4 s (29%) faster**, from the smaller reranker pool (below). The p95 did not move (14.1 s against 14.5 s): the slowest queries are the ones with a corrective search (below) and the cross-document ones.
- **The generation scores are lower by about one question each, and that is not a regression I can attribute to Phase 6.** Five verdicts changed: `q19` and `q46` went up, `q04`, `q29` and `q44` went down. `q29` is an answer-model flip: its top four chunks and its validation grade (correct, no search) are identical to `v3`, only the irrelevant fifth chunk differs because of the smaller pool (a Transformer-paper chunk instead of a resume chunk), and the model then named DevPilot AI instead of IntelliFusion. `q04` and `q44` are borderline answers: the answer text of `q04` is identical in both runs and only its completeness verdict moved (1.0 to 0.5), because the judge also reads the context and the fifth chunk changed. Re-judging the same 50 `v4` answers a second time (`v4-rejudged`) changed **no** verdict, so the judge is repeatable on identical input and its verdicts only move when its input moves. One run of each version cannot separate a change of the system from this sensitivity; a few questions flip in both directions every time something upstream changes slightly.
- **Repeat check on the four generation failures** (`v4`: `q04`, `q25`, `q29`, `q44`): each question answered 5 times on the chunks `v3` retrieved and 5 times on the chunks `v4` retrieved, same prompt, same judge. `q29`: fully correct 5 of 5 on both chunk sets, so the wrong answer in the `v4` run was a one-off sample of the answer model, not an effect of `v4` (this also corrects the idea above that the changed fifth chunk caused it). `q04`: the same answer every time ("Fixing the grader alone moved measured recall from 68% to 95% with no model change") and judged incomplete 5 of 5 on both chunk sets, because it leaves out the LLM-judge fallback: a real, stable generation gap (the "keep it short" rule), and the single `v3` verdict of complete (1.0) was the lucky one, so `v3`'s completeness of 0.977 was a little too high. `q25`: correct only 2 of 5 on both chunk sets (the same wrong attachment of audit logging to the GPT-4 calls every time): a real, stable error. `q44`: correct 3 of 5 on `v3`'s chunks and 1 of 5 on `v4`'s, with answers that give both latencies but do not always say "No"; 5 repeats cannot tell a small effect of the changed tail chunk from chance. So of the four, two are stable generation weaknesses (`q04`, `q25`), one is borderline phrasing (`q44`) and one was a flake (`q29`).
- **Prompt `v2.3` (three targeted rules) and the first `v5` candidate** (`v4` with `PROMPT_VERSION = "v2.3"`, kept as the run `v5-v2.3`; the current `v5` uses `v2.4`, below). `src/generate/prompts/v2.3.txt` replaces "keep the answer short and exact" with "be concise, but include every fact from the context that answers the question", adds "attribute each fact only to the item it belongs to, do not combine facts from different parts of a sentence, for requirement-matching questions name the requirement (for example FR-10) and the matching evidence", and "for yes/no questions, start with Yes or No, then explain". Because single runs drift (see the repeat checks above), it was compared with `python -m eval.prompt_repeats --source v4 --prompts v2,v2.3 --repeats 3`: all 50 questions, 3 answers each per prompt on the same `v4` chunks, every answer judged (150 answers per prompt).

| Prompt | Correct | Complete | Faithful | Fully correct | Fully complete | Answer chars |
|---|---|---|---|---|---|---|
| `v2` | 0.957 | 0.947 | 0.980 | 0.933 | 0.917 | 117 |
| **`v2.3`** | 0.967 | 0.962 | 1.000 | 0.953 | 0.924 | 121 |

Questions that differ in at least 2 of 3 repeats: **better** `q25` (1 to 3 of 3: it now says the candidate built audit logging in the AI ATP Toolkit instead of attaching it to the GPT-4 call tracking) and `q40` (0 to 3 of 3: `v2` said "Yes, ... more than 2 consecutive days requires a document", a wrong answer to "does a two-day sick leave need one?"; `v2.3` says "No, ... only if more than 2 days"); **worse** `q44` (2 to 0 of 3: the Yes/No rule made it begin "Yes, the fine-tuned model had a lower median ... but slower at p95", which the judge scores 0 because the answer is no: a question about "every percentile" is not a yes/no on one of the facts). **Unchanged: `q04`** (still 0 of 3 complete: "be concise but include every fact" did not make the model add the LLM-judge fallback). So the rules fix two stable errors and introduce one, a small net gain, and the real cost is 72 more input tokens per query (+5%).

The real end-to-end run of `v5` (`eval/results/v5`, 50 questions) gave correctness 0.98, completeness 0.97, faithfulness 1.00, recall 0.97 (unchanged), median latency 7.3 s and p95 13.4 s, against 0.96, 0.94, 0.98, 0.97, 8.3 s and 14.1 s for `v4`. Five verdicts moved (`q25`, `q29`, `q44` up; `q19`, `q38` down), and two of them contradict the repeats: `q44` was correct in this run but 0 of 3 in the repeat, and `q29` had been a flake in the `v4` run. The repeat on identical chunks is the evidence to trust; the two agree on the direction (a small gain), not on which questions moved.
- **Prompt `v2.4`: `v2.3` with the yes/no rule refined** (`src/generate/prompts/v2.4.txt`: "answer Yes only if it holds for everything the question asks; if it holds for only part of what was asked, for example not for 'every' or 'all' of them, answer No and say which part holds"). Compared with `v2` and `v2.3` by the same repeat (50 questions x 3 repeats, 150 answers per prompt; the earlier `v2` against `v2.3` result is kept in `prompt_repeats_v4_v2_v2.3.md`):

| Prompt | Correct | Complete | Faithful | Fully correct | Fully complete |
|---|---|---|---|---|---|
| `v2` | 0.957 | 0.943 | 0.987 | 0.933 | 0.909 |
| `v2.3` | 0.963 | 0.958 | 1.000 | 0.953 | 0.924 |
| **`v2.4`** | 0.973 | 0.966 | 0.993 | 0.953 | 0.939 |

The eight questions that moved, 5 more repeats each (answers correct and complete, of 5): `q44` 4 / 0 / 5 for `v2` / `v2.3` / `v2.4`; `q40` 0 / 4 / 5; `q25` 0 / 4 / 5; `q38` 5 / 5 / **0**; `q29` 4 / 5 / 5; `q04` 2 / 0 / 1; `q19` 2 / 2 / 3; `q35` 5 / 5 / 5. Total 22, 25 and 29 of 40. So the refinement does what it was meant to: `q44` (the "every percentile" question) is fixed (5 of 5 against 0 of 5 under `v2.3`), and `q40` and `q25` stay fixed, but it creates a new regression, `q38`: "which leave types cannot be carried forward?" listed only Casual and Maternity Leave and dropped Sick and Paternity Leave (0 of 5 and 0 of 3, the same answer every time in that session; `v2` and `v2.3` listed all four). **That cause was wrong, see the correction below.** At the time: each new prompt had exactly one regression (`v2.3`: `q44`, `v2.4`: `q38`) and fixed `q25` and `q40`, against `v2`'s two errors there; `q04` is not fixed by any of them. Net against `v2`: about one question out of 50 (+0.01 to +0.02 on the aggregates), +72 input tokens per query.
- **Correction: `q38` is not caused by a rule of `v2.4`; it sits on a knife edge that the prompt does not control.** Taking the `v2.4` rule apart on the same five chunks (5 answers each) showed the prompt without any yes/no rule listing all four leave types only 1 of 5 times, and each piece of the rule 2 to 4 of 5, no pattern. Then 10 answers per prompt, in two rounds: `v2` all four 8 and 9 of 10, `v2.3` 1 and 1 of 10, `v2.4` 8 and 9 of 10, the reverse of the repeat run made before (`v2.3` listed all four, `v2.4` did not, 0 of 8 across two sessions). The prompts are byte-identical in both procedures and the chunks are the same in `v3`, `v4` and `v5`, yet identical calls gave different answers depending on how they were sent (six calls at once: `v2.4` 10 of 10, `v2` 10 of 10, `v2.3` 3 of 10; one at a time: 9, 7 and 0 of 10), and six identical calls at temperature 0 gave two different replies. The cause was not found; the plausible one is that the model's choice between listing 2 and 4 rows is nearly tied for this table, so numerical differences on the provider's side flip it (that is a guess, not measured). Consequence: no prompt can be ranked on `q38`, and the earlier "stable regressions" (`q38` for `v2.4`, `q44` for `v2.3`) are not properties of the prompts. What does repeat in both repeat runs (50 questions x 3, made hours apart, same `v4` chunks): `v2.3` and `v2.4` both turn `q25` and `q40` from 0 of 3 to 3 of 3, `v2.3` breaks `q44` (to 0 of 3) and `v2.4` does not, and the aggregates keep the same order: correctness 0.957 and 0.963 for `v2`, 0.963 and 0.963 for `v2.3`, **0.973 and 0.980 for `v2.4`**; completeness 0.943 and 0.947, 0.958 and 0.958, **0.966 and 0.970**. So `v2.4` is the best prompt measured, by about 2 points of correctness and completeness over `v2` (3 to 4 answers of 150 each time), at about +100 input tokens per query; `q04` is not fixed by any prompt. Making `q38` reliable would need a change in how the model reads the table (a stronger answer model, or answering twice and comparing), not another rule.

**`v5` = `v4` + prompt `v2.4`, adopted as the default version on request.** Real end-to-end run on the 50 questions (`eval/results/v5`, judge prompt v2), with the earlier versions:

| Run | Recall@5 | Correct | Complete | Faithful | Abstention | Refusal on unanswerable | False abstention | Input tokens |
|---|---|---|---|---|---|---|---|---|
| `v3` | 0.97 | 0.97 | 0.98 | 1.00 | 1.00 | 1.00 | 0.00 | 1,566 |
| `v4` | 0.97 | 0.96 | 0.94 | 0.98 | 1.00 | 1.00 | 0.00 | 1,527 |
| `v5-v2.3` | 0.97 | 0.98 | 0.97 | 1.00 | 1.00 | 1.00 | 0.00 | 1,599 |
| **`v5`** | 0.97 | **0.99** | **0.98** | 1.00 | 1.00 | 1.00 | 0.00 | 1,641 |

Against `v4`, four verdicts moved: `q25`, `q29` and `q44` up (as the repeats predicted for `q25` and `q44`; `q29` was a flake in the `v4` run), and `q38` down (the knife-edge question described above, not a rule of the prompt). Failures: `q04` (stable, leaves out the LLM-judge fallback) and `q38`. Retrieval and validation are identical to `v4`, so recall (0.97) is unchanged. **Latency is not a property of this change**: the median was 10.1 s (p95 18.1 s) in this run against 8.3 s and 7.3 s for `v4` and `v5-v2.3`, with the same retrieval code; the machine was shared with other applications (CPU load 41%) and the reranker and the answer call both ran slower (the median answer call 1.8 s against 1.3 to 1.4 s). A latency comparison between runs needs an idle machine and the stage timings of `eval/latency_breakdown.py`. The remaining caveats are those above: the answer model drifts between runs, one question is 0.02, and `q38` gives different answers to identical input.
- **A narrower prompt (`v2.2`) did not help, and the repeat shows why single runs mislead.** `src/generate/prompts/v2.2.txt` is `v2` plus three narrow rules: begin a yes/no question with Yes or No (a stated "No" is an answer), name every item for "what changed" / "which items" questions, attach each fact only to its own item. One run on the saved `v4` chunks looked better (correctness 0.96 to 0.97, completeness 0.943 to 0.966, faithfulness 0.98 to 1.00, no false abstention) with five verdicts moving in both directions (`q44`, `q47`, `q19` up; `q29`, `q38` down) and +173 input tokens per query (+11%) for the longer prompt. Repeating 9 questions (the four failures plus `q38`, `q47`, `q24`, `q13`, `q41`) 5 times with each prompt on the same chunks gave **identical results: 45 of 45 fully correct and 40 of 45 complete for both prompts**; the only incomplete answers are `q04` (the answer still leaves out the LLM-judge fallback, `v2.2` did not change it). Not adopted. The same repeat also contradicts the repeat before it: `q44` and `q25`, 1 of 5 and 2 of 5 correct on these chunks half an hour earlier, were 5 of 5 under both prompts, so the answer model's behaviour on an identical prompt drifts between runs made at different times, even at temperature 0. The consequence is that the differences between two single runs (`v3` against `v4`, and every earlier version comparison with changes of one or two questions) cannot be told from this drift; only large moves (retrieval recall of a few questions, latency) and repeats on the same chunks within one session are reliable. `q04` is the one stable weakness left, and it is arguably a question about "the grader fix" that the answer does answer.
- **Retrieval lists differ from `v3` on 29 questions, but only in the tail.** Recall is identical (0.966), MRR 0.898 against 0.909. That is the same as the offline result below: candidates 8 to 10 of the hybrid list almost never rank into the top 5, and when one does it is not the evidence.
- **The validation layer decided on 6 of 50 questions** (the best score under 0.1): `q01`, `q07`, `q40` (answerable, scores 0.0045, 0.019, 0.0033), `q26`, `q28`, `q50` (unanswerable). All six were searched again, `q26` was stopped (score 0.0002 after the second search), the other five were answered or refused by the prompt as before, and none of their judge verdicts changed. So in this run the corrective search changed no answer and cost 12 to 15 s on each of those queries (against 5 to 9 s for the others): the tail did not improve, and `RETRIES` = 0 in the `v4` profile would remove that cost and keep only the free score block.
- **What was not measured:** the corrective search was never needed for a query where evidence was missing from a first search in this set, so its value on such a case is unproven here; the cross-document misses (`q46`, `q47`) score too high to be retried; there is no faithfulness check at answer time (the judge scores faithfulness 0.98 offline, and an extra model call per query was not worth adding).

**Step 4: where the time goes, and what to send.** `python -m eval.latency_breakdown --reranker-sweep` times each stage of the real retrieval (24 questions):

| Stage | Ordinary question (median 10.7 s) | Cross-document question (median 9.6 s) |
|---|---|---|
| cross-encoder reranker (local CPU) | 10.1 s (95%) | 6.8 s (71%, two reranks) |
| query-plan model call | - | 2.6 s (28%) |
| query embedding (OpenAI) | 0.4 s | 0.5 s |
| vector search + BM25 | 0.02 s | 0.04 s |

The reranker reads about 1 s per chunk (512 tokens) on this CPU. The answer call is only 1.3 s, so **sending fewer chunks to the model saves tokens, not time**: `python -m eval.context_size_experiments` replays the `v3` retrieval with k = 5 and k = 3 (input tokens 1,558 to 1,099, -30%; the model call 1.3 s to 1.2 s; completeness 0.955 to 0.932, `q04`, `q19` and `q35` lower, `q47` higher), so it was not adopted. The lever is how many candidates the reranker reads (`python -m eval.rerank_cost_experiments`, scores computed once for the hybrid top 10 of the 44 labelled questions, smaller pools derived from the same scores):

| Candidates reranked | Max tokens per chunk | Recall | MRR | nDCG | Rerank time |
|---|---|---|---|---|---|
| 10 (v3) | 512 | 0.977 | 0.920 | 0.923 | 10.0 s |
| **7 (v4)** | 512 | 0.977 | 0.920 | 0.923 | 7.0 s |
| 5 | 512 | 0.947 | 0.936 | 0.912 | 5.0 s |
| 3 | 512 | 0.913 | 0.939 | 0.893 | 3.0 s |
| 10 | 256 | 0.920 | 0.894 | 0.868 | 4.9 s |
| 5 | 256 | 0.947 | 0.905 | 0.881 | 2.5 s |

7 candidates give exactly the quality of 10 on these questions and save 3 s; 5 loses about one question of recall (0.03); cutting each chunk to 256 tokens halves the time but loses evidence in long chunks (recall 0.977 to 0.920), so chunks stay at 512. The index has only 56 chunks, so "candidates 8 to 10 never matter" is a statement about this document set. Cross-document questions rerank 5 candidates per document (`PER_DOCUMENT_POOL` = 5), as before, so lowering the pool for ordinary questions does not touch them. A remaining option not tried here: quantising or exporting the cross-encoder to run faster on the CPU, since it is 95% of the retrieval time.

**Review follow-up: the new golden set, a held-out set, the keyword check and prompt `v2.5`.**
- **New golden set** (`eval/golden_set.json`, written by you; the earlier one is kept as `golden_set_original.json`): 50 questions, 44 with a `required_answer` (26 before; 18 added, my 26 unchanged), alternatives in `must_include` (a fact can be a list of wordings), `must_not_include`, `notes` and an `expected_verdict` ("yes"/"no") on 9 questions. `q04` now requires the fallback. Your `eval/keyword_check.py` (matching ignores spaces, hyphens and case, strips `[S#]`; checks facts, forbidden values, the first word Yes/No and `found`) is now the answer-level keyword check in the runner (`metrics.keyword_check` delegates to it; a wrong first word counts as a failure when no judge is used). Fixes made around it: your file now reads the golden set as UTF-8 (it failed on Windows), `contains()` accepts alternative lists and lets a decimal take a unit (`2.86` finds `2.86s`: without this `q44` lost all its relevance labels and would have dropped out of the retrieval metrics silently; after it all 44 answerable questions are labelled, and `q25`, `q46`, `q47`, `q05`, `q13`, `q37`, `q43` get slightly different labels from the sharper facts), and the golden consistency test now accepts a fact that is in the document or, for a derived fact (a total, a name), in the reference answer. 180 tests pass.
- **Held-out set** (`eval/holdout_set.json`, 18 new questions `h01` to `h18`, your file, not used for any prompt written before this test): the current version `v5` scores correctness 0.92, completeness 0.93, faithfulness 0.92, recall 0.98, refusal 1.00 end to end (`holdout18-v5`), against 0.99 / 0.98 / 1.00 on the golden set, as expected for questions that were not tuned on. Its three failures: `h03` ("does every leave type give at least 10 days?": answers No, wrongly, with confused reasoning), `h08` (right verdict, 41.0 instead of 41.8 BLEU, the same conflicting number as `q19`) and `h12` (a judge error: the answer is a correct refusal and was scored faithfulness 0; it was correct 15 of 15 times in the repeats). `h17` ("does the resume show email notifications?", it does not) is a systematic gap in every prompt: the model refuses with "Information not found" instead of saying the resume does not mention it, which your notes accept as the answer; it was not fixed, because writing a rule for it would turn the held-out set into a tuning set (a fix needs a fresh held-out question or two).
- **`v2.5`** (`v2.4` with rule 6 reworded: "keep each fact attached to the item it describes; do not move a fact from one item to another, even when both appear in the same sentence"). The hypothesis that the old wording caused `q04` is **not supported**: `q04` is 0 of 10 complete under `v2`, `v2.4` and `v2.5` alike, always the same answer ("Fixing the grader alone moved measured recall from 68% to 95% with no model change"). `q38` is 9, 6 and 10 of 10 (`v2`, `v2.4`, `v2.5`), the same coin flip as before. The `q38` table chunk is clean (one row per line, 5 cells, the wrapped cell intact), so it is not a parsing problem.
- **Prompt comparison on the new golden set and the held-out set (repeats on identical chunks).**

| Check | `v2` | `v2.4` | `v2.5` |
|---|---|---|---|
| Judge, golden (50 q x 3): correct / complete | 0.990 / 0.981 | 0.987 / 0.981 | 0.983 / 0.973 |
| Judge, held-out (18 q x 5): fully correct | 0.889 | 0.889 | 0.889 |
| Keyword pass, golden (no LLM) | 0.758 | 0.833 | 0.848 |
| Keyword pass, held-out (no LLM) | 0.760 | 0.800 | 0.867 |
| Starts with the expected Yes/No, golden (27 yes/no answers x 3) | 0.778 | 1.000 | 1.000 |
| Starts with the expected Yes/No, held-out (30 answers) | 0.667 | 0.833 | 0.833 |

With the corrected references the judge no longer separates the prompts (all 0.98 to 0.99 on the golden set; `q25` is now 10 of 10 even under `v2`, because your corrected reference accepts its answer, so the earlier `q25` "failure" was mostly the reference). The earlier claim that `v2.4` was the best prompt by 2 points of correctness did **not** replicate (it was measured against the old references and old chunks) and on the held-out set `v2`, `v2.4` and `v2.5` are equal by the judge. What does separate them is the deterministic check, which does not depend on the model's mood: both new prompts put Yes or No first as your `expected_verdict` field requires (`v2` does not for `q14`, `q44`, `h11`), and `v2.5` has the best keyword pass on both sets. The yes/no clause that the review called test-specific ("not every or all") did not produce cautious wrong "No" answers on the held-out set (`h02`, `h08`, `h14` pass; `h03` fails under `v2` too). So the honest ranking is `v2.5` >= `v2.4` > `v2`, by a small margin that is visible only in the deterministic check, at about +100 input tokens per query.

**`v6` = `v5` + prompt `v2.5`, adopted as the default version on request.** Real end-to-end runs, judge prompt v2, new golden set: golden set (`eval/results/v6`) correctness 0.99, completeness 0.99, faithfulness 1.00, abstention 1.00, false abstention 0.00, median latency 8.7 s (p95 16.7 s), 1,661 input tokens; held-out set of 18 questions (`holdout18-v6`; the 41-question run below is now `holdout-v6`) correctness 0.89, completeness 0.87, faithfulness 0.97, abstention 0.94, false abstention 0.07. Against `v5` (`v5` golden run 0.99 / 0.98 / 1.00; `holdout18-v5` 0.92 / 0.93 / 0.92): the golden verdicts moved on `q04` (0.5 to 1.0), `q38` (0.5 to 1.0) and `q19` (1.0 to 0.5), draws of questions that flip between runs (`q04` was 0 of 10 complete for every prompt in the repeats, so its pass here is a one-off); the held-out verdicts moved on `h12` (the identical correct refusal scored faithfulness 0 once and 1.0 once: judge noise), `h17` (`v5` happened to answer "the resume does not show that", `v6` said "Information not found": the unstable question) and `h03` (wrong in both). Recall is not comparable between the `v5` and `v6` golden runs because the relevance labels changed with the new golden set (0.97 and 0.98). So `v6` is not measurably different from `v5`; it was chosen because it has the best deterministic keyword result, no wrong first word on a yes/no question on either set (`h03` is a wrong verdict, not a missing one), and no rule that conflicts with "include every fact". Open items that no prompt in this section fixes: `q04` (leaves out the fallback), `q19` (conflicting 41.0 against 41.8 in the paper, also `h08`), `h03` (a numeric "at least 10" slip) and `h17` (refuses instead of saying the resume does not mention it). Fixing `h17` would need a rule tested on a fresh held-out set, since this one has now been looked at.

**Overfitting check (golden hard tier against held-out, and the prompts against each other on the held-out set).** Test A, same prompt and judge: `v2.5` scores correctness 0.985 / fully correct 0.970 on the golden hard tier (`q29` to `q50`, 22 questions, 3 repeats) and 0.889 / 0.886 on the held-out set (18 questions, two independent runs of 5 repeats); by the rule "golden hard about 0.98 and held-out about 0.89 means overfitting" this reads as overfitting, **but the untuned `v2` shows the same gap** (0.992 and 0.889): so it is not caused by the `v2.3` to `v2.5` rules, it is the whole system (chunking, retrieval, gating, references) having been tuned on the golden questions, plus two held-out questions that fail under every prompt (`h03`, `h17`). Two failed questions of 18 is also what chance gives about 1 time in 5 if the true failure rate were that of the golden hard tier, so the gap is suggestive, not proven. Test B, `v2`, `v2.4`, `v2.5` on the held-out chunks (2 runs x 5 repeats = 180 answers each): correctness 0.889 / 0.883 / 0.886, completeness 0.877 / 0.877 / 0.863, fully correct 0.889 / 0.872 / 0.883: **`v2.5` is equal to `v2` on correctness**, `v2.4` is slightly worse (`h08`: 10, 6 and 9 of 10 good answers for `v2`, `v2.4`, `v2.5`), and the rules did not make anything worse either (no cautious wrong "No": `h02`, `h14` pass 10 of 10 under all three). By your rule ("equal to `v2` means the rule gains were golden-set overfitting") the correctness gains of the new rules did not generalize. What did generalize is the format: the answer starts with the expected Yes or No in 83% of the yes/no answers under `v2.4` and `v2.5` against 67% under `v2`, and the deterministic keyword pass is 0.860 (`v2.5`), 0.793 (`v2.4`) and 0.767 (`v2`) on the held-out set. So `v6` is justified by the Yes/No-first format, not by accuracy, and costs about 100 more input tokens per query; if that format does not matter, `v2` is as accurate and simpler.

**A larger held-out set, `v6` run end to end** (`eval/holdout_set.json` now has 41 questions: `h01` to `h18` are the 18 used above, so they are no longer unseen, plus 23 new ones, `h19` to `h41`, tier "easy", never run before; the first 18-question file was replaced, its results are kept as `holdout18-v5` and `holdout18-v6`; the 41-question run is `eval/results/holdout-v6`, and the by-tier table of its report is the split: tier "easy" = the 23 unseen questions, tier "hard" = the 18 seen ones; a test now guards the file and checks that no question is shared with the golden set).

| `v6` | n | Correct | Complete | Faithful | Abstention | False abstention | Fully correct | Keyword pass |
|---|---|---|---|---|---|---|---|---|
| New held-out questions `h19` to `h41` (unseen) | 23 | 0.978 | 0.976 | 1.000 | 1.000 | 0.000 | 0.957 | 0.905 |
| `h01` to `h18` (seen before) | 18 | 0.917 | 0.900 | 1.000 | 0.944 | 0.067 | 0.889 | 0.933 |
| All 41 | 41 | 0.951 | 0.944 | 1.000 | 0.976 | 0.028 | 0.927 | 0.917 |
| Golden set, run 1 / run 2 (50) | 50 | 0.990 / 0.980 | 0.989 / 0.977 | 1.000 / 0.990 | 1.000 | 0.000 | 0.980 / 0.980 | 0.864 / 0.864 |

On the 23 unseen questions the single failure is `h26` ("what two tasks was the Transformer evaluated on?": the answer lists only the two translation tasks and misses English constituency parsing, a multi-section question whose evidence chunk was not retrieved; recall 0.33). The retrieval recall of the unseen questions is 0.88, against 0.98 on the seen ones, and it is mostly a labelling effect: `h25`, `h33` and `h40` score recall 0.17 to 0.5 yet are answered correctly, because multi-section questions have many chunks that hold at least half of the key facts. The keyword check misses `h41` ("Visual Placemat" is not named in an otherwise correct answer). These questions are the "easy" tier, so 0.98 is not comparable with the hard tier. The two golden runs of the identical `v6` code differ by 0.01 (correctness 0.990 and 0.980; the hard tier 1.000 and 0.955): three verdicts flipped (`q04` 1.0 to 0.5, `q19` 0.5 to 1.0 and `q43` 1.0 to 0.0, a retrieval miss in the second run), which is the run-to-run noise to keep in mind for any difference of one or two questions. The same 18 held-out questions in two runs of `v6` scored 0.889 and 0.917 (one flip, `h08`). Overall there is no sign of overfitting on the new unseen questions; the weak spots are the ones already known (`h03`, `h17`, `q04`, `q19`, multi-section retrieval).

**Phase 6 status:** built and measured: the relevance grade, the corrective search, the citation form check (Phase 5), the validation span in the trace, the `validation` block in the API, and a badge in the chat and a Validation section in the trace view. Tried and removed: an LLM sufficiency check, RAGAS. Not built: a faithfulness check at answer time. `v6` (`v5` plus the `v2.5` answer prompt) was the default version at this point; since 22 October it is `v12` (update below).

**Current design: CRAG-style grade from the reranker score, with a corrective search (no LLM check).** Code: `src/validate/threshold.py` (grade of the best reranker score: correct at or above `RELEVANCE_HIGH` = 0.1, incorrect below `RELEVANCE_LOW` = 0.0005, ambiguous between) and `src/validate/gate.py` (`validate_with_retry`: a correct context is answered as it is; an ambiguous or incorrect one gets `RETRIES` = 1 corrective search, the question rewritten by gpt-4o-mini and searched again with the same reranked retrieval; the two chunk lists are merged by score; if the merged chunks are still incorrect the query is not answered, "not found", and the model is never called). It is wired into `src/pipeline.py` as a `validate` span before `build_prompt` and `llm_call`. CRAG's "search another source" has nothing to point at here: the project has only these 3 documents and no web search, so the only corrective source is the same documents with a rewritten query. Experiment: `python -m eval.validation_experiments` (57 questions: the 50 golden ones on the chunks the saved `v3` run retrieved, plus 7 extra traps of the Phase 5 drill; the real rewrite and search run for the 28 questions scoring under 0.9 (27 searched, 1 rewrite repeated the question), then the cutoffs are swept offline; report in `eval/results/validation_experiments.md`).

- **The corrective search brought no gain on this set.** All 16 answerable questions it searched again (those scoring under 0.9; a 17th, `q44`, got a rewrite that repeated the question) already had their labelled evidence chunk in the first search: the low or middling score was not a retrieval miss, so there was nothing to recover (0 evidence chunks found, 0 lost). It raised the best score for 18 of 27 questions, but also for the unanswerable ones: "Programming language of QuickLeave" went from 0.294 to 0.787 and "Hindi support" from 0.067 to 0.215. The reranker score is not a measure of whether the answer is in the documents (see below).
- **The score-only block is nearly useless.** With 0.1 and 0.0005: no false abstention, but only 2 of 13 unanswerable questions are stopped without the search (the ones scoring under 0.0005), and 1 of 13 with it, because the rewrite lifted one unanswerable question above the cutoff. The other unanswerable questions reach the answer model, which refused all of them in the Phase 5 drill (26 of 26).
- **It costs latency.** 11 of 57 questions get a corrective search at 0.1 (9.6 s median: the rewrite and a second reranked retrieval), an average of +1.7 s per query; at 0.9 it is 28 of 57 and +4.2 s. Setting `RETRIES` to 0 in the `v4` profile keeps only the free score block.
- **What it cannot do:** `q46` and `q47` miss evidence while scoring 0.98 and 0.76, so a score-based grade never retries them. Only the LLM check below could see those misses.
- **Why the score fails:** the answerable `q01`, `q40` and `q07` (answered correctly) score 0.001, 0.003 and 0.019 because the answer sits in a table or a short header chunk, while the unanswerable `q48` and `q49` score 0.95 and 0.94 (the reranker likes a chunk that is on the topic). No cutoff separates them; `0.0005` is just below the lowest answerable score (`q01` 0.0008), so it holds on this document set and would need the sweep again on another.

**Tried and removed: RAGAS instead of the home-made judge.** RAGAS is itself an LLM judge (the model splits each answer into claims and checks every claim), so it does not remove the judge, it replaces my one-prompt judge with claim-level metrics (faithfulness, factual correctness against the reference, answer relevancy). It needs its own langchain and `openai` pins (a plain install would have downgraded the project's `openai` package), so it ran in a separate virtual environment through a worker process. It was dropped because of the cost: about 12 to 16 model calls per answer against 1 for the judge, 145 s for a 6-question smoke test and over 20 minutes for the 50 saved `v3` answers (against about 3 minutes), roughly 10 times the tokens, and no abstention metric. The run was stopped before it finished, so there are no RAGAS scores to compare. The judge stays the generation metric. All of its code, its requirements file and the virtual environment were deleted.

**Tried and removed: an LLM sufficiency check** (results kept in `eval/results/validation_sufficiency_experiments.md` and `validation_sufficiency_records.json`; the code was deleted). One gpt-4o-mini yes/no call asked whether the retrieved chunks contain what the question asks. On all 57 questions it said "insufficient" for 13 of 13 unanswerable ones (both trials agree on every question) but also for 6 of 44 answerable ones (`q13`, `q14`, `q41`, `q44`, `q46`, `q47`), five of them answered correctly by `v3`: it reads "No, ..." answers ("No, half-day leave is out of scope") as "not stated". Run only when the score was ambiguous it gave 0 false abstentions and stopped 8 of 13 unanswerable questions at 9 of 57 checks of 1.1 s, but the cutoff (0.1) sat just below `q41` and `q44`, so that rested on two questions. Combined with a retry written from what the check said was missing, the retry rescued `q46` and `q47` (and no unanswerable question) but only when the check ran on every query, which also blocked 4 answerable questions (`q13`, `q14`, `q41`, `q44`). It was removed on request in favour of the score-only grade above.

**First attempt, as it was built:** the sections below describe the removed check and its retry.

**Step 1 (removed): relevance grade (CRAG-style) and sufficiency check.** Code: `src/validate/threshold.py` (grade from the best reranker score: correct at or above `RELEVANCE_HIGH`, incorrect below `RELEVANCE_LOW`, ambiguous in between), `src/validate/sufficiency.py` (one gpt-4o-mini yes/no call on the retrieved chunks; deleted), `src/validate/gate.py` (correct: answer; incorrect: "not found" with no model call; ambiguous: the check decides), wired into `src/pipeline.py` as a `validate` span that can end the query before `build_prompt` and `llm_call`. Experiment: the check run twice on 57 questions: the 50 golden ones on the chunks the saved `v3` run retrieved, plus the 7 extra traps of the Phase 5 drill retrieved fresh; the cutoffs were then swept offline.

- **The reranker score alone cannot be the threshold.** The best score of the answerable questions ranges from 0.001 to 1.000 (median 0.98); `q01`, `q40` and `q07` (answered correctly) score 0.001, 0.003 and 0.019 because their answer sits in a table or a short header chunk, while the unanswerable `q48` and `q49` score 0.95 and 0.94. Any cutoff that stops `q26`, `q50` and `q28` also stops those answerable questions.
- **The sufficiency check is accurate on unanswerable questions and too strict on "No" answers.** On all 57 questions it said "insufficient" for 13 of 13 unanswerable ones (both trials agree on every question) but also for 6 of 44 answerable ones (`q13`, `q14`, `q41`, `q44`, `q46`, `q47`): five were answered correctly by `v3`, so running it on every question gives 5 false abstentions. Most of them are answers like "No, half-day leave is out of scope" or "No, not before 3 months", which the check reads as "not stated".
- **So the check only runs when the score is not decisive.** With `RELEVANCE_HIGH = 0.1` and `RELEVANCE_LOW = 0.0005`: 9 of 57 questions get a check (about 1.1 s each), 0 false abstentions, 8 of 13 unanswerable questions are stopped before the answer model, the other 5 reach it and the grounding prompt refuses them (26 of 26 in the Phase 5 drill). On the 7 held-out traps (not used to choose the cutoffs by hand, but retrieved after the golden questions were seen) 5 are blocked.
- **How far to trust these cutoffs:** 0.1 sits just under the scores of `q41` (0.233) and `q44` (0.111), the two answerable questions the check wrongly rejects, so the zero false abstentions rests on 2 questions and a different document set would need the sweep run again. The gate also cannot improve accuracy on this set: the grounding prompt already refuses every unanswerable question. What it saves is the answer call (about 1.3 s and 1,400 input tokens) on the unanswerable queries it stops; what it costs is a check call on about 16% of queries.

**Step 2 (removed): corrective retry after the check.** Code: `src/validate/retry.py` (deleted: a new search query written from what the check said was missing) and `validate_with_retry` in `src/validate/gate.py` (after an "insufficient" verdict, up to `RETRIES` = 1 extra search; the new chunks were checked against the user's own question, always by the check and never by the score alone; if they passed, the answer was generated from them; otherwise "not found"). A grade "incorrect" (score near zero) was not retried. The experiment ran the real retry (rewrite, fresh reranked search, check) for every question the check rejected (19 of 57).

- **Found a flaw in my first gate and fixed it:** a retry that scored 0.1 or more was graded "correct" and answered without a check, so the unanswerable `q49` (retry score 0.998) would have slipped through. A retry now always goes through the check.
- **The retry rescues the two cross-document retrieval misses.** `q46` and `q47`, which the check rejected, get a sufficient context after the retry (the new queries name the specific item, for example "RBAC ... in relation to the three defined roles", and the reranked search returns the evidence chunk). No unanswerable question became sufficient after a retry (13 of 13 stay blocked).
- **It cannot rescue the check's own mistakes.** `q13`, `q14`, `q41` and `q44` were rejected although the first search already held the labelled evidence chunk (and still does after the retry): the check, not the search, is wrong ("No, ..." answers). These four stay blocked.
- **But with the v4 cutoffs the retry has nothing to do on this set.** `q46` and `q47` score 0.98 and 0.76, so they are graded "correct" and never reach the check; the questions that do reach the check at 0.1 / 0.0005 are the unanswerable ones, which the retry cannot rescue (0 rescued) and which then pay for it: a retry's search takes a median 10.6 s (the reranked retrieval) plus the rewrite and a second check, on top of the 1.1 s check. The retry rescues questions only if the check is applied to high-scoring contexts too (`all` in the table: 2 rescued, but 4 false abstentions remain because of the check's strictness on "No" answers).
- **So the retry's real value is bound to a better sufficiency check**: the check must stop rejecting "No" answers, and it would then have to run on every query (+1.1 s for each).

### Update (22 Oct): completing CRAG with knowledge refinement (`v10`, `v12`)

**What CRAG has here, and what was missing.** Since `v4`: a retrieval evaluator (the best reranker score, not a trained evaluator), the three actions (correct / ambiguous / incorrect) and a corrective search (a rewritten query on the same documents: there is no web search). Missing was CRAG's **knowledge refinement**: the gate grades the context as a whole by its best chunk, so every chunk next to a good best chunk passed.

**`v10` = `v9` + refinement** (`src/validate/refine.py`, `REFINE`): after validation, one gpt-4o-mini call grades every chunk except the best one (and each named document's best one on a cross-document question) and drops the ones that do not help; it leans towards keeping (only an explicit "drop" removes a chunk; a failed call keeps everything). Simulated first on the `v9` chunks (the real function, twice, identical results, about $0.05): context precision 0.54 to 0.74 (golden), 0.51 to 0.70 (holdout). Full run: context precision 0.75 / 0.72, correctness 0.97 to 0.99 (golden) and 0.96 to 1.00 (holdout), 18-21% fewer input tokens, about 2.4 s slower (the extra call); it fixed `q29` (the misleading DevPilot chunk is dropped), `n12` and `q34` in that run.

**The regression it hid, and `v12`.** Re-judging showed `q37` ("which version introduced the error code ...") incomplete in `v10`: the filter graded the revision history ("1.2 | ... Added error code QL-1006") on its own, where it looks off topic, because only the kept error-code table says QL-1006 is that code. `v8` and `v9` had answered "version 1.2". Checking every chunk the filter dropped that a `v9` answer had cited: only `q37` was hurt (`q29` was helped, `q46` and `n08` were harmless). **`v12` = `v11` + the grader shown the always-kept passages** (`REFINE_WITH_CONTEXT`), and asked to keep a passage that says when, why or by whom something in a kept passage was added or changed. Real model on `q37`: 3 of 3 complete (the revision chunk kept every time). Full run: golden correctness 0.990, completeness 0.977, MRR 0.951, nDCG 0.933, recall 0.962 (the best ranking of any version); holdout correctness 0.960, MRR 0.974; failures `q04`, `q35`, `n12`, `n13`, all shared with earlier versions (`n13`: the same incomplete answer in `v10`, `v11` and `v12`, passed by the earlier judges).

**CRAG grades per chunk** (`grade_chunks` in `src/validate/threshold.py`): every retrieved chunk is graded with the cut-offs of the context grade (correct at 0.1 or more, ambiguous / "partial" from 0.0005, incorrect below), marked as read or filtered out, stored in the trace (`context_grades`) and sent to the chat.

**What "partial" means, measured on the `v12` runs.** CRAG answers an ambiguous context after trying to correct it; only incorrect stops the answer. Of 8 questions graded partial, the 4 answerable ones were answered correctly (`q01`, `q07`, `q40`, `n08`) and the 4 unanswerable ones were refused by the answer model; refusing on partial would have lost 4 correct answers and gained nothing. Three unanswerable questions were even graded "correct". The corrective search ran 10 times (about 6 s each) and never gave an answerable question a chunk it used: its rewrite often repeats the question ("projects user worked on"). Candidate changes, not made: skip the second search when the rewrite is nearly the question (saves about 6 s, changes no answer), treat "user", "my" and "I" as the candidate for questions about jobs, skills or projects, and word the badge "low reranker confidence" when retrieval looks fine.

---

## Phase 7 — Productionize (1–2 weeks)

**Goal:** Turn the notebook-style pipeline into a service.

**Build:**
- **FastAPI** endpoints: `POST /ingest`, `POST /query`, `GET /health`
- **Incremental ingestion** — hash files; only re-index new/changed docs
- Move to **Qdrant in Docker**
- **Config-driven** pipeline (`config.yaml`: chunk size, top-k, models, thresholds)
- **Move from the home-made tracer to Langfuse** — keep the same span names (`embed_query`, `retrieve`, `rerank`, `build_prompt`, `llm_call`, `validate`) so nothing changes conceptually; add user feedback scores to traces
- **Semantic cache** with TTL (and invalidate when docs change)
- **Eval in CI** — run golden set on every change; fail if metrics drop below baseline
- **Docker Compose** for API + Qdrant + Langfuse
- Track **p95 latency** and **cost per query**

**Done when:** `docker compose up` runs the whole system, and every query is traceable.

**Built so far (22 Oct), ahead of the rest of this phase:**
- **Streaming with TTFT.** `POST /api/chat/stream` (Server-Sent Events: `stage`, `validation`, `context`, `token`, `done`, `error`); the old polling endpoints still work. The model clients stream (`stream()` for OpenAI and Ollama, `stream_completion` falls back to one piece). Because the cited prompts reply in JSON, `AnswerStream` (`src/generate/answer.py`) extracts only the `answer` text as it is written, escapes split across pieces included, so the user never sees braces or the `checks` scratchpad. The trace records `ttft_ms` (question to the first answer character) and the model's own first token. Measured: TTFT about 6 s against 13.8 s total for a normal question (search about 4 s, validation and filter about 1 s, model first token about 0.9 s); the first question after a restart waits for the search models to load (25 to 75 s).
- **Chat UI:** the live pipeline version in the header; the CRAG grade of every query (Correct / Partial / Incorrect, with what was done: answered, corrected by a second search, not answered) shown before the answer starts; the per-chunk counts ("Retrieved 5: 2 correct, 1 partial, 2 incorrect, 3 read by the model"); sources listed by chunk name with grade, score, citation and "not read (filtered out)", without the chunk text (still in the Traces view).
- **Reports:** context precision as a card and a per-type column.
- **Run the app:** `python -m uvicorn api.main:app --host 127.0.0.1 --port 8765` (serves `ui/dist`; rebuild the UI with `npm run build` in `ui/`). A config change needs a restart; the app reads `PIPELINE_VERSION` at start.
- **Not built yet:** Qdrant, incremental ingestion, Langfuse, semantic cache, eval in CI, Docker Compose.

---

## Phase 8 — Optional advanced

- **CRAG-style** retrieval evaluator with fallback source (partly done in Phase 6: score-based evaluator, corrective search and knowledge refinement; a fallback source such as web search is not)
- Multi-turn chat with query condensation
- Simple **Streamlit** UI with clickable citations
- Parent-document (small-to-big) retrieval
- Agentic retrieval with tools

---

## Debugging Checklist — when an answer is wrong

Always pull the trace first (`scripts/show_trace.py <trace_id>`). Then go in order:

```
Wrong answer → Pull trace
      ↓
1. Is the correct doc in the index?            No → INDEXING (ingestion / parsing failed, not re-indexed)
      ↓ Yes
2. Is the answer intact in one chunk?           No → CHUNKING (split at boundary, table broken, no context)
      ↓ Yes
3. Search WITHOUT filters — doc appears?        Yes → FILTER (wrong metadata, too strict)
      ↓ No
4. Doc retrieved in top-20 but not top-5?       Yes → RANKING (add / tune reranker, increase k)
      ↓ No
5. Not retrieved at all?                        → SEARCH (vocabulary mismatch, exact terms → hybrid, query rewrite)
      ↓ (if retrieved correctly)
6. Is the retrieved doc current and correct?    No → DATA QUALITY (stale / conflicting versions)
      ↓ Yes
7. Was it in the final prompt (not truncated)?  No → CONTEXT ASSEMBLY
      ↓ Yes
8. Answer doesn't match the chunk               → GENERATION (prompt, grounding rules, model)
```

**After every fix:** re-run the golden set, compare with baseline, and add the failing question to the golden set if it isn't there.

## How to know an improvement is real

- Same golden set, **one change at a time**, compare against the baseline.
- Check metrics **by question type**, not only the average (avoid hidden regressions).
- Check **latency and cost** didn't get worse beyond budget.
- Label each improvement with a version: add a profile to `PROFILES` in `src/config.py` (v1, v2, ...) and set `PIPELINE_VERSION` before re-indexing and running the eval. Traces and eval runs are tagged with it, and the UI can filter by version.
- **Never change a measured version in place:** a new idea gets a new profile (`v11` = `v10` + ...), so earlier runs stay valid and switching back is one line.
- **Check on the held-out set, not only the golden set.** A gain that shows only on golden is tuning; keep the held-out questions unseen (no prompt written from them).
- **Simulate first, for free, on saved runs** (retrieval cut-offs, filters, ranking orders), then run the full eval once. Use a check that matters for answers ("would this drop a chunk the answer cited?"), not only relevance labels.
- **Compare like with like:** runs judged by the same judge in one pass, and repeats on identical chunks for anything worth one or two questions (Phase 2 lessons).
- Keep a simple experiment log:

| Run | Change | Recall@5 | Precision@5 | MRR | nDCG@5 | Correctness | Relevance | Completeness | Faithfulness | Abstention acc. | p95 latency | Keep? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | naive, 500-token chunks | | | | | | | | | | | — |

## Timeline

~8–10 weeks alongside a full-time job. Each phase produces something concrete to talk about in interviews:
*"I found X failure, changed Y, and recall@5 went from A to B."*
