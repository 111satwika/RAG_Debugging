# Project walkthrough — detailed script

*A first-person script for explaining the project in a technical interview or demo, about 15–20 minutes. Each version follows the same pattern: what I built, how I debugged it, what improved, and what still went wrong.*

---

## Opening

I built a retrieval-augmented question-answering system over three different kinds of documents: my own resume, a product specification for a leave-management app called QuickLeave, and the "Attention Is All You Need" paper. I picked them on purpose: a resume has short sections and bullet points, the spec has numbered requirements and tables, and the paper has dense text, equations and results tables.

My goal wasn't just to get answers. I wanted to understand how a RAG system fails and how to prove that a change really improves it. So I built everything myself, without LangChain, measured every change, and changed one thing at a time.

Before writing any pipeline code, I built the test: a **golden set** of questions with reference answers, the key facts each answer must contain, and the source section. It started at 28 questions and grew to 50, covering factual, table, multi-section, exact-term, cross-document and unanswerable questions. I also added traps: an old value still present in the spec, a number the paper contradicts itself on, and questions the model could answer from its own knowledge but that aren't in the documents.

---

## v1 — The naive baseline

**What I built.** The simplest possible pipeline: parse the PDFs, cut them into fixed 500-token chunks, embed with a small local model (`bge-small`), store in Chroma, retrieve the top 5, and ask a local 7-billion-parameter Qwen model. From day one, every query wrote a **trace**: the chunks with their scores, the exact prompt, the answer and the time spent in each step.

**How I debugged it.** I wrote an evaluation harness that runs every golden question and reports retrieval metrics (Recall@5, MRR, nDCG) and answer metrics scored by an LLM judge (correctness, completeness, faithfulness, abstention). It also labels each failure with the stage that failed: chunking, retrieval or generation.

**Results.** Recall 0.91, MRR 0.69, correctness 0.91, with a p95 latency of **184 seconds**, because the model ran on a CPU.

**What went wrong.**
- One broad "hub" chunk at the start of the spec outranked the right chunk for 7 questions.
- 7 of 38 chunks mixed two or more sections, and 14 had no context saying which document or section they came from.
- `q07` failed because its answer sat in a mixed tail chunk that was never retrieved.

---

## v2 — Chunking (Phase 3)

**What I built.** I switched to OpenAI embeddings and gpt-4o-mini, then tested chunking strategies one at a time: smaller chunks, overlap, recursive splitting, and structure-aware chunks (one chunk per resume section, paper heading or spec section). Each chunk got a header such as "QuickLeave spec > 5. Leave Policy", tables were kept whole, and tiny pieces were merged.

**How I debugged it.** A chunk-diagnostics script counted structural defects: mixed sections, orphan headings, mid-sentence cuts, and chunks with no context. In a drill, 300-token chunks split the number 0.831 into "0.83" and "1", and the model answered 0.83.

**What improved.** Contextual headers were the big win: MRR went from 0.79 to 0.86. The structural defects dropped to almost zero.

**What went wrong.** Ranking improved, but answers got worse (correctness 0.86). Cross-document questions like "QuickLeave requires Azure AD; does the candidate have Azure AD experience?" broke. Every spec chunk now started with "QuickLeave", so a question naming QuickLeave pulled all 11 spec chunks into ranks 1 to 11 and pushed the resume evidence to rank 14. Cross-document correctness was **0.00**.

---

## v2-extended — Retrieval (Phase 4)

**What I built.** I tested BM25 keyword search, hybrid search (dense plus BM25, merged with reciprocal rank fusion), cross-encoder rerankers, query rewriting, and document routing.

**How I debugged it.** I made the golden set harder, with paraphrases, two-chunk answers and comparison questions, because hybrid was already ranking the right chunk first on 20 of 25 of the original questions.

**What improved, in three steps:**
1. **Hybrid search** was the only cheap technique that clearly helped. BM25 matches exact words like "Azure AD" in the resume, so the resume chunk came back into the top 5. Cross-document correctness went from 0.00 to 0.75.
2. **A reranker** improved the hard questions: their MRR rose from 0.92 to 1.00. But on its own it **hurt** cross-document questions (0.75 to 0.63). It re-sorts one shared list and prefers chunks that match the question's wording, which is mostly the spec's, so the resume chunk dropped out again.
3. **A gated query transformation** fixed that. A free rule detects questions that name two documents, about 8% of them; only for those, a small model call writes one search per document. Each document is searched and reranked separately and gets its own three places, so the reranker can no longer push one document out. Cross-document correctness went to 1.00, and overall correctness from 0.81 to 0.92.

**What went wrong.**
- The reranker hurt cross-document questions until the per-document search was added (step 2 above). It also adds seconds per query on a CPU.
- BM25 initially matched filler words until I added a stopword list.
- Query rewriting on its own did nothing, and transforming every question cost 2.1 seconds instead of 0.4 for no gain, which is why it's gated.

---

## v3 — Grounded, cited answers (Phase 5)

**What I built.** A versioned prompt with grounding rules: answer only from the sources, say "Information not found" when the answer isn't there, cite every claim as `[S1]`, `[S2]`, and reply in JSON (answer, sources, found) validated with Pydantic.

**How I debugged it.** A drill with 13 unanswerable questions, run twice each. Without the grounding rule, the model answered from its own knowledge about a third of the time. For example, it said "the documents do not say, however it is widely known that GPT-3 has 175 billion parameters".

**What improved.** Hallucination on the drill went from 8 of 26 to **0 of 26**; correctness 0.97, faithfulness 1.00, abstention 1.00.

**What went wrong.** The "keep the answer short" rule made answers drop details: `q04` left out a second fact every time. I also found two bugs in my own experiment: the model copied a placeholder from the JSON example, and the cross-document search plan varied between runs.

---

## v4 to v6 — Validation layer (Phase 6) and the first prompt rounds

**What I built.** A CRAG-style validation layer: the reranker's best score grades the retrieved context as correct, ambiguous or incorrect. Ambiguous or incorrect contexts get a corrective search with a rewritten question, and if the context is still incorrect, the system answers "not found" without calling the model. I also tested an LLM sufficiency check and RAGAS, and removed both: the check rejected correct "No" answers, and RAGAS cost about 10 times the tokens.

**How I debugged it.**
- I timed every stage. The reranker was 95% of the retrieval time, so I cut its candidates from 10 to 7, which saved 3 seconds with identical quality.
- Then I iterated on the prompt, versions `v2.1` to `v2.5`, each rule fixing a specific golden question.

**What improved.** The golden set reached **0.99** correctness.

**What went wrong.** This is where the most important lesson came from:
- On a held-out set of new questions, the same version scored only **0.89**.
- The plain `v2` prompt did just as well on the held-out set. My rules had been tuned to the golden questions.
- Some questions gave different answers to identical input at temperature 0. One table question listed 2 or 4 leave types depending on the run.

---

## Fixing the measurement before fixing the system

At this point I stopped changing the system and looked at the measurement, because many "failures" weren't real:
- **Reference answers asked for more than the question.** One required "as reported in the abstract and Table 2".
- **The judge made mistakes.** It scored a correct refusal as irrelevant, overruled the reference with its own misreading of a broken table, and penalised a true extra detail.
- **Identical answers got different scores** in different runs.

I rebuilt the judge in three steps. The final version only lists the reference's key facts and the answer's claims, each claim with an exact quote from the context, and **code** computes the scores and verifies the quotes. Correctness and completeness are separate: an incomplete but right answer isn't "wrong". Every answer is judged three times and the median counts. I checked it on 17 answers whose right score I knew, and it scored all 17 correctly, both on correctness and on faithfulness.

I also retired the held-out set, because I had looked at its questions while designing prompts, and wrote 25 fresh questions that are never used for tuning.

---

## v7 and v8 — Prompts that generalize

**What I built.**
- `v7` (prompt `v2.9`): plain `v2` plus a scratchpad for comparison questions ("every", "all", "at least"), and an answer like "No, the resume doesn't mention Terraform" instead of "not found".
- `v8` (prompt `v2.11`): "be concise but give every fact, with its numbers, and name the item it belongs to", plus attribution checks.

**How I debugged it.** Every prompt was compared with repeats: 3–5 answers per question on identical chunks, on both the golden set and the fresh held-out set. I also tried an "evidence first" design, which fixed one comparison question but made answers drop facts everywhere else.

**What improved.** `v2.11` beat `v2.9` on both sets: held-out correctness 0.92 → 0.97, golden 0.97 → 0.99. It fixed attribution errors such as "precision rose to 100%", where 100% actually belonged to a different metric in the same sentence.

**What went wrong.**
- Questions that list items from a table still stop early sometimes.
- One answer always leaves out a second fact (`q04`).
- A malformed scratch field once made the app show raw JSON; I fixed the parser.

---

## v9 — Precision (Phase 4 revisited)

**The problem.** Precision@5 was stuck at 0.28. I calculated the ceiling: 31 of 44 questions have only one relevant chunk, so sending five chunks caps it at about 0.29. It was already at the maximum.

**What I built.** An adaptive cut-off after reranking: keep chunks scoring at least 2% of the best one. It has two guards, both found in a free simulation on saved runs:
- per-document searches are cut within their own list, because their scores aren't comparable;
- cross-document questions keep each document's top two chunks, because the reranker scores real evidence very low there.

**How I debugged it.** I tested every threshold against the question "would this drop a chunk the answer actually cited?", not just the relevance labels. That also showed that Precision@5 always divides by 5 and could never show this gain, so I added a **context precision** metric: relevant chunks divided by the chunks the model actually read.

**What improved.** Context precision 0.28 → **0.55**, about 20% fewer tokens, the same answer quality.

---

## v10 — Completing CRAG with knowledge refinement (Phase 6 revisited)

**What I built.** CRAG's missing step: an LLM filter that reads each retrieved chunk and drops the ones that don't help, before the answer is written.

**What improved.** Context precision 0.54 → **0.75**. Correctness rose on both sets, and it fixed answers that had a misleading chunk in their context.

**What went wrong.** A later re-judge showed a hidden regression. The filter graded each chunk alone and dropped a revision-history row, "version 1.2 added QL-1006". On its own it looks irrelevant, but it's the answer once you know from another chunk that QL-1006 is the error code being asked about. Earlier versions got that question right.

---

## v11 and v12 — Ranking and the multi-step fix

**v11.** Golden MRR (0.90) was below the held-out set's (0.97). I diagnosed each miss:
- **Table chunks carried the wrong heading.** The parser put Table 2 under "5.4 Regularization".
- **Cross-document results started with the spec**, although the question asks about the candidate.

I headed table chunks with their own captions and put the asked-about document first. Golden MRR went from 0.90 to **0.95** and nDCG from 0.89 to **0.93**, with the held-out set unchanged.

Earlier I had also found that the PDF parser broke Table 2 itself: the header came out in fragments and one value shifted into the wrong column. Both the answer model and the judge misread it until I repaired it during ingestion.

**v12.** The filter now sees the chunks it already keeps, so it recognises a passage that completes them. The version-1.2 question went from 0 of 3 to 3 of 3 with the real model.

**v12 final results:**

| | Golden | Held-out |
|---|---|---|
| Correctness | 0.99 | 0.96 |
| MRR | 0.95 | 0.97 |
| nDCG | 0.93 | 0.94 |
| Context precision | 0.75 | 0.72 |

---

## The app (Phase 7, started)

The system runs as a FastAPI service with a React UI. Answers **stream** over server-sent events, and I measure **time to first token**: the answer starts at about 6 seconds instead of arriving whole at about 14. The breakdown shows retrieval, not the model, is the bottleneck. Each answer shows:
- its CRAG grade (correct, partial, incorrect);
- the grade of every retrieved chunk;
- which chunks were read or filtered out;
- clickable citations that open the source.

---

## What still gives wrong answers

- **Lists:** answers that stop before naming every item. HR Admin permissions and `q04`'s second fact miss items in every version.
- **Tables:** reading them reliably. "Which model has the highest score" is right about 4 times in 5, and the paper's own 41.0 vs 41.8 BLEU conflict still confuses answers.
- **Question wording:** "the user's" in a question pulls the product spec instead of the resume.
- **Reranker scale:** resume chunks get low reranker scores even when they're exactly right, so many correct answers show a "partial" grade.
- **Wasted retries:** the corrective search rarely finds anything new, and costs about 6 seconds when it runs.
- **Ranking:** the BLEU-table questions still rank the Results text above the table.

I've stopped adding prompt rules for these. After two or three attempts on the same weakness, the next lever is a stronger model for those question types, or a design change.

---

## Closing — what I'd say I learned

1. **Fix the measurement first.** Half my "failures" were wrong references, a biased judge, or a metric that couldn't move.
2. **Keep a held-out set you never tune on.** My golden set said 0.99 while unseen questions said 0.89.
3. **Single runs lie.** Use repeats, and judge competing versions in the same pass.
4. **Let the model extract and let code decide** when building an LLM judge.
5. **Fix data at the source.** One broken table misled both my answer model and my judge.
6. **Simulate for free before paying,** and never change a measured version in place: every idea got its own version, so every result stayed reproducible.
