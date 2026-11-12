# Project walkthrough — brief script

*A first-person script, about 2–3 minutes spoken. The companion detailed script covers each version in depth.*

---

I built a RAG system from scratch, without LangChain, over three very different documents: my resume, a product spec with tables and numbered requirements, and the "Attention Is All You Need" paper. The goal was to learn how RAG fails and how to prove an improvement is real, so I wrote a 50-question golden set first and changed one thing at a time.

**Baseline (`v1`).** Fixed 500-token chunks, a small local embedding model and a local 7B model: correctness 0.91, but retrieval ranking was weak (MRR 0.69) and the p95 latency was 184 seconds. Traces on every query showed broad "hub" chunks outranking the right ones.

**Chunking (`v2`).** Structure-aware chunks with "document > section" headers raised MRR to 0.86, but broke cross-document questions: every spec chunk mentioned QuickLeave, so the resume evidence fell to rank 14.

**Retrieval (`v2-extended`).** Hybrid search (dense plus BM25) fixed that, because BM25 matches exact terms like "Azure AD". A reranker then improved the hard questions but pushed the resume evidence out again, so for the 8% of questions that name two documents, I search and rerank each document separately. Correctness went from 0.81 to 0.92, and cross-document questions from 0.00 to 1.00.

**Grounded answers (`v3`).** Grounding rules, citations and JSON output took hallucinations on unanswerable questions from 8 of 26 to 0.

**Validation (`v4`–`v6`).** A CRAG-style grade and corrective search. Prompt tuning pushed the golden set to 0.99, **but a held-out set showed only 0.89**: I had tuned the rules to my test questions.

**Fixing the measurement.** Many failures were evaluation bugs: wrong reference answers, a judge that overruled the reference, and a precision metric that couldn't move. I rebuilt the judge so the model only extracts facts and quotes and code computes the scores, and calibrated it until it scored 17 of 17 known answers right. I also wrote a fresh held-out set I never tune on.

**Improvements that held on both sets (`v7`–`v12`):**

| Version | Change | Result |
|---|---|---|
| `v8` | A prompt that gives every fact and names its item | Held-out correctness 0.92 → 0.97 |
| `v9` | An adaptive reranker cut-off | Share of relevant chunks the model reads 0.28 → 0.55 |
| `v10` | CRAG knowledge refinement: an LLM filter drops unhelpful chunks | 0.55 → 0.75 |
| `v11` | Table chunks headed by their own caption; the asked-about document first | Golden MRR 0.90 → 0.95 |
| `v12` | The filter sees the chunks it keeps | Fixed a multi-step question it had broken |

**Final (`v12`).** Correctness 0.99 golden and 0.96 held-out. Answers stream with time-to-first-token of about 6 seconds, and the UI shows a CRAG grade and clickable citations.

**Still wrong sometimes.** Answers that stop before listing every item, reading tables (about 4 in 5 right on one question), and "user" in a question pulling the wrong document. The next lever there is the model, not more prompt rules.

**What I learned:** fix the measurement first, keep a held-out set you never tune on, don't trust single runs, and fix data problems at the source.
