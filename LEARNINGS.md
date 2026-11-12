# Learnings from building this RAG system, phase by phase

What each phase of this project taught, with the evidence from this repository. Each learning has: what happened, the lesson, and how to apply it next time. The full record (results, tables, versions) is in `.github/skills/rag-learning-plan/SKILL.md`.

| Phase | Main learning |
|---|---|
| 0. Data + test sets | Reference answers can be wrong, and any question you look at while designing becomes tuning data |
| 1. Naive RAG + tracing | Traces are what make every later diagnosis possible |
| 2. Evaluation | The measurement was wrong more often than the system; let the model extract and code decide |
| 3. Chunking + data | Fix data problems at the source; a broken table misled everything downstream |
| 4. Retrieval | Metrics have ceilings; judge retrieval by what answers actually used |
| 5. Prompting | Every rule changes every answer; some limits belong to the model |
| 6. Validation (CRAG) | Measure what each step really changes; filters need context |
| 7. Production | Stream and measure TTFT; show the evidence; be strict only on what users see |

---

## Phase 0 — Data and test sets

### Reference answers can be wrong
**What happened.** Several "failures" were the golden set's fault:
- `q19` required where a value was reported ("as reported in the abstract and Table 2"), which the question didn't ask;
- `q25` worded an alternative as two required facts ("A and B; naming either is fine"), so the judge demanded both;
- `q24`'s key phrase "Azure AD" didn't match the evidence "Azure (AD SSO …)".

**Lesson.** A reference answer is a measurement instrument, and it can be miscalibrated.

**Apply it.**
- The required answer should contain only what the question asks; put everything else in optional details.
- Write alternatives explicitly ("X or Y; either is enough").
- When a correct-looking answer fails, read the reference before the answer.

### Tuning on the test set happens by accident
**What happened.** Prompts `v2.1`–`v2.5` each added a rule to fix one golden question. They reached 0.99 on the golden set but were no better than plain `v2` on the held-out set. Later, the old held-out set was used to design `v2.9`'s rules (`h03`, `h17`), so it stopped being a fair test and was replaced by fresh questions (`n01`–`n25`).

**Lesson.** Every question you look at while designing a change becomes training data. A gain that shows only on the set you tuned on isn't a gain.

**Apply it.**
- Keep a held-out set that is never used to write a prompt or choose a setting; it only decides.
- Once you've studied held-out failures, write fresh held-out questions before the next decision.
- Before freezing new questions, check every key fact sits in a chunk, every "not mentioned" term is really absent, and every unanswerable question really has no answer.

---

## Phase 1 — Naive RAG and tracing

### Traces make every later diagnosis possible
**What happened.** Almost every finding in later phases came from reading one query's trace:
- the filter dropping the revision-history chunk (`q37`);
- a corrective search whose rewrite just repeated the question;
- "User's current job" ranking QuickLeave chunks first;
- the model's own first-token time versus the time the user waits.

**Lesson.** You can only fix what you can see per query.

**Apply it.**
- Every stage gets a span with its inputs, outputs and timings.
- Record what each decision step did: the grade, the retry, which chunks were dropped and which were cited.
- When adding a new stage, add its trace data in the same change.

---

## Phase 2 — Evaluation and debugging harness

### The measurement was wrong more often than the system
**What happened.** For most of the prompt work, the scores pointed at the answer prompt, but many failures were in the evaluation:
- **The judge** scored a correct refusal as irrelevant (`h12`), overruled the reference with its own misreading of a broken table (`q19`), and penalised a true extra detail (`n02`).
- **Precision@5 couldn't move:** it always divides by 5, so sending fewer, better chunks (`v9`) showed 0.27 while the real share of relevant chunks was 0.55.
- **Context precision first read the wrong list:** the search results, not the chunks the model got after the filter.
- **The comparison tool didn't use the pipeline's prompt:** `prompt_repeats` labelled every chunk "page 1".

**Lesson.** Before trusting a score, check that the reference, the judge and the metric are right. A wrong measurement sends you to fix the wrong component.

**Apply it.**
- Read the failing answers yourself before changing anything.
- When a metric doesn't move after a change that should move it, suspect the metric definition first.
- Make sure experiment tools build exactly what the pipeline builds.

### Let the model extract, let code decide
**What happened.** Asking the judge for scores directly gave erratic results: it mixed correctness with faithfulness and punished extra true facts. It became reliable (17 of 17 on known answers) once the model only *listed* things and code did the scoring:
- the model lists the reference's key facts as present, missing or contradicted, and code computes correctness and completeness;
- the model quotes the context for each claim, and code verifies the quote is really there;
- Yes/No flips, refusals (`found: false`) and calculations from the answer's own numbers are checked in code.

**Lesson.** LLMs are better at extracting and comparing than at giving calibrated scores. Move every rule you can state precisely into code, where it's deterministic and testable.

**Apply it.**
- Ask the judge for structured evidence and compute the score yourself.
- Separate checks that need different inputs: correctness without the context, faithfulness without the reference.
- Keep a calibration set of answers with known scores (`eval/judge_calibration.py`) and re-run it after every judge change.
- Define correctness as "is anything wrong" and completeness as "how much is covered", so an incomplete but right answer isn't scored as wrong.

### LLM outputs and judges are noisy, even at temperature 0
**What happened.**
- Identical prompts gave different answers: `q38` listed 2 or 4 leave types depending on the run, and `n12` was right 4 times in 5.
- Word-for-word identical answers from identical chunks were judged pass in one run and fail in another (`q34`, `q37`).
- Two single runs of the same code differed by one to two questions.

**Lesson.** A difference of one or two questions between two single runs isn't evidence.

**Apply it.**
- Compare prompts with repeats on identical chunks: 3–5 answers per question (`eval/prompt_repeats.py`).
- Judge each answer several times and take the median (3 votes).
- Compare runs only when they were judged together in one pass; re-judge saved answers rather than comparing separately judged runs.
- A re-judge can also expose a hidden regression. Re-judging `v10` revealed `q37`'s real failure, which a lenient first verdict had hidden.

---

## Phase 3 — Chunking and data

### Fix data problems at the source
**What happened.** The PDF parser broke the paper's Table 2: the header came out in fragments, underscores sat inside numbers, and one value shifted into the wrong column. Nothing said which column was English-to-French BLEU, so both the answer model and the judge misread it across many experiments. Repairing it during normalization fixed the misreading for every later step at once.

**Lesson.** A parsing error looks like a retrieval, generation or judge problem downstream.

**Apply it.**
- Read the parsed text of every table before chunking.
- Record data versions so you know which runs used which text.

### Chunk headers decide how chunks rank
**What happened.** The parser places a table under the section that precedes it on the page. Table 2 (BLEU scores) was headed "5.4 Regularization", and Table 1 "3.4 Embeddings and Softmax", so the reranker preferred the Results text. Heading each table chunk with its own caption moved `q16` to rank 1. The BLEU questions didn't move, because the text and the table still scored almost equally.

**Lesson.** The header you prepend to a chunk is part of what gets matched. A wrong header hides a right chunk.

**Apply it.**
- Check what header each table and figure chunk carries.
- Prefer the element's own caption over the surrounding section.

### Don't rebuild a shared index for an experiment
**What happened.** The live versions shared one index. The caption experiment got its own index (`rag_chunks_v11`), so the live version wasn't silently changed.

**Apply it.** Give any chunking experiment its own index and leave the measured versions' data untouched.

---

## Phase 4 — Retrieval

### Know a metric's ceiling before chasing it
**What happened.** 31 of 44 golden questions have one relevant chunk, so always sending 5 chunks caps Precision@5 near 0.29; `v8` was already at 0.28. Only sending fewer chunks could raise it. The adaptive cut-off (`v9`) doubled the share of relevant chunks the model reads, 0.28 → 0.55.

**Lesson.** A metric stuck below where you want it may already be at its maximum for the current design.

**Apply it.** Compute the best possible value on your data before tuning.

### Reranker scores aren't comparable everywhere
**What happened.**
- Scores from different sub-queries (one search per document) aren't on the same scale.
- On cross-document questions the reranker scores real evidence very low (`q24`: 2.2% of the top score).
- A single cut-off over everything would have dropped evidence, so it needed two guards: cut within each sub-query's list, and keep each named document's top 2 chunks.

**Lesson.** A threshold on a score assumes the score means the same thing everywhere. Check that first.

### Judge retrieval changes by what the answers used
**What happened.** Relevance labels misled both ways. A cut-off "lost" chunks labelled relevant but not needed, and kept chunks the answers actually used. Checking "would this drop a chunk the answer **cited**?" found the safe threshold (2%) and showed where stricter ones broke: 5% dropped `q37`'s evidence and 10% `q16`'s.

**Apply it.** Simulate retrieval changes for free on saved runs, using the cited-chunk check alongside the labels.

### Ranking problems came from data and ordering
**What happened.** Golden MRR (0.90) was below the holdout's (0.97) for every recent version. The causes weren't the model:
- table chunks headed by the wrong section;
- cross-document result lists starting with the spec although the question asked about the candidate;
- a generic skills chunk ranking first.

Fixing the headings and putting the asked-about document first raised golden MRR from 0.90 to 0.95 and nDCG from 0.89 to 0.93, with the holdout unchanged.

**Lesson.** Diagnose each ranking miss individually before reaching for a bigger reranker.

### Question wording matters
**What happened.** "What is User's current job?" pulled QuickLeave chunks ("Users and Roles"). "The candidate's" or the name found the resume. Even then, resume chunks scored low (0.02 at best), because the reranker scores short tables and bullet points conservatively.

**Lesson.** Users won't phrase questions like your test set. Words such as "user", "my" and "I" need handling.

---

## Phase 5 — Prompting and generation

### Every prompt rule changes every answer
**What happened.**
- Each new rule fixed one question and often broke another: `v2.3` fixed `q25` and `q40` but broke `q44`; `v2.10` fixed `n07` and `n16` but listed only some rows of `q38`.
- "Write the evidence first" fixed a comparison question (`h03`, 0/5 → 5/5) but made answers drop facts everywhere else. The model extracted evidence lossily, then treated anything missing as "not specified".
- What generalized: replacing "keep it short" with "give every fact and name its item" (`v2.11`), and answering "No, the resume does not mention X" instead of "not found".

**Lesson.** A prompt rule is a change to every answer, not just the one you're fixing.

**Apply it.**
- Test every new rule on all questions, with repeats, on both sets.
- Prefer scratch steps that apply only to the question types that need them, like the `checks` list for every/all/at-least questions.

### Some limits belong to the model
**What happened.** `q04` leaves out a second fact in every version, and table reading stays at about 4 in 5 (`n12`, `q35`), whatever the wording. The superlative rule in `v2.12` didn't change `n12` (2 of 3, as before).

**Lesson.** After two or three failed rules for the same weakness, stop adding rules.

**Apply it.** Consider a stronger model for those question types, answering twice and comparing, or a design change.

### Be strict on what users see, lenient on scratch work
**What happened.** When the model wrote its `checks` scratch list in an unexpected shape (a list of lists), the whole reply failed validation and the raw JSON was shown as the answer (`q34`).

**Apply it.** Validate the answer, sources and `found` strictly; accept any shape in fields the user never sees.

---

## Phase 6 — Validation (CRAG)

### "Partial" doesn't mean "don't answer"
**What happened.** Of 8 questions graded partial (ambiguous), the 4 answerable ones were answered correctly and the 4 unanswerable ones were refused by the answer model. Refusing on "partial" would have lost 4 correct answers and gained nothing. Three unanswerable questions were even graded "correct".

**Lesson.** A retrieval score is a hint to try harder, not proof the answer is or isn't in the documents. CRAG itself answers ambiguous contexts after correcting them.

### Measure what each step actually changes
**What happened.** The corrective search ran 10 times, at about 6 s each, and never gave an answerable question a chunk it used. Its rewrite often repeated the question ("projects user worked on"). Knowledge refinement, by contrast, paid off: the LLM filter that drops unhelpful chunks (`v10`) raised context precision from 0.54 to 0.75 and fixed answers that had a misleading chunk in their context (`q29`).

**Apply it.** Count how often each step changes an answer. Remove or improve the ones that don't, such as skipping a retry when the rewrite is nearly the same question.

### Filters that judge items one by one miss connections
**What happened.** The refinement filter graded each chunk on its own and dropped the revision history ("version 1.2 added QL-1006"). That chunk only matters once you know, from another chunk, that QL-1006 is the error code being asked about (`q37`). Showing the grader the passages already kept fixed it (`v12`: 3 of 3 complete).

**Lesson.** Independent per-item filtering breaks multi-step questions.

**Apply it.** Give a filter the context of what's already kept.

---

## Phase 7 — Production

### Stream the answer and measure TTFT
**What happened.** With streaming, a normal answer starts appearing at about 6 s instead of arriving whole at 14 s. TTFT showed where the wait really is: search about 4 s, validation and filtering about 1 s, and the model's first token only about 0.9 s. The first question after a restart waits 25–75 s while the search models load.

**Lesson.** Users feel time to first token, not total time. Its breakdown tells you what to optimize, which here is retrieval, not the model.

### Structured output needs care when streaming
**What happened.** The cited prompts reply in JSON, which can't be shown raw while it streams. The answer field is extracted as it arrives, including escape characters split across pieces, so users never see braces or the scratch fields.

### Show the evidence, not just the answer
**What happened.** Showing the CRAG grade per query and per chunk, which chunks were read or filtered out, and what each answer cites made the system's behaviour understandable to its user. It also surfaced the "user" wording problem directly in the chat.

**Apply it.** Show users and developers *why* an answer came out the way it did, with chunk names, grades and citations.

---

## Across all phases — habits that kept the work recoverable

- **Versioned profiles, never edited in place.** Each idea became a new profile (`v9` = `v8` + one change). Earlier runs stayed valid, comparisons changed one thing at a time, and switching back was one line.
- **Simulate for free before paying.** Cut-offs, filters and ranking orders were tried on saved runs first; only promising ones got a full evaluation (about $0.35 each).
- **Back up before overwriting.** Verdicts were backed up before re-judging, so `v10` could be restored exactly, checked equal to the backup.
- **A test for every rule.** Unit tests caught real bugs: invisible control characters that disabled two regexes, a missing import, and a test that broke when `v9` was created. The suite grew to 213 tests.
- **Budget awareness.** With a few dollars of credit, every paid run was estimated in advance, judged with gpt-4o-mini, and replaced by free analysis where possible.

## Mistakes worth remembering (including the assistant's)

- **Stating a result before checking it.** It was reported that `v10` "lost no evidence an answer needed"; a re-judge showed it had (`q37`). Verify a claim on the data before stating it.
- **Bugs in the experiment, not the system.** A simulation capped every list at 5 and "lost" the 6th chunk of cross-document questions; a tool labelled every chunk page 1. Test experiment code as carefully as product code.
- **Fixing the judge by adding rules.** One rule ("list alternatives as one fact") fixed one case and broke another (`h26`). The better fix was correcting the reference wording.
- **Shell escaping.** Several edits written through a shell turned `\n` or `\b` into real characters. Use an editor tool for code, and check files for control characters.

## Interview talking points

- **Phase 0/2:** "My golden-set scores hit 0.99, but a held-out set showed 0.89. The prompt rules had been tuned to the golden questions. I built a fresh held-out set and only adopted changes that improved both."
- **Phase 2:** "Half my 'failures' were evaluation bugs. I rebuilt the LLM judge so the model only extracts facts and quotes and code computes the scores, then calibrated it on 17 known answers until it scored all of them right."
- **Phase 3/4:** "A broken PDF table made both my answer model and my judge misread BLEU scores. Fixing it at ingestion fixed both. Fixing table headings and document order raised golden MRR from 0.90 to 0.95."
- **Phase 4:** "Precision@5 was stuck at 0.28 because most questions have one relevant chunk. An adaptive reranker cut-off with guards for cross-document questions doubled the share of relevant chunks the model reads, with no evidence lost, checked against the chunks answers actually cited."
- **Phase 6:** "I completed a CRAG pipeline with knowledge refinement: context precision went from 0.54 to 0.75. Re-judging exposed a multi-step failure, a chunk that only made sense next to another, which I fixed by giving the filter the kept passages as context."
- **Phase 7:** "I stream answers over SSE and measure time to first token: the answer starts at about 6 s instead of 14 s, and the TTFT breakdown showed retrieval, not the model, is the bottleneck."
