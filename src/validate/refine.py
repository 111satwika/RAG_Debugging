"""Knowledge refinement, the CRAG step the validation gate (src/validate/gate.py) does not do: judge each retrieved chunk
on its content and drop the ones that do not help answer the question, before the answer model reads them.

The gate grades the context as a whole by its best reranker score, so the chunks next to a good best chunk always pass, and
v9's score cut-off (src/retrieve/retriever.py cut_low_scores) can only drop chunks the reranker already scores far below
the best. On the v9 runs half of the chunks the model read were still irrelevant: 34 of 148 scored high but were off topic,
36 sat in a 2-20% band where evidence and noise score alike (eval: relevant / chunks given 0.55).

One model call per question grades every chunk except the ones that are always kept:
  - the best chunk (the gate already judged it);
  - for a cross-document question, the best chunk of each document it names.
The prompt leans towards keeping: a chunk that holds part of the answer, or that the answer may need, is kept. A model
failure keeps every chunk.
"""
import json
import logging
import time

log = logging.getLogger("rag.refine")

REFINE_PROMPT = """You decide which passages a question answering system should read. The question will be answered from the passages you keep.

Question: {question}

Passages:
{passages}

For each numbered passage, answer "keep" if it states anything needed to answer the question, fully or in part: a value, a name, a rule, a row of a table, a heading that says which item a fact belongs to, or a fact that shows the answer is "no". Answer "drop" only if the passage is about something else and the answer cannot use it. When unsure, keep it.

Reply with JSON only, one entry per passage, keyed by its number, in exactly this shape:
{{"passages": {{"1": "keep", "2": "drop"}}}}
"""

# With context: the grader also sees the passages that are kept anyway, so it can tell when a passage completes them. Graded
# alone, the revision history ("1.2 ... Added error code QL-1006") looked off topic for "which version introduced the error
# code for requests dated too far in the past?" because only the kept error-code table says QL-1006 is that code (q37).
REFINE_WITH_CONTEXT_PROMPT = """You decide which passages a question answering system should read. The question will be answered from the passages you keep.

Question: {question}

Already kept (for reference only, do not grade these):
{kept}

Passages to grade:
{passages}

For each numbered passage, answer "keep" if it states anything needed to answer the question, fully or in part: a value, a name, a rule, a row of a table, a heading that says which item a fact belongs to, or a fact that shows the answer is "no". Also keep a passage that completes the kept passages: one that says when, where, why or by whom something in a kept passage was added, changed or decided (for example a code, a value or an item the kept passage names). Answer "drop" only if the passage is about something else and the answer cannot use it. When unsure, keep it.

Reply with JSON only, one entry per passage, keyed by its number, in exactly this shape:
{{"passages": {{"1": "keep", "2": "drop"}}}}
"""


def always_kept(chunks: list[dict]) -> set[str]:
    """The best chunk, and for a cross-document question the best chunk of each document it names."""
    if not chunks:
        return set()
    keep = {max(chunks, key=lambda c: c.get("score") or 0.0)["chunk_id"]}
    info = next((c["transform"] for c in chunks if c.get("transform")), None) or {}
    named = [d.strip() for r in info.get("reasons", []) if r.startswith("cross_document:") for d in r.split(":", 1)[1].split(",")]
    for doc in named:
        in_doc = [c for c in chunks if c.get("doc_id") == doc]
        if in_doc:
            keep.add(max(in_doc, key=lambda c: c.get("score") or 0.0)["chunk_id"])
    return keep


def refine_chunks(question: str, chunks: list[dict], llm, with_context: bool = False) -> tuple[list[dict], dict]:
    """The chunks worth reading, in their original order, and a record for the trace. `with_context` shows the grader the
    always-kept passages, so a passage that completes them is kept (REFINE_WITH_CONTEXT_PROMPT)."""
    started = time.perf_counter()
    fixed = always_kept(chunks)
    graded = [c for c in chunks if c["chunk_id"] not in fixed]
    record = {"kept_always": sorted(fixed), "graded": len(graded), "dropped": [], "tokens_in": 0, "tokens_out": 0}
    if not graded:
        record["ms"] = round((time.perf_counter() - started) * 1000)
        return chunks, record
    passages = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(graded, start=1))
    try:
        if with_context:
            kept = "\n\n".join(c["text"] for c in chunks if c["chunk_id"] in fixed)
            prompt = REFINE_WITH_CONTEXT_PROMPT.format(question=question, kept=kept, passages=passages)
        else:
            prompt = REFINE_PROMPT.format(question=question, passages=passages)
        reply = llm.complete(prompt, json_mode=True)
        verdicts = json.loads(reply["text"])["passages"]
        if isinstance(verdicts, list):
            verdicts = {str(i): v for i, v in enumerate(verdicts, start=1)}
        record.update(tokens_in=reply.get("input_tokens") or 0, tokens_out=reply.get("output_tokens") or 0)
    except Exception:                                     # a failed or unreadable reply keeps every chunk
        log.exception("refinement failed; keeping every chunk")
        record.update(error=True, ms=round((time.perf_counter() - started) * 1000))
        return chunks, record
    # only an explicit "drop" removes a chunk: a missing or unclear verdict keeps it
    dropped = {c["chunk_id"] for i, c in enumerate(graded, start=1) if str(verdicts.get(str(i), "keep")).strip().lower() == "drop"}
    record.update(dropped=sorted(dropped), ms=round((time.perf_counter() - started) * 1000))
    return [c for c in chunks if c["chunk_id"] not in dropped], record
