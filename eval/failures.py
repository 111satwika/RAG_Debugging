"""Label the failing stage of a wrong answer, following the Debugging Checklist in the skill.

indexing | chunking | filter | retrieval | generation | data_quality
(`filter` is never assigned yet: there are no metadata filters until Phase 4.)
`retrieval` covers both a key chunk ranked too low (it is in the top-20 but not the top-k) and one that is not
found at all; the note says which. These labels are heuristics from the retrieved chunks and the key facts.
Open the trace to confirm.
"""
from eval.metrics import SINGLE_FACT_TYPES, chunk_body, contains, coverage, source_doc_ids


def _union_coverage(chunk_ids: list[str], by_id: dict, keys: list[str]) -> float:
    text = "\n".join(chunk_body(by_id[c]) for c in chunk_ids if c in by_id)
    return coverage(text, keys)


NEGATIVE_EVIDENCE = ("not mention", "doesn't mention", "no evidence", "does not show", "doesn't show", "not shown",
                     "not found", "no mention", "does not state", "not stated", "does not say")


def is_negative_evidence(keys: list) -> bool:
    """True when every key fact is a phrase like "not mention": the answer is that a document lacks something, so the
    facts can never be in a chunk and keyword coverage says nothing about retrieval (h17)."""
    groups = [k if isinstance(k, list) else [k] for k in keys]
    return bool(groups) and all(any(n in alt.lower() for n in NEGATIVE_EVIDENCE) for g in groups for alt in g)


def classify(question: dict, result: dict, chunks: list[dict], deep_ids: list[str]) -> tuple[str, str]:
    """Return (stage, note). `deep_ids` are the chunk ids of a wider search (top-20)."""
    by_id = {c["chunk_id"]: c for c in chunks}
    keys = question.get("must_include") or []
    if is_negative_evidence(keys):
        keys = []           # source docs are still checked below; past that it is a generation failure
    answer = result.get("answer") or ""
    judge = result.get("judge") or {}

    if question["type"] == "unanswerable":
        return "generation", "answered a question the documents cannot answer"

    source = source_doc_ids(question)
    missing_docs = source - {c["doc_id"] for c in chunks}
    if missing_docs:
        return "indexing", f"not in the index: {sorted(missing_docs)}"

    retrieved = [h["chunk_id"] for h in result["retrieved"]]
    context = " ".join(by_id[c]["text"] for c in retrieved if c in by_id)
    stale = [f for f in question.get("must_not_include") or [] if contains(answer, f) and contains(context, f)]
    if stale:
        return "data_quality", f"answer used a stale/conflicting value that was in the context: {stale}"

    if keys and question["type"] in SINGLE_FACT_TYPES:
        best = max((coverage(chunk_body(c), keys) for c in chunks if c["doc_id"] in source), default=0.0)
        if best < 1:
            return "chunking", f"no single chunk holds all key facts (best chunk covers {best:.0%})"

    if keys and _union_coverage(retrieved, by_id, keys) < 1:
        if _union_coverage(deep_ids, by_id, keys) >= 1:
            return "retrieval", "a key chunk is in the top-20 but not all key facts are in the top-k (ranked too low)"
        return "retrieval", "the key facts are not all retrieved even in the top-20 (not found)"

    if judge.get("abstained"):
        return "generation", "the facts were retrieved but the model abstained"
    return "generation", "the facts were retrieved but the answer is wrong, incomplete or unsupported"
