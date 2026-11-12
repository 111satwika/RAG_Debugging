"""Relevance grade of the retrieved context (the CRAG idea: grade the evidence before answering).

The grade comes from the reranker's score for the best chunk (0 to 1):
    correct     the best chunk scores at least `high`: answer directly
    ambiguous   between the two cutoffs: correct the retrieval (rewrite the query and search again, src/validate/gate.py)
    incorrect   below `low`: the same correction; if the chunks are still incorrect, answer "not found" without calling the model

The reranker's score is not calibrated across question styles: on the golden set a few answerable questions score under 0.02
(the answer sits in a table or a short header chunk) while unanswerable ones can score above 0.9. So `low` is meant to be very
low. See eval/validation_experiments.py.
"""


def best_score(chunks: list[dict]) -> float | None:
    scores = [c["score"] for c in chunks if c.get("score") is not None]
    return max(scores) if scores else None


def grade_score(score: float | None, high: float, low: float) -> str:
    """The CRAG grade of one retrieved chunk from its reranker score, with the cutoffs of the context grade."""
    if score is None:
        return "incorrect"
    if score >= high:
        return "correct"
    return "ambiguous" if score >= low else "incorrect"


def grade_chunks(chunks: list[dict], high: float, low: float) -> list[dict]:
    """Every retrieved chunk with its CRAG grade (CRAG grades each retrieved document, not only the best one)."""
    return [{"chunk_id": c["chunk_id"], "doc_id": c.get("doc_id"), "page": c.get("page"), "score": c.get("score"),
             "grade": grade_score(c.get("score"), high, low)} for c in chunks]


def grade_context(chunks: list[dict], high: float, low: float) -> str:
    """'correct', 'ambiguous' or 'incorrect' from the best reranker score. No chunks at all is incorrect."""
    if low > high:
        raise ValueError("the low cutoff must not be above the high cutoff")
    top = best_score(chunks)
    if top is None:
        return "incorrect"
    if top >= high:
        return "correct"
    return "ambiguous" if top >= low else "incorrect"
