"""The validation layer in front of the answer (Phase 6), CRAG-style: grade the retrieved context from the reranker's score and
correct a weak retrieval before answering.

    correct    the best chunk scores at least `high`: answer with the chunks as they are
    ambiguous  between the cutoffs: rewrite the query, search again, merge the two result lists by score, grade again
    incorrect  below `low`: the same corrective search; if the merged chunks are still graded incorrect, do not answer
               ("not found"), without calling the model

The only corrective source is the same documents with a rewritten query: this project has no second source (CRAG's web search).
The scores come from the reranker, so this only makes sense for a version that reranks (RERANK_TOP > 0).
"""
import time

from src.retrieve.rewrite import rewrite_query
from src.validate.threshold import best_score, grade_context


def merge_chunks(first: list[dict], second: list[dict], k: int) -> list[dict]:
    """The k best chunks of two result lists by score; a chunk found by both keeps its better score."""
    best = {}
    for chunk in first + second:
        score = chunk.get("score") or 0.0
        if chunk["chunk_id"] not in best or score > (best[chunk["chunk_id"]].get("score") or 0.0):
            best[chunk["chunk_id"]] = chunk
    return sorted(best.values(), key=lambda c: c.get("score") or 0.0, reverse=True)[:k]


def is_new_query(query: str, question: str) -> bool:
    return " ".join(query.lower().split()) != " ".join(question.lower().split())


def validate_with_retry(question: str, chunks: list[dict], llm, search, k: int, high: float, low: float,
                        retries: int = 1) -> tuple[dict, list[dict]]:
    """(verdict, the chunks the answer must use).

    verdict: {'grade' (after the corrections), 'first_grade', 'best_score', 'action': 'answer' | 'abstain', 'reason',
    'retries': one entry per corrective search (query, grade before and after, search time)}.
    """
    first_grade = grade = grade_context(chunks, high, low)
    attempts = []
    for _ in range(retries if search else 0):
        if grade == "correct":
            break
        query = rewrite_query(question, llm)
        if not is_new_query(query, question):
            attempts.append({"query": query, "skipped": "the rewrite repeats the question"})
            break
        start = time.perf_counter()
        extra = search(query, k)
        search_ms = round((time.perf_counter() - start) * 1000)
        chunks = merge_chunks(chunks, extra, k)
        attempts.append({"query": query, "search_ms": search_ms, "grade_before": grade, "best_retry_score": best_score(extra)})
        grade = grade_context(chunks, high, low)
        attempts[-1]["grade_after"] = grade
    verdict = {"grade": grade, "first_grade": first_grade, "best_score": best_score(chunks), "action": "answer", "reason": "",
               "retries": attempts}
    if grade == "incorrect":
        verdict.update(action="abstain", reason=f"the best chunk scores below {low}, also after rewriting the query: "
                                                "nothing relevant was retrieved")
    return verdict, chunks
