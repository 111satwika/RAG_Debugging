"""Hybrid search: merge the dense ranking and the BM25 ranking, and merge rankings in other ways."""
RRF_K = 60     # the usual constant: a hit at rank 1 scores 1/61, at rank 10 1/70, so the top ranks count but do not dominate


def rrf(rankings: dict[str, list[dict]], k: int = RRF_K, top: int | None = None) -> list[dict]:
    """Reciprocal Rank Fusion. Every list gives a chunk 1 / (k + rank); the scores are added.

    It uses ranks only, so lists with different score scales (cosine similarity against BM25) combine without tuning.
    Each hit records the rank it had in every list as `from`.
    """
    scores: dict[str, float] = {}
    hits: dict[str, dict] = {}
    for name, ranking in rankings.items():
        for rank, hit in enumerate(ranking, start=1):
            chunk_id = hit["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            hits.setdefault(chunk_id, {**hit, "from": {}})["from"][name] = rank
    ordered = sorted(scores, key=lambda chunk_id: -scores[chunk_id])[:top]
    return [{**hits[chunk_id], "rank": rank, "score": round(scores[chunk_id], 4)}
            for rank, chunk_id in enumerate(ordered, start=1)]


def round_robin(lists: list[list[dict]], k: int) -> list[dict]:
    """Take the best chunk of each list in turn (skipping repeats) until k are chosen, so every list gets a fair share of
    the first k places. Each list keeps its own order. Used to merge the results of several queries."""
    out, seen, position = [], set(), 0
    while len(out) < k and any(position < len(items) for items in lists):
        for items in lists:
            if position < len(items) and items[position]["chunk_id"] not in seen and len(out) < k:
                seen.add(items[position]["chunk_id"])
                out.append(items[position])
        position += 1
    return [{**hit, "rank": rank} for rank, hit in enumerate(out, start=1)]


def with_minimum_share(ranking: list[dict], docs: list[str], k: int, minimum: int = 2) -> list[dict]:
    """The top k of `ranking`, except that every document in `docs` gets at least `minimum` places when the ranking holds
    that many of its chunks (a per-document quota). The best chunks stay in score order; the lowest-ranked chunks of a
    document that has more than its share make room."""
    minimum = max(1, min(minimum, k // max(len(docs), 1)))
    position = {hit["chunk_id"]: i for i, hit in enumerate(ranking)}
    chosen, rest = list(ranking[:k]), list(ranking[k:])

    def count(doc):
        return sum(hit["doc_id"] == doc for hit in chosen)

    for doc in docs:
        for extra in [hit for hit in rest if hit["doc_id"] == doc]:
            if count(doc) >= minimum:
                break
            victim = next((hit for hit in reversed(chosen) if hit["doc_id"] != doc and count(hit["doc_id"]) > minimum), None)
            if victim is None:
                break
            chosen.remove(victim)
            chosen.append(extra)
            rest.remove(extra)
    chosen.sort(key=lambda hit: position[hit["chunk_id"]])
    return [{**hit, "rank": rank} for rank, hit in enumerate(chosen, start=1)]
