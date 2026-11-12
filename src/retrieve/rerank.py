"""Reranking: a cross-encoder reads the question and one chunk together and scores how well the chunk answers it.

It is more accurate than comparing two separate embeddings, but slow, so it only re-orders a shortlist
(over-retrieve 10 or 20, rerank, keep 5). Every reranker here is an open model that runs locally, so no document text
leaves the machine.
"""
RERANKERS = {
    "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",     # small and fast, trained on web-search passages
    "bge-base": "BAAI/bge-reranker-base",                 # multilingual, 278M parameters
    "bge-v2-m3": "BAAI/bge-reranker-v2-m3",               # the larger BGE reranker, 568M parameters
    "mxbai-base": "mixedbread-ai/mxbai-rerank-base-v1",   # too slow on this CPU (about 10 s per chunk), so not in the experiments
}
DEFAULT_RERANKER = "minilm"
RERANKER = RERANKERS[DEFAULT_RERANKER]
_loaded: dict = {}


class CrossEncoderReranker:
    def __init__(self, name: str = RERANKER):
        self.name = name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            try:
                # A model that is already on disk is loaded without any network call: a request to the model hub that never
                # returns would otherwise hang the first query for good.
                self._model = CrossEncoder(self.name, max_length=512, local_files_only=True)
            except OSError:
                self._model = CrossEncoder(self.name, max_length=512)       # not downloaded yet: fetch it
        return self._model

    def warm_up(self) -> None:
        """Load the model now, so the first query does not wait for it."""
        self._load().predict([("warm up", "warm up")], show_progress_bar=False)

    def rerank(self, query: str, hits: list[dict], k: int) -> list[dict]:
        """The k hits that answer the query best. `score` becomes the cross-encoder score (not a cosine similarity);
        the score the hit had before is kept as `retrieval_score`."""
        if not hits:
            return []
        scores = self._load().predict([(query, hit["text"]) for hit in hits], show_progress_bar=False)
        ranked = sorted(zip(scores, hits), key=lambda pair: -float(pair[0]))[:k]
        return [{**hit, "rank": rank, "score": round(float(score), 4), "retrieval_score": hit["score"]}
                for rank, (score, hit) in enumerate(ranked, start=1)]


def get_reranker(key: str = DEFAULT_RERANKER) -> CrossEncoderReranker:
    """The reranker for a key in RERANKERS; each model is loaded once and reused."""
    if key not in RERANKERS:
        raise ValueError(f"unknown reranker '{key}'. Choose from {list(RERANKERS)}")
    if key not in _loaded:
        _loaded[key] = CrossEncoderReranker(RERANKERS[key])
    return _loaded[key]
