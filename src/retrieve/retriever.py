"""Phase 4 retrieval: one retriever that can search dense, BM25 or hybrid, filter by document, over-retrieve and rerank,
rewrite the query, and transform hard questions (rewriting, expansion, multi-query), according to a RetrievalSpec.
`Retriever.searcher(spec)` returns a `search(question, k)` function that the pipeline and the eval harness accept in place
of plain dense search.
"""
import re
import threading
import time
from dataclasses import dataclass, replace

from src.config import ASKED_FIRST, QUERY_TRANSFORM, RERANK_MIN_RATIO, RERANK_TOP, RERANKER, RETRIEVAL_MODE
from src.generate.llm import load_llm
from src.ingest.embed import Embedder, load_embedder
from src.ingest.index import load_parents
from src.observability.logger import get_logger
from src.retrieve.bm25 import BM25Index
from src.retrieve.dense import doc_filter, get_collection, search as dense_search
from src.retrieve.filters import build_catalog, route_with_llm
from src.retrieve.hybrid import RRF_K, round_robin, rrf, with_minimum_share
from src.retrieve.rerank import DEFAULT_RERANKER, RERANKERS, CrossEncoderReranker, get_reranker
from src.retrieve.rewrite import rewrite_query
from src.retrieve.transform import (MAX_SUB_QUERIES, QueryPlan, SubQuery, document_names, documents_named, gate, make_plan,
                                    with_expansions)

log = get_logger("retrieve")
MODES = ("dense", "bm25", "hybrid")
ROUTES = ("none", "oracle", "llm")
TRANSFORMS = ("none", "auto", "always")    # auto: only the questions the gate in transform.py selects
HYBRID_FETCH = 20      # each list contributes this many candidates to the fusion
SHARE_POOL = 20        # when a question needs several documents, look this deep so each can have its share
MIN_SHARE = 2          # the places in the top k each of those documents is guaranteed
FUSE_QUESTION = False  # multi-query: also search each document with the question itself and fuse the two (no gain, see SKILL.md)
PER_DOCUMENT_SLOTS = 3  # multi-query: places each document gets in the answer. 3 each (6 chunks for two documents) rather than
                        # splitting k as 3 + 2: replaying the 9 recorded plans, the evidence chunk is third in its document's list
                        # in the failing ones, and this fixes both (7 of 9 became 9 of 9)
PER_DOCUMENT_POOL = 5   # multi-query with a reranker: candidates reranked per document. 5 is what rerank_top 10 gave two documents; fixed so that
                        # lowering rerank_top for ordinary questions (v4 uses 7: same quality, 3 s faster) leaves cross-document questions as they were
CACHE_LIMIT = 256      # query vectors, rewrites and routes remembered per retriever (oldest dropped first)


def _remember(cache: dict, key, value):
    if len(cache) >= CACHE_LIMIT:
        cache.pop(next(iter(cache)))
    cache[key] = value
    return value


@dataclass(frozen=True)
class RetrievalSpec:
    mode: str = "dense"
    rerank_top: int = 0       # > 0: over-retrieve this many candidates, rerank them, keep k
    rewrite: bool = False
    route: str = "none"       # which documents to search: "none" = all, "llm" = the model decides, "oracle" = golden set
    reranker: str = DEFAULT_RERANKER     # which model reorders the candidates (a key of RERANKERS)
    transform: str = "none"   # query transformation: "auto" only for the questions the gate selects, "always" for every one
    min_ratio: float = 0.0    # > 0 (with a reranker): drop chunks scoring under this share of the best one (cut_low_scores)
    asked_first: bool = False  # a per-document search lists the document the question asks about first (asked_first)

    def __post_init__(self):
        if self.mode not in MODES or self.route not in ROUTES:
            raise ValueError(f"mode must be one of {MODES} and route one of {ROUTES}")
        if self.rerank_top < 0:
            raise ValueError("rerank_top cannot be negative")
        if self.reranker not in RERANKERS:
            raise ValueError(f"reranker must be one of {list(RERANKERS)}")
        if self.transform not in TRANSFORMS:
            raise ValueError(f"transform must be one of {TRANSFORMS}")
        if not 0 <= self.min_ratio < 1:
            raise ValueError("min_ratio must be at least 0 and below 1")

    @property
    def label(self) -> str:
        parts = [self.mode]
        if self.route != "none":
            parts.append(f"route-{self.route}")
        if self.rerank_top:
            parts.append(f"rerank{self.rerank_top}" + ("" if self.reranker == DEFAULT_RERANKER else f"-{self.reranker}"))
        if self.rewrite:
            parts.append("rewrite")
        if self.transform != "none":
            parts.append(f"transform-{self.transform}")
        if self.min_ratio and self.rerank_top:
            parts.append(f"cut{self.min_ratio:g}")
        if self.asked_first:
            parts.append("asked-first")
        return "-".join(parts)

    @property
    def llm_calls(self) -> int:
        """Model calls added to every query (before the answer itself). With transform "auto" it is the most a query can
        add: only the questions the gate selects make the call."""
        return int(self.rewrite) + int(self.route == "llm") + int(self.transform != "none")


def asked_document(question: str, names: dict[str, set[str]]) -> str | None:
    """The document the question asks about: the one named in its last sentence ("QuickLeave requires Azure AD. Does the
    candidate have experience with Azure AD?" asks about the resume; the spec's part is already stated in the question)."""
    sentences = [s for s in re.split(r"(?<=[.?!])\s+", question.strip()) if s]
    named = documents_named(sentences[-1], names) if sentences else []
    return named[0] if len(named) == 1 else None


def asked_first(question: str, sub_queries: list, lists: list[list[dict]], names: dict[str, set[str]]) -> list[list[dict]]:
    """Put the per-document list of the asked-about document first, so the interleaved ranking starts with its best chunk.
    In plan order the spec came first, so the resume's evidence could never be at rank 1 (q46, q47)."""
    asked = asked_document(question, names)
    order = sorted(range(len(lists)), key=lambda i: sub_queries[i].document != asked)     # stable: the rest keep their order
    return [lists[i] for i in order]


def cut_low_scores(hits: list[dict], ratio: float, per_document: int = MIN_SHARE) -> list[dict]:
    """Drop the reranked chunks that score under `ratio` x the best score, so the model reads fewer unrelated chunks.

    Most questions have one relevant chunk, so always sending k=5 caps Precision@5 near 0.2-0.3 (eval: 0.28 on the golden
    set). Replaying the v8 runs, a ratio of 0.02 doubled precision (0.28 -> 0.55) and kept every chunk an answer cited
    except one optional detail; 0.05 already dropped evidence (q37) and 0.1 more (q16). Two guards:
      - a per-document search (multi-query, hits carry "query") is cut within its own list: scores reranked against
        different queries are not comparable;
      - a question the transform gate marks as cross-document keeps the best `per_document` chunks of each document it
        names: the reranker scores real evidence low there (q24: the Azure AD SSO chunk had 2.2% of the top score).
    The best chunk of every list is always kept.
    """
    if ratio <= 0 or not hits:
        return hits
    info = next((h["transform"] for h in hits if h.get("transform")), None) or {}
    named = [d.strip() for r in info.get("reasons", []) if r.startswith("cross_document:") for d in r.split(":", 1)[1].split(",")]
    protected = {h["chunk_id"] for d in named for h in [h for h in hits if h["doc_id"] == d][:per_document]}
    groups: dict = {}
    for h in hits:
        groups.setdefault(h.get("query"), []).append(h)
    keep = set(protected)
    for group in groups.values():
        best = max(h["score"] for h in group)
        keep |= {h["chunk_id"] for h in group if h["score"] >= ratio * best}
        keep.add(max(group, key=lambda h: h["score"])["chunk_id"])
    return [{**h, "rank": rank} for rank, h in enumerate((h for h in hits if h["chunk_id"] in keep), start=1)]


def load_corpus(collection, parents: dict | None = None) -> list[dict]:
    """Every chunk the model can read, shaped like a search hit. With parent-child retrieval these are the parents."""
    if collection.metadata.get("parent_child"):
        return [{"chunk_id": p["chunk_id"], "doc_id": p["doc_id"], "doc_type": p["doc_type"], "page": p["page"],
                 "text": p["text"], "parent_id": ""} for p in (parents or load_parents(collection.name)).values()]
    data = collection.get(include=["documents", "metadatas"])
    return [{"chunk_id": i, "doc_id": m["doc_id"], "doc_type": m["doc_type"], "page": m["page"], "text": t,
             "parent_id": m.get("parent_id", "")} for i, t, m in zip(data["ids"], data["documents"], data["metadatas"])]


class Retriever:
    def __init__(self, embedder: Embedder, collection, parents: dict | None = None, llm=None,
                 reranker: CrossEncoderReranker | None = None, oracle: dict[str, list[str]] | None = None):
        """`reranker` replaces the model of every spec (used by tests); by default each spec names its own."""
        self.embedder, self.collection, self.parents = embedder, collection, parents
        self.llm, self.reranker, self.oracle = llm, reranker, oracle or {}
        self.corpus = load_corpus(collection, parents)
        self.bm25 = BM25Index(self.corpus)
        self.catalog = build_catalog(self.corpus)
        self.names = document_names(self.corpus, self.catalog)    # words that point at each document (for the gate)
        self.llm_ms: list[float] = []            # how long each model call (rewrite, route or plan) took
        self._vectors, self._rewrites, self._routes, self._plans = {}, {}, {}, {}

    # -- the pieces

    def _need_llm(self):
        if self.llm is None:
            raise RuntimeError("rewrite and llm routing need an LLM: pass llm=...")
        return self.llm

    def rewrite(self, question: str) -> str:
        if question not in self._rewrites:
            started = time.perf_counter()
            _remember(self._rewrites, question, rewrite_query(question, self._need_llm()))
            self.llm_ms.append((time.perf_counter() - started) * 1000)
        return self._rewrites[question]

    def route(self, question: str, how: str) -> list[str] | None:
        """The documents to search, or None for all of them."""
        if how == "oracle":
            return self.oracle.get(question)
        if how == "llm":
            if question not in self._routes:
                started = time.perf_counter()
                _remember(self._routes, question, route_with_llm(question, self.catalog, self._need_llm()))
                self.llm_ms.append((time.perf_counter() - started) * 1000)
            return self._routes[question]
        return None

    def _dense(self, query: str, n: int, docs: list[str] | None) -> list[dict]:
        if query not in self._vectors:
            _remember(self._vectors, query, self.embedder.embed_queries([query])[0])
        return dense_search(self._vectors[query], n, self.embedder, self.collection, self.parents, doc_filter(docs))

    def _candidates(self, mode: str, query: str, n: int, docs: list[str] | None) -> list[dict]:
        if mode == "dense":
            return self._dense(query, n, docs)
        if mode == "bm25":
            return self.bm25.search(query, n, docs)
        fetch = max(n, HYBRID_FETCH)
        return rrf({"dense": self._dense(query, fetch, docs), "bm25": self.bm25.search(query, fetch, docs)}, RRF_K, n)

    # -- query transformation (hard questions only)

    def plan_for(self, spec: RetrievalSpec, question: str) -> tuple[QueryPlan, dict] | None:
        """The transformation plan for this question, or None when it is searched as it is: transformation is off, the
        gate (a free rule, no model) does not select it, or the model's plan is unusable."""
        if spec.transform == "none":
            return None
        reasons = ["always"] if spec.transform == "always" else gate(question, self.names)
        if not reasons:
            return None
        if question not in self._plans:
            llm = self._need_llm()                               # a missing model is a setup error, not something to hide
            try:
                plan = make_plan(question, self.catalog, llm)
            except Exception:                                    # a model failure must not fail the search
                log.exception("query plan failed; searching the question as it is")
                return None
            if plan is None:
                log.warning("unusable query plan; searching the question as it is")
                return None
            self.llm_ms.append(plan.ms)
            _remember(self._plans, question, plan)
        plan = self._plans[question]
        named = documents_named(question, self.names)
        if not plan.multi_query and len(named) >= 2:
            # The model gave no per-document queries (for example the question already states the spec's half), but the
            # question names two documents: search each of them, with the rewritten question.
            plan = replace(plan, sub_queries=[SubQuery(doc, plan.rewrite) for doc in named[:MAX_SUB_QUERIES]])
        info = {"reasons": reasons, "rewrite": plan.rewrite, "expansions": plan.expansions,
                "sub_queries": [{"document": s.document, "query": s.query} for s in plan.sub_queries],
                "llm_ms": round(plan.ms), "tokens_in": plan.tokens_in, "tokens_out": plan.tokens_out}
        return plan, info

    def _search_with_plan(self, spec: RetrievalSpec, question: str, plan: QueryPlan, info: dict, k: int) -> list[dict]:
        reranker = (self.reranker or get_reranker(spec.reranker)) if spec.rerank_top else None
        limit = k
        if plan.multi_query:
            # One search per document the question needs (a metadata filter keeps each search inside its document), each
            # with its own small pool and reranked against its own query (the reranker reads the same number of chunks as
            # a normal search), then merged so that every query gets a fair share of the top k.
            n = len(plan.sub_queries)
            share = PER_DOCUMENT_SLOTS or max(2, -(-k // n))
            total = share * n if PER_DOCUMENT_SLOTS else k
            limit = total
            pool = (PER_DOCUMENT_POOL if spec.rerank_top else 0) or (max(share, spec.rerank_top // n) if spec.rerank_top else share)
            lists = []
            for sub in plan.sub_queries:
                query = with_expansions(sub.query, plan.expansions)
                docs = [sub.document] if sub.document else None
                hits = self._candidates(spec.mode, query, pool, docs)
                if docs and FUSE_QUESTION:
                    # The model words each sub-query a little differently on every run, so a sub-query alone can miss the
                    # chunk that has the question's own words. Search the same document with the question as well.
                    hits = rrf({"sub_query": hits, "question": self._candidates(spec.mode, question, pool, docs)}, RRF_K, pool)
                if reranker:
                    hits = reranker.rerank(query, hits, len(hits))
                lists.append([{**hit, "query": query} for hit in hits])
            if spec.asked_first:
                lists = asked_first(question, plan.sub_queries, lists, self.names)
            merged = round_robin(lists, total)
        else:
            # One need: search the question and its rewrite (with the expansion terms), merge, and rerank by the rewrite.
            query = with_expansions(plan.rewrite, plan.expansions)
            pool = max(k, spec.rerank_top)
            fused = rrf({"original": self._candidates(spec.mode, question, pool, None),
                         "rewrite": self._candidates(spec.mode, query, pool, None)}, RRF_K, pool)
            merged = reranker.rerank(query, fused, k) if reranker else fused[:k]
        return [{**hit, "rank": rank, "transform": info} for rank, hit in enumerate(merged[:limit], start=1)]

    # -- the whole search

    def search(self, spec: RetrievalSpec, question: str, k: int) -> list[dict]:
        hits = self._search(spec, question, k)
        return cut_low_scores(hits, spec.min_ratio) if spec.rerank_top else hits     # only reranker scores are comparable

    def _search(self, spec: RetrievalSpec, question: str, k: int) -> list[dict]:
        planned = self.plan_for(spec, question)
        if planned:
            return self._search_with_plan(spec, question, *planned, k)
        query = self.rewrite(question) if spec.rewrite else question
        docs = self.route(question, spec.route)
        several = bool(docs) and len(docs) > 1
        pool = max(k, spec.rerank_top, SHARE_POOL if several else 0)       # a wide enough pool to give each document its share
        hits = self._candidates(spec.mode, query, pool, docs)
        if spec.rerank_top:
            hits = (self.reranker or get_reranker(spec.reranker)).rerank(query, hits, len(hits))
        hits = with_minimum_share(hits, docs, k, MIN_SHARE) if several else hits[:k]
        return [{**hit, "rank": rank} for rank, hit in enumerate(hits, start=1)]

    def searcher(self, spec: RetrievalSpec):
        return lambda question, k: self.search(spec, question, k)


# ---------------------------------------------------------------- the active pipeline version

_shared: Retriever | None = None
_shared_lock = threading.Lock()


def configured_spec() -> RetrievalSpec:
    """The retrieval of the active pipeline version (src/config.py)."""
    return RetrievalSpec(RETRIEVAL_MODE, RERANK_TOP, reranker=RERANKER, transform=QUERY_TRANSFORM, min_ratio=RERANK_MIN_RATIO,
                         asked_first=ASKED_FIRST)


def shared_retriever() -> Retriever:
    """One retriever for the whole process: it reads the index once and builds the BM25 index once."""
    global _shared
    with _shared_lock:
        if _shared is None:
            embedder = load_embedder()
            retriever = Retriever(embedder, get_collection(embedder), llm=load_llm() if QUERY_TRANSFORM != "none" else None)
            if RERANK_TOP:
                get_reranker(RERANKER).warm_up()
            _shared = retriever
        return _shared


def reset_shared_retriever() -> None:
    """Forget the shared retriever, so the next search reads the index again (after the index is rebuilt)."""
    global _shared
    with _shared_lock:
        _shared = None


def configured_search():
    """A `search(question, k)` function for the active version, or None when it is plain dense search
    (the pipeline then uses src/retrieve/dense.py directly)."""
    spec = configured_spec()
    if spec.mode == "dense" and not spec.rerank_top and spec.transform == "none":
        return None
    return lambda question, k: shared_retriever().search(spec, question, k)
