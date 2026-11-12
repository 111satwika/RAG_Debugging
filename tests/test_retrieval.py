import json
import re
import zlib

import chromadb
import pytest

from src.ingest.index import build_index
from src.retrieve.bm25 import BM25Index, tokenize
from src.retrieve.dense import doc_filter, search
from src.retrieve.filters import build_catalog, oracle_routes, parse_doc_ids, route_with_llm
from src.retrieve.hybrid import rrf, with_minimum_share
from src.retrieve.rerank import RERANKERS, CrossEncoderReranker, get_reranker
from src.retrieve.retriever import RetrievalSpec, Retriever
from src.retrieve.rewrite import rewrite_query


def hit(chunk_id, text="", doc_id="d1", score=0.0):
    return {"chunk_id": chunk_id, "doc_id": doc_id, "doc_type": "spec", "page": 1, "text": text, "score": score, "parent_id": ""}


# ---------------------------------------------------------------- BM25

def test_tokenize_keeps_codes_whole_and_also_split():
    assert tokenize("Error QL-1003: d_model=512, 0.831!") == ["error", "ql-1003", "ql", "1003", "d_model", "d", "model", "512", "0.831", "0", "831"]


def test_tokenize_drops_filler_words_unless_asked_to_keep_them():
    assert tokenize("What is the candidate's current job title?") == ["candidate", "current", "job", "title"]
    assert tokenize("What is the", drop_stopwords=False) == ["what", "is", "the"]


def test_a_question_of_filler_words_matches_nothing():
    chunks = [hit("a", "what is the leave policy"), hit("b", "the rules")]
    assert BM25Index(chunks).search("what is the", 5) == []


def test_bm25_finds_an_exact_code_and_ignores_chunks_without_the_query_words():
    chunks = [hit("a", "QL-1003 means the balance is too low"), hit("b", "leave requests and approvals"),
              hit("c", "balance of leave days")]
    results = BM25Index(chunks).search("what does QL-1003 mean", 5)
    assert results[0]["chunk_id"] == "a"
    assert {r["chunk_id"] for r in BM25Index(chunks).search("zzzz", 5)} == set()      # no word in common, no hit


def test_bm25_prefers_rare_words_and_can_filter_by_document():
    chunks = [hit("a", "leave leave policy", doc_id="x"), hit("b", "leave warmup", doc_id="y"), hit("c", "leave", doc_id="y")]
    index = BM25Index(chunks)
    assert index.search("leave warmup", 3)[0]["chunk_id"] == "b"         # 'warmup' is rare, 'leave' is everywhere
    assert [r["chunk_id"] for r in index.search("leave", 3, ["x"])] == ["a"]


# ---------------------------------------------------------------- fusion

def test_rrf_rewards_a_chunk_that_both_lists_like_and_records_where_it_came_from():
    dense = [hit("a"), hit("b"), hit("c")]
    keyword = [hit("c"), hit("d")]
    fused = rrf({"dense": dense, "bm25": keyword})
    assert [h["chunk_id"] for h in fused] == ["c", "a", "b", "d"]         # c: rank 3 and 1; a: rank 1 only; b and d tie at rank 2
    assert fused[0]["from"] == {"dense": 3, "bm25": 1} and [h["rank"] for h in fused] == [1, 2, 3, 4]
    assert len(rrf({"dense": dense, "bm25": keyword}, top=2)) == 2


def test_minimum_share_keeps_score_order_and_gives_each_document_its_places():
    ranking = [hit("s1", doc_id="spec"), hit("s2", doc_id="spec"), hit("s3", doc_id="spec"), hit("s4", doc_id="spec"),
               hit("s5", doc_id="spec"), hit("r1", doc_id="resume"), hit("r2", doc_id="resume")]
    out = with_minimum_share(ranking, ["spec", "resume"], 5, minimum=2)
    assert [h["chunk_id"] for h in out] == ["s1", "s2", "s3", "r1", "r2"]       # the best stay first; the weakest make room
    assert [h["rank"] for h in out] == [1, 2, 3, 4, 5]
    already = [hit("s1", doc_id="spec"), hit("r1", doc_id="resume"), hit("s2", doc_id="spec"), hit("r2", doc_id="resume"),
               hit("s3", doc_id="spec"), hit("r3", doc_id="resume")]
    assert [h["chunk_id"] for h in with_minimum_share(already, ["spec", "resume"], 5, 2)] == ["s1", "r1", "s2", "r2", "s3"]
    assert [h["chunk_id"] for h in with_minimum_share(ranking[:5], ["spec", "resume"], 5, 2)] == ["s1", "s2", "s3", "s4", "s5"]
    three = with_minimum_share(ranking, ["spec", "resume", "paper"], 5, 2)         # 5 places cannot give 3 documents 2 each
    assert len(three) == 5 and sum(h["doc_id"] == "resume" for h in three) == 1


# ---------------------------------------------------------------- filters and rewrite

CATALOG = [{"doc_id": "resume", "doc_type": "resume", "title": "Jane Doe", "sections": ["SKILLS"]},
           {"doc_id": "spec", "doc_type": "spec", "title": "Spec", "sections": ["1. Intro"]},
           {"doc_id": "paper", "doc_type": "paper", "title": "Paper", "sections": []}]


class ScriptedLLM:
    def __init__(self, text):
        self.text, self.prompts = text, []

    def complete(self, prompt, json_mode=False):
        self.prompts.append(prompt)
        return {"text": self.text}


def test_build_catalog_reads_title_and_sections_from_chunk_headers():
    chunks = [hit("a", "Spec Title > 1. Intro\n\n## 1. Intro\ntext", doc_id="spec"),
              hit("b", "Spec Title > 2. Rules\n\ntext", doc_id="spec"), hit("c", "plain text", doc_id="no_headers")]
    catalog = {d["doc_id"]: d for d in build_catalog(chunks)}
    assert catalog["spec"]["title"] == "Spec Title" and catalog["spec"]["sections"] == ["1. Intro", "2. Rules"]
    assert catalog["no_headers"]["title"] == "no headers"


def test_parse_doc_ids_accepts_only_known_ids_and_falls_back_to_no_filter():
    assert parse_doc_ids('{"doc_ids": ["spec", "resume"]}', CATALOG) == ["spec", "resume"]
    assert parse_doc_ids('{"doc_ids": ["spec", "made_up"]}', CATALOG) == ["spec"]
    assert parse_doc_ids('{"doc_ids": ["made_up"]}', CATALOG) is None
    assert parse_doc_ids('{"doc_ids": ["spec", "resume", "paper"]}', CATALOG) is None     # all documents = no filter
    assert parse_doc_ids("not json", CATALOG) is None and parse_doc_ids('{"doc_ids": "spec"}', CATALOG) is None


def test_route_with_llm_puts_the_documents_and_question_in_the_prompt():
    llm = ScriptedLLM('{"doc_ids": ["spec"]}')
    assert route_with_llm("What is QL-1003?", CATALOG, llm) == ["spec"]
    assert "- spec (spec): Spec; sections: 1. Intro" in llm.prompts[0] and "What is QL-1003?" in llm.prompts[0]


def test_oracle_routes_come_from_the_answer_key_and_skip_questions_with_no_filter():
    golden = [{"question": "one", "source_docs": ["spec.pdf"]}, {"question": "two", "source_docs": ["spec.pdf", "resume.pdf"]},
              {"question": "none", "source_docs": []}, {"question": "all", "source_docs": ["spec.pdf", "resume.pdf", "paper.pdf"]}]
    assert oracle_routes(golden, CATALOG) == {"one": ["spec"], "two": ["spec", "resume"]}


def test_rewrite_query_returns_the_models_query_or_the_question_if_it_is_empty():
    assert rewrite_query("what's QL-1003?", ScriptedLLM('"error code QL-1003 meaning"\n')) == "error code QL-1003 meaning"
    assert rewrite_query("what's QL-1003?", ScriptedLLM("  ")) == "what's QL-1003?"


# ---------------------------------------------------------------- the retriever

class WordEmbedder:
    name, dim = "words", 64

    def _vec(self, text):
        vec = [0.0] * self.dim
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vec[zlib.crc32(word.encode()) % self.dim] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_queries(self, texts):
        return [self._vec(t) for t in texts]


class LengthReranker(CrossEncoderReranker):
    """Scores a chunk by how many query words it contains: no model download."""
    def __init__(self):
        super().__init__("fake")

    def rerank(self, query, hits, k):
        words = set(query.lower().split())
        scored = sorted(hits, key=lambda h: -len(words & set(h["text"].lower().split())))[:k]
        return [{**h, "rank": r, "retrieval_score": h["score"]} for r, h in enumerate(scored, start=1)]


def make_index(docs: dict[str, list[str]], types: dict[str, str] | None = None):
    chunks = []
    for doc_id, texts in docs.items():
        for n, text in enumerate(texts):
            chunks.append({"chunk_id": f"{doc_id}_c{n}", "doc_id": doc_id, "doc_type": (types or {}).get(doc_id, "spec"),
                           "page": 1, "n_tokens": 10, "tokenizer": "t", "text": f"{doc_id} title > S{n}\n\n{text}"})
    return build_index(chunks, WordEmbedder(), chromadb.EphemeralClient(), collection_name="test_chunks")


DOCS = {"spec": ["leave balance rules for employees", "error code QL-1003 means low balance", "audit log of approvals"],
        "resume": ["built audit logging for approvals", "azure ad single sign on experience", "skills python sql"]}


def test_dense_search_can_be_limited_to_documents():
    collection = make_index(DOCS)
    vector = WordEmbedder().embed_queries(["audit log approvals"])[0]
    assert {h["doc_id"] for h in search(vector, 5, collection=collection, where=doc_filter(["resume"]))} == {"resume"}
    assert len(search(vector, 50, collection=collection)) == 6                 # asking for more than exist is fine


def test_retrieval_spec_validates_and_labels_itself():
    assert RetrievalSpec("hybrid", route="llm", rerank_top=20, rewrite=True).label == "hybrid-route-llm-rerank20-rewrite"
    assert RetrievalSpec().label == "dense" and RetrievalSpec("hybrid", rewrite=True).llm_calls == 1
    assert RetrievalSpec("hybrid", route="llm", rewrite=True).llm_calls == 2 and RetrievalSpec("bm25").llm_calls == 0
    for bad in (dict(mode="nope"), dict(route="nope"), dict(rerank_top=-1)):
        with pytest.raises(ValueError):
            RetrievalSpec(**bad)


def test_hybrid_finds_an_exact_code_that_a_different_ranking_would_bury():
    retriever = Retriever(WordEmbedder(), make_index(DOCS), reranker=LengthReranker())
    hits = retriever.search(RetrievalSpec("bm25"), "QL-1003", 3)
    assert hits[0]["chunk_id"] == "spec_c1"
    assert retriever.search(RetrievalSpec("hybrid"), "QL-1003", 3)[0]["chunk_id"] == "spec_c1"
    assert "from" in retriever.search(RetrievalSpec("hybrid"), "QL-1003", 3)[0]


def test_routing_filters_the_search_and_a_two_document_route_gets_a_share_each():
    llm = ScriptedLLM('{"doc_ids": ["resume"]}')
    retriever = Retriever(WordEmbedder(), make_index(DOCS), llm=llm, reranker=LengthReranker())
    assert {h["doc_id"] for h in retriever.search(RetrievalSpec("dense", route="llm"), "audit log", 3)} == {"resume"}
    both = Retriever(WordEmbedder(), make_index(DOCS), llm=ScriptedLLM('{"doc_ids": ["spec", "resume"]}'),
                     reranker=LengthReranker())
    docs = [h["doc_id"] for h in both.search(RetrievalSpec("dense", route="llm"), "audit log of approvals", 2)]
    assert sorted(docs) == ["resume", "spec"]
    assert len(llm.prompts) == 1                                         # the routing answer is remembered


def test_rerank_over_retrieves_then_keeps_k_and_rewrite_changes_the_query_used():
    retriever = Retriever(WordEmbedder(), make_index(DOCS), llm=ScriptedLLM("azure ad single sign on"), reranker=LengthReranker())
    hits = retriever.search(RetrievalSpec("dense", rerank_top=6, rewrite=True), "does she know it", 2)
    assert len(hits) == 2 and hits[0]["chunk_id"] == "resume_c1" and "retrieval_score" in hits[0]
    assert len(retriever.llm_ms) == 1


def test_llm_features_need_an_llm():
    retriever = Retriever(WordEmbedder(), make_index(DOCS), reranker=LengthReranker())
    with pytest.raises(RuntimeError):
        retriever.search(RetrievalSpec("dense", rewrite=True), "x", 2)


def test_the_oracle_route_uses_the_answer_key():
    retriever = Retriever(WordEmbedder(), make_index(DOCS), reranker=LengthReranker(), oracle={"audit log": ["spec"]})
    assert {h["doc_id"] for h in retriever.search(RetrievalSpec("dense", route="oracle"), "audit log", 3)} == {"spec"}
    assert len({h["doc_id"] for h in retriever.search(RetrievalSpec("dense", route="oracle"), "unknown question", 6)}) == 2


def test_the_reranker_experiments_retrieve_10_or_20_and_use_open_local_models():
    from eval.retrieval_experiments import RERANKER_EXPERIMENTS, SETS

    reranked = [e for e in RERANKER_EXPERIMENTS if e.spec.rerank_top and e.spec.reranker != "minilm"]
    assert reranked and all(e.spec.rerank_top in (10, 20) for e in reranked)
    assert {e.spec.reranker for e in reranked} == {"bge-base", "bge-v2-m3"}
    assert all(RERANKERS[e.spec.reranker].split("/")[0] != "openai" for e in RERANKER_EXPERIMENTS if e.spec.rerank_top)
    names = [e.name for e in RERANKER_EXPERIMENTS]
    assert len(names) == len(set(names)) and "p4-dense" in names and "p4-hybrid" in names
    assert set(SETS) == {"main", "rerankers", "transform"}


def test_the_transform_experiments_compare_every_question_with_the_gated_ones():
    from eval.retrieval_experiments import TRANSFORM_EXPERIMENTS

    modes = {(e.spec.rerank_top, e.spec.transform) for e in TRANSFORM_EXPERIMENTS}
    assert {(0, "none"), (0, "always"), (0, "auto"), (10, "none"), (10, "auto")} == modes
    names = [e.name for e in TRANSFORM_EXPERIMENTS]
    assert len(names) == len(set(names)) and "p4-hybrid-transform-auto" in names


def test_the_reranker_is_part_of_the_technique_name_and_must_be_known():
    assert RetrievalSpec("hybrid", rerank_top=10, reranker="bge-base").label == "hybrid-rerank10-bge-base"
    assert RetrievalSpec("hybrid", rerank_top=20).label == "hybrid-rerank20"      # the default model keeps its old name
    with pytest.raises(ValueError):
        RetrievalSpec("hybrid", rerank_top=10, reranker="nope")
    with pytest.raises(ValueError):
        get_reranker("nope")
    assert get_reranker("bge-base") is get_reranker("bge-base") and get_reranker("bge-base").name == "BAAI/bge-reranker-base"


def test_the_experiment_set_covers_the_plan_without_hyde():
    from eval.retrieval_experiments import CONTROL, EXPERIMENTS

    names = [e.name for e in EXPERIMENTS]
    assert len(names) == len(set(names)) and CONTROL in names and not any("hyde" in n for n in names)
    specs = {e.spec.label for e in EXPERIMENTS}
    assert {"dense", "bm25", "hybrid", "dense-route-oracle", "dense-route-llm", "dense-rerank20", "hybrid-rerank20",
            "dense-rewrite"} <= specs


def test_the_api_warms_up_retrieval_in_the_background_only_for_versions_that_need_it(monkeypatch):
    import threading

    import api.main as api

    loaded = threading.Event()
    monkeypatch.setattr(api, "shared_retriever", loaded.set)
    monkeypatch.setattr(api, "configured_search", lambda: None)
    api.warm_up_retrieval()
    assert not loaded.wait(0.3)                                    # plain dense search needs no warm-up
    monkeypatch.setattr(api, "configured_search", lambda: (lambda question, k: []))
    api.warm_up_retrieval()
    assert loaded.wait(5)


def test_a_reranker_loads_its_model_once_when_warmed_up(monkeypatch):
    loads = []

    class FakeModel:
        def predict(self, pairs, show_progress_bar=False):
            return [0.0] * len(pairs)

    reranker = CrossEncoderReranker("fake")
    monkeypatch.setattr(reranker, "_load", lambda: loads.append(1) or FakeModel())
    reranker.warm_up()
    assert loads == [1]


# ---------------------------------------------------------------- query transformation for hard questions

TRANSFORM_DOCS = {"spec": ["leave balance rules for employees", "error code QL-1003 means low balance", "audit log of approvals"],
                  "resume": ["built audit logging for approvals", "azure ad single sign on experience", "skills python sql"]}
TYPES = {"spec": "spec", "resume": "resume"}
TWO_DOCUMENTS = "The spec requires an audit log of approvals. Has the candidate built audit logging before?"
ONE_DOCUMENT = "What does error code QL-1003 mean?"
PLAN = ('{"rewrite": "audit logging for approvals", "expansions": ["audit trail"], "sub_queries": ['
        '{"document": "spec", "query": "audit log requirement"}, {"document": "resume", "query": "built audit logging"}]}')


def transforming_retriever(llm, reranker=None):
    return Retriever(WordEmbedder(), make_index(TRANSFORM_DOCS, TYPES), llm=llm, reranker=reranker or LengthReranker())


def test_the_gate_selects_only_the_questions_that_name_two_documents():
    from src.retrieve.transform import document_names, documents_named, gate

    corpus = [{"doc_id": "quickleave", "text": "QuickLeave - Product Specification leave"},
              {"doc_id": "cv", "text": "VALUMUDI SATWIKA experience, we need agents"},
              {"doc_id": "paper", "text": "Attention Is All You Need transformer"}]
    catalog = [{"doc_id": "quickleave", "doc_type": "spec", "title": "QuickLeave - Product Specification"},
               {"doc_id": "cv", "doc_type": "resume", "title": "VALUMUDI SATWIKA"},
               {"doc_id": "paper", "doc_type": "paper", "title": "Attention Is All You Need"}]
    names = document_names(corpus, catalog)
    assert {"quickleave", "spec", "specification"} <= names["quickleave"] and "product" not in names["quickleave"]
    assert {"valumudi", "satwika", "candidate"} <= names["cv"]
    assert "need" not in names["paper"] and "attention" in names["paper"]      # a short common word is not a document name
    assert documents_named("Does the candidate know QuickLeave?", names) == ["quickleave", "cv"]
    assert gate("QuickLeave requires Azure AD. Does the candidate have it?", names) == ["cross_document: quickleave, cv"]
    assert gate("What does error code QL-1003 mean in QuickLeave?", names) == []         # one document
    assert gate("Which leave types need a document?", names) == []                      # no document named
    assert gate("What is the candidate's CGPA?", names) == []


def test_parse_plan_keeps_what_is_usable_and_refuses_the_rest():
    from src.retrieve.transform import parse_plan

    known = {"spec", "resume"}
    plan = parse_plan(PLAN, "q?", known)
    assert plan.rewrite == "audit logging for approvals" and plan.expansions == ["audit trail"] and plan.multi_query
    assert [(s.document, s.query) for s in plan.sub_queries] == [("spec", "audit log requirement"), ("resume", "built audit logging")]
    odd = parse_plan('{"rewrite": " ", "expansions": ["a", "a", 3, "b", "c", "d"], "sub_queries": ['
                     '{"document": "nope", "query": "x"}, {"document": "spec", "query": "y"}, {"document": "spec", "query": "z"}, '
                     '{"document": "resume", "query": ""}, "junk"]}', "the question", known)
    assert odd.rewrite == "the question" and odd.expansions == ["a", "b", "c"]                # a blank rewrite falls back
    assert [(s.document, s.query) for s in odd.sub_queries] == [(None, "x"), ("spec", "y")]     # unknown id -> every document
    assert parse_plan('{"rewrite": "r", "sub_queries": [{"document": "spec", "query": "only one"}]}', "q", known).sub_queries == []
    assert parse_plan("not json", "q", known) is None and parse_plan("[1, 2]", "q", known) is None


def test_make_plan_puts_the_documents_and_the_question_in_the_prompt_and_records_the_cost():
    from src.retrieve.transform import make_plan

    class CostedLLM(ScriptedLLM):
        def complete(self, prompt, json_mode=False):
            self.prompts.append(prompt)
            return {"text": PLAN, "input_tokens": 500, "output_tokens": 90, "ms": 1200}

    llm = CostedLLM("")
    catalog = [{"doc_id": "spec", "doc_type": "spec", "title": "Spec", "sections": ["1. Intro"]},
               {"doc_id": "resume", "doc_type": "resume", "title": "Jane", "sections": ["SKILLS"]}]
    plan = make_plan(TWO_DOCUMENTS, catalog, llm)
    assert "- spec (spec): Spec; sections: 1. Intro" in llm.prompts[0] and TWO_DOCUMENTS in llm.prompts[0]
    assert (plan.tokens_in, plan.tokens_out, plan.ms) == (500, 90, 1200)


def test_round_robin_gives_every_query_a_fair_share_without_repeats():
    from src.retrieve.hybrid import round_robin

    out = round_robin([[hit("a1"), hit("a2"), hit("a3")], [hit("b1"), hit("b2")]], 4)
    assert [h["chunk_id"] for h in out] == ["a1", "b1", "a2", "b2"] and [h["rank"] for h in out] == [1, 2, 3, 4]
    assert [h["chunk_id"] for h in round_robin([[hit("a1")], [hit("a1"), hit("b1")]], 5)] == ["a1", "b1"]
    assert round_robin([[], []], 3) == []


def test_a_question_that_names_two_documents_is_split_and_searched_in_each():
    llm = ScriptedLLM(PLAN)
    retriever = transforming_retriever(llm)
    hits = retriever.search(RetrievalSpec("hybrid", transform="auto"), TWO_DOCUMENTS, 4)
    assert {h["doc_id"] for h in hits} == {"spec", "resume"}                                   # a share for each document
    assert [h["doc_id"] for h in hits].count("spec") == 3 == [h["doc_id"] for h in hits].count("resume")   # 3 places each
    assert [h["doc_id"] for h in hits[:2]] == ["spec", "resume"]
    info = hits[0]["transform"]
    assert len(info["reasons"]) == 1 and set(info["reasons"][0].removeprefix("cross_document: ").split(", ")) == {"spec", "resume"}
    assert info["expansions"] == ["audit trail"]
    assert [s["document"] for s in info["sub_queries"]] == ["spec", "resume"]
    assert "(audit trail)" in hits[0]["query"] and len(llm.prompts) == 1
    retriever.search(RetrievalSpec("hybrid", transform="auto"), TWO_DOCUMENTS, 4)
    assert len(llm.prompts) == 1                                                                # the plan is remembered


def test_the_reranker_loads_a_cached_model_without_the_network_and_downloads_only_a_missing_one(monkeypatch):
    import sentence_transformers

    calls = []

    class FakeCrossEncoder:
        cached = True

        def __init__(self, name, max_length=512, local_files_only=False):
            calls.append(local_files_only)
            if local_files_only and not FakeCrossEncoder.cached:
                raise OSError("not in the cache")

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", FakeCrossEncoder)
    CrossEncoderReranker("a-model")._load()
    assert calls == [True]                                    # on disk: no request to the model hub at all
    calls.clear()
    FakeCrossEncoder.cached = False
    reranker = CrossEncoderReranker("another-model")
    reranker._load()
    assert calls == [True, False]                             # not on disk: one more try that may download it
    reranker._load()
    assert calls == [True, False]                             # and it is loaded only once


def test_a_cross_document_question_reranks_five_candidates_per_document_whatever_rerank_top_is():
    retriever = transforming_retriever(ScriptedLLM(PLAN))
    pools, original = [], retriever._candidates
    retriever._candidates = lambda mode, query, n, docs: (pools.append(n), original(mode, query, n, docs))[1]
    for top in (10, 7):                                    # v2-extended and v3 use 10, v4 uses 7: the two documents keep 5 each
        pools.clear()
        retriever.search(RetrievalSpec("hybrid", rerank_top=top, transform="auto"), TWO_DOCUMENTS, 4)
        assert pools == [5, 5], top


def test_without_sub_queries_from_the_model_each_named_document_is_still_searched():
    llm = ScriptedLLM('{"rewrite": "audit logging for approvals", "expansions": [], "sub_queries": []}')
    hits = transforming_retriever(llm).search(RetrievalSpec("hybrid", transform="auto"), TWO_DOCUMENTS, 4)
    assert {h["doc_id"] for h in hits} == {"spec", "resume"}
    assert {s["document"] for s in hits[0]["transform"]["sub_queries"]} == {"spec", "resume"}


def test_a_simple_question_makes_no_model_call_and_is_not_marked_transformed():
    llm = ScriptedLLM(PLAN)
    plain = transforming_retriever(llm)
    hits = plain.search(RetrievalSpec("hybrid", transform="auto"), ONE_DOCUMENT, 3)
    assert llm.prompts == [] and plain.llm_ms == [] and all("transform" not in h for h in hits)
    same = plain.search(RetrievalSpec("hybrid"), ONE_DOCUMENT, 3)
    assert [h["chunk_id"] for h in hits] == [h["chunk_id"] for h in same]                       # identical to no transformation


def test_transform_always_rewrites_a_single_need_question_and_reranks_by_the_rewrite():
    llm = ScriptedLLM('{"rewrite": "QL-1003 error code meaning", "expansions": ["low balance"], "sub_queries": []}')
    retriever = transforming_retriever(llm)
    hits = retriever.search(RetrievalSpec("hybrid", rerank_top=4, transform="always"), ONE_DOCUMENT, 2)
    assert len(hits) == 2 and hits[0]["chunk_id"] == "spec_c1" and hits[0]["transform"]["reasons"] == ["always"]
    assert len(llm.prompts) == 1 and len(retriever.llm_ms) == 1


def test_an_unusable_plan_or_a_failing_model_falls_back_to_the_normal_search():
    class FailingLLM:
        def complete(self, prompt, json_mode=False):
            raise RuntimeError("model unavailable")

    for llm in (ScriptedLLM("not json"), FailingLLM()):
        retriever = transforming_retriever(llm)
        hits = retriever.search(RetrievalSpec("hybrid", transform="auto"), TWO_DOCUMENTS, 3)
        plain = retriever.search(RetrievalSpec("hybrid"), TWO_DOCUMENTS, 3)
        assert [h["chunk_id"] for h in hits] == [h["chunk_id"] for h in plain] and all("transform" not in h for h in hits)
    with pytest.raises(RuntimeError):                                                           # no model at all is a setup error
        transforming_retriever(None).search(RetrievalSpec("hybrid", transform="auto"), TWO_DOCUMENTS, 3)


def test_the_transform_setting_is_validated_and_part_of_the_technique_name():
    assert RetrievalSpec("hybrid", transform="auto").label == "hybrid-transform-auto"
    assert RetrievalSpec("hybrid", rerank_top=10, reranker="bge-base", transform="always").label == "hybrid-rerank10-bge-base-transform-always"
    assert RetrievalSpec("hybrid", transform="auto").llm_calls == 1 and RetrievalSpec("hybrid").llm_calls == 0
    with pytest.raises(ValueError):
        RetrievalSpec("hybrid", transform="sometimes")


def test_cut_low_scores_drops_weak_chunks_but_keeps_the_evidence_guards():
    from src.retrieve.retriever import RetrievalSpec, cut_low_scores
    hit = lambda cid, doc, score, **kw: {"chunk_id": cid, "doc_id": doc, "score": score, **kw}
    # an ordinary question: chunks under 2% of the best score go, the best one always stays
    hits = [hit("a", "spec", 0.8), hit("b", "spec", 0.03), hit("c", "spec", 0.001), hit("d", "cv", 0.0)]
    assert [h["chunk_id"] for h in cut_low_scores(hits, 0.02)] == ["a", "b"]
    assert [h["rank"] for h in cut_low_scores(hits, 0.02)] == [1, 2]
    assert cut_low_scores(hits, 0.0) == hits                                         # off
    # a per-document search: each sub-query's list is cut on its own (its scores are not comparable with the other's)
    multi = [hit("s1", "spec", 0.9, query="q1"), hit("r1", "cv", 0.004, query="q2"), hit("s2", "spec", 0.001, query="q1"),
             hit("r2", "cv", 0.003, query="q2")]
    assert [h["chunk_id"] for h in cut_low_scores(multi, 0.02)] == ["s1", "r1", "r2"]
    # a cross-document question: the best 2 chunks of each named document stay whatever their score (q24)
    info = {"reasons": ["cross_document: cv, spec"]}
    cross = [hit("skills", "cv", 0.25, transform=info), hit("fr", "spec", 0.08, transform=info),
             hit("sso", "cv", 0.0055, transform=info), hit("roles", "spec", 0.0001, transform=info),
             hit("certs", "cv", 0.0001, transform=info)]
    assert [h["chunk_id"] for h in cut_low_scores(cross, 0.05)] == ["skills", "fr", "sso", "roles"]
    assert RetrievalSpec("hybrid", 7, min_ratio=0.02).label.endswith("cut0.02")
    with pytest.raises(ValueError):
        RetrievalSpec(min_ratio=1.5)


def test_asked_first_lists_the_document_the_question_asks_about():
    from src.retrieve.retriever import RetrievalSpec, asked_document, asked_first
    from src.retrieve.transform import SubQuery
    names = {"resume": {"candidate", "resume"}, "spec": {"quickleave", "spec"}}
    q = "QuickLeave exposes its API under /api/v1. Does the candidate have REST API design experience?"
    assert asked_document(q, names) == "resume"
    assert asked_document("Which leave types exist in QuickLeave?", names) == "spec"
    assert asked_document("Does the candidate match the QuickLeave spec?", names) is None     # both named: keep plan order
    subs = [SubQuery("spec", "api"), SubQuery("resume", "rest api")]
    assert asked_first(q, subs, [["spec hits"], ["resume hits"]], names) == [["resume hits"], ["spec hits"]]
    assert "asked-first" in RetrievalSpec("hybrid", 7, asked_first=True).label
