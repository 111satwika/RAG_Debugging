"""Tracing tests. A fake LLM keeps them fast; retrieval uses the real Chroma index."""
import pytest

from scripts.show_trace import show
from src.observability.tracer import Trace, load_trace
from src.pipeline import run_query


class FakeLLM:
    name = "fake-llm"

    def complete(self, prompt: str) -> dict:
        return {"text": "fake answer", "input_tokens": 123, "output_tokens": 4, "ms": 1, "temperature": 0}


class BrokenLLM:
    name = "broken-llm"

    def complete(self, prompt: str) -> dict:
        raise RuntimeError("model unavailable")


def test_trace_has_all_spans_and_is_saved(tmp_path):
    path = tmp_path / "traces.jsonl"
    out = run_query("What does error code QL-1003 mean?", k=3, llm=FakeLLM(), trace_path=path)

    saved = load_trace(out["trace_id"], path)
    assert list(saved["spans"]) == ["embed_query", "retrieve", "build_prompt", "llm_call"]
    assert all("ms" in s for s in saved["spans"].values())
    assert len(saved["spans"]["retrieve"]["results"]) == 3
    assert "QL-1003" in saved["spans"]["build_prompt"]["final_prompt"]
    assert saved["spans"]["llm_call"]["input_tokens"] == 123
    assert saved["answer"] == "fake answer"
    assert saved["error"] is None
    assert saved["total_ms"] >= 0
    for key in ("version", "chunk_size", "top_k", "embed_model", "llm", "prompt_version"):
        assert key in saved["config"]


def test_a_version_with_its_own_search_skips_the_embedding_step_and_traces_the_search(tmp_path, monkeypatch):
    fake_hits = [{"chunk_id": "c1", "doc_id": "d", "doc_type": "spec", "page": 1, "score": 0.5, "text": "QL-1003 means low balance"}]
    monkeypatch.setattr("src.pipeline.configured_search", lambda: (lambda question, k: fake_hits))
    path = tmp_path / "traces.jsonl"
    out = run_query("What does QL-1003 mean?", k=3, llm=FakeLLM(), trace_path=path)
    saved = load_trace(out["trace_id"], path)
    assert list(saved["spans"]) == ["retrieve", "build_prompt", "llm_call"]
    assert saved["spans"]["retrieve"]["results"][0]["chunk_id"] == "c1"
    assert "QL-1003 means low balance" in saved["spans"]["build_prompt"]["final_prompt"]


def test_dense_versions_have_no_configured_search_and_hybrid_ones_search_through_the_shared_retriever(monkeypatch):
    import src.retrieve.retriever as r

    assert r.configured_search() is None                      # the tests run as v1: plain dense search
    calls = []

    class FakeRetriever:
        def search(self, spec, question, k):
            calls.append((spec.mode, spec.rerank_top, question, k))
            return []

    monkeypatch.setattr(r, "RETRIEVAL_MODE", "hybrid")
    monkeypatch.setattr(r, "shared_retriever", lambda: FakeRetriever())
    search = r.configured_search()
    assert search("q?", 5) == [] and calls == [("hybrid", 0, "q?", 5)]
    monkeypatch.setattr(r, "RETRIEVAL_MODE", "dense")
    monkeypatch.setattr(r, "RERANK_TOP", 10)
    assert r.configured_search() is not None                  # reranking needs the retriever even with dense search


def test_a_transformed_question_records_why_and_what_the_model_planned_in_its_trace(tmp_path, monkeypatch):
    info = {"reasons": ["cross_document: spec, resume"], "rewrite": "r", "expansions": ["RBAC"],
            "sub_queries": [{"document": "spec", "query": "a"}, {"document": "resume", "query": "b"}],
            "llm_ms": 1500, "tokens_in": 500, "tokens_out": 90}
    hits = [{"chunk_id": "c1", "doc_id": "d", "doc_type": "spec", "page": 1, "score": 0.5, "text": "text", "transform": info,
             "query": "a (RBAC)"},
            {"chunk_id": "c2", "doc_id": "e", "doc_type": "resume", "page": 1, "score": 0.4, "text": "text", "transform": info}]
    monkeypatch.setattr("src.pipeline.configured_search", lambda: (lambda question, k: hits))
    path = tmp_path / "traces.jsonl"
    out = run_query("two documents?", k=2, llm=FakeLLM(), trace_path=path)
    span = load_trace(out["trace_id"], path)["spans"]["retrieve"]
    assert span["transform"]["reasons"] == ["cross_document: spec, resume"] and span["transform"]["llm_ms"] == 1500
    assert span["results"][0]["query"] == "a (RBAC)" and "query" not in span["results"][1]


def test_failed_query_is_still_traced(tmp_path):
    path = tmp_path / "traces.jsonl"
    with pytest.raises(RuntimeError):
        run_query("anything", k=2, llm=BrokenLLM(), trace_path=path)

    saved = load_trace(None, path)
    assert saved["error"] == "RuntimeError: model unavailable"
    assert "llm_call" in saved["spans"] and "error" in saved["spans"]["llm_call"]
    assert saved["answer"] is None


def test_load_trace_by_id_and_latest(tmp_path):
    path = tmp_path / "traces.jsonl"
    first, second = Trace("one"), Trace("two")
    for t in (first, second):
        t.finish(answer="x")
        t.save(path)
    assert load_trace(first.trace_id, path)["query"] == "one"
    assert load_trace(None, path)["query"] == "two"
    with pytest.raises(KeyError):
        load_trace("nope", path)


def test_show_trace_prints_chunks_and_prompt(tmp_path, capsys):
    path = tmp_path / "traces.jsonl"
    out = run_query("What does error code QL-1003 mean?", k=2, llm=FakeLLM(), trace_path=path)
    show(load_trace(out["trace_id"], path))
    printed = capsys.readouterr().out
    assert out["trace_id"] in printed
    assert "RETRIEVED CHUNKS" in printed and "FINAL PROMPT" in printed and "fake answer" in printed
