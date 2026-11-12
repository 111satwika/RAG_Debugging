"""Version profiles: each pipeline version is a complete, reproducible set of settings."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_config(version: str | None, expression: str) -> subprocess.CompletedProcess:
    env = {**os.environ}
    env.pop("RAG_PIPELINE_VERSION", None)
    if version:
        env["RAG_PIPELINE_VERSION"] = version
    return subprocess.run([sys.executable, "-c", f"import src.config as c; print({expression})"], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=60)


def test_v1_is_the_frozen_baseline_and_keeps_its_own_index():
    out = read_config("v1", "(c.COLLECTION_NAME, c.EMBED_MODEL, c.CHUNK_STRATEGY, c.CHUNK_SIZE, c.CHUNK_OVERLAP, c.LLM_MODEL)")
    assert out.stdout.strip() == "('rag_chunks', 'BAAI/bge-small-en-v1.5', 'fixed', 500, 0, 'qwen2.5:7b-instruct')"


def test_v2_is_the_phase_3_pipeline_with_its_own_index():
    out = read_config("v2", "(c.COLLECTION_NAME, c.EMBED_PROVIDER, c.EMBED_MODEL, c.TOKENIZER_NAME, c.CHUNK_STRATEGY, "
                            "c.CHUNK_OVERLAP, c.CHUNK_HEADERS, c.KEEP_TABLES, c.CHILD_SIZE, c.LLM_MODEL)")
    assert out.stdout.strip() == ("('rag_chunks_v2', 'openai', 'text-embedding-3-small', 'cl100k_base', 'structure', 50, True, "
                                  "True, 0, 'gpt-4o-mini')")


def test_v2_extended_is_v2_plus_hybrid_retrieval_with_reranking_and_shares_the_v2_index():
    v2 = read_config("v2", "(c.COLLECTION_NAME, c.EMBED_MODEL, c.CHUNK_STRATEGY, c.CHUNK_SIZE, c.CHUNK_OVERLAP, c.CHUNK_HEADERS, "
                           "c.KEEP_TABLES, c.CHUNK_MIN_TOKENS, c.LLM_MODEL)")
    extended = read_config("v2-extended", "(c.COLLECTION_NAME, c.EMBED_MODEL, c.CHUNK_STRATEGY, c.CHUNK_SIZE, c.CHUNK_OVERLAP, "
                                          "c.CHUNK_HEADERS, c.KEEP_TABLES, c.CHUNK_MIN_TOKENS, c.LLM_MODEL)")
    assert extended.stdout == v2.stdout and "'rag_chunks_v2'" in extended.stdout          # the same chunks and the same index
    assert read_config("v2-extended", "(c.RETRIEVAL_MODE, c.RERANK_TOP, c.RERANKER, c.QUERY_TRANSFORM)").stdout.strip() == (
        "('hybrid', 10, 'bge-base', 'auto')")
    assert read_config("v2", "(c.RETRIEVAL_MODE, c.RERANK_TOP, c.QUERY_TRANSFORM)").stdout.strip() == "('dense', 0, 'none')"
    assert read_config("v1", "(c.RETRIEVAL_MODE, c.RERANK_TOP, c.QUERY_TRANSFORM)").stdout.strip() == "('dense', 0, 'none')"


def test_v3_is_v2_extended_plus_the_cited_prompt_and_shares_the_v2_index():
    fields = ("c.COLLECTION_NAME, c.EMBED_MODEL, c.CHUNK_STRATEGY, c.RETRIEVAL_MODE, c.RERANK_TOP, c.RERANKER, "
              "c.QUERY_TRANSFORM, c.LLM_MODEL, c.LLM_TEMPERATURE")
    v3, extended = read_config("v3", f"({fields})"), read_config("v2-extended", f"({fields})")
    assert v3.stdout == extended.stdout and "'rag_chunks_v2'" in v3.stdout          # only the prompt differs
    assert read_config("v3", "c.PROMPT_VERSION").stdout.strip() == "v2"
    for older in ("v1", "v2", "v2-extended"):
        assert read_config(older, "c.PROMPT_VERSION").stdout.strip() == "v1"         # the earlier versions keep the baseline prompt
    assert read_config("v3", "c.snapshot()['prompt_version']").stdout.strip() == "v2"


def test_the_default_version_is_v12_and_an_unknown_version_is_refused():
    assert read_config(None, "c.PIPELINE_VERSION").stdout.strip() == "v12"
    bad = read_config("v99", "c.PIPELINE_VERSION")
    assert bad.returncode != 0 and "Unknown pipeline version" in bad.stderr


def test_v4_is_v3_plus_the_validation_layer_and_earlier_versions_do_not_validate():
    fields = ("c.COLLECTION_NAME, c.EMBED_MODEL, c.CHUNK_STRATEGY, c.RETRIEVAL_MODE, c.RERANKER, "
              "c.QUERY_TRANSFORM, c.LLM_MODEL, c.PROMPT_VERSION")
    assert read_config("v4", f"({fields})").stdout == read_config("v3", f"({fields})").stdout
    assert read_config("v4", "c.RERANK_TOP").stdout.strip() == "7" and read_config("v3", "c.RERANK_TOP").stdout.strip() == "10"   # v4 reranks fewer
    assert read_config("v4", "(c.VALIDATE, c.RELEVANCE_HIGH, c.RELEVANCE_LOW, c.RETRIES)").stdout.strip() == "(True, 0.1, 0.0005, 1)"
    snap = read_config("v4", "(c.snapshot()['validate'], c.snapshot()['relevance_high'], c.snapshot()['relevance_low'], c.snapshot()['retries'])")
    assert snap.stdout.strip() == "(True, 0.1, 0.0005, 1)"
    for older in ("v1", "v2", "v2-extended", "v3"):
        assert read_config(older, "(c.VALIDATE, c.snapshot()['relevance_high'], c.snapshot()['retries'])").stdout.strip() == "(False, None, None)"
    assert read_config("v1", "(c.with_legacy_defaults({})['validate'], c.with_legacy_defaults({})['relevance_low'])").stdout.strip() == "(False, None)"


def test_v5_is_v4_with_only_the_answer_prompt_changed():
    fields = ("c.COLLECTION_NAME, c.EMBED_MODEL, c.CHUNK_STRATEGY, c.RETRIEVAL_MODE, c.RERANK_TOP, c.RERANKER, c.QUERY_TRANSFORM, "
              "c.LLM_MODEL, c.VALIDATE, c.RELEVANCE_HIGH, c.RELEVANCE_LOW, c.RETRIES")
    assert read_config("v5", f"({fields})").stdout == read_config("v4", f"({fields})").stdout
    assert read_config("v5", "c.PROMPT_VERSION").stdout.strip() == "v2.4" and read_config("v4", "c.PROMPT_VERSION").stdout.strip() == "v2"
    assert read_config("v5", "c.snapshot()['prompt_version']").stdout.strip() == "v2.4"


def test_v6_is_v5_with_only_the_answer_prompt_changed():
    fields = ("c.COLLECTION_NAME, c.EMBED_MODEL, c.CHUNK_STRATEGY, c.RETRIEVAL_MODE, c.RERANK_TOP, c.RERANKER, c.QUERY_TRANSFORM, "
              "c.LLM_MODEL, c.VALIDATE, c.RELEVANCE_HIGH, c.RELEVANCE_LOW, c.RETRIES")
    assert read_config("v6", f"({fields})").stdout == read_config("v5", f"({fields})").stdout
    assert read_config("v6", "c.PROMPT_VERSION").stdout.strip() == "v2.5"
    assert read_config("v6", "c.snapshot()['prompt_version']").stdout.strip() == "v2.5"


def test_v7_is_v4_with_only_the_answer_prompt_changed():
    fields = ("c.COLLECTION_NAME, c.EMBED_MODEL, c.CHUNK_STRATEGY, c.RETRIEVAL_MODE, c.RERANK_TOP, c.RERANKER, c.QUERY_TRANSFORM, "
              "c.LLM_MODEL, c.VALIDATE, c.RELEVANCE_HIGH, c.RELEVANCE_LOW, c.RETRIES")
    assert read_config("v7", f"({fields})").stdout == read_config("v4", f"({fields})").stdout
    assert read_config("v7", "c.PROMPT_VERSION").stdout.strip() == "v2.9"


def test_v8_is_v7_with_only_the_answer_prompt_changed():
    fields = ("c.COLLECTION_NAME, c.EMBED_MODEL, c.CHUNK_STRATEGY, c.RETRIEVAL_MODE, c.RERANK_TOP, c.RERANKER, c.QUERY_TRANSFORM, "
              "c.LLM_MODEL, c.VALIDATE, c.RELEVANCE_HIGH, c.RELEVANCE_LOW, c.RETRIES")
    assert read_config("v8", f"({fields})").stdout == read_config("v7", f"({fields})").stdout
    assert read_config("v8", "c.PROMPT_VERSION").stdout.strip() == "v2.11"
    assert read_config("v8", "c.RERANK_MIN_RATIO").stdout.strip() == "0.0"          # v8 stays without the cut-off


def test_v9_is_v8_with_only_the_precision_cut_off():
    fields = ("c.COLLECTION_NAME, c.EMBED_MODEL, c.CHUNK_STRATEGY, c.RETRIEVAL_MODE, c.RERANK_TOP, c.RERANKER, c.QUERY_TRANSFORM, "
              "c.LLM_MODEL, c.PROMPT_VERSION, c.VALIDATE, c.RELEVANCE_HIGH, c.RELEVANCE_LOW, c.RETRIES")
    assert read_config("v9", f"({fields})").stdout == read_config("v8", f"({fields})").stdout
    assert read_config("v9", "c.RERANK_MIN_RATIO").stdout.strip() == "0.02"
    assert read_config("v9", "c.REFINE").stdout.strip() == "False"                  # v9 stays without the refinement


def test_v10_is_v9_with_only_the_refinement():
    fields = ("c.COLLECTION_NAME, c.EMBED_MODEL, c.CHUNK_STRATEGY, c.RETRIEVAL_MODE, c.RERANK_TOP, c.RERANKER, c.QUERY_TRANSFORM, "
              "c.RERANK_MIN_RATIO, c.LLM_MODEL, c.PROMPT_VERSION, c.VALIDATE, c.RELEVANCE_HIGH, c.RELEVANCE_LOW, c.RETRIES")
    assert read_config("v10", f"({fields})").stdout == read_config("v9", f"({fields})").stdout
    assert read_config("v10", "c.REFINE").stdout.strip() == "True"
    assert read_config("v10", "(c.REFINE_WITH_CONTEXT, c.ASKED_FIRST, c.CHUNK_TABLE_CAPTIONS)").stdout.strip() == "(False, False, False)"


def test_snapshot_records_the_version_and_every_chunking_and_retrieval_setting():
    out = read_config("v2-extended", "sorted(c.snapshot())")
    for key in ("version", "chunk_strategy", "chunk_size", "chunk_overlap", "chunk_headers", "keep_tables", "child_size",
                "embed_model", "tokenizer", "llm", "retrieval_mode", "rerank_top", "reranker", "query_transform"):
        assert f"'{key}'" in out.stdout
    assert read_config("v2-extended", "(c.snapshot()['version'], c.snapshot()['retrieval_mode'], c.snapshot()['rerank_top'], "
                       "c.snapshot()['reranker'], c.snapshot()['query_transform'])").stdout.strip() == (
        "('v2-extended', 'hybrid', 10, 'bge-base', 'auto')")
    assert read_config("v2", "c.snapshot()['reranker']").stdout.strip() == "None"       # no reranker, so none is recorded


def test_runs_made_before_retrieval_was_recorded_count_as_dense_search():
    out = read_config("v1", "(c.with_legacy_defaults({'version': 'v2'})['retrieval_mode'], "
                            "c.with_legacy_defaults({})['rerank_top'], c.with_legacy_defaults({})['query_transform'])")
    assert out.stdout.strip() == "('dense', 0, 'none')"


def test_v12_is_v11_with_only_the_refinement_context():
    fields = ("c.COLLECTION_NAME, c.EMBED_MODEL, c.CHUNK_STRATEGY, c.CHUNK_TABLE_CAPTIONS, c.RETRIEVAL_MODE, c.RERANK_TOP, "
              "c.RERANKER, c.QUERY_TRANSFORM, c.RERANK_MIN_RATIO, c.ASKED_FIRST, c.LLM_MODEL, c.PROMPT_VERSION, c.VALIDATE, c.REFINE")
    assert read_config("v12", f"({fields})").stdout == read_config("v11", f"({fields})").stdout
    assert read_config("v12", "(c.PROMPT_VERSION, c.REFINE_WITH_CONTEXT)").stdout.strip() == "('v2.11', True)"
    assert read_config("v11", "(c.PROMPT_VERSION, c.REFINE_WITH_CONTEXT)").stdout.strip() == "('v2.11', False)"
