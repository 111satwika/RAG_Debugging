"""Central settings. Every query trace and eval run records these values.

Each pipeline version is a complete profile below, so a version can be switched back to or reproduced at any time:
    v1           the baseline: bge-small embeddings, fixed 500-token chunks, a local Qwen model
    v2           Phase 3: OpenAI embeddings, structure-aware chunks with headers, tables kept intact and overlap, gpt-4o-mini
    v2-extended  Phase 4: v2 (same chunks, same index) with hybrid retrieval (dense + BM25 merged with RRF), then the top 10
                 reranked by a cross-encoder (bge-reranker-base) down to the best 5, and, for the hard questions only
                 (the ones that name two documents), a model-planned rewrite / expansion / multi-query search
    v3           Phase 5: v2-extended with the cited prompt (v2 prompt): grounding rules, sources labelled [S1]..., citations
                 required and a structured JSON answer (answer, sources, found)
    v4           Phase 6: v3 with a CRAG-style validation layer (the reranker score grades the context; a weak one gets a corrective
                 search with a rewritten query; still no relevant chunk: "not found" without calling the model) and 7 candidates
                 reranked instead of 10 (3 s faster, same retrieval quality)
    v5           v4 with the v2.4 answer prompt (concise but every fact that answers, each fact attributed to its own item, yes/no
                 questions start with Yes or No, and No when it holds for only part of what was asked)
    v6           v5 with the v2.5 answer prompt: the attribution rule is reworded (keep each fact attached to the item it describes,
                 do not move a fact from one item to another)
    v7           v4 with the v2.9 answer prompt: the v2 rules plus item-by-item "checks" for every/all/both/at-least questions,
                 "No, <document> does not mention it" for yes/no questions (instead of "not found"), plain-text answers and a
                 citation in every sentence. v2.1 to v2.5 were tuned on the golden set and do no better than v2 on the held-out
                 set; v2.6 to v2.8 (an evidence list before every answer) dropped facts (see eval/results/prompt_repeats_*gpt4o*)
                 On the fresh held-out set (n01-n25) and the golden set it beats v2 on both (eval/phase5_final.py)
    v8           v7 with the v2.11 answer prompt (every answering fact with its numbers and its item named; attribution
                 checks for which-item questions)
    v9           v8 with the precision cut-off after reranking (chunks under 2% of the best score dropped; cross-document
                 questions keep the best 2 chunks of each document). Relevant / chunks given 0.28 -> 0.55, 17-22% fewer
                 input tokens, same answer quality on the held-out set
    v10          v9 with CRAG knowledge refinement (a gpt-4o-mini call drops chunks that do not help before the answer).
                 Context precision 0.54 -> 0.75, correctness 0.97 -> 0.99 (golden) and 0.96 -> 1.00 (holdout), about
                 2.4 s slower
    v11          v10 with table chunks headed by their caption and the asked-about document first: golden MRR 0.90 -> 0.95
    v12          v11 with the refinement grader shown the kept passages (q37 fixed). The default version, put in the app
                 at the user's request before a full eval (only q37 and n12 were checked with the real model)
To make a new version, add a profile and set PIPELINE_VERSION. RAG_PIPELINE_VERSION in the environment overrides it
(the tests set it to v1 so they keep using the v1 index).
"""
import os

PIPELINE_VERSION = os.environ.get("RAG_PIPELINE_VERSION", "v12")

_V2 = {
    "EMBED_PROVIDER": "openai", "EMBED_MODEL": "text-embedding-3-small",
    "TOKENIZER_PROVIDER": "tiktoken", "TOKENIZER_NAME": "cl100k_base",
    "CHUNK_STRATEGY": "structure", "CHUNK_SIZE": 500, "CHUNK_OVERLAP": 50,
    "CHUNK_HEADERS": True, "KEEP_TABLES": True, "CHUNK_MIN_TOKENS": 50, "CHILD_SIZE": 0,
    "LLM_PROVIDER": "openai", "LLM_MODEL": "gpt-4o-mini",
}
PROFILES = {
    "v1": {
        "EMBED_PROVIDER": "sentence-transformers", "EMBED_MODEL": "BAAI/bge-small-en-v1.5",
        "TOKENIZER_PROVIDER": "huggingface", "TOKENIZER_NAME": "BAAI/bge-small-en-v1.5",
        "CHUNK_STRATEGY": "fixed", "CHUNK_SIZE": 500, "CHUNK_OVERLAP": 0,
        "CHUNK_HEADERS": False, "KEEP_TABLES": False, "CHUNK_MIN_TOKENS": 0, "CHILD_SIZE": 0,
        "LLM_PROVIDER": "ollama", "LLM_MODEL": "qwen2.5:7b-instruct",
    },
    "v2": _V2,
    # Retrieval keys are optional: a profile without them searches with dense search only (v1, v2).
    # INDEX_FROM reuses another version's index instead of building a copy of the same chunks.
    # PROMPT_VERSION (a file in src/generate/prompts/) defaults to v1, the plain baseline prompt.
    "v2-extended": {**_V2, "RETRIEVAL_MODE": "hybrid", "RERANK_TOP": 10, "RERANKER": "bge-base", "QUERY_TRANSFORM": "auto",
                    "INDEX_FROM": "v2"},
}
PROFILES["v3"] = {**PROFILES["v2-extended"], "PROMPT_VERSION": "v2"}
# Phase 6: v3 plus the CRAG-style validation layer. VALIDATE grades the retrieved context from the reranker score
# (src/validate/gate.py): at or above RELEVANCE_HIGH answer; below it rewrite the query and search again (RETRIES times),
# merge the chunks by score and grade again; still under RELEVANCE_LOW: do not answer. The cutoffs come from the sweep in
# eval/validation_experiments.py (see SKILL.md for how far to trust them).
PROFILES["v4"] = {**PROFILES["v3"], "VALIDATE": True, "RELEVANCE_HIGH": 0.1, "RELEVANCE_LOW": 0.0005, "RETRIES": 1,
                  "RERANK_TOP": 7}     # 7 candidates reranked instead of 10: same retrieval quality, 3 s faster (eval/rerank_cost_experiments.py)
PROFILES["v5"] = {**PROFILES["v4"], "PROMPT_VERSION": "v2.4"}      # only the answer prompt differs from v4
PROFILES["v6"] = {**PROFILES["v5"], "PROMPT_VERSION": "v2.5"}      # only the answer prompt differs from v5
# v7 candidate: v4 (prompt v2) with v2.9 = v2 plus item-by-item checks for comparison questions and a "No, not mentioned"
# rule for yes/no questions. Measured with eval/phase5_final.py: better than v2 on the golden and the fresh held-out set.
PROFILES["v7"] = {**PROFILES["v4"], "PROMPT_VERSION": "v2.9"}
# v8: v7 with the v2.11 answer prompt: v2.9 plus "be concise but give every fact, with its numbers, and name the item it
# belongs to" (instead of "keep the answer short") and attribution checks (copy every sentence or table row that answers a
# which-item question, with its heading). v2.10 (the same without "every sentence or row") beat v2.9 on the fresh held-out
# set (correctness 0.932 -> 0.968; n07, n16 fixed) but listed only some rows of q38; v2.11 fixes that wording and is not
# measured yet (see eval/results/prompt_repeats_*_v2.10.md). No precision cut-off yet.
PROFILES["v8"] = {**PROFILES["v7"], "PROMPT_VERSION": "v2.11"}
# v9: v8 with the precision cut-off: after reranking, drop chunks scoring under 2% of the best one (per-document searches
# cut within their own list; cross-document questions keep the best 2 chunks of each document), see
# src/retrieve/retriever.py cut_low_scores. Measured against v8 (eval/results/v9, fresh-v9): precision doubled, cheaper and
# faster, no evidence an answer needed was lost.
PROFILES["v9"] = {**PROFILES["v8"], "RERANK_MIN_RATIO": 0.02}
# v10: v9 with CRAG knowledge refinement: after validation, one gpt-4o-mini call grades every chunk except the best one
# (and each named document's best one on a cross-document question) and drops the ones that do not help.
# Measured against v9 (eval/results/v10, fresh-v10): context precision 0.54 -> 0.75 (golden), 0.51 -> 0.72 (holdout);
# correctness 0.97 -> 0.99 and 0.96 -> 1.00; 18-21% fewer input tokens; about 2.4 s slower (the extra call).
PROFILES["v10"] = {**PROFILES["v9"], "REFINE": True}
# v11 (ranking): v10 with (1) table chunks headed by their caption instead of the section before them on the page
# (src/ingest/chunkers.py table_caption), in its own index rag_chunks_v11, and (2) the asked-about document listed first on a
# per-document search (src/retrieve/retriever.py asked_first). Measured against v10 (eval/results/v11, fresh-v11): golden
# MRR 0.90 -> 0.95, nDCG 0.89 -> 0.93 (q16, q24, q46, q47 now at rank 1); holdout unchanged; answers the same within noise.
PROFILES["v11"] = {**PROFILES["v10"], "CHUNK_TABLE_CAPTIONS": True, "ASKED_FIRST": True, "INDEX_FROM": "v11"}
# v12: v11 with the refinement grader shown the always-kept passages, so a passage that completes them is kept (q37: the
# revision history saying version 1.2 added QL-1006 was dropped by v10/v11, which graded it alone). Checked with the real
# model on q37: 3 of 3 complete answers (v11: 0). Prompt v2.12 (superlatives in checks, for n12) was tried and left out:
# n12 stayed at 2 of 3 and a prompt change needs a full eval. Not yet measured on the full sets.
PROFILES["v12"] = {**PROFILES["v11"], "REFINE_WITH_CONTEXT": True}
if PIPELINE_VERSION not in PROFILES:
    raise ValueError(f"Unknown pipeline version '{PIPELINE_VERSION}'. Add a profile to PROFILES: {list(PROFILES)}")
_PROFILE = PROFILES[PIPELINE_VERSION]

# Embeddings. The tokenizer counts chunk sizes, so it should match the embedding model.
EMBED_PROVIDER = _PROFILE["EMBED_PROVIDER"]
EMBED_MODEL = _PROFILE["EMBED_MODEL"]
TOKENIZER_PROVIDER = _PROFILE["TOKENIZER_PROVIDER"]    # "huggingface" or "tiktoken"
TOKENIZER_NAME = _PROFILE["TOKENIZER_NAME"]
# bge models retrieve better when the *query* (not the documents) has this prefix.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Chunking. Strategies are described in src/ingest/chunkers.py.
CHUNK_SIZE = _PROFILE["CHUNK_SIZE"]               # tokens, counted with the tokenizer above
CHUNK_OVERLAP = _PROFILE["CHUNK_OVERLAP"]         # tokens carried over from the previous chunk when a text must be cut
CHUNK_STRATEGY = _PROFILE["CHUNK_STRATEGY"]       # "fixed", "recursive" or "structure"
CHUNK_HEADERS = _PROFILE["CHUNK_HEADERS"]         # prepend "doc title > section" to each chunk ("structure" only)
KEEP_TABLES = _PROFILE["KEEP_TABLES"]             # never split a table, up to the embedding model's limit ("structure" only)
CHUNK_MIN_TOKENS = _PROFILE["CHUNK_MIN_TOKENS"]   # merge pieces smaller than this with a neighbour in the same section
CHILD_SIZE = _PROFILE["CHILD_SIZE"]               # parent-child retrieval: search children of this size, answer with the parent
CHUNK_TABLE_CAPTIONS = _PROFILE.get("CHUNK_TABLE_CAPTIONS", False)   # head a captioned table chunk with its caption ("structure" only)
TOP_K = 5
SPECIAL_TOKENS = 2    # [CLS] and [SEP] that the embedder adds on top of each chunk

# Retrieval (Phase 4). Strategies are in src/retrieve/retriever.py.
RETRIEVAL_MODE = _PROFILE.get("RETRIEVAL_MODE", "dense")    # "dense", "bm25" or "hybrid" (dense + BM25 merged with RRF)
RERANK_TOP = _PROFILE.get("RERANK_TOP", 0)                  # > 0: rerank this many candidates and keep TOP_K
RERANKER = _PROFILE.get("RERANKER", "minilm")               # a key of src/retrieve/rerank.py RERANKERS
QUERY_TRANSFORM = _PROFILE.get("QUERY_TRANSFORM", "none")   # "auto": rewrite / expand / split only the hard questions (src/retrieve/transform.py)
RERANK_MIN_RATIO = _PROFILE.get("RERANK_MIN_RATIO", 0.0)        # > 0: drop reranked chunks under this share of the best score
ASKED_FIRST = _PROFILE.get("ASKED_FIRST", False)             # per-document search: list the asked-about document first

CHROMA_DIR = "chroma_db"
# Each pipeline version keeps its own index, so v1 stays on disk while a later version is built.
# A version with the same chunks as an earlier one (INDEX_FROM) shares that index.
INDEX_VERSION = _PROFILE.get("INDEX_FROM", PIPELINE_VERSION)
COLLECTION_NAME = "rag_chunks" if INDEX_VERSION == "v1" else f"rag_chunks_{INDEX_VERSION}"

LLM_PROVIDER = _PROFILE["LLM_PROVIDER"]
LLM_MODEL = _PROFILE["LLM_MODEL"]
LLM_TEMPERATURE = 0           # low temperature for factual Q&A
LLM_CONTEXT_WINDOW = 8192     # Ollama defaults to a small window; 5 chunks need more
OLLAMA_URL = "http://localhost:11434"
PROMPT_VERSION = _PROFILE.get("PROMPT_VERSION", "v1")     # the answer prompt: a file in src/generate/prompts/

# Validation (Phase 6). Off for v1 to v3.
VALIDATE = _PROFILE.get("VALIDATE", False)
RELEVANCE_HIGH = _PROFILE.get("RELEVANCE_HIGH", 0.1)      # best reranker score at or above this: the context is graded "correct"
RELEVANCE_LOW = _PROFILE.get("RELEVANCE_LOW", 0.0005)     # below this: "incorrect"; in between: "ambiguous" (both get a corrective search)
RETRIES = _PROFILE.get("RETRIES", 0)                      # corrective searches (rewritten query) for a context that is not "correct"
REFINE = _PROFILE.get("REFINE", False)                    # CRAG knowledge refinement: a grading call drops chunks that do not help (src/validate/refine.py)
REFINE_WITH_CONTEXT = _PROFILE.get("REFINE_WITH_CONTEXT", False)   # the grader also sees the always-kept passages (q37)

# The judge is gpt-4o-mini, the same model as the answers, so it may favour its own style. With judge prompt v2 it broke its
# own rules now and then (it marked down missing optional details (q19), a value the reference accepts (h08) and correct
# refusals (h12)); v3 states those rules explicitly, a correct refusal is fixed in parse_verdict, and each answer is judged
# JUDGE_VOTES times (the median counts). Compare runs judged with the same model and judge prompt only.
JUDGE_PROVIDER = "openai"
JUDGE_MODEL = os.environ.get("RAG_JUDGE_MODEL", "gpt-4o-mini")
JUDGE_VOTES = int(os.environ.get("RAG_JUDGE_VOTES", 3))   # judge calls per answer; the verdict is their median (eval/judge.py combine_votes)
JUDGE_WORKERS = int(os.environ.get("RAG_JUDGE_WORKERS", 6))   # answers judged in parallel (same calls and cost, faster)
JUDGE_PROMPT_VERSION = os.environ.get("RAG_JUDGE_PROMPT", "v4")   # v2: one call; v3: a reference call (no context)
                                                                  # and a faithfulness call (no reference); v4: v3 with the
                                                                  # reference call listing key facts, scored in code (eval/judge.py)


# Settings that traces and eval runs made before they were recorded used. Missing keys count as these.
LEGACY_SETTINGS = {"version": "v1", "chunk_strategy": "fixed", "chunk_headers": False, "keep_tables": False,
                   "chunk_min_tokens": 0, "child_size": 0, "chunk_table_captions": False, "retrieval_mode": "dense", "rerank_top": 0, "reranker": None,
                   "query_transform": "none", "rerank_min_ratio": 0.0, "asked_first": False, "validate": False, "refine": False, "refine_with_context": False, "relevance_high": None, "relevance_low": None, "retries": None}


def with_legacy_defaults(config: dict) -> dict:
    return {**LEGACY_SETTINGS, **config}


def snapshot() -> dict:
    """Settings that shape an answer. Stored in every trace and eval run."""
    return {
        "version": PIPELINE_VERSION,
        "chunk_strategy": CHUNK_STRATEGY,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunk_headers": CHUNK_HEADERS,
        "keep_tables": KEEP_TABLES,
        "chunk_min_tokens": CHUNK_MIN_TOKENS,
        "child_size": CHILD_SIZE,
        "chunk_table_captions": CHUNK_TABLE_CAPTIONS,
        "retrieval_mode": RETRIEVAL_MODE,
        "rerank_top": RERANK_TOP,
        "reranker": RERANKER if RERANK_TOP else None,
        "query_transform": QUERY_TRANSFORM,
        "rerank_min_ratio": RERANK_MIN_RATIO,
        "asked_first": ASKED_FIRST,
        "top_k": TOP_K,
        "embed_model": EMBED_MODEL,
        "tokenizer": TOKENIZER_NAME,
        "llm": LLM_MODEL,
        "temperature": LLM_TEMPERATURE,
        "prompt_version": PROMPT_VERSION,
        "validate": VALIDATE,
        "relevance_high": RELEVANCE_HIGH if VALIDATE else None,
        "relevance_low": RELEVANCE_LOW if VALIDATE else None,
        "retries": RETRIES if VALIDATE else None,
        "refine": REFINE,
        "refine_with_context": REFINE_WITH_CONTEXT,
    }
