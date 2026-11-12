"""Retrieval metrics, chunk-level relevance labels and the cheap keyword check. No LLM needed."""
import math
import re
from pathlib import Path

from eval.keyword_check import keyword_check as strict_keyword_check
RELEVANCE_THRESHOLD = 0.5   # a chunk is relevant when it holds at least this share of the key facts
SINGLE_FACT_TYPES = ("factual", "exact_term", "table")   # the answer should sit in one chunk


def normalize(text: str) -> str:
    text = text.lower().replace("\u00b2", "^2").replace("\u00b3", "^3").replace("\u2212", "-")
    text = re.sub(r"\s*([\u00b7^])\s*", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def contains(text: str, needle: str | list[str]) -> bool:
    """Whole-token match, so '6' does not match '16' or '2.6'. A word also matches its plural.
    A list of alternatives matches when any one of them does."""
    if isinstance(needle, list):
        return any(contains(text, alternative) for alternative in needle)
    needle = normalize(needle)
    if not needle:
        return False
    pattern = re.escape(needle)
    if needle[0].isalnum():
        pattern = r"(?<![a-z0-9.])" + pattern
    if needle[-1].isalpha():
        pattern += r"(?:e?s)?"          # 'gradient' should find 'gradients'
    elif needle[-1].isdigit() and "." in needle:
        pattern += r"(?:ms|s|x|k|m|b)?"   # '2.86' should find '2.86s' (a bare '6' still does not find '6s')
    if needle[-1].isalnum():
        pattern += r"(?![a-z0-9]|\.\d)"
    return re.search(pattern, normalize(text)) is not None


def coverage(text: str, needles: list) -> float:
    """Share of the key facts that appear in the text (a key fact can be a list of alternative wordings)."""
    if not needles:
        return 0.0
    return sum(contains(text, n) for n in needles) / len(needles)


def fact_label(fact) -> str:
    """A key fact for display: a plain string, or its alternatives joined with ' / '."""
    return " / ".join(fact) if isinstance(fact, list) else fact


def chunk_body(chunk: dict) -> str:
    """The chunk text without its contextual header, so a header cannot create or hide a key fact."""
    header, text = chunk.get("header") or "", chunk["text"]
    return text[len(header):] if header and text.startswith(header) else text


def source_doc_ids(question: dict) -> set[str]:
    return {Path(d).stem for d in question["source_docs"]}


def label_relevant(question: dict, chunks: list[dict], threshold: float = RELEVANCE_THRESHOLD) -> dict:
    """Chunk-level relevance, judged from the facts in `must_include`.

    A chunk is relevant if it comes from one of the question's source documents and holds at
    least `threshold` of the key facts. For single-fact question types only the chunks holding
    the most key facts count, because generic words ("Pending", "softmax") appear everywhere.
    If no chunk reaches the threshold (the answer is split across chunks), the best-covering
    chunks are used and the labels are flagged weak.
    Labels come from chunk text, never chunk ids, so they stay valid when chunking changes.
    """
    keys = question.get("must_include") or []
    if question["type"] == "unanswerable" or not keys:
        return {"ids": [], "weak": False, "coverage": {}}
    scored = {c["chunk_id"]: coverage(chunk_body(c), keys)
              for c in chunks if c["doc_id"] in source_doc_ids(question)}
    best = max(scored.values(), default=0.0)
    if question["type"] in SINGLE_FACT_TYPES and best >= threshold:
        ids = [cid for cid, cov in scored.items() if cov == best]
    else:
        ids = [cid for cid, cov in scored.items() if cov >= threshold]
    weak = False
    if not ids and best > 0:
        ids = [cid for cid, cov in scored.items() if cov == best]
        weak = True
    return {"ids": sorted(ids), "weak": weak, "coverage": {cid: round(scored[cid], 2) for cid in ids}}


def hit_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return float(any(c in relevant for c in ranked[:k]))


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(relevant & set(ranked[:k])) / len(relevant)


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return sum(c in relevant for c in ranked[:k]) / k


def mrr(ranked: list[str], relevant: set[str], k: int) -> float:
    for rank, cid in enumerate(ranked[:k], start=1):
        if cid in relevant:
            return 1 / rank
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1 / math.log2(rank + 1) for rank, cid in enumerate(ranked[:k], start=1) if cid in relevant)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def context_precision(given: list[str], relevant: set[str]) -> float:
    """Share of the chunks the model was given that are relevant. Unlike Precision@k (which always divides by k), it rises
    when fewer, better chunks are sent: with v9's cut-off a question with 1 relevant chunk in 1 given scores 1.0, not 0.2."""
    return sum(c in relevant for c in given) / len(given) if given else 0.0


def retrieval_metrics(ranked: list[str], relevant: set[str], k: int) -> dict | None:
    if not relevant:
        return None
    return {
        "recall": recall_at_k(ranked, relevant, k),
        "precision": precision_at_k(ranked, relevant, k),
        "context_precision": context_precision(ranked, relevant),
        "mrr": mrr(ranked, relevant, k),
        "ndcg": ndcg_at_k(ranked, relevant, k),
        "hit": hit_at_k(ranked, relevant, k),
    }


def keyword_check(answer: str | None, question: dict, found: bool | None = None) -> dict | None:
    """Cheap correctness proxy (no LLM), from eval/keyword_check.py: key facts present (alternatives allowed), stale or wrong
    values absent, and for a question with an `expected_verdict` the answer must start with that Yes or No.
    `found` is the answer's own found flag when it has one (cited prompts); without it the found check is skipped."""
    if answer is None or question["type"] == "unanswerable":
        return None
    check = strict_keyword_check(question, answer, found=True if found is None else found)
    keys = question.get("must_include") or []
    return {
        "include_rate": (len(keys) - len(check["missing"])) / len(keys) if keys else None,
        "missing": check["missing"],
        "forbidden": check["wrong"],
        "verdict_ok": check["verdict_ok"],
        "found_ok": check["found_ok"] if found is not None else None,
    }


def citation_check(answer: dict, relevant: set[str]) -> dict | None:
    """What a cited, structured answer did with its sources (None for a plain-text answer). `answer` is an answers.jsonl
    record: `status`, `found` and `cited` (the chunk ids it cited) are present only for the cited prompts."""
    if answer.get("status") is None:
        return None
    cited = answer.get("cited") or []
    return {
        "status": answer["status"],
        "found": answer.get("found"),
        "cited": cited,
        "has_citation": bool(cited),
        "cites_relevant": bool(set(cited) & relevant) if relevant else None,      # None: no chunk is labelled relevant
        "issues": answer.get("issues") or [],
    }


def citation_summary(results: list[dict]) -> dict:
    """Citation metrics over a list of results; every value is None when no answer was a cited one.

    citation_rate         of the answerable questions the model answered ("found"), the share that cite at least one source
    cites_relevant_rate   of the answerable questions that have relevant chunks and were answered, the share that cite one of them
    format_ok_rate        share of answers whose JSON was valid, had no unknown source, and cited every source it listed
    format_invalid_rate   share of replies that were not valid JSON of the right shape
    found_agreement       share of judged answers where the `found` flag agrees with the judge on whether it abstained
    """
    rows = [r for r in results if r.get("citation")]
    empty = {"citation_rate": None, "cites_relevant_rate": None, "format_ok_rate": None, "format_invalid_rate": None,
             "found_agreement": None}
    if not rows:
        return empty
    answered = [r for r in rows if r["type"] != "unanswerable" and r["citation"]["found"]]
    labelled = [r for r in answered if r["citation"]["cites_relevant"] is not None]
    judged = [r for r in rows if r.get("judge") and r["citation"]["found"] is not None]

    def share(items, test):
        return sum(1 for r in items if test(r)) / len(items) if items else None

    return {
        "citation_rate": share(answered, lambda r: r["citation"]["has_citation"]),
        "cites_relevant_rate": share(labelled, lambda r: r["citation"]["cites_relevant"]),
        "format_ok_rate": share(rows, lambda r: r["citation"]["status"] == "ok"),
        "format_invalid_rate": share(rows, lambda r: r["citation"]["status"] == "invalid"),
        "found_agreement": share(judged, lambda r: r["citation"]["found"] == (not r["judge"]["abstained"])),
    }
