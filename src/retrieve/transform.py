"""Query transformation for the hard questions only.

Most questions are answered well by the plain search, and a transformation costs a model call and time. So a free,
rule-based gate decides first (no model involved), and only a question that needs it is transformed:

  cross_document   the question names two or more documents ("QuickLeave requires ... Does the candidate ...").
                   The plain search lets the document the question's words match best fill the top 5, so the
                   other document's chunk never arrives.

One model call then plans the search (a QueryPlan) with the three standard techniques:
  rewriting        one clear, self-contained query that uses the documents' own words
  expansion        extra terms the documents may use instead of the question's words (acronyms, synonyms)
  multi-query      one short query per need, each aimed at one document, searched separately and merged fairly

I also tried two other gates on the golden set and dropped them: "the question has words that appear nowhere in the
documents" fires on two thirds of the questions (mostly word endings), and "the reranker is unsure" never fires on a
weak question (a question that finds only half of its evidence still has a confident top result).
"""
import json
import re
from dataclasses import dataclass, field

from src.retrieve.bm25 import STOPWORDS

MAX_EXPANSIONS = 3
MAX_SUB_QUERIES = 3
# Words that name a kind of document, whatever its title is.
DOCUMENT_HINTS = {
    "resume": {"candidate", "applicant", "resume", "cv"},
    "spec": {"spec", "specification"},
    "paper": {"paper", "authors"},
}
GENERIC_TITLE_WORDS = {"product", "specification"}

PLAN_PROMPT = """You prepare search queries for a document search engine that combines meaning-based and keyword search.

Documents (the id is what you must use to refer to one):
{catalog}

Question: {question}

Return JSON: {{"rewrite": "...", "expansions": ["..."], "sub_queries": [{{"document": "<id>", "query": "..."}}]}}
- rewrite: one clear, self-contained search query. Keep every name, ID, number and technical term exactly as written. Spell
  out vague wording and use the words the documents are likely to use. Do not answer the question.
- expansions: up to {max_expansions} short keywords the documents may use instead of the question's own words: an acronym
  or abbreviation for a phrase in the question (for example "role-based access control" -> "RBAC"), or an exact synonym.
  Not general topic words. [] if there are none.
- sub_queries: when the question needs facts from two or more documents, one entry per document it needs (at most
  {max_sub_queries}). "document" is that document's id from the list. "query" says what to look for IN that document, in the
  words that document is likely to use; it never contains a document id or name, and it does not mention the other documents.
  [] if the question needs only one document."""


@dataclass
class SubQuery:
    document: str | None      # the document to search (a metadata filter), None to search them all
    query: str


@dataclass
class QueryPlan:
    rewrite: str
    expansions: list[str] = field(default_factory=list)
    sub_queries: list[SubQuery] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    ms: float = 0.0

    @property
    def multi_query(self) -> bool:
        return len(self.sub_queries) >= 2


def document_names(corpus: list[dict], catalog: list[dict]) -> dict[str, set[str]]:
    """The words that point at each document: words of its title that no other document uses, plus the words that name
    its kind of document (candidate, specification, paper)."""
    from src.retrieve.bm25 import tokenize

    words_of = {}
    for chunk in corpus:
        words_of.setdefault(chunk["doc_id"], set()).update(tokenize(chunk["text"]))
    names = {}
    for doc in catalog:
        others = set().union(*(w for d, w in words_of.items() if d != doc["doc_id"])) if len(words_of) > 1 else set()
        title = {t for t in re.findall(r"[a-z0-9]+", doc["title"].lower()) if t not in STOPWORDS and len(t) > 4}
        names[doc["doc_id"]] = (title - GENERIC_TITLE_WORDS - others) | DOCUMENT_HINTS.get(doc["doc_type"], set())
    return names


def documents_named(question: str, names: dict[str, set[str]]) -> list[str]:
    words = set(re.findall(r"[a-z0-9]+", question.lower()))
    return [doc_id for doc_id, doc_names in names.items() if doc_names & words]


def gate(question: str, names: dict[str, set[str]]) -> list[str]:
    """Why this question should be transformed (empty: search it as it is). Needs no model."""
    reasons = []
    named = documents_named(question, names)
    if len(named) >= 2:
        reasons.append("cross_document: " + ", ".join(named))
    return reasons


def parse_plan(text: str, question: str, known_docs: set[str]) -> QueryPlan | None:
    """The model's JSON as a QueryPlan, or None if it is unusable (the question is then searched as it is).
    A sub-query for a document that does not exist is searched in every document; two sub-queries for the same document
    keep only the first, because the point of several queries is to reach different documents."""
    try:
        data = json.loads(text)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None

    expansions = data.get("expansions", [])
    expansions = [s.strip() for s in expansions if isinstance(s, str) and s.strip()] if isinstance(expansions, list) else []
    rewrite = data.get("rewrite")
    rewrite = rewrite.strip() if isinstance(rewrite, str) and rewrite.strip() else question

    sub_queries, seen = [], set()
    raw = data.get("sub_queries", [])
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict) or not isinstance(item.get("query"), str) or not item["query"].strip():
            continue
        document = item.get("document") if item.get("document") in known_docs else None
        if document in seen and document is not None:
            continue
        seen.add(document)
        sub_queries.append(SubQuery(document, item["query"].strip()))
    sub_queries = sub_queries[:MAX_SUB_QUERIES]
    return QueryPlan(rewrite, list(dict.fromkeys(expansions))[:MAX_EXPANSIONS], sub_queries if len(sub_queries) >= 2 else [])


def make_plan(question: str, catalog: list[dict], llm) -> QueryPlan | None:
    from src.retrieve.filters import describe_catalog

    prompt = PLAN_PROMPT.format(catalog=describe_catalog(catalog), question=question, max_expansions=MAX_EXPANSIONS,
                                max_sub_queries=MAX_SUB_QUERIES)
    result = llm.complete(prompt, json_mode=True)
    plan = parse_plan(result["text"], question, {d["doc_id"] for d in catalog})
    if plan:
        plan.tokens_in, plan.tokens_out, plan.ms = result.get("input_tokens", 0), result.get("output_tokens", 0), result.get("ms", 0.0)
    return plan


def with_expansions(query: str, expansions: list[str]) -> str:
    """The query with the expansion terms added, so both the keyword and the meaning-based search see them."""
    return f"{query} ({', '.join(expansions)})" if expansions else query
