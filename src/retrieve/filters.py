"""Metadata filters: decide which documents a question is about, so the search only looks inside them.

A filter is only as good as the decision behind it. `route_with_llm` asks the model which documents the question needs
(both, for a question that combines two documents); `oracle_routes` uses the golden set's answer key and is only a
ceiling for what a perfect filter would give, never something the app can do.
"""
import json
import re

ROUTE_PROMPT = """You pick which documents are needed to answer a question.

Documents:
{catalog}

Rules:
- Return every document the question needs. A question that combines facts from two documents needs both.
- If the question could be about any document, or you cannot tell, return all of them.
- Use only the document ids listed above.

Question: {question}

Return JSON: {{"doc_ids": ["..."]}}"""


def build_catalog(chunks: list[dict]) -> list[dict]:
    """One entry per document, read from the indexed chunks: id, type, title and section names."""
    catalog: dict[str, dict] = {}
    for chunk in chunks:
        entry = catalog.setdefault(chunk["doc_id"], {"doc_id": chunk["doc_id"], "doc_type": chunk["doc_type"],
                                                     "title": "", "sections": []})
        first_line = chunk["text"].split("\n", 1)[0]
        if " > " in first_line:
            title, section = first_line.split(" > ", 1)
            entry["title"] = entry["title"] or title
            if section not in entry["sections"]:
                entry["sections"].append(section)
    for entry in catalog.values():
        entry["title"] = entry["title"] or re.sub(r"[_\s]+", " ", entry["doc_id"])
    return list(catalog.values())


def describe_catalog(catalog: list[dict], max_sections: int = 8) -> str:
    return "\n".join(f"- {d['doc_id']} ({d['doc_type']}): {d['title']}; sections: {', '.join(d['sections'][:max_sections])}"
                     for d in catalog)


def parse_doc_ids(text: str, catalog: list[dict]) -> list[str] | None:
    """The document ids in the model's answer. None (no filter) if the answer is unusable, empty or names every document."""
    known = {d["doc_id"] for d in catalog}
    try:
        ids = json.loads(text).get("doc_ids", [])
    except (ValueError, AttributeError):
        return None
    ids = [i for i in dict.fromkeys(ids) if i in known] if isinstance(ids, list) else []
    return ids if ids and len(ids) < len(known) else None


def route_with_llm(question: str, catalog: list[dict], llm) -> list[str] | None:
    prompt = ROUTE_PROMPT.format(catalog=describe_catalog(catalog), question=question)
    return parse_doc_ids(llm.complete(prompt, json_mode=True)["text"], catalog)


def oracle_routes(golden: list[dict], catalog: list[dict]) -> dict[str, list[str]]:
    """question text -> the document ids in the golden set's `source_docs` (those that exist in the index)."""
    known = {d["doc_id"] for d in catalog}
    routes = {}
    for q in golden:
        ids = [re.sub(r"\.pdf$", "", name) for name in q.get("source_docs", [])]
        ids = [i for i in ids if i in known]
        if ids and len(ids) < len(known):
            routes[q["question"]] = ids
    return routes
