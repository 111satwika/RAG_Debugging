"""Query rewriting: the model turns the user's wording into a clear search query."""
REWRITE_PROMPT = """Rewrite the question as a clear, self-contained search query for searching documents.
- Keep every name, ID, error code, number and technical term exactly as written.
- Spell out vague wording and abbreviations only when the meaning is certain. Do not add facts.
- Do not answer the question.
Return only the query.

Question: {question}"""


def rewrite_query(question: str, llm) -> str:
    text = llm.complete(REWRITE_PROMPT.format(question=question))["text"].strip().strip('"').strip()
    return text or question
