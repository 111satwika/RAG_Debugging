"""Build the LLM prompt from retrieved chunks, using a versioned template.

v1 is the baseline: the chunks are pasted in and the model answers in plain text.
v2 and later are the cited prompts: every chunk is delimited and labelled with a source id ([S1], [S2], ... in rank order),
the model must cite them, and it replies with JSON that src/generate/answer.py validates.
"""
from pathlib import Path

from src.config import PROMPT_VERSION

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
PLAIN_PROMPTS = {"v1"}      # every other prompt version is a cited, structured one


def is_cited(version: str) -> bool:
    return version not in PLAIN_PROMPTS


def source_ids(chunks: list[dict]) -> list[str]:
    """The ids the cited prompts give the chunks: S1 for the best chunk, S2 for the next, and so on."""
    return [f"S{i}" for i in range(1, len(chunks) + 1)]


def format_context(chunks: list[dict], cited: bool) -> str:
    if not cited:
        return "\n\n".join(c["text"] for c in chunks)
    return "\n\n".join(f"[{sid}] (document: {c.get('doc_id', '?')}, page {c.get('page', '?')})\n{c['text']}"
                       for sid, c in zip(source_ids(chunks), chunks))


def build_prompt(question: str, chunks: list[dict], version: str = PROMPT_VERSION) -> str:
    template = (PROMPT_DIR / f"{version}.txt").read_text(encoding="utf-8")
    return template.format(context=format_context(chunks, is_cited(version)), question=question)
