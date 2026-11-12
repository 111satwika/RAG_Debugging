"""Tokenizer abstraction so the chunker is not tied to one model.

To support a new tokenizer, add a class with `name`, `max_tokens`, `offsets()` and
`count()`, then register it in PROVIDERS.
"""
from functools import lru_cache
from typing import Protocol


class Tokenizer(Protocol):
    name: str
    max_tokens: int | None  # hard input limit of the matching embedding model, if known

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """(start_char, end_char) of every token, without special tokens."""

    def count(self, text: str) -> int: ...


class HFTokenizer:
    """Any Hugging Face tokenizer; use the same name as the embedding model."""

    def __init__(self, name: str):
        from transformers import AutoTokenizer

        self.name = name
        self._tok = AutoTokenizer.from_pretrained(name)
        limit = self._tok.model_max_length
        self.max_tokens = limit if limit < 100_000 else None

    def offsets(self, text: str) -> list[tuple[int, int]]:
        enc = self._tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return [tuple(o) for o in enc["offset_mapping"]]

    def count(self, text: str) -> int:
        return len(self._tok(text, add_special_tokens=False)["input_ids"])


class TiktokenTokenizer:
    """OpenAI tokenizers, e.g. name='cl100k_base' for text-embedding-3-*."""

    def __init__(self, name: str):
        import tiktoken

        self.name = name
        self._enc = tiktoken.get_encoding(name)
        self.max_tokens = 8191

    def offsets(self, text: str) -> list[tuple[int, int]]:
        tokens = self._enc.encode(text)
        _, starts = self._enc.decode_with_offsets(tokens)
        ends = starts[1:] + [len(text)]
        return list(zip(starts, ends))

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))


PROVIDERS = {"huggingface": HFTokenizer, "tiktoken": TiktokenTokenizer}


@lru_cache(maxsize=None)
def load_tokenizer(provider: str, name: str) -> Tokenizer:
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown tokenizer provider '{provider}'. Choose from {list(PROVIDERS)}")
    return PROVIDERS[provider](name)
