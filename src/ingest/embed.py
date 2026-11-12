"""Step 3a: embedders. Swap the model by changing EMBED_MODEL/EMBED_PROVIDER in config.

To add a provider, write a class with `name`, `dim`, `embed_documents()` and
`embed_queries()`, then register it in PROVIDERS.
"""
from functools import lru_cache
import os
from typing import Protocol

from src.config import EMBED_MODEL, EMBED_PROVIDER, QUERY_PREFIX


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_queries(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    def __init__(self, name: str, query_prefix: str = ""):
        from sentence_transformers import SentenceTransformer

        self.name = name
        self.query_prefix = query_prefix
        self._model = SentenceTransformer(name)
        dim_fn = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self.dim = dim_fn()

    def _encode(self, texts: list[str]) -> list[list[float]]:
        # Unit-length vectors, so cosine similarity equals the dot product.
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._encode([self.query_prefix + t for t in texts])


PROVIDERS = {"sentence-transformers": SentenceTransformerEmbedder}


class OpenAIEmbedder:
    """OpenAI embedding models. Inputs may be up to 8191 tokens, so the 512-token limit of bge no longer applies."""

    BATCH = 96
    DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072, "text-embedding-ada-002": 1536}

    def __init__(self, name: str, query_prefix: str = ""):
        from src.generate.llm import _load_env

        _load_env()
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set. Put it in a .env file in the project root.")
        from openai import OpenAI

        self.name = name
        self.dim = self.DIMS.get(name, 1536)
        self._client = OpenAI()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for start in range(0, len(texts), self.BATCH):
            batch = [t if t.strip() else " " for t in texts[start:start + self.BATCH]]
            response = self._client.embeddings.create(model=self.name, input=batch)
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda item: item.index))
        return vectors      # OpenAI vectors are already unit length

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)


PROVIDERS["openai"] = OpenAIEmbedder


@lru_cache(maxsize=None)
def load_embedder(provider: str = EMBED_PROVIDER, name: str = EMBED_MODEL) -> Embedder:
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown embedder provider '{provider}'. Choose from {list(PROVIDERS)}")
    kwargs = {"query_prefix": QUERY_PREFIX} if "bge" in name.lower() else {}
    return PROVIDERS[provider](name, **kwargs)
