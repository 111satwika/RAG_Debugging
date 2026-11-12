"""Step 3b: embed chunks and store them in Chroma with their metadata.

With parent-child retrieval the collection holds the small *children* (what is searched) and each child's metadata
names its parent. The parents (what the model reads) are saved next to the index in data/processed/parents_<collection>.json.
"""
import json
from pathlib import Path

import chromadb

from src.config import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL
from src.ingest.chunk import chunk_all, current_spec, default_tokenizer
from src.ingest.chunkers import make_children
from src.ingest.embed import Embedder, load_embedder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARENTS_DIR = PROJECT_ROOT / "data" / "processed"
BATCH_SIZE = 64


def get_client(path: str = CHROMA_DIR) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(PROJECT_ROOT / path))


def parents_path(collection_name: str = COLLECTION_NAME) -> Path:
    return PARENTS_DIR / f"parents_{collection_name}.json"


def save_parents(parents: list[dict], collection_name: str = COLLECTION_NAME) -> Path:
    path = parents_path(collection_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({p["chunk_id"]: p for p in parents}, ensure_ascii=False), encoding="utf-8")
    return path


def load_parents(collection_name: str = COLLECTION_NAME) -> dict[str, dict]:
    path = parents_path(collection_name)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Re-run: python -m src.ingest.index")
    return json.loads(path.read_text(encoding="utf-8"))


def build_index(chunks: list[dict], embedder: Embedder | None = None, client=None, parents: list[dict] | None = None,
                persist_parents: bool = False, collection_name: str = COLLECTION_NAME):
    """Rebuild the collection from scratch (incremental updates come in Phase 7).

    `chunks` are what gets embedded. For parent-child retrieval pass the children here and the parents in `parents`.
    """
    embedder = embedder or load_embedder()
    client = client or get_client()

    existing = {c if isinstance(c, str) else c.name for c in client.list_collections()}
    if collection_name in existing:
        client.delete_collection(collection_name)
    collection = client.create_collection(
        collection_name,
        metadata={"hnsw:space": "cosine", "embed_model": embedder.name, "parent_child": bool(parents),
                  "chunking": chunks[0].get("strategy", "") if chunks else ""},
    )
    if parents and persist_parents:
        save_parents(parents, collection_name)

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=embedder.embed_documents([c["text"] for c in batch]),
            metadatas=[
                {
                    "doc_id": c["doc_id"],
                    "doc_type": c["doc_type"],
                    "page": c["page"],
                    "n_tokens": c["n_tokens"],
                    "tokenizer": c["tokenizer"],
                    "strategy": c.get("strategy", ""),
                    "header": c.get("header", ""),
                    "parent_id": c.get("parent_id", ""),
                }
                for c in batch
            ],
        )
    return collection


if __name__ == "__main__":
    spec, embedder = current_spec(), load_embedder()
    parents = chunk_all()
    if spec.child_size:
        children = make_children(parents, default_tokenizer(), spec)
        collection = build_index(children, embedder, parents=parents, persist_parents=True)
        print(f"indexed {collection.count()} children of {len(parents)} parent chunks")
    else:
        collection = build_index(parents, embedder)
    print(f"indexed {collection.count()} chunks with {EMBED_MODEL} ({spec.label}) -> {CHROMA_DIR}/{COLLECTION_NAME}")
