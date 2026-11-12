"""Step 4: dense retrieval. Embed the query, return the top-k closest chunks."""
import argparse
import json
from pathlib import Path

from src.config import COLLECTION_NAME, TOP_K
from src.ingest.embed import Embedder, load_embedder
from src.ingest.index import get_client, load_parents
from src.observability.logger import get_logger

log = get_logger("retrieve")
CHILD_FANOUT = 4    # with parent-child retrieval, look at this many children per wanted parent
GOLDEN_PATH = Path(__file__).resolve().parents[2] / "eval" / "golden_set.json"


def get_collection(embedder: Embedder):
    collection = get_client().get_collection(COLLECTION_NAME)
    indexed_with = collection.metadata.get("embed_model")
    if indexed_with != embedder.name:
        raise RuntimeError(
            f"Index was built with '{indexed_with}' but queries use '{embedder.name}'. "
            "Re-run: python -m src.ingest.index"
        )
    return collection


def embed_query(query: str, embedder: Embedder | None = None) -> list[float]:
    return (embedder or load_embedder()).embed_queries([query])[0]


def doc_filter(doc_ids: list[str] | None) -> dict | None:
    """A metadata filter that keeps only chunks of these documents (None = no filter)."""
    return {"doc_id": {"$in": list(doc_ids)}} if doc_ids else None


def search(query_vector: list[float], k: int = TOP_K, embedder: Embedder | None = None, collection=None,
           parents: dict | None = None, where: dict | None = None) -> list[dict]:
    """Return the k most similar chunks, best first. score = cosine similarity (1.0 = identical).

    With parent-child retrieval the index holds small children: the best children are found, then replaced by their
    parents (each parent once, scored by its best child), so the model reads whole sections.
    `where` is a metadata filter (see doc_filter); only matching chunks can be returned.
    """
    embedder = embedder or load_embedder()
    collection = collection or get_collection(embedder)
    parent_child = bool(collection.metadata.get("parent_child"))
    available = collection.count()
    wanted = min(k * CHILD_FANOUT, available) if parent_child else min(k, available)
    res = collection.query(
        query_embeddings=[query_vector],
        n_results=max(wanted, 1),
        include=["documents", "metadatas", "distances"],
        **({"where": where} if where else {}),
    )
    hits = [
        {
            "rank": rank,
            "chunk_id": chunk_id,
            "doc_id": meta["doc_id"],
            "doc_type": meta["doc_type"],
            "page": meta["page"],
            "score": round(1 - dist, 4),
            "text": text,
            "parent_id": meta.get("parent_id", ""),
        }
        for rank, (chunk_id, meta, dist, text) in enumerate(
            zip(res["ids"][0], res["metadatas"][0], res["distances"][0], res["documents"][0]), start=1
        )
    ]
    if parent_child:
        hits = _parents_of(hits, parents or load_parents(collection.name))[:k]
    for hit in hits:
        log.debug("hit #%d %s score=%.3f text=%r", hit["rank"], hit["chunk_id"], hit["score"], hit["text"])
    return hits


def _parents_of(child_hits: list[dict], parents: dict) -> list[dict]:
    """Replace ranked children by their parents, keeping the first (best) child's score for each."""
    seen, out = set(), []
    for hit in child_hits:
        parent = parents[hit["parent_id"]]
        if parent["chunk_id"] in seen:
            continue
        seen.add(parent["chunk_id"])
        out.append({"rank": len(out) + 1, "chunk_id": parent["chunk_id"], "doc_id": parent["doc_id"],
                    "doc_type": parent["doc_type"], "page": parent["page"], "score": hit["score"],
                    "text": parent["text"], "parent_id": "", "matched_child": hit["chunk_id"]})
    return out


def retrieve(query: str, k: int = TOP_K, embedder: Embedder | None = None, collection=None,
             parents: dict | None = None, where: dict | None = None) -> list[dict]:
    """Embed the query, then search."""
    embedder = embedder or load_embedder()
    return search(embed_query(query, embedder), k, embedder, collection, parents, where)


def load_question(arg: str) -> tuple[str, dict | None]:
    """Accept a free-text question or a golden-set id like q11."""
    golden = {q["id"]: q for q in json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))}
    if arg in golden:
        return golden[arg]["question"], golden[arg]
    return arg, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Top-k dense search over the Chroma index")
    parser.add_argument("query", help="a question, or a golden-set id such as q11")
    parser.add_argument("-k", type=int, default=TOP_K)
    parser.add_argument("--full", action="store_true", help="print full chunk text instead of a preview")
    args = parser.parse_args()

    question, golden = load_question(args.query)
    print(f"Q: {question}")
    if golden:
        print(f"expected: {golden['expected_answer']}")
        print(f"source:   {golden['source_docs']} | {golden['source_section']}\n")
    for hit in retrieve(question, args.k):
        body = hit["text"] if args.full else hit["text"][:220].replace("\n", " ") + " ..."
        print(f"#{hit['rank']} score={hit['score']:.3f} {hit['chunk_id']} (page {hit['page']})\n    {body}")
