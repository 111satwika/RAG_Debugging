"""Step 2: chunking. Cut each normalized document into chunks with the strategy chosen in src/config.py."""
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from src.config import (CHILD_SIZE, CHUNK_HEADERS, CHUNK_MIN_TOKENS, CHUNK_OVERLAP, CHUNK_SIZE, CHUNK_STRATEGY, CHUNK_TABLE_CAPTIONS,
                        KEEP_TABLES,
                        SPECIAL_TOKENS, TOKENIZER_NAME, TOKENIZER_PROVIDER)
from src.ingest.chunkers import ChunkSpec, make_pieces
from src.ingest.normalize import normalize_document
from src.ingest.parse import PROCESSED_DIR, parse_all, save_parsed
from src.ingest.tokenization import Tokenizer, load_tokenizer
from src.observability.logger import LOGS_DIR, get_logger

log = get_logger("ingest")
INGEST_LOG_PATH = LOGS_DIR / "ingest.jsonl"
CHUNKS_PATH = PROCESSED_DIR.parent / "chunks.json"
NORMALIZED_DIR = PROCESSED_DIR.parent / "normalized"


def default_tokenizer() -> Tokenizer:
    return load_tokenizer(TOKENIZER_PROVIDER, TOKENIZER_NAME)


def current_spec() -> ChunkSpec:
    """The chunking setup chosen in src/config.py."""
    return ChunkSpec(strategy=CHUNK_STRATEGY, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP, headers=CHUNK_HEADERS,
                     keep_tables=KEEP_TABLES, min_tokens=CHUNK_MIN_TOKENS, child_size=CHILD_SIZE,
                     table_captions=CHUNK_TABLE_CAPTIONS)


def check_size(tokenizer: Tokenizer, size: int) -> None:
    """Fail early if a chunk could be truncated by the embedding model."""
    if tokenizer.max_tokens and size + SPECIAL_TOKENS > tokenizer.max_tokens:
        raise ValueError(
            f"chunk size {size} + {SPECIAL_TOKENS} special tokens exceeds "
            f"{tokenizer.name} limit of {tokenizer.max_tokens}"
        )


def chunk_document(doc: dict, tokenizer: Tokenizer | None = None, size: int | None = None,
                   spec: ChunkSpec | None = None) -> list[dict]:
    spec = spec or current_spec()
    if size is not None:
        spec = replace(spec, size=size)
    tokenizer = tokenizer or default_tokenizer()
    check_size(tokenizer, spec.size)
    chunks = []
    for piece in make_pieces(doc, tokenizer, spec):
        chunks.append({
            "chunk_id": f"{doc['doc_id']}_c{len(chunks)}",
            "doc_id": doc["doc_id"],
            "doc_type": doc["doc_type"],
            "page": piece.page,
            "n_tokens": tokenizer.count(piece.text),
            "tokenizer": tokenizer.name,
            "strategy": spec.label,
            "header": piece.header,
            "text": piece.text,
        })
    return chunks


def log_ingest(doc: dict, chunks: list[dict], path: Path = INGEST_LOG_PATH) -> dict:
    """One line per file: catches parsing problems (empty pages, zero chunks) early."""
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "file": doc["source"],
        "doc_type": doc["doc_type"],
        "pages": len(doc["pages"]),
        "empty_pages": sum(1 for p in doc["pages"] if not p["text"].strip()),
        "chars": sum(len(p["text"]) for p in doc["pages"]),
        "chunks": len(chunks),
        "error": doc["error"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    if record["error"] or record["empty_pages"] or not record["chunks"]:
        log.warning("ingest problem: %s", record)
    else:
        log.info("ingested %s: %d pages, %d chunks", doc["source"], record["pages"], record["chunks"])
    return record


def chunk_all(tokenizer: Tokenizer | None = None, size: int | None = None, spec: ChunkSpec | None = None) -> list[dict]:
    chunks = []
    for doc in parse_all():
        doc = normalize_document(doc)
        save_parsed(doc, NORMALIZED_DIR)
        doc_chunks = chunk_document(doc, tokenizer, size, spec)
        log_ingest(doc, doc_chunks)
        chunks.extend(doc_chunks)
    return chunks


if __name__ == "__main__":
    all_chunks = chunk_all()
    CHUNKS_PATH.write_text(json.dumps(all_chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    by_doc = {}
    for c in all_chunks:
        by_doc.setdefault(c["doc_id"], []).append(c["n_tokens"])
    for doc_id, toks in by_doc.items():
        print(f"{doc_id:40} chunks={len(toks):3} max_tokens={max(toks):3} min_tokens={min(toks):3}")
    print(f"strategy: {current_spec().label} | total chunks: {len(all_chunks)}  ->  {CHUNKS_PATH}")
