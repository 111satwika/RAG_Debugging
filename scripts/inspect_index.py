"""Snapshot what is stored in Chroma into versioned, readable files.

Run after every improvement (new chunking, normalization, embedding model, ...):
    python -m scripts.inspect_index --note "what changed"

Each run creates the next version in data/processed/versions/ and never overwrites:
    chunks_report_v1.md, embeddings_v1.csv, ...  plus a row in versions.md
"""
import argparse
import csv
import math
import re
from datetime import datetime
from pathlib import Path

from src.config import (CHUNK_HEADERS, CHUNK_OVERLAP, CHUNK_SIZE, CHUNK_STRATEGY, COLLECTION_NAME, EMBED_MODEL, KEEP_TABLES,
                        TOKENIZER_NAME)
from src.ingest.index import get_client

VERSIONS_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "versions"
LOG_PATH = VERSIONS_DIR / "versions.md"
PREVIEW_DIMS = 8


def next_version() -> int:
    found = [int(m.group(1)) for p in VERSIONS_DIR.glob("chunks_report_v*.md")
             if (m := re.search(r"_v(\d+)\.md$", p.name))]
    return max(found, default=0) + 1


def main(note: str) -> None:
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    collection = get_client().get_collection(COLLECTION_NAME)
    data = collection.get(include=["documents", "metadatas", "embeddings"])
    rows = sorted(
        zip(data["ids"], data["documents"], data["metadatas"], data["embeddings"]),
        key=lambda r: (r[2]["doc_id"], r[2]["page"], r[0]),
    )
    dim = len(rows[0][3])
    tokens = [r[2]["n_tokens"] for r in rows]
    version = next_version()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    index_model = collection.metadata.get("embed_model")

    lines = [
        f"# Chroma chunk report - v{version}",
        f"- created: {stamp}",
        f"- change: {note}",
        f"- chunks: {len(rows)} | tokens per chunk: avg {sum(tokens) / len(tokens):.0f}, "
        f"min {min(tokens)}, max {max(tokens)} | vector dimension: {dim}",
        f"- embedding model in index: `{index_model}` | distance: {collection.metadata.get('hnsw:space')}",
        f"- tokenizer in index: `{rows[0][2]['tokenizer']}` | chunking: `{collection.metadata.get('chunking') or 'fixed-500 (recorded before strategies existed)'}` | "
        f"config: strategy={CHUNK_STRATEGY}, chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, headers={CHUNK_HEADERS}, keep_tables={KEEP_TABLES}",
        "",
    ]
    if index_model != EMBED_MODEL or rows[0][2]["tokenizer"] != TOKENIZER_NAME:
        lines.insert(4, "- WARNING: index was built with different settings than src/config.py; re-run src.ingest.index")
    for chunk_id, text, meta, vec in rows:
        norm = math.sqrt(sum(v * v for v in vec))
        preview = ", ".join(f"{v:+.4f}" for v in vec[:PREVIEW_DIMS])
        lines += [
            f"## {chunk_id}",
            f"- doc: `{meta['doc_id']}` ({meta['doc_type']}) | page: {meta['page']} | tokens: {meta['n_tokens']}",
            f"- vector (first {PREVIEW_DIMS} of {dim}): [{preview}, ...] | length: {norm:.3f}",
            "",
            "```text",
            text,
            "```",
            "",
        ]
    report_path = VERSIONS_DIR / f"chunks_report_v{version}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    csv_path = VERSIONS_DIR / f"embeddings_v{version}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["chunk_id", "doc_id", "page"] + [f"d{i}" for i in range(dim)])
        for chunk_id, _, meta, vec in rows:
            writer.writerow([chunk_id, meta["doc_id"], meta["page"]] + [f"{v:.6f}" for v in vec])

    if not LOG_PATH.exists():
        LOG_PATH.write_text(
            "# Version log\n\n| Version | Date | Chunks | Avg tokens | Embedding model | Change |\n|---|---|---|---|---|---|\n",
            encoding="utf-8",
        )
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"| v{version} | {stamp} | {len(rows)} | {sum(tokens) / len(tokens):.0f} | {index_model} | {note} |\n")

    print(f"v{version}: {report_path.name}, {csv_path.name} (logged in {LOG_PATH.name})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--note", default="(no note)", help="short description of what changed in this version")
    main(parser.parse_args().note)
