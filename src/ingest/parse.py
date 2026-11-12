"""Step 1: parse PDFs in data/raw/ into per-page markdown text."""
import re
from pathlib import Path

import pymupdf
import pymupdf4llm

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "parsed"

# The sub-folder name tells us the document type.
DOC_TYPES = {"resumes": "resume", "specs": "spec", "papers": "paper"}


def parse_pdf(path: Path, doc_type: str) -> dict:
    """Return {doc_id, doc_type, source, pages: [{page, text}], error}."""
    result = {
        "doc_id": path.stem,
        "doc_type": doc_type,
        "source": path.name,
        "pages": [],
        "error": None,
    }
    try:
        with pymupdf.open(path) as pdf:
            chunks = pymupdf4llm.to_markdown(pdf, page_chunks=True)
        for i, ch in enumerate(chunks, start=1):
            text = re.sub(r"\n{3,}", "\n\n", ch["text"]).strip()
            result["pages"].append({"page": i, "text": text})
    except Exception as exc:  # a bad file must not stop the whole ingest
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def parse_all(raw_dir: Path = RAW_DIR) -> list[dict]:
    docs = []
    for folder, doc_type in DOC_TYPES.items():
        for pdf_path in sorted((raw_dir / folder).glob("*.pdf")):
            docs.append(parse_pdf(pdf_path, doc_type))
    return docs


def save_parsed(doc: dict, out_dir: Path = PROCESSED_DIR) -> Path:
    """Write one readable .md file per document, with a marker before each page."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{doc['doc_id']}.md"
    parts = [f"<!-- page {p['page']} -->\n\n{p['text']}" for p in doc["pages"]]
    out_path.write_text("\n\n---\n\n".join(parts), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    for doc in parse_all():
        save_parsed(doc)
        chars = sum(len(p["text"]) for p in doc["pages"])
        print(f"{doc['source']:45} type={doc['doc_type']:7} "
              f"pages={len(doc['pages']):3} chars={chars:6} error={doc['error']}")
