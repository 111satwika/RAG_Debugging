"""Logging setup: INFO for pipeline steps, DEBUG for details (chunk text), ERROR for failures."""
import logging
import os
from pathlib import Path

# RAG_LOGS_DIR lets the tests write somewhere else, so they never touch the real logs.
LOGS_DIR = Path(os.environ.get("RAG_LOGS_DIR") or Path(__file__).resolve().parents[2] / "logs")
_FORMAT = "%(asctime)s %(levelname)-5s %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    root = logging.getLogger("rag")
    if not root.handlers:
        root.setLevel(logging.DEBUG)
        LOGS_DIR.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(LOGS_DIR / "pipeline.log", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_FORMAT))
        console = logging.StreamHandler()
        console.setLevel(logging.WARNING)  # keep CLI output clean; details go to the log file
        console.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(file_handler)
        root.addHandler(console)
        root.propagate = False
    return logging.getLogger(f"rag.{name}")
