"""A small tracer: one trace per query, one span per pipeline stage, saved as JSON lines."""
import json
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from src.config import snapshot
from src.observability.logger import LOGS_DIR

TRACES_PATH = LOGS_DIR / "traces.jsonl"


class Trace:
    def __init__(self, query: str, config: dict | None = None):
        self.data = {
            "trace_id": uuid.uuid4().hex[:8],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "config": config or snapshot(),
            "query": query,
            "spans": {},
            "answer": None,
            "total_ms": None,
            "error": None,
        }
        self._start = time.perf_counter()

    @property
    def trace_id(self) -> str:
        return self.data["trace_id"]

    @contextmanager
    def span(self, name: str):
        """Time a stage. Put its inputs/outputs in the yielded dict; ms is added automatically."""
        record: dict = {}
        start = time.perf_counter()
        try:
            yield record
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["ms"] = round((time.perf_counter() - start) * 1000)
            self.data["spans"][name] = record

    def elapsed_ms(self) -> int:
        """Milliseconds since the question came in (used for the time to the first answer token)."""
        return round((time.perf_counter() - self._start) * 1000)

    def finish(self, answer: str | None = None, error: str | None = None) -> None:
        self.data["answer"] = answer
        self.data["error"] = error
        self.data["total_ms"] = round((time.perf_counter() - self._start) * 1000)

    def save(self, path: Path = TRACES_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self.data, ensure_ascii=False) + "\n")
        return path


def load_trace(trace_id: str | None = None, path: Path = TRACES_PATH) -> dict:
    """Find a trace by id, or the most recent one when trace_id is None."""
    if not path.exists():
        raise FileNotFoundError(f"No traces yet: {path}")
    traces = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if trace_id is None:
        return traces[-1]
    for t in traces:
        if t["trace_id"] == trace_id:
            return t
    raise KeyError(f"trace_id '{trace_id}' not found in {path}")
