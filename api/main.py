"""Local API for the web UI: chat, traces, logs and eval reports.

Run from the project root:  python -m uvicorn api.main:app --host 127.0.0.1 --port 8765
It also serves the built UI (ui/dist) when that folder exists.
"""
import json
import queue
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi import Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from eval.metrics import label_relevant
from eval.run_eval import load_index_chunks
from src.config import (COLLECTION_NAME, LLM_MODEL, LLM_PROVIDER, OLLAMA_URL, PIPELINE_VERSION, TOP_K, snapshot,
                        with_legacy_defaults)
from src.generate.llm import load_llm
from src.ingest.index import get_client
from src.observability.logger import LOGS_DIR
from src.observability.tracer import TRACES_PATH
from src.pipeline import run_query
from src.retrieve.retriever import configured_search, shared_retriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
GOLDEN_PATH = PROJECT_ROOT / "eval" / "golden_set.json"
UI_DIST = PROJECT_ROOT / "ui" / "dist"
MAX_JOBS = 50
MAX_LOG_MESSAGE = 600

LOG_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ (\w+)\s+([\w.]+): (.*)$")
TRACE_ID = r"^[a-f0-9]{8}$"
RUN_NAME = r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$"
VERSION = r"^[A-Za-z0-9_.-]{1,20}$"
LEGACY_VERSION = "v1"   # traces and runs made before versions were recorded came from the baseline setup

def warm_up_retrieval() -> None:
    """Load the index, the BM25 index and the reranker in the background, so the first question does not wait for them."""
    if configured_search() is not None:
        threading.Thread(target=shared_retriever, daemon=True, name="retrieval-warm-up").start()


@asynccontextmanager
async def lifespan(_app):
    warm_up_retrieval()
    yield


app = FastAPI(title="RAG Project API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def get_llm():
    return load_llm()


# ---------------------------------------------------------------- health

@app.get("/api/health")
def health() -> dict:
    try:
        chunks = get_client().get_collection(COLLECTION_NAME).count()
    except Exception:
        chunks = None
    reachable = None
    if LLM_PROVIDER == "ollama":
        try:
            reachable = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2).ok
        except requests.RequestException:
            reachable = False
    return {"status": "ok", "config": snapshot(), "indexed_chunks": chunks,
            "llm": {"provider": LLM_PROVIDER, "model": LLM_MODEL, "reachable": reachable}}


# ---------------------------------------------------------------- chat (job + polling)

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=TOP_K, ge=1, le=10)


JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
EXECUTOR = ThreadPoolExecutor(max_workers=1)   # one query at a time: the local model is the bottleneck


def _chat_result(out: dict) -> dict:
    trace = out["trace"]
    llm_span = trace["spans"].get("llm_call", {})
    return {
        "question": trace["query"],
        "answer": out["answer"],
        "trace_id": out["trace_id"],
        "chunks": out["chunks"],
        "timings": {name: span["ms"] for name, span in trace["spans"].items()},
        "total_ms": trace["total_ms"],
        "model": llm_span.get("model"),
        "input_tokens": llm_span.get("input_tokens"),
        "output_tokens": llm_span.get("output_tokens"),
        # Streamed queries only: time from the question to the first answer character the user sees, and from the model
        # call to the model's first token (the answer starts later when the model writes its "checks" first).
        "ttft_ms": trace.get("ttft_ms"),
        "model_ttft_ms": llm_span.get("ttft_ms"),
        # Cited prompts only: did the documents hold the answer, how well the reply followed the format, what it cites.
        "found": out.get("found"),
        "answer_status": out.get("answer_status"),
        "sources": out.get("sources"),
        # Versions with the validation layer: how the retrieved context was graded, whether a second search corrected it, and
        # whether the query was stopped before the model was called.
        "validation": out.get("validation"),
        # The CRAG grade of each retrieved chunk (correct / ambiguous / incorrect from its reranker score) and whether the
        # answer model read it (the refinement filter can drop it).
        "context": out.get("context"),
    }


def _run_job(job_id: str, question: str, k: int) -> None:
    job = JOBS[job_id]
    job["status"], job["started"] = "running", time.time()
    try:
        out = run_query(question, k, get_llm(), trace_path=TRACES_PATH)
        job["result"], job["status"] = _chat_result(out), "done"
    except Exception as exc:
        job["error"], job["status"] = f"{type(exc).__name__}: {exc}", "error"


@app.post("/api/chat")
def start_chat(req: ChatRequest) -> dict:
    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "queued", "created": time.time(), "started": None, "result": None, "error": None}
        for old in [j for j, v in JOBS.items() if v["status"] in ("done", "error")][:-MAX_JOBS]:
            JOBS.pop(old, None)
    EXECUTOR.submit(_run_job, job_id, req.question.strip(), req.k)
    return {"job_id": job_id, "status": "queued"}


STREAM_TIMEOUT_S = 600


@app.post("/api/chat/stream")
def stream_chat(req: ChatRequest) -> StreamingResponse:
    """Answer one question as Server-Sent Events: {"type": "stage"} when a step starts, {"type": "token"} for each piece of
    the answer as the model writes it, then {"type": "done", "result": ...} (the same result as the job endpoint, with
    ttft_ms) or {"type": "error"}. Queries still run one at a time, in the same queue as the job endpoint."""
    events: queue.Queue = queue.Queue()
    question = req.question.strip()

    def work() -> None:
        try:
            out = run_query(question, req.k, get_llm(), trace_path=TRACES_PATH, on_event=events.put)
            events.put({"type": "done", "result": _chat_result(out)})
        except Exception as exc:
            events.put({"type": "error", "error": f"{type(exc).__name__}: {exc}"})

    events.put({"type": "stage", "stage": "queued"})
    EXECUTOR.submit(work)

    def sse():
        while True:
            try:
                event = events.get(timeout=STREAM_TIMEOUT_S)
            except queue.Empty:
                event = {"type": "error", "error": f"no answer within {STREAM_TIMEOUT_S} s"}
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if event["type"] in ("done", "error"):
                return

    return StreamingResponse(sse(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/chat/{job_id}")
def chat_status(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return {"status": job["status"], "elapsed_s": round(time.time() - job["created"], 1),
            "result": job["result"], "error": job["error"]}


def _version_of(config: dict | None) -> str:
    return str((config or {}).get("version") or LEGACY_VERSION)


def _version_key(version: str) -> tuple:
    """Natural order: v1, v2, ..., v10."""
    return tuple(int(n) for n in re.findall(r"\d+", version)), version


# ---------------------------------------------------------------- traces

def _read_traces() -> list[dict]:
    if not TRACES_PATH.exists():
        return []
    traces = []
    for line in TRACES_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                traces.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return traces


@app.get("/api/traces")
def list_traces(limit: int = Query(200, ge=1, le=1000), version: str | None = Query(None, pattern=VERSION)) -> list[dict]:
    summaries = []
    for t in reversed(_read_traces()):
        trace_version = _version_of(t.get("config"))
        if version and trace_version != version:
            continue
        results = t.get("spans", {}).get("retrieve", {}).get("results", [])
        summaries.append({
            "trace_id": t["trace_id"], "timestamp": t["timestamp"], "query": t["query"],
            "total_ms": t["total_ms"], "error": t["error"], "version": trace_version,
            "answer_preview": (t["answer"] or "")[:160],
            "top_score": results[0]["score"] if results else None,
            "llm": t.get("config", {}).get("llm"),
        })
    return summaries[:limit]


GOLDEN_FIELDS = ("id", "type", "expected_answer", "source_docs", "source_section", "must_include", "must_not_include", "notes")


def _normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _golden_for(query: str) -> dict | None:
    """The golden-set entry (with the expected answer) whose question matches this trace's query."""
    try:
        golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    wanted = _normalize_question(query)
    for entry in golden:
        if _normalize_question(entry["question"]) == wanted:
            return {field: entry.get(field) for field in GOLDEN_FIELDS}
    return None


INDEX_SETTINGS = ("chunk_strategy", "chunk_size", "chunk_overlap", "chunk_headers", "keep_tables", "chunk_min_tokens",
                  "child_size", "embed_model", "tokenizer")


def _relevance_for(trace: dict, golden: dict | None) -> dict | None:
    """Which indexed chunks hold the question's key facts (the same text check the eval uses).

    Chunk ids only mean something for the index they came from, so this is skipped when the
    trace was made with different chunking or embedding settings than the current index.
    """
    if not golden or golden["type"] == "unanswerable":
        return None
    config, current = with_legacy_defaults(trace.get("config", {})), snapshot()
    if config["version"] != PIPELINE_VERSION or any(config.get(key) != current[key] for key in INDEX_SETTINGS):
        return {"available": False, "reason": "The index has changed since this trace was made, so its chunk ids no longer match."}
    try:
        label = label_relevant(golden, load_index_chunks())
    except Exception:
        return None
    retrieved = [r["chunk_id"] for r in trace.get("spans", {}).get("retrieve", {}).get("results", [])]
    return {"available": True, "relevant_ids": label["ids"], "weak": label["weak"],
            "missing": [c for c in label["ids"] if c not in retrieved]}


@app.get("/api/traces/{trace_id}")
def get_trace(trace_id: str = PathParam(pattern=TRACE_ID)) -> dict:
    for t in _read_traces():
        if t["trace_id"] == trace_id:
            golden = _golden_for(t["query"])
            return {**t, "golden": golden, "version": _version_of(t.get("config")),
                    "relevance": _relevance_for(t, golden)}
    raise HTTPException(404, "trace not found")


# ---------------------------------------------------------------- logs

def _tail_lines(path: Path, n: int, max_bytes: int = 1_000_000) -> list[str]:
    if not path.exists():
        return []
    size = path.stat().st_size
    with open(path, "rb") as f:
        f.seek(max(0, size - max_bytes))
        data = f.read()
    lines = data.decode("utf-8", errors="replace").splitlines()
    if size > max_bytes:
        lines = lines[1:]   # the first line may be cut in half
    return lines[-n:]


@app.get("/api/logs/pipeline")
def pipeline_log(lines: int = Query(300, ge=10, le=2000)) -> dict:
    path = LOGS_DIR / "pipeline.log"
    entries: list[dict] = []
    for raw in _tail_lines(path, lines):
        match = LOG_LINE.match(raw)
        if match:
            entries.append({"time": match[1], "level": match[2], "logger": match[3], "message": match[4]})
        elif entries:   # a traceback line continues the previous entry
            entries[-1]["message"] += "\n" + raw
    for e in entries:
        if len(e["message"]) > MAX_LOG_MESSAGE:
            e["message"] = e["message"][:MAX_LOG_MESSAGE] + " ..."
    return {"entries": entries, "size_bytes": path.stat().st_size if path.exists() else 0}


@app.get("/api/logs/ingest")
def ingest_log(limit: int = Query(200, ge=1, le=1000)) -> list[dict]:
    records = []
    for raw in _tail_lines(LOGS_DIR / "ingest.jsonl", limit):
        try:
            records.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return list(reversed(records))


# ---------------------------------------------------------------- reports

def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


@app.get("/api/reports")
def list_reports(version: str | None = Query(None, pattern=VERSION)) -> list[dict]:
    runs = []
    if RESULTS_DIR.exists():
        for run_dir in RESULTS_DIR.iterdir():
            meta, summary = _read_json(run_dir / "config.json"), _read_json(run_dir / "summary.json")
            if run_dir.is_dir() and meta and summary:
                run_version = _version_of(meta.get("config"))
                if version and run_version != version:
                    continue
                runs.append({"name": run_dir.name, "created": meta.get("created", ""), "note": meta.get("note", ""),
                             "k": meta.get("k"), "config": meta.get("config", {}), "judge": meta.get("judge"),
                             "version": run_version, "overall": summary["overall"]})
    return sorted(runs, key=lambda r: r["created"], reverse=True)


@app.get("/api/versions")
def list_versions() -> dict:
    """Every pipeline version that has traces or eval runs, plus the current one."""
    found = {PIPELINE_VERSION} | {_version_of(t.get("config")) for t in _read_traces()}
    if RESULTS_DIR.exists():
        for run_dir in RESULTS_DIR.iterdir():
            meta = _read_json(run_dir / "config.json")
            if run_dir.is_dir() and meta:
                found.add(_version_of(meta.get("config")))
    return {"versions": sorted(found, key=_version_key), "current": PIPELINE_VERSION}


@app.get("/api/reports/{name}")
def get_report(name: str = PathParam(pattern=RUN_NAME)) -> dict:
    run_dir = RESULTS_DIR / name
    meta, summary = _read_json(run_dir / "config.json"), _read_json(run_dir / "summary.json")
    if not (run_dir.is_dir() and meta and summary):
        raise HTTPException(404, "report not found")
    results = []
    results_path = run_dir / "results.jsonl"
    if results_path.exists():
        results = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    report_md = run_dir / "report.md"
    return {"meta": meta, "overall": summary["overall"], "by_type": summary["by_type"], "results": results,
            "version": _version_of(meta.get("config")),
            "markdown": report_md.read_text(encoding="utf-8") if report_md.exists() else ""}


# ---------------------------------------------------------------- built UI

if UI_DIST.exists():
    app.mount("/assets", StaticFiles(directory=UI_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_ui(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(404)
        return FileResponse(UI_DIST / "index.html")
