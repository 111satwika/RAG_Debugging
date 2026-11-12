"""API tests. A fake LLM and temp folders keep them fast and away from the real traces/results."""
import json
import time

import pytest
from fastapi.testclient import TestClient

import api.main as api_main


class FakeLLM:
    name = "fake-llm"

    def complete(self, prompt, json_mode=False):
        return {"text": "Insufficient leave balance, status 409.", "input_tokens": 100, "output_tokens": 8,
                "ms": 2, "temperature": 0}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main, "TRACES_PATH", tmp_path / "traces.jsonl")
    monkeypatch.setattr(api_main, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(api_main, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(api_main, "get_llm", lambda: FakeLLM())
    return TestClient(api_main.app)


def wait_for(client, job_id, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/chat/{job_id}").json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.3)
    raise AssertionError("job did not finish")


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["indexed_chunks"] == 38
    assert body["config"]["top_k"] == 5 and "model" in body["llm"]


def test_chat_job_returns_answer_chunks_and_a_trace(client):
    job = client.post("/api/chat", json={"question": "What does error code QL-1003 mean?", "k": 3}).json()
    body = wait_for(client, job["job_id"])
    assert body["status"] == "done", body
    result = body["result"]
    assert result["answer"].startswith("Insufficient leave balance")
    assert len(result["chunks"]) == 3 and result["chunks"][0]["text"]
    assert set(result["timings"]) == {"embed_query", "retrieve", "build_prompt", "llm_call"}
    assert result["input_tokens"] == 100 and result["model"] == "fake-llm"

    summaries = client.get("/api/traces").json()
    assert [s["trace_id"] for s in summaries] == [result["trace_id"]]
    assert summaries[0]["top_score"] > 0.5
    detail = client.get(f"/api/traces/{result['trace_id']}").json()
    assert "final_prompt" in detail["spans"]["build_prompt"] and detail["answer"] == result["answer"]


def test_trace_includes_expected_answer_when_question_is_in_the_golden_set(client, tmp_path, monkeypatch):
    golden = [{"id": "q99", "question": "What does error code QL-1003 mean?", "expected_answer": "Insufficient leave balance. HTTP 409.",
               "source_docs": ["quickleave_spec_v1.2.pdf"], "source_section": "8. Error Codes", "type": "exact_term",
               "must_include": ["409"], "notes": "a note"}]
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(json.dumps(golden), encoding="utf-8")
    monkeypatch.setattr(api_main, "GOLDEN_PATH", golden_path)

    matched = wait_for(client, client.post("/api/chat", json={"question": "  what does ERROR code QL-1003 mean? ", "k": 2}).json()["job_id"])
    other = wait_for(client, client.post("/api/chat", json={"question": "Something not in the golden set", "k": 2}).json()["job_id"])

    with_golden = client.get(f"/api/traces/{matched['result']['trace_id']}").json()["golden"]
    assert with_golden["id"] == "q99" and with_golden["expected_answer"] == "Insufficient leave balance. HTTP 409."
    assert with_golden["source_section"] == "8. Error Codes" and with_golden["notes"] == "a note"
    assert client.get(f"/api/traces/{other['result']['trace_id']}").json()["golden"] is None


def test_trace_without_a_golden_file_still_loads(client, tmp_path, monkeypatch):
    monkeypatch.setattr(api_main, "GOLDEN_PATH", tmp_path / "missing.json")
    job = wait_for(client, client.post("/api/chat", json={"question": "anything", "k": 1}).json()["job_id"])
    assert client.get(f"/api/traces/{job['result']['trace_id']}").json()["golden"] is None


def test_trace_marks_which_retrieved_chunks_are_relevant(client, tmp_path, monkeypatch):
    golden = [{"id": "q99", "question": "What does error code QL-1003 mean?", "expected_answer": "Insufficient leave balance. 409.",
               "source_docs": ["quickleave_spec_v1.2.pdf"], "source_section": "8", "type": "exact_term",
               "must_include": ["409", "Insufficient leave balance"]},
              {"id": "q98", "question": "Which database does QuickLeave use?", "expected_answer": "ABSTAIN: not specified",
               "source_docs": [], "source_section": None, "type": "unanswerable"}]
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(json.dumps(golden), encoding="utf-8")
    monkeypatch.setattr(api_main, "GOLDEN_PATH", golden_path)

    found = wait_for(client, client.post("/api/chat", json={"question": "What does error code QL-1003 mean?", "k": 5}).json()["job_id"])
    relevance = client.get(f"/api/traces/{found['result']['trace_id']}").json()["relevance"]
    assert relevance["available"] is True and "quickleave_spec_v1.2_c4" in relevance["relevant_ids"]
    assert relevance["missing"] == []                      # the answer chunk was among the 5 retrieved

    # A real miss: for the education question the answer chunk (the resume's last chunk) is not in the top 5.
    golden.append({"id": "q97", "question": "Where did the candidate study, and what was her CGPA?", "expected_answer": "SASTRA, 9.13",
                   "source_docs": ["Satwika_Agentic_AI_Engineer_Resume.pdf"], "source_section": "Education", "type": "factual",
                   "must_include": ["SASTRA", "9.13"]})
    golden_path.write_text(json.dumps(golden), encoding="utf-8")
    missed = wait_for(client, client.post("/api/chat", json={"question": "Where did the candidate study, and what was her CGPA?", "k": 5}).json()["job_id"])
    miss = client.get(f"/api/traces/{missed['result']['trace_id']}").json()["relevance"]
    assert miss["available"] is True and miss["missing"] == ["Satwika_Agentic_AI_Engineer_Resume_c4"]

    unanswerable = wait_for(client, client.post("/api/chat", json={"question": "Which database does QuickLeave use?", "k": 2}).json()["job_id"])
    assert client.get(f"/api/traces/{unanswerable['result']['trace_id']}").json()["relevance"] is None
    other = wait_for(client, client.post("/api/chat", json={"question": "not in the golden set", "k": 2}).json()["job_id"])
    assert client.get(f"/api/traces/{other['result']['trace_id']}").json()["relevance"] is None


def test_relevance_is_withheld_when_the_index_has_changed(client, tmp_path, monkeypatch):
    golden = [{"id": "q99", "question": "old question", "expected_answer": "x", "source_docs": ["quickleave_spec_v1.2.pdf"],
               "source_section": "s", "type": "factual", "must_include": ["409"]}]
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(json.dumps(golden), encoding="utf-8")
    monkeypatch.setattr(api_main, "GOLDEN_PATH", golden_path)
    config = {**api_main.snapshot(), "chunk_size": 123}      # made with different chunking than the current index
    old = {"trace_id": "0000000a", "timestamp": "2026-10-19T10:00:00", "query": "old question", "config": config,
           "spans": {"retrieve": {"ms": 1, "results": [{"chunk_id": "x_c0"}]}}, "answer": "a", "total_ms": 1, "error": None}
    api_main.TRACES_PATH.write_text(json.dumps(old) + "\n", encoding="utf-8")

    relevance = client.get("/api/traces/0000000a").json()["relevance"]
    assert relevance["available"] is False and "index has changed" in relevance["reason"]


def test_chat_validation_and_unknown_job(client):
    assert client.post("/api/chat", json={"question": ""}).status_code == 422
    assert client.post("/api/chat", json={"question": "x", "k": 99}).status_code == 422
    assert client.get("/api/chat/doesnotexist").status_code == 404


def test_the_chat_result_carries_the_validation_verdict_only_when_there_is_one():
    trace = {"query": "q", "spans": {"validate": {"ms": 5}}, "total_ms": 5}
    verdict = {"grade": "incorrect", "first_grade": "incorrect", "best_score": 0.0001, "action": "abstain",
               "reason": "nothing relevant was retrieved", "retries": [{"query": "q2", "search_ms": 90, "grade_before": "incorrect",
                                                                       "grade_after": "incorrect", "best_retry_score": 0.0002}]}
    out = {"trace": trace, "answer": "Information not found in the provided documents.", "trace_id": "t1", "chunks": [], "validation": verdict}
    result = api_main._chat_result(out)
    assert result["validation"] == verdict and result["model"] is None and result["input_tokens"] is None   # no model call was made
    assert api_main._chat_result({**out, "validation": None})["validation"] is None
    assert api_main._chat_result({k: v for k, v in out.items() if k != "validation"})["validation"] is None


def test_failed_query_reports_error(client, monkeypatch):
    class Broken:
        name = "broken"

        def complete(self, prompt, json_mode=False):
            raise RuntimeError("model unavailable")

    monkeypatch.setattr(api_main, "get_llm", lambda: Broken())
    job = client.post("/api/chat", json={"question": "anything"}).json()
    body = wait_for(client, job["job_id"])
    assert body["status"] == "error" and "model unavailable" in body["error"]


def test_trace_id_is_validated(client):
    assert client.get("/api/traces/not-an-id").status_code == 422
    assert client.get("/api/traces/deadbeef").status_code == 404
    assert client.get("/api/traces").json() == []


def test_pipeline_log_parsing_and_truncation(client, tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    long_text = "x" * 2000
    (logs / "pipeline.log").write_text(
        "2026-10-19 01:00:00,123 INFO  rag.pipeline: trace abc: hello\n"
        f"2026-10-19 01:00:01,456 DEBUG rag.retrieve: hit #1 text={long_text}\n"
        "2026-10-19 01:00:02,789 ERROR rag.pipeline: trace abc failed\n"
        "Traceback (most recent call last):\n  File \"x.py\", line 1\n", encoding="utf-8")
    entries = client.get("/api/logs/pipeline?lines=50").json()["entries"]
    assert [e["level"] for e in entries] == ["INFO", "DEBUG", "ERROR"]
    assert entries[0]["logger"] == "rag.pipeline" and entries[0]["time"] == "2026-10-19 01:00:00"
    assert len(entries[1]["message"]) < 700
    assert "Traceback" in entries[2]["message"]


def test_missing_logs_are_empty(client):
    assert client.get("/api/logs/pipeline").json()["entries"] == []
    assert client.get("/api/logs/ingest").json() == []


def test_ingest_log_newest_first(client, tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    rows = [{"file": "a.pdf", "chunks": 1}, {"file": "b.pdf", "chunks": 2}]
    (logs / "ingest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    assert [r["file"] for r in client.get("/api/logs/ingest").json()] == ["b.pdf", "a.pdf"]


def make_run(results_dir, name, created, recall, version=None):
    run = results_dir / name
    run.mkdir(parents=True)
    config = {"top_k": 5, **({"version": version} if version else {})}
    (run / "config.json").write_text(json.dumps({"name": name, "created": created, "note": "n", "k": 5,
                                                 "config": config, "judge": "j"}), encoding="utf-8")
    (run / "summary.json").write_text(json.dumps({"overall": {"recall": recall, "n": 1}, "by_type": {}}), encoding="utf-8")
    (run / "results.jsonl").write_text(json.dumps({"id": "q1", "type": "factual"}) + "\n", encoding="utf-8")
    (run / "report.md").write_text("# report", encoding="utf-8")


def test_traces_and_reports_filter_by_version(client, tmp_path, monkeypatch):
    def trace(trace_id, version):
        config = {"llm": "m", **({"version": version} if version else {})}
        return {"trace_id": trace_id, "timestamp": "2026-10-19T10:00:00", "query": f"q {trace_id}", "config": config,
                "spans": {}, "answer": "a", "total_ms": 1, "error": None}

    api_main.TRACES_PATH.write_text(
        "\n".join(json.dumps(t) for t in [trace("00000001", "v1"), trace("00000002", "v2"),
                                          trace("00000003", None), trace("00000010", "v10")]) + "\n", encoding="utf-8")
    ids = lambda params="": [t["trace_id"] for t in client.get(f"/api/traces{params}").json()]   # noqa: E731

    assert ids() == ["00000010", "00000003", "00000002", "00000001"]
    assert ids("?version=v1") == ["00000003", "00000001"]       # a trace with no version counts as v1
    assert ids("?version=v2") == ["00000002"] and ids("?version=v9") == []
    assert client.get("/api/traces/00000003").json()["version"] == "v1"
    assert client.get("/api/traces/00000010").json()["version"] == "v10"
    assert client.get("/api/traces?version=bad%20v").status_code == 422

    results = tmp_path / "results"
    make_run(results, "run_a", "2026-10-18 10:00", 0.5, "v1")
    make_run(results, "run_b", "2026-10-19 10:00", 0.9, "v2")
    make_run(results, "run_legacy", "2026-10-17 10:00", 0.4)
    names = lambda params="": [r["name"] for r in client.get(f"/api/reports{params}").json()]   # noqa: E731
    assert names("?version=v2") == ["run_b"]
    assert names("?version=v1") == ["run_a", "run_legacy"]
    assert client.get("/api/reports/run_legacy").json()["version"] == "v1"

    monkeypatch.setattr(api_main, "PIPELINE_VERSION", "v3")   # the current version is listed even with no data yet
    assert client.get("/api/versions").json() == {"versions": ["v1", "v2", "v3", "v10"], "current": "v3"}


def test_versions_default_to_current_when_empty(client):
    assert client.get("/api/versions").json() == {"versions": [api_main.PIPELINE_VERSION], "current": api_main.PIPELINE_VERSION}


def test_reports_list_and_detail(client, tmp_path):
    results = tmp_path / "results"
    make_run(results, "older", "2026-10-18 10:00", 0.5)
    make_run(results, "newer", "2026-10-19 10:00", 0.9)
    (results / "incomplete").mkdir()
    assert [r["name"] for r in client.get("/api/reports").json()] == ["newer", "older"]
    detail = client.get("/api/reports/newer").json()
    assert detail["overall"]["recall"] == 0.9 and detail["results"][0]["id"] == "q1" and detail["markdown"] == "# report"
    assert client.get("/api/reports/incomplete").status_code == 404
    assert client.get("/api/reports/nope").status_code == 404
    assert client.get("/api/reports/bad name").status_code == 422
    assert client.get("/api/reports/.hidden").status_code == 422


class StreamingLLM(FakeLLM):
    """Writes its answer in pieces, like a real streamed reply."""

    def stream(self, prompt, json_mode, on_text):
        pieces = ["Insufficient ", "leave balance, ", "status 409."]
        for piece in pieces:
            on_text(piece)
        return {"text": "".join(pieces), "input_tokens": 100, "output_tokens": 8, "ms": 3, "ttft_ms": 1, "temperature": 0}


def sse_events(response) -> list[dict]:
    return [json.loads(line[len("data: "):]) for line in response.text.splitlines() if line.startswith("data: ")]


def test_chat_stream_sends_stages_tokens_and_ttft(client, monkeypatch):
    monkeypatch.setattr(api_main, "get_llm", lambda: StreamingLLM())
    response = client.post("/api/chat/stream", json={"question": "What does QL-1003 mean?", "k": 3})
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/event-stream")
    events = sse_events(response)
    stages = [e["stage"] for e in events if e["type"] == "stage"]
    assert stages[0] == "queued" and "retrieve" in stages and "generate" in stages
    tokens = [e["text"] for e in events if e["type"] == "token"]
    assert len(tokens) == 3 and "".join(tokens) == "Insufficient leave balance, status 409."
    done = events[-1]
    assert done["type"] == "done" and done["result"]["answer"] == "".join(tokens)
    assert done["result"]["ttft_ms"] is not None and done["result"]["model_ttft_ms"] == 1
    assert done["result"]["ttft_ms"] <= done["result"]["total_ms"]


def test_chat_stream_works_with_a_model_that_cannot_stream(client):
    events = sse_events(client.post("/api/chat/stream", json={"question": "What does QL-1003 mean?", "k": 3}))
    assert [e["text"] for e in events if e["type"] == "token"] == ["Insufficient leave balance, status 409."]
    assert events[-1]["type"] == "done" and events[-1]["result"]["ttft_ms"] is not None


def test_chat_stream_reports_an_error(client, monkeypatch):
    class Broken(FakeLLM):
        def complete(self, prompt, json_mode=False):
            raise RuntimeError("model down")
    monkeypatch.setattr(api_main, "get_llm", lambda: Broken())
    events = sse_events(client.post("/api/chat/stream", json={"question": "What does QL-1003 mean?", "k": 3}))
    assert events[-1]["type"] == "error" and "model down" in events[-1]["error"]
