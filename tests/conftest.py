"""Keep the tests away from the real logs/ folder, and on the v1 pipeline (bge index, no network calls), with one judge call per answer."""
import os
import tempfile

os.environ.setdefault("RAG_LOGS_DIR", tempfile.mkdtemp(prefix="rag_test_logs_"))
os.environ.setdefault("RAG_PIPELINE_VERSION", "v1")
os.environ.setdefault("RAG_JUDGE_VOTES", "1")      # the fake judges reply once per answer
os.environ.setdefault("RAG_JUDGE_WORKERS", "1")    # the fake judges reply in a fixed order
os.environ.setdefault("RAG_JUDGE_PROMPT", "v2")    # one judge call per vote; the split v3 judge has its own test
