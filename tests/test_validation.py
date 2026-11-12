"""Phase 6 validation layer: the CRAG-style relevance grade, the corrective search and the experiment helpers."""
import json

import pytest

from eval.validation_experiments import render_report, retry_summary, simulate
from src.generate.answer import NOT_FOUND
from src.observability.tracer import load_trace
from src.pipeline import run_query
from src.validate.gate import is_new_query, merge_chunks, validate_with_retry
from src.validate.threshold import best_score, grade_context

HIGH, LOW = 0.1, 0.0005
ANSWER = '{"answer": "Insufficient leave balance [S1].", "sources": ["S1"], "found": true}'


def chunk(score, cid="first_c1"):
    return {"chunk_id": cid, "doc_id": "spec", "page": 1, "text": f"text of {cid}", "score": score}


class RoutedLLM:
    """The rewrite prompt gets `query`, any other prompt (the answer) gets `answer`."""
    name = "routed"

    def __init__(self, query="a rewritten question", answer=ANSWER):
        self.query, self.answer, self.kinds = query, answer, []

    def complete(self, prompt, json_mode=False):
        kind = "rewrite" if "Rewrite the question" in prompt else "answer"
        self.kinds.append(kind)
        return {"text": self.query if kind == "rewrite" else self.answer, "input_tokens": 100, "output_tokens": 10, "ms": 5,
                "temperature": 0}


def retry_search(score, cid="second_c1"):
    """The corrective search: records its queries and returns one chunk."""
    queries = []

    def search(query, k):
        queries.append(query)
        return [chunk(score, cid)][:k]

    search.queries = queries
    return search


# ---------------------------------------------------------------- grade

def test_the_grade_comes_from_the_best_score_and_the_two_cutoffs():
    assert grade_context([chunk(0.9), chunk(0.0)], HIGH, LOW) == "correct"
    assert grade_context([chunk(0.1)], HIGH, LOW) == "correct"                      # the cutoff itself is correct
    assert grade_context([chunk(0.0), chunk(0.03)], HIGH, LOW) == "ambiguous"       # the best chunk counts, not the first
    assert grade_context([chunk(0.0004)], HIGH, LOW) == "incorrect"
    assert grade_context([], HIGH, LOW) == "incorrect" and best_score([]) is None
    with pytest.raises(ValueError):
        grade_context([chunk(0.5)], 0.1, 0.5)


# ---------------------------------------------------------------- corrective search

def test_chunks_from_two_searches_are_merged_by_score_and_a_duplicate_keeps_its_better_score():
    merged = merge_chunks([chunk(0.2, "a"), chunk(0.05, "b")], [chunk(0.9, "c"), chunk(0.3, "a")], 3)
    assert [(c["chunk_id"], c["score"]) for c in merged] == [("c", 0.9), ("a", 0.3), ("b", 0.05)]
    assert len(merge_chunks([chunk(0.2, "a")], [chunk(0.1, "b")], 1)) == 1
    assert is_new_query("A  New wording", "a new wording") is False and is_new_query("different", "a new wording") is True


def test_a_correct_grade_is_answered_as_it_is_without_any_search_or_rewrite():
    llm, search = RoutedLLM(), retry_search(0.9)
    verdict, chunks = validate_with_retry("q?", [chunk(0.9)], llm, search, 3, HIGH, LOW, retries=1)
    assert verdict["action"] == "answer" and verdict["grade"] == "correct" and verdict["retries"] == []
    assert llm.kinds == [] and search.queries == [] and [c["chunk_id"] for c in chunks] == ["first_c1"]


def test_an_ambiguous_grade_searches_again_and_answers_from_the_merged_chunks():
    llm, search = RoutedLLM(query="the better query"), retry_search(0.8)
    verdict, chunks = validate_with_retry("q?", [chunk(0.02)], llm, search, 3, HIGH, LOW, retries=1)
    assert llm.kinds == ["rewrite"] and search.queries == ["the better query"]
    assert verdict["first_grade"] == "ambiguous" and verdict["grade"] == "correct" and verdict["action"] == "answer"
    assert [c["chunk_id"] for c in chunks] == ["second_c1", "first_c1"]               # the best first, the first search kept
    attempt = verdict["retries"][0]
    assert attempt["grade_before"] == "ambiguous" and attempt["grade_after"] == "correct" and attempt["best_retry_score"] == 0.8


def test_an_ambiguous_grade_that_stays_ambiguous_is_still_answered():
    verdict, _ = validate_with_retry("q?", [chunk(0.02)], RoutedLLM(), retry_search(0.03), 3, HIGH, LOW, retries=1)
    assert verdict["grade"] == "ambiguous" and verdict["action"] == "answer"             # the grounding prompt decides, not the score


def test_an_incorrect_grade_is_blocked_only_when_the_search_does_not_help():
    verdict, chunks = validate_with_retry("q?", [chunk(0.0001)], RoutedLLM(), retry_search(0.0002), 3, HIGH, LOW, retries=1)
    assert verdict["action"] == "abstain" and verdict["first_grade"] == "incorrect" and "also after rewriting" in verdict["reason"]
    rescued, _ = validate_with_retry("q?", [chunk(0.0001)], RoutedLLM(), retry_search(0.4), 3, HIGH, LOW, retries=1)
    assert rescued["action"] == "answer" and rescued["first_grade"] == "incorrect" and rescued["grade"] == "correct"
    no_retry, _ = validate_with_retry("q?", [chunk(0.0001)], RoutedLLM(), retry_search(0.4), 3, HIGH, LOW, retries=0)
    assert no_retry["action"] == "abstain" and no_retry["retries"] == []


def test_a_rewrite_that_repeats_the_question_is_skipped_without_a_search_and_there_is_no_search_without_a_function():
    search = retry_search(0.9)
    verdict, _ = validate_with_retry("What is it?", [chunk(0.02)], RoutedLLM(query="what is  it?"), search, 3, HIGH, LOW, retries=1)
    assert search.queries == [] and "skipped" in verdict["retries"][0] and verdict["action"] == "answer"
    verdict, _ = validate_with_retry("q?", [chunk(0.0001)], RoutedLLM(), None, 3, HIGH, LOW, retries=1)
    assert verdict["retries"] == [] and verdict["action"] == "abstain"


# ---------------------------------------------------------------- pipeline

def first_then_retry(first_score, second_score):
    """The pipeline's search: the first call is the normal retrieval, the next ones are corrective searches."""
    calls = []

    def search(query, k):
        calls.append(query)
        return [chunk(first_score, "first_c1")][:k] if len(calls) == 1 else [chunk(second_score, "second_c1")][:k]

    search.calls = calls
    return search


def test_a_blocked_query_never_reaches_the_answer_model_and_is_traced(tmp_path):
    llm = RoutedLLM()
    out = run_query("What is the salary?", k=1, llm=llm, version="v2", trace_path=tmp_path / "t.jsonl",
                    search_fn=first_then_retry(0.0001, 0.0002), validate=True, retries=1)
    assert llm.kinds == ["rewrite"]                                                      # no answer prompt was sent
    assert out["answer"] == NOT_FOUND and out["found"] is False and out["sources"] == [] and out["answer_status"] == "ok"
    assert out["validation"]["action"] == "abstain"
    spans = load_trace(out["trace_id"], tmp_path / "t.jsonl")["spans"]
    assert "validate" in spans and "llm_call" not in spans and "build_prompt" not in spans


def test_a_weak_retrieval_is_corrected_and_the_answer_uses_the_merged_chunks(tmp_path):
    llm = RoutedLLM(query="the better query")
    out = run_query("What is QL-1003?", k=2, llm=llm, version="v2", trace_path=tmp_path / "t.jsonl",
                    search_fn=first_then_retry(0.02, 0.6), validate=True, retries=1)
    assert llm.kinds == ["rewrite", "answer"] and out["answer"] == "Insufficient leave balance [S1]."
    assert [c["chunk_id"] for c in out["chunks"]] == ["second_c1", "first_c1"]
    trace = load_trace(out["trace_id"], tmp_path / "t.jsonl")
    prompt = trace["spans"]["build_prompt"]["final_prompt"]
    assert "text of second_c1" in prompt and "text of first_c1" in prompt
    assert trace["spans"]["validate"]["retries"][0]["query"] == "the better query"
    assert trace["spans"]["retrieve"]["results"][0]["chunk_id"] == "first_c1"            # the first retrieval stays in the trace


def test_a_correct_query_is_answered_directly_and_validation_is_off_by_default(tmp_path):
    llm = RoutedLLM()
    out = run_query("What is QL-1003?", k=1, llm=llm, version="v2", trace_path=tmp_path / "t.jsonl",
                    search_fn=first_then_retry(0.9, 0.9), validate=True, retries=1)
    assert llm.kinds == ["answer"] and out["validation"]["grade"] == "correct"
    off = run_query("What is QL-1003?", k=1, llm=RoutedLLM(), version="v2", trace_path=tmp_path / "t.jsonl",
                    search_fn=first_then_retry(0.0, 0.0), validate=False)
    assert "validation" not in off and off["answer"] == "Insufficient leave balance [S1]."          # a zero score is not blocked


# ---------------------------------------------------------------- experiment helpers

def record(qid, answerable, top, retry=None, v3_correctness=1.0):
    r = {"id": qid, "question": qid, "answerable": answerable, "top": top, "chunks": [], "v3_correctness": v3_correctness,
         "first_had_relevant": True if answerable else None}
    if retry is not None:
        r["retry"] = retry
    return r


def again(top, has_relevant=True):
    return {"query": "x", "rewrite_ms": 1000, "search_ms": 9000, "top": top, "chunks": [], "has_relevant": has_relevant}


RECORDS = [
    record("a_high", True, 0.9),
    record("a_tiny", True, 0.0001, again(0.4)),                  # answerable, scored near zero: the search rescues it
    record("a_stays", True, 0.0001, again(0.0002)),              # still incorrect: a false abstention
    record("u_zero", False, 0.0, again(0.0)),
    record("u_lifted", False, 0.0001, again(0.7)),               # the rewrite raises an unanswerable question's score
    record("a_mid", True, 0.03, again(0.02)),
]


def test_simulate_counts_what_the_cutoffs_block_with_and_without_the_corrective_search():
    off = simulate(RECORDS, HIGH, LOW, retry=False)
    assert off["retried"] == 0 and off["answerable_blocked"] == 2 and off["false_abstentions"] == 2 and off["unanswerable_blocked"] == 2
    on = simulate(RECORDS, HIGH, LOW, retry=True)
    assert on["retried"] == 5 and on["rescued"] == 2                                        # a_tiny and u_lifted
    assert on["answerable_blocked"] == 1 and on["false_abstentions"] == 1 and on["unanswerable_blocked"] == 1
    assert on["retry_s"] == 50.0                                                           # 5 searches of 10 s


def test_a_skipped_rewrite_adds_no_search_and_the_false_abstention_count_needs_a_correct_saved_answer():
    records = [record("a", True, 0.0001, {"query": "q", "skipped": True, "rewrite_ms": 800}, v3_correctness=0.5),
               record("b", True, 0.0001, {"query": "q", "skipped": True, "rewrite_ms": 800}, v3_correctness=1.0)]
    s = simulate(records, HIGH, LOW)
    assert s["retried"] == 2 and s["answerable_blocked"] == 2 and s["false_abstentions"] == 1 and s["rescued"] == 0


def test_the_summary_and_the_report_describe_what_the_search_did():
    missed = record("a_missed", True, 0.05, again(0.6, has_relevant=True))
    missed["first_had_relevant"] = False
    s = retry_summary(RECORDS + [missed])
    assert s["first_search_missed_evidence"] == ["a_missed"] and s["found_missing_evidence"] == ["a_missed"] and s["lost_evidence"] == []
    report = render_report(RECORDS + [missed])
    assert "## What each pair of cutoffs would have done" in report and "False abstention" in report
    assert "a_missed" in report


class _GradingLLM:
    name = "fake-grader"

    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def complete(self, prompt, json_mode=False):
        self.prompts.append(prompt)
        return {"text": self.reply, "input_tokens": 10, "output_tokens": 5, "ms": 1, "temperature": 0}


def test_refine_keeps_the_best_chunk_and_drops_only_explicit_drops():
    from src.validate.refine import refine_chunks
    info = {"reasons": ["cross_document: cv, spec"]}
    chunks = [{"chunk_id": "best", "doc_id": "spec", "score": 0.9, "text": "BEST", "transform": info},
              {"chunk_id": "cv1", "doc_id": "cv", "score": 0.01, "text": "CV EVIDENCE", "transform": info},
              {"chunk_id": "noise", "doc_id": "spec", "score": 0.3, "text": "NOISE", "transform": info},
              {"chunk_id": "unsure", "doc_id": "spec", "score": 0.2, "text": "UNSURE", "transform": info}]
    llm = _GradingLLM(json.dumps({"passages": {"1": "drop"}}))       # passage 2 has no verdict: kept
    kept, record = refine_chunks("q?", chunks, llm)
    assert [c["chunk_id"] for c in kept] == ["best", "cv1", "unsure"] and record["dropped"] == ["noise"]
    assert "BEST" not in llm.prompts[0] and "CV EVIDENCE" not in llm.prompts[0]    # the always-kept chunks are not graded
    kept, record = refine_chunks("q?", chunks, _GradingLLM("not json"))            # a broken reply keeps every chunk
    assert kept == chunks and record["error"]
    single = chunks[:1]
    assert refine_chunks("q?", single, _GradingLLM("unused"))[0] == single           # nothing to grade: no call needed


def test_refine_with_context_shows_the_kept_passage_to_the_grader():
    from src.validate.refine import refine_chunks
    chunks = [{"chunk_id": "codes", "doc_id": "spec", "score": 0.3, "text": "QL-1006 | 422 | start date more than 90 days in the past"},
              {"chunk_id": "history", "doc_id": "spec", "score": 0.01, "text": "1.2 | Added error code QL-1006."}]
    alone, linked = _GradingLLM(json.dumps({"passages": {"1": "keep"}})), _GradingLLM(json.dumps({"passages": {"1": "keep"}}))
    refine_chunks("Which version added the code?", chunks, alone)
    refine_chunks("Which version added the code?", chunks, linked, with_context=True)
    assert "QL-1006 | 422" not in alone.prompts[0]                  # v10/v11: each passage graded on its own (q37)
    assert "Already kept" in linked.prompts[0] and "QL-1006 | 422" in linked.prompts[0]
    assert "1.2 | Added error code QL-1006." in linked.prompts[0].split("Passages to grade:")[1]   # still graded, not kept blindly


def test_the_stream_sends_the_crag_grade_before_the_answer(tmp_path):
    events = []
    out = run_query("What is QL-1003?", k=1, llm=RoutedLLM(), version="v2", trace_path=tmp_path / "t.jsonl",
                    search_fn=first_then_retry(0.9, 0.9), validate=True, retries=1, refine=False, on_event=events.append)
    kinds = [e["type"] for e in events]
    assert "validation" in kinds and kinds.index("validation") < kinds.index("token")      # the chat shows it first
    graded = next(e for e in events if e["type"] == "validation")["validation"]
    assert graded["grade"] == "correct" and graded["action"] == "answer" and graded == out["validation"]
    # a blocked query: graded incorrect, and the "not found" text is the only token
    events = []
    run_query("What is the salary?", k=1, llm=RoutedLLM(), version="v2", trace_path=tmp_path / "t.jsonl",
              search_fn=first_then_retry(0.0001, 0.0002), validate=True, retries=1, refine=False, on_event=events.append)
    graded = next(e for e in events if e["type"] == "validation")["validation"]
    assert graded["grade"] == "incorrect" and graded["action"] == "abstain"
    assert [e["text"] for e in events if e["type"] == "token"] == [NOT_FOUND]


def test_every_retrieved_chunk_gets_a_crag_grade(tmp_path):
    from src.validate.threshold import grade_chunks, grade_score
    assert [grade_score(s, 0.1, 0.0005) for s in (0.9, 0.1, 0.05, 0.0005, 0.0001, None)] == \
        ["correct", "correct", "ambiguous", "ambiguous", "incorrect", "incorrect"]
    graded = grade_chunks([{"chunk_id": "a", "doc_id": "d", "page": 1, "score": 0.5}], 0.1, 0.0005)
    assert graded == [{"chunk_id": "a", "doc_id": "d", "page": 1, "score": 0.5, "grade": "correct"}]
    events = []
    out = run_query("What is QL-1003?", k=1, llm=RoutedLLM(), version="v2", trace_path=tmp_path / "t.jsonl",
                    search_fn=first_then_retry(0.9, 0.9), validate=True, retries=1, refine=False, on_event=events.append)
    assert out["context"][0]["grade"] == "correct" and out["context"][0]["used"] is True
    assert next(e for e in events if e["type"] == "context")["context"] == out["context"]
    assert load_trace(out["trace_id"], tmp_path / "t.jsonl")["context_grades"] == out["context"]
    blocked = run_query("What is the salary?", k=1, llm=RoutedLLM(), version="v2", trace_path=tmp_path / "t.jsonl",
                        search_fn=first_then_retry(0.0001, 0.0002), validate=True, retries=1, refine=False)
    assert {c["grade"] for c in blocked["context"]} == {"incorrect"} and not any(c["used"] for c in blocked["context"])
    plain = run_query("What is QL-1003?", k=1, llm=RoutedLLM(), version="v2", trace_path=tmp_path / "t.jsonl",
                      search_fn=first_then_retry(0.9, 0.9), validate=False)
    assert "context" not in plain                                                   # no grades without validation
