"""Phase 6 step 4: latency breakdown and reranker cost experiments (the pure parts)."""
import time

from eval.latency_breakdown import Timers, render_report as latency_report, sample_questions, summarize
from eval.rerank_cost_experiments import evaluate, rank, render_report as cost_report


# ---------------------------------------------------------------- latency breakdown

def test_timers_add_up_the_time_per_stage_and_count_the_calls():
    timers = Timers()
    slow = timers.wrap("rerank", lambda x: (time.sleep(0.02), x * 2)[1])
    fast = timers.wrap("bm25", lambda: "hits")
    assert slow(2) == 4 and slow(3) == 6 and fast() == "hits"
    taken = timers.reset()
    assert taken["calls"] == {"rerank": 2, "bm25": 1} and taken["ms"]["rerank"] >= 40
    assert timers.reset() == {"ms": {}, "calls": {}}                    # reset starts the next query from zero


def test_a_failing_stage_is_still_timed_and_the_error_is_not_hidden():
    timers = Timers()

    def boom():
        raise ValueError("x")

    try:
        timers.wrap("embed query", boom)()
    except ValueError:
        pass
    assert timers.reset()["calls"] == {"embed query": 1}


ROWS = [{"id": "a", "type": "factual", "total_ms": 11000, "ms": {"rerank": 10000, "embed query": 400}, "calls": {"rerank": 1, "embed query": 1}},
        {"id": "b", "type": "factual", "total_ms": 9000, "ms": {"rerank": 8000, "embed query": 600}, "calls": {"rerank": 1, "embed query": 1}},
        {"id": "c", "type": "cross_doc", "total_ms": 12000, "ms": {"rerank": 7000, "query plan (model call)": 2600},
         "calls": {"rerank": 2, "query plan (model call)": 1}}]


def test_the_summary_is_a_median_per_stage_and_counts_a_stage_a_query_never_used_as_zero():
    s = summarize(ROWS[:2])
    assert s["rerank"]["median_ms"] == 9000 and s["embed query"]["calls_per_query"] == 1 and s["rerank"]["queries"] == 2
    mixed = summarize(ROWS)
    assert mixed["query plan (model call)"]["queries"] == 1 and mixed["query plan (model call)"]["median_ms"] == 0   # median over 3 queries


def test_the_sample_takes_every_cross_document_question_and_every_second_other_answerable_one():
    golden = [{"id": f"q{i}", "type": "factual"} for i in range(10)] + [{"id": "u", "type": "unanswerable"}, {"id": "x", "type": "cross_doc"}]
    ids = [q["id"] for q in sample_questions(golden)]
    assert "u" not in ids and "x" in ids and ids.count("q0") == 1 and "q1" not in ids and "q2" in ids


def test_the_latency_report_names_the_stage_that_takes_the_time():
    report = latency_report(ROWS, [{"pool": 10, "max_length": 512, "median_ms": 9700.0}])
    assert "| rerank | 9000 |" in report and "Cross-document questions" in report and "| 10 | 512 | 9700 | 970 |" in report


# ---------------------------------------------------------------- reranker cost

RECORD = {"id": "q1", "chunks": ["a", "b", "c", "d"], "scores": {"512": [0.1, 0.9, 0.2, 0.95], "256": [0.5, 0.4, 0.3, 0.2]},
          "ms": {"512": 4000.0, "256": 2000.0}}


def test_a_smaller_pool_only_reranks_its_first_candidates_and_the_length_changes_the_scores():
    assert rank(RECORD, 4, 512) == ["d", "b", "c", "a"]
    assert rank(RECORD, 2, 512) == ["b", "a"]                           # candidate d is never read, so it cannot be promoted
    assert rank(RECORD, 4, 256) == ["a", "b", "c", "d"]


def test_a_setting_is_measured_on_the_questions_with_labels_and_its_time_scales_with_the_pool():
    labels = {"q1": {"ids": ["d"]}, "q2": {"ids": []}}
    records = [RECORD, {**RECORD, "id": "q2"}]
    full = evaluate(records, labels, pool=4, k=1, length=512)
    assert full["n"] == 1 and full["recall"] == 1.0 and full["mrr"] == 1.0 and full["rerank_s"] == 4.0   # 1 s per chunk
    small = evaluate(records, labels, pool=2, k=1, length=512)
    assert small["recall"] == 0.0 and small["rerank_s"] == 2.0


def test_the_cost_report_lists_every_setting_with_its_time():
    report = cost_report([RECORD], {"q1": {"ids": ["d"]}})
    assert "Rerank s per query" in report and "| 3 | 3 | 256 |" in report and "Baseline" in report
