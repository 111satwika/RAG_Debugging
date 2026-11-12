"""Phase 2 tests. Fake LLMs keep them fast; the end-to-end test uses the real Chroma index."""
import json
import math

import pytest

from eval.failures import classify
from eval.judge import combine_votes, judge_answer, parse_verdict
from eval.keyword_check import keyword_check as strict_keyword_check, matches as strict_matches
from eval.metrics import (contains, coverage, keyword_check, label_relevant, mrr, ndcg_at_k, precision_at_k,
                          recall_at_k, retrieval_metrics)
from eval.run_eval import is_failure, judge_vs_keyword, percentile, run_evaluation


def q(**kw):
    base = {"id": "qx", "question": "?", "expected_answer": "x", "source_docs": ["doc_a.pdf"],
            "source_section": "s", "type": "factual", "must_include": ["alpha"]}
    return {**base, **kw}


# ---------------------------------------------------------------- metrics

def test_perfect_ranking():
    ranked, rel = ["a", "b", "c"], {"a"}
    assert mrr(ranked, rel, 3) == 1 and ndcg_at_k(ranked, rel, 3) == 1
    assert recall_at_k(ranked, rel, 3) == 1 and precision_at_k(ranked, rel, 3) == pytest.approx(1 / 3)


def test_lower_rank_scores_lower():
    rel = {"c"}
    assert mrr(["a", "b", "c"], rel, 3) == pytest.approx(1 / 3)
    assert ndcg_at_k(["a", "b", "c"], rel, 3) == pytest.approx(1 / math.log2(4))
    assert mrr(["a", "b", "c"], rel, 2) == 0 and recall_at_k(["a", "b", "c"], rel, 2) == 0


def test_ndcg_with_several_relevant():
    ranked, rel = ["x", "a", "b"], {"a", "b"}
    expected = (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(ranked, rel, 3) == pytest.approx(expected)
    assert retrieval_metrics(ranked, set(), 3) is None


# ---------------------------------------------------------------- matching and labels

@pytest.mark.parametrize("text,needle,expected", [
    ("it has 16 heads", "6", False),
    ("N = 6 layers", "6", True),
    ("BLEU of 2.6", "6", False),
    ("O(n^2 \u00b7 d) per layer", "O(n\u00b2\u00b7d)", True),
    ("beta2 = 0.98 and", "0.98", True),
    ("limit is 12 days.", "12", True),
    ("Insufficient LEAVE balance", "insufficient leave balance", True),
])
def test_contains_whole_tokens(text, needle, expected):
    assert contains(text, needle) is expected


def test_coverage_and_keyword_check():
    assert coverage("alpha and beta", ["alpha", "beta", "gamma"]) == pytest.approx(2 / 3)
    check = keyword_check("the value is 10 days", q(must_include=["12"], must_not_include=["10 days"]))
    assert check["missing"] == ["12"] and check["forbidden"] == ["10 days"]
    assert keyword_check("anything", q(type="unanswerable")) is None


CHUNKS = [
    {"chunk_id": "a_c0", "doc_id": "doc_a", "text": "alpha beta"},
    {"chunk_id": "a_c1", "doc_id": "doc_a", "text": "alpha only"},
    {"chunk_id": "a_c2", "doc_id": "doc_a", "text": "nothing here"},
    {"chunk_id": "b_c0", "doc_id": "doc_b", "text": "alpha beta"},
]


def test_label_relevant_single_fact_keeps_only_best_chunks():
    labels = label_relevant(q(must_include=["alpha", "beta"]), CHUNKS)
    assert labels["ids"] == ["a_c0"] and not labels["weak"]      # not a_c1 (half), not doc_b


def test_label_relevant_multi_section_keeps_chunks_above_threshold():
    labels = label_relevant(q(type="multi_section", must_include=["alpha", "beta"]), CHUNKS)
    assert labels["ids"] == ["a_c0", "a_c1"]


def test_label_relevant_weak_and_unanswerable():
    weak = label_relevant(q(type="multi_section", must_include=["alpha", "x1", "x2", "x3", "x4"]), CHUNKS)
    assert weak["weak"] and weak["ids"] == ["a_c0", "a_c1"]
    assert label_relevant(q(type="unanswerable"), CHUNKS)["ids"] == []


# ---------------------------------------------------------------- judge

class FakeJudge:
    name = "fake-judge"

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), 0

    def complete(self, prompt, json_mode=False):
        self.calls += 1
        return {"text": self.replies.pop(0), "input_tokens": 10, "output_tokens": 5, "ms": 1, "temperature": 0}


GOOD = json.dumps({"correctness": 0.9, "relevance": 1, "completeness": 0.4, "faithfulness": 1,
                   "abstained": False, "unsupported_claims": [], "reasoning": "ok"})


def test_parse_verdict_snaps_scores_to_allowed_values():
    verdict = parse_verdict(GOOD, q())
    assert verdict["correctness"] == 1.0 and verdict["completeness"] == 0.5 and verdict["abstained"] is False


def test_unanswerable_correctness_follows_abstention():
    refused = json.dumps({"correctness": 0, "relevance": 1, "completeness": 1, "faithfulness": 1, "abstained": True})
    verdict = parse_verdict(refused, q(type="unanswerable"))
    assert verdict["correctness"] == 1.0 and verdict["completeness"] is None


def test_correct_refusal_is_relevant_and_faithful():
    refused = json.dumps({"correctness": 1, "relevance": 0, "completeness": 1, "faithfulness": 0, "abstained": True,
                          "unsupported_claims": []})
    verdict = parse_verdict(refused, q(type="unanswerable"))
    assert verdict["relevance"] == verdict["faithfulness"] == 1.0
    padded = json.dumps({"correctness": 1, "relevance": 0, "completeness": 1, "faithfulness": 0, "abstained": True,
                         "unsupported_claims": ["made-up number"]})
    assert parse_verdict(padded, q(type="unanswerable"))["faithfulness"] == 0.0     # a refusal with a made-up claim is not
    assert parse_verdict(refused, q())["relevance"] == 0.0                          # an answerable question keeps the score


def test_judge_votes_take_the_median():
    reply = lambda c: json.dumps({"correctness": c, "relevance": 1, "completeness": c, "faithfulness": 1, "abstained": False})
    judge = FakeJudge(reply(0.5), reply(1), reply(1))
    verdict = judge_answer(judge, q(), "answer", [], votes=3)
    assert judge.calls == 3 and verdict["correctness"] == verdict["completeness"] == 1.0
    assert len(verdict["votes"]) == 3 and verdict["judge_tokens"] == {"in": 30, "out": 15}
    assert combine_votes([{"x": 1}]) == {"x": 1}


def test_judge_retries_then_fails_on_bad_json():
    judge = FakeJudge("not json", GOOD)
    verdict = judge_answer(judge, q(), "answer", [{"chunk_id": "c", "text": "t"}])
    assert judge.calls == 2 and verdict["judge_model"] == "fake-judge"
    with pytest.raises(ValueError):
        judge_answer(FakeJudge("bad", "worse"), q(), "answer", [])


# ---------------------------------------------------------------- failure stages

def test_judge_grades_against_the_required_answer_and_shows_optional_details_separately():
    class Capture(FakeJudge):
        def complete(self, prompt, json_mode=False):
            self.prompt = prompt
            return super().complete(prompt, json_mode)

    split = q(expected_answer="FULL ANSWER WITH EXTRAS", required_answer="ONLY THE ASKED PART", optional_details="THE EXTRAS")
    judge = Capture(GOOD)
    judge_answer(judge, split, "a", [])
    assert "Reference answer: ONLY THE ASKED PART" in judge.prompt and "FULL ANSWER WITH EXTRAS" not in judge.prompt
    assert "THE EXTRAS" in judge.prompt and "do not lower any score" in judge.prompt

    plain = Capture(GOOD)
    judge_answer(plain, q(expected_answer="WHOLE"), "a", [])
    assert "Reference answer: WHOLE\n\nContext" in plain.prompt and "Optional details" not in plain.prompt   # the v1 text


def test_rescore_keeps_old_verdicts_for_unsplit_questions_and_rejudges_split_ones(tmp_path):
    from eval.rescore import collection_for, rescore, saved_search, verdict_changes

    def verdict(completeness):
        return {"correctness": 1.0, "relevance": 1.0, "completeness": completeness, "faithfulness": 1.0, "abstained": False,
                "unsupported_claims": [], "reasoning": "", "judge_model": "m", "judge_prompt": "v1"}

    questions = [q(id="a", question="qa", required_answer="r", optional_details="o", expected_answer="r and o"),
                 q(id="b", question="qb", expected_answer="b")]
    (tmp_path / "golden.json").write_text(json.dumps(questions), encoding="utf-8")
    src = tmp_path / "old"
    src.mkdir()
    retrieved = [{"chunk_id": "c1", "doc_id": "doc_a", "page": 1, "score": 0.5}]
    (src / "config.json").write_text(json.dumps({"name": "old", "note": "", "k": 5, "created": "x", "judge": "j",
                                                  "config": {"chunk_strategy": "structure"}}), encoding="utf-8")
    old_rows = [{"id": i, "question": f"q{i}", "retrieved": retrieved, "judge": verdict(0.5)} for i in "ab"]
    (src / "results.jsonl").write_text("\n".join(json.dumps(r) for r in old_rows), encoding="utf-8")
    answers = [{"id": i, "answer": "ans", "error": None, "chunks": [{"chunk_id": "c1", "text": "alpha"}]} for i in "ab"]
    (src / "answers.jsonl").write_text("\n".join(json.dumps(r) for r in answers), encoding="utf-8")
    (src / "judgments.jsonl").write_text("\n".join(json.dumps({"id": i, **verdict(0.5)}) for i in "ab"), encoding="utf-8")
    (src / "relevance_labels.json").write_text(json.dumps({i: {"ids": ["c1"], "weak": False} for i in "ab"}), encoding="utf-8")

    judge = FakeJudge(json.dumps({"correctness": 1, "relevance": 1, "completeness": 1, "faithfulness": 1,
                                  "abstained": False, "unsupported_claims": [], "reasoning": "ok"}))
    chunks = [{"chunk_id": "c1", "doc_id": "doc_a", "text": "alpha", "header": ""}]
    out = rescore("old", "new", judge_llm=judge, results_dir=tmp_path, golden_path=tmp_path / "golden.json", chunks=chunks)

    assert judge.calls == 1                                       # only the split question is judged again
    by_id = {r["id"]: r for r in out["results"]}
    assert by_id["a"]["judge"]["completeness"] == 1.0 and by_id["b"]["judge"]["completeness"] == 0.5
    assert out["changes"] == [{"id": "a", "completeness": (0.5, 1.0)}]
    assert (tmp_path / "new" / "report.md").exists() and (tmp_path / "old" / "judgments.jsonl").exists()
    assert json.loads((tmp_path / "new" / "config.json").read_text(encoding="utf-8"))["rescored_from"] == "old"
    assert collection_for({"chunk_strategy": "fixed"}) == "rag_chunks" and collection_for({"chunk_strategy": "structure"}) == "rag_chunks_v2"
    assert saved_search({"a": {"question": "qa", "retrieved": retrieved * 6}})("qa", 5) == retrieved * 5
    assert verdict_changes({}, by_id) == []                       # no old verdicts: nothing to compare


def result(retrieved, answer="a", abstained=False):
    return {"answer": answer, "retrieved": [{"chunk_id": c} for c in retrieved], "judge": {"abstained": abstained}}


def test_classify_stages():
    question = q(must_include=["alpha", "beta"])
    assert classify(question, result(["a_c0"]), CHUNKS, [])[0] == "generation"
    low_rank = classify(question, result(["a_c2"]), CHUNKS, ["a_c0"])
    not_found = classify(question, result(["a_c2"]), CHUNKS, ["a_c2"])
    assert low_rank[0] == not_found[0] == "retrieval"
    assert "ranked too low" in low_rank[1] and "not found" in not_found[1]
    assert classify(q(must_include=["alpha", "zzz"]), result(["a_c0"]), CHUNKS, [])[0] == "chunking"
    assert classify(q(source_docs=["ghost.pdf"]), result(["a_c0"]), CHUNKS, [])[0] == "indexing"
    assert classify(q(type="unanswerable"), result([]), CHUNKS, [])[0] == "generation"


def test_classify_negative_evidence_is_generation_not_retrieval():
    # "the resume does not mention X": the key phrase can never be in a chunk, so it says nothing about retrieval
    question = q(must_include=[["not mention", "no evidence"]])
    assert classify(question, result(["a_c2"], abstained=True), CHUNKS, ["a_c2"]) ==         ("generation", "the facts were retrieved but the model abstained")


def test_classify_data_quality_needs_stale_value_in_context_and_answer():
    stale_chunks = CHUNKS + [{"chunk_id": "a_c3", "doc_id": "doc_a", "text": "alpha old value 10 days"}]
    question = q(must_include=["alpha"], must_not_include=["10 days"])
    assert classify(question, result(["a_c3"], answer="it is 10 days"), stale_chunks, [])[0] == "data_quality"
    assert classify(question, result(["a_c0"], answer="it is 10 days"), stale_chunks, [])[0] != "data_quality"


def test_judge_vs_keyword_compares_against_correct_and_complete():
    def res(rid, include_rate, correctness, completeness):
        return {"id": rid, "keyword": {"include_rate": include_rate, "forbidden": []},
                "judge": {"correctness": correctness, "completeness": completeness}}

    agreement, disagree = judge_vs_keyword([
        res("agree_ok", 1.0, 1.0, 1.0),
        res("agree_incomplete", 0.5, 1.0, 0.5),     # correct but incomplete: both checks say "not perfect"
        res("keyword_false_negative", 0.5, 1.0, 1.0),
        res("judge_stricter", 1.0, 0.5, 0.5),
    ])
    assert disagree == ["keyword_false_negative", "judge_stricter"] and agreement == 0.5


def test_is_failure_rules():
    base = {"error": None, "type": "factual", "keyword": None,
            "judge": {"abstained": False, "correctness": 1, "faithfulness": 1, "completeness": 1}}
    assert not is_failure(base)
    assert is_failure({**base, "judge": {**base["judge"], "completeness": 0.5}})
    assert is_failure({**base, "judge": {**base["judge"], "abstained": True}})
    assert is_failure({**base, "type": "unanswerable"})            # answered an unanswerable question
    assert is_failure({**base, "judge": None, "keyword": {"include_rate": 0.5, "forbidden": []}})
    assert not is_failure({**base, "judge": None})                 # retrieval-only: nothing to judge
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10 and percentile([], 95) is None


# ---------------------------------------------------------------- end to end

class FakeLLM:
    name = "fake-llm"

    def complete(self, prompt, json_mode=False):
        return {"text": "Insufficient leave balance, status 409.", "input_tokens": 100, "output_tokens": 8,
                "ms": 2, "temperature": 0}


def test_full_run_with_fake_llms(tmp_path):
    verdict = json.dumps({"correctness": 1, "relevance": 1, "completeness": 1, "faithfulness": 1,
                          "abstained": False, "unsupported_claims": [], "reasoning": "ok"})
    judge = FakeJudge(*[verdict] * 5)
    out = run_evaluation("t", ids=["q11", "q12"], llm=FakeLLM(), judge_llm=judge, results_dir=tmp_path,
                         trace_path=tmp_path / "traces.jsonl")

    run_dir = tmp_path / "t"
    for name in ("answers.jsonl", "judgments.jsonl", "results.jsonl", "summary.json", "report.md",
                 "relevance_labels.json", "config.json"):
        assert (run_dir / name).exists(), name
    summary = out["summary"]
    assert summary["n"] == 2 and summary["n_judged"] == 2 and summary["recall"] == 1.0
    assert summary["correctness"] == 1.0 and summary["abstention_accuracy"] == 1.0
    assert "Recall@5" in out["report"] and "Failure report" in out["report"]
    assert (tmp_path / "experiment_log.md").read_text(encoding="utf-8").count("| t |") == 1

    # Resuming does not call the fake judge again and does not duplicate the log row.
    again = run_evaluation("t", ids=["q11", "q12"], llm=FakeLLM(), judge_llm=judge, results_dir=tmp_path,
                           trace_path=tmp_path / "traces.jsonl")
    assert judge.calls == 2 and again["summary"]["n_judged"] == 2
    assert (tmp_path / "experiment_log.md").read_text(encoding="utf-8").count("| t |") == 1
    assert len((run_dir / "answers.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_resuming_a_run_made_before_versions_were_recorded(tmp_path):
    verdict = json.dumps({"correctness": 1, "relevance": 1, "completeness": 1, "faithfulness": 1,
                          "abstained": False, "unsupported_claims": [], "reasoning": "ok"})
    kwargs = dict(ids=["q11"], llm=FakeLLM(), judge_llm=FakeJudge(*[verdict] * 3), results_dir=tmp_path,
                  trace_path=tmp_path / "traces.jsonl")
    run_evaluation("old", **kwargs)

    config_path = tmp_path / "old" / "config.json"
    meta = json.loads(config_path.read_text(encoding="utf-8"))
    del meta["config"]["version"]                                   # what a run from before versions looks like
    config_path.write_text(json.dumps(meta), encoding="utf-8")
    run_evaluation("old", **kwargs)                                 # still resumes: a missing version means v1

    meta["config"]["chunk_size"] = 123                              # a genuinely different setup is still refused
    config_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(RuntimeError):
        run_evaluation("old", **kwargs)


# ---------------------------------------------------------------- the golden set and tiers

def result_for(rid, tier, mrr_value, qtype="factual"):
    retrieval = {"recall": mrr_value, "precision": 0.2, "mrr": mrr_value, "ndcg": mrr_value, "hit": 1.0}
    return {"id": rid, "type": qtype, "tier": tier, "retrieval": retrieval, "judge": None, "keyword": None,
            "failure": None, "total_ms": None, "input_tokens": None, "output_tokens": None}


def test_summaries_split_standard_and_hard_questions():
    from eval.run_eval import summarize_by_tier

    results = [result_for("a", "standard", 1.0), result_for("b", "standard", 1.0), result_for("c", "hard", 0.5)]
    by_tier = summarize_by_tier(results, 5)
    assert list(by_tier) == ["standard", "hard"]
    assert by_tier["standard"]["n"] == 2 and by_tier["hard"]["mrr"] == 0.5
    assert summarize_by_tier([result_for("a", "standard", 1.0)], 5) == {}      # one tier: nothing to compare


def test_the_golden_set_is_consistent_and_every_key_fact_is_in_its_source_document():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    golden = json.loads((root / "eval" / "golden_set.json").read_text(encoding="utf-8"))
    ids = [g["id"] for g in golden]
    assert len(ids) == len(set(ids)) and len(golden) >= 50
    assert {g.get("tier", "standard") for g in golden} == {"standard", "hard"}
    for g in golden:
        for field in ("question", "expected_answer", "must_include", "source_docs", "source_section", "type"):
            assert field in g, (g["id"], field)
        if g["type"] == "unanswerable":
            assert not g["must_include"] and not g["source_docs"], g["id"]
            assert g["expected_answer"].startswith("ABSTAIN"), g["id"]
        # The split: optional details only come with a required answer; the required part is never an ABSTAIN question.
        if "optional_details" in g:
            assert "required_answer" in g and g["optional_details"].strip(), g["id"]
        if "required_answer" in g:
            assert g["type"] != "unanswerable" and g["required_answer"].strip(), g["id"]
        if g.get("expected_verdict"):
            assert g["expected_verdict"] in ("yes", "no") and g["type"] != "unanswerable", g["id"]
            reference = g.get("required_answer") or g["expected_answer"]
            assert reference.lower().startswith(g["expected_verdict"]), g["id"]      # the reference answer follows its own verdict
        if g["type"] != "unanswerable":
            # Every reference answer (required + optional) passes the keyword check: the check and the key facts agree.
            reference = g.get("required_answer", g["expected_answer"]) + " " + g.get("optional_details", "")
            check = strict_keyword_check(g, reference, found=True)
            assert check["pass"], (g["id"], check)
    normalized = root / "data" / "processed" / "normalized"
    if not normalized.exists():
        pytest.skip("the documents have not been parsed")
    texts = {p.stem: p.read_text(encoding="utf-8") for p in normalized.glob("*.md")}
    for g in golden:
        for fact in g["must_include"]:
            # A key fact is in the source document, or (a derived fact such as a total or a version name) in the reference answer.
            reference = " ".join([g["expected_answer"], g.get("required_answer", ""), g.get("optional_details", "")])
            assert (any(strict_matches(fact, texts[Path(d).stem]) for d in g["source_docs"]) or strict_matches(fact, reference)), (g["id"], fact)


def test_the_holdout_set_is_consistent_and_never_overlaps_the_golden_set():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    golden = json.loads((root / "eval" / "golden_set.json").read_text(encoding="utf-8"))
    held = json.loads((root / "eval" / "holdout_set.json").read_text(encoding="utf-8"))
    assert len(held) >= 18 and len({h["id"] for h in held}) == len(held)
    assert not {h["id"] for h in held} & {g["id"] for g in golden}                  # ids differ (h.. against q..)
    assert not {h["question"] for h in held} & {g["question"] for g in golden}     # and no question is shared: it must stay held out
    texts = {p.stem: p.read_text(encoding="utf-8") for p in (root / "data" / "processed" / "normalized").glob("*.md")} \
        if (root / "data" / "processed" / "normalized").exists() else None
    for h in held:
        for field in ("question", "expected_answer", "must_include", "source_docs", "type"):
            assert field in h, (h["id"], field)
        if h["type"] == "unanswerable":
            assert not h["must_include"] and not h["source_docs"] and h["expected_answer"].startswith("ABSTAIN"), h["id"]
            continue
        if h.get("expected_verdict"):
            assert h["expected_verdict"] in ("yes", "no"), h["id"]
        reference = h.get("required_answer", h["expected_answer"]) + " " + h.get("optional_details", "")
        assert strict_keyword_check(h, reference, found=True)["pass"], h["id"]      # the reference answer passes its own check
        if texts:
            for fact in h["must_include"]:
                everything = " ".join([h["expected_answer"], h.get("required_answer", ""), h.get("optional_details", "")])
                assert any(strict_matches(fact, texts[Path(d).stem]) for d in h["source_docs"]) or strict_matches(fact, everything), (h["id"], fact)


def test_a_key_fact_can_be_a_list_of_alternatives_and_a_yes_no_question_checks_the_first_word():
    assert contains("median 2.86s and p95 8.99s", "2.86") and not contains("12.86s", "2.86") and not contains("2.867", "2.86")   # a decimal takes a unit
    assert contains("the model has six layers", ["6 layers", "six layers"]) and not contains("the model has 16 layers", ["6 layers"])
    assert coverage("alpha and gamma", ["alpha", ["beta", "gamma"]]) == 1.0
    question = q(must_include=[["out of scope", "not supported"], "version 1"], expected_verdict="no", must_not_include=["planned for version 1"])
    good = keyword_check("No. Half-day leave is not supported in version 1 [S1].", question, found=True)
    assert good["include_rate"] == 1.0 and good["verdict_ok"] is True and good["forbidden"] == [] and good["found_ok"] is True
    assert keyword_check("Yes, it is out of scope for version 1.", question)["verdict_ok"] is False             # starts with the wrong word
    assert keyword_check("No. It is planned for version 1.", question)["forbidden"] == ["planned for version 1"]
    assert keyword_check("No, not supported.", question)["missing"] == ["version 1"]
    assert keyword_check("No. Out of scope in version 1.", question, found=False)["found_ok"] is False
    assert keyword_check("No. Out of scope in version 1.", question)["found_ok"] is None                        # no found flag: not checked
    assert is_failure({"error": None, "judge": None, "type": "factual", "keyword": keyword_check("Yes, out of scope version 1", question)})


def test_the_evaluation_uses_the_versions_own_retrieval_and_the_answer_path_pays_for_its_search(tmp_path, monkeypatch):
    from src.retrieve.dense import retrieve

    calls = []

    def configured():
        def search(question, k):
            calls.append((question, k))
            return retrieve(question, k)
        return search

    monkeypatch.setattr("eval.run_eval.configured_search", configured)
    monkeypatch.setattr("src.pipeline.configured_search", configured)
    verdict = json.dumps({"correctness": 1, "relevance": 1, "completeness": 1, "faithfulness": 1,
                          "abstained": False, "unsupported_claims": [], "reasoning": "ok"})
    run_evaluation("own", ids=["q11"], llm=FakeLLM(), judge_llm=FakeJudge(*[verdict] * 2), results_dir=tmp_path,
                   trace_path=tmp_path / "traces.jsonl")
    # Once to score the retrieval and once inside the answer's trace, so the reported latency includes the search.
    assert [c for c in calls if c[1] == 5] == [("What does error code QL-1003 mean?", 5)] * 2


def test_a_prompt_experiment_replays_the_chunks_the_model_read_even_when_they_differ_from_the_retrieval_pass(tmp_path):
    from eval.prompt_experiments import saved_retrieval

    run = tmp_path / "r"
    run.mkdir()
    retrieved = [{"chunk_id": "spec_c1", "doc_id": "spec", "page": 2, "score": 0.9}]
    (run / "results.jsonl").write_text(json.dumps({"id": "q1", "question": "what?", "retrieved": retrieved}), encoding="utf-8")
    answered = [{"chunk_id": "spec_c1", "text": "one"}, {"chunk_id": "resume_c4", "text": "four"}]      # a corrective search added resume_c4
    (run / "answers.jsonl").write_text(json.dumps({"id": "q1", "answer": "a", "error": None, "chunks": answered}), encoding="utf-8")
    hits = saved_retrieval("r", tmp_path)["what?"]
    assert [h["chunk_id"] for h in hits] == ["spec_c1", "resume_c4"] and [h["rank"] for h in hits] == [1, 2]
    assert hits[0]["score"] == 0.9 and hits[0]["page"] == 2 and hits[0]["text"] == "one"        # known chunks keep their metadata
    assert hits[1]["doc_id"] == "resume" and hits[1]["text"] == "four"                         # the others get it from their id


def test_split_judge_keeps_the_context_from_the_reference_call():
    from eval.judge import _judge_once
    graded = json.dumps({"reasoning": "matches", "correctness": 1, "relevance": 1, "completeness": 1, "abstained": False})
    checked = json.dumps({"reasoning": "supported", "faithfulness": 0.5, "unsupported_claims": ["x"]})
    judge = FakeJudge(graded, checked)
    prompts = []
    judge.complete = (lambda complete: lambda prompt, json_mode=False: prompts.append(prompt) or complete(prompt, json_mode))(judge.complete)
    verdict = _judge_once(judge, q(), "the answer", [{"chunk_id": "c", "text": "SECRET CONTEXT"}], 1, version="v3")
    assert judge.calls == 2 and verdict["correctness"] == 1.0 and verdict["faithfulness"] == 0.5
    assert "SECRET CONTEXT" not in prompts[0] and "SECRET CONTEXT" in prompts[1]     # the reference call never sees the context
    assert verdict["judge_prompt"] == "v3" and verdict["judge_tokens"] == {"in": 20, "out": 10}


def test_split_judge_counts_the_pipeline_refusal_as_abstained():
    from eval.judge import _judge_once
    from src.generate.answer import NOT_FOUND
    graded = json.dumps({"reasoning": "no answer", "correctness": 0, "relevance": 0, "completeness": 0, "abstained": False})
    checked = json.dumps({"reasoning": "nothing claimed", "faithfulness": 1, "unsupported_claims": []})
    verdict = _judge_once(FakeJudge(graded, checked), q(type="unanswerable"), NOT_FOUND, [], 1, version="v3")
    assert verdict["abstained"] and verdict["correctness"] == verdict["relevance"] == verdict["faithfulness"] == 1.0


def test_v4_scores_facts_in_code():
    from eval.judge import score_facts, verdict_flipped
    yes_q = q(expected_verdict="yes")
    present, missing = {"status": "present"}, {"status": "missing"}
    assert score_facts(q(), "a", [present, present], []) == {"correctness": 1.0, "completeness": 1.0}
    assert score_facts(q(), "a", [present, missing], []) == {"correctness": 1.0, "completeness": 0.5}      # incomplete, not wrong
    assert score_facts(q(), "a", [missing, missing], []) == {"correctness": 0.0, "completeness": 0.0}      # does not answer
    assert score_facts(q(), "a", [present, present], ["a third, wrong item"])["correctness"] == 0.5        # n16
    assert score_facts(q(), "a", [{"status": "contradicted"}], [])["correctness"] == 0.0
    assert score_facts(yes_q, "No, not every leave type", [present], [])["correctness"] == 0.0           # h03: Yes/No flip
    assert verdict_flipped(yes_q, "Yes, it does") is False and verdict_flipped(q(), "No") is False       # no expected verdict


def test_v4_faithfulness_needs_a_verified_quote():
    from eval.judge import claim_supported, quote_in_context, score_claims
    context = "|Transformer (big)|28.4|41.8|2.3·10^19||\nOur big model achieves a BLEU score of 41.0"
    assert quote_in_context("Transformer (big) | 28.4 | 41.8", context)                 # table pipes and spaces do not matter
    assert not quote_in_context("Transformer (big) 28.4 42.9", context)                  # a number that is not there
    assert claim_supported({"claim": "41.0 BLEU", "quote": "achieves a BLEU score of 41.0"}, context)
    assert claim_supported({"claim": "EN-FR is much larger than EN-DE", "quote": "", "derived_from": []}, context)
    assert claim_supported({"claim": "The resume does not mention Terraform", "quote": "", "supported_by_absence": True}, context)
    assert not claim_supported({"claim": "ReleaseForge AI uses Ollama", "quote": "", "supported_by_absence": True}, context)
    assert score_claims([], context) == (1.0, [])                                         # a refusal claims nothing
    assert score_claims([{"claim": "x", "quote": "28.4"}, {"claim": "y", "quote": "made up"}], context) == (0.5, ["y"])


def test_v4_judge_end_to_end_with_a_fake_judge():
    from eval.judge import _judge_once
    facts = json.dumps({"reasoning": "r", "facts": [{"fact": "41.8", "status": "present"}], "wrong_claims": [],
                        "relevance": 1, "abstained": False})
    claims = json.dumps({"claims": [{"claim": "41.8 BLEU", "quote": "41.8"}]})
    v = _judge_once(FakeJudge(facts, claims), q(), "41.8 BLEU", [{"chunk_id": "c", "text": "|big|28.4|41.8|"}], 1, version="v4")
    assert (v["correctness"], v["completeness"], v["faithfulness"], v["judge_prompt"]) == (1.0, 1.0, 1.0, "v4")
    # found=False from the answer model: a refusal, whatever the judge reads (h17)
    v = _judge_once(FakeJudge(facts, claims), q(), "Nothing on that.", [], 1, version="v4", found=False)
    assert v["abstained"] and v["correctness"] == v["completeness"] == 0.0 and v["faithfulness"] == 1.0


def test_v4_accepts_a_calculation_from_the_answers_own_numbers():
    from eval.judge import calculated_from_answer
    context = "English-French dataset consisting of 36M sentences ... about 4.5 million sentence pairs"
    answer = "36 million vs about 4.5 million, so larger by approximately 31.5 million"
    assert calculated_from_answer("larger by approximately 31.5 million", context, answer)          # 36 - 4.5 (q34)
    assert not calculated_from_answer("larger by 42 million", context, answer)                        # invented number
    assert not calculated_from_answer("ReleaseForge AI uses Ollama", context, answer)                 # no number: not this rule


def test_context_precision_rewards_sending_fewer_better_chunks():
    from eval.metrics import context_precision, precision_at_k, retrieval_metrics
    relevant = {"a"}
    assert precision_at_k(["a"], relevant, 5) == 0.2 and context_precision(["a"], relevant) == 1.0          # v9 sends 1 chunk
    assert precision_at_k(["a", "b", "c", "d", "e"], relevant, 5) == context_precision(["a", "b", "c", "d", "e"], relevant) == 0.2
    assert context_precision([], relevant) == 0.0
    assert retrieval_metrics(["a", "b"], relevant, 5)["context_precision"] == 0.5
