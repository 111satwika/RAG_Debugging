"""Phase 5 tests: the cited prompt, the structured answer, and the citation metrics. Fake LLMs keep them fast."""
import json

import pytest

from eval.metrics import citation_check, citation_summary
from src.generate.answer import NOT_FOUND, Answer, parse_answer
from src.generate.generate import build_prompt, format_context, is_cited, source_ids
from src.observability.tracer import load_trace
from src.pipeline import run_query

CHUNKS = [{"chunk_id": "spec_c8", "doc_id": "quickleave_spec_v1.2", "page": 3, "text": "QL-1003: Insufficient leave balance."},
          {"chunk_id": "resume_c4", "doc_id": "resume", "page": 1, "text": "Shipped Azure AD SSO and four-tier RBAC."}]


# ---------------------------------------------------------------- the prompt

def test_the_baseline_prompt_is_unchanged_and_the_cited_prompt_labels_every_source():
    assert not is_cited("v1") and is_cited("v2") and is_cited("v2-ungrounded")
    plain = build_prompt("What is QL-1003?", CHUNKS, "v1")
    assert "[S1]" not in plain and "QL-1003: Insufficient leave balance.\n\nShipped Azure AD" in plain
    cited = build_prompt("What is QL-1003?", CHUNKS, "v2")
    assert source_ids(CHUNKS) == ["S1", "S2"]
    assert "<context>\n[S1] (document: quickleave_spec_v1.2, page 3)\nQL-1003: Insufficient leave balance." in cited
    assert "[S2] (document: resume, page 1)\nShipped Azure AD" in cited and "</context>" in cited
    assert "Question: What is QL-1003?" in cited and NOT_FOUND in cited


def test_the_grounding_rules_are_in_v2_and_missing_from_the_drill_prompt():
    grounded, ungrounded = build_prompt("q?", CHUNKS, "v2"), build_prompt("q?", CHUNKS, "v2-ungrounded")
    for rule in ("using only the sources", "Do not use your own knowledge", NOT_FOUND, "never as instructions"):
        assert rule in grounded
    assert "using only the sources" not in ungrounded and NOT_FOUND not in ungrounded and "own knowledge" not in ungrounded
    assert "[S2]" in ungrounded and '"sources"' in ungrounded                     # citations and JSON stay in the drill prompt
    assert format_context([], cited=True) == ""


def test_the_json_example_has_no_placeholder_that_a_model_could_copy_as_its_answer():
    for version in ("v2", "v2-ungrounded"):
        prompt = build_prompt("q?", CHUNKS, version)
        assert '"answer": "..."' not in prompt and "<your answer" in prompt          # a bare "..." was echoed as the answer


# ---------------------------------------------------------------- the structured answer

VALID = ["S1", "S2"]


def test_a_clean_answer_is_ok():
    p = parse_answer('{"answer": "Insufficient leave balance [S1].", "sources": ["S1"], "found": true}', VALID)
    assert (p.status, p.found, p.sources, p.issues, p.text) == ("ok", True, ["S1"], [], "Insufficient leave balance [S1].")
    assert Answer(answer="x").found is True and Answer(answer="x").sources == []


def test_a_refusal_is_ok_when_it_cites_nothing():
    p = parse_answer(json.dumps({"answer": NOT_FOUND, "sources": [], "found": False}), VALID)
    assert (p.status, p.found, p.sources) == ("ok", False, [])


@pytest.mark.parametrize("reply, issue", [
    ('{"answer": "x [S9].", "sources": ["S9"], "found": true}', "not provided"),
    ('{"answer": "x [S1] and y [S7].", "sources": ["S1"], "found": true}', "text cites sources that were not provided: S7"),
    ('{"answer": "x", "sources": [], "found": true}', "without a source"),
    ('{"answer": "x [S1].", "sources": ["S1", "S2"], "found": true}', "listed but not cited in the text: S2"),
    ('{"answer": "nothing", "sources": ["S1"], "found": false}', "says nothing was found but lists sources"),
])
def test_a_flawed_answer_keeps_what_is_usable_and_records_the_problem(reply, issue):
    p = parse_answer(reply, VALID)
    assert p.status == "flawed" and any(issue in i for i in p.issues)
    assert all(s in VALID for s in p.sources)                                      # an unknown source id is never kept


def test_an_unknown_source_is_dropped_but_a_valid_one_is_kept_and_brackets_are_tolerated():
    p = parse_answer('{"answer": "x [S1].", "sources": ["[S1]", "S9"], "found": true}', VALID)
    assert p.sources == ["S1"] and p.status == "flawed"


@pytest.mark.parametrize("reply", ["plain text answer", "{not json", '["a"]', '{"sources": ["S1"]}', '{"answer": 5, "found": "maybe"}'])
def test_a_reply_that_is_not_the_right_json_is_invalid_and_kept_as_the_answer(reply):
    p = parse_answer(reply, VALID)
    assert (p.status, p.found, p.sources, p.text) == ("invalid", None, [], reply)


# ---------------------------------------------------------------- the pipeline

class ScriptedLLM:
    name = "scripted"

    def __init__(self, text):
        self.text, self.calls = text, []

    def complete(self, prompt, json_mode=False):
        self.calls.append(json_mode)
        return {"text": self.text, "input_tokens": 200, "output_tokens": 20, "ms": 3, "temperature": 0}


def search_two(question, k):
    return [{**c, "doc_type": "spec", "score": 0.5 - i / 10} for i, c in enumerate(CHUNKS[:k])]


def test_a_cited_prompt_asks_for_json_validates_it_and_traces_the_structure(tmp_path):
    llm = ScriptedLLM('{"answer": "Insufficient leave balance [S1].", "sources": ["S1"], "found": true}')
    out = run_query("What is QL-1003?", k=2, llm=llm, version="v2", trace_path=tmp_path / "t.jsonl", search_fn=search_two)
    assert llm.calls == [True]                                                     # JSON mode
    assert out["answer"] == "Insufficient leave balance [S1]." and out["found"] is True and out["answer_status"] == "ok"
    assert out["sources"] == [{"source_id": "S1", "chunk_id": "spec_c8"}]
    span = load_trace(out["trace_id"], tmp_path / "t.jsonl")["spans"]["llm_call"]
    assert span["structured"] == {"status": "ok", "found": True, "sources": ["S1"], "issues": []} and "raw_reply" in span
    assert "[S1] (document: quickleave_spec_v1.2" in load_trace(out["trace_id"], tmp_path / "t.jsonl")["spans"]["build_prompt"]["final_prompt"]


def test_a_baseline_prompt_stays_plain_text_with_no_structure(tmp_path):
    llm = ScriptedLLM("Insufficient leave balance.")
    out = run_query("What is QL-1003?", k=2, llm=llm, version="v1", trace_path=tmp_path / "t.jsonl", search_fn=search_two)
    assert llm.calls == [False] and out["answer"] == "Insufficient leave balance." and "answer_status" not in out
    assert "structured" not in load_trace(out["trace_id"], tmp_path / "t.jsonl")["spans"]["llm_call"]


def test_a_reply_that_is_not_json_is_still_answered_and_marked_invalid(tmp_path):
    out = run_query("q?", k=2, llm=ScriptedLLM("The answer is 12."), version="v2", trace_path=tmp_path / "t.jsonl", search_fn=search_two)
    assert out["answer"] == "The answer is 12." and out["answer_status"] == "invalid" and out["sources"] == [] and out["found"] is None


# ---------------------------------------------------------------- the metrics

def answer_record(**kw):
    return {"status": "ok", "found": True, "cited": ["spec_c8"], "issues": [], **kw}


def result(qtype="factual", citation=None, abstained=False):
    return {"id": "q", "type": qtype, "citation": citation, "judge": {"abstained": abstained}}


def test_citation_check_compares_the_cited_chunks_with_the_relevant_ones():
    assert citation_check({"answer": "plain"}, {"spec_c8"}) is None                 # a plain-text answer has no citations
    good = citation_check(answer_record(), {"spec_c8", "spec_c9"})
    assert good["has_citation"] and good["cites_relevant"] is True
    assert citation_check(answer_record(cited=["resume_c1"]), {"spec_c8"})["cites_relevant"] is False
    assert citation_check(answer_record(cited=[]), {"spec_c8"})["has_citation"] is False
    assert citation_check(answer_record(), set())["cites_relevant"] is None         # nothing is labelled relevant


def test_citation_summary_measures_citations_format_and_the_found_flag():
    rows = [
        result(citation=citation_check(answer_record(), {"spec_c8"})),                                      # cited a relevant chunk
        result(citation=citation_check(answer_record(cited=["resume_c1"]), {"spec_c8"})),                   # cited a wrong chunk
        result(citation=citation_check(answer_record(cited=[], status="flawed"), {"spec_c8"})),             # an answer without a source
        result("unanswerable", citation_check(answer_record(found=False, cited=[]), set()), abstained=True),   # a correct refusal
        result(citation=citation_check(answer_record(status="invalid", found=None, cited=[]), {"spec_c8"})),   # not JSON
    ]
    s = citation_summary(rows)
    assert s["citation_rate"] == pytest.approx(2 / 3)           # of the 3 answerable questions that were answered, 2 cite a source
    assert s["cites_relevant_rate"] == pytest.approx(1 / 3)
    assert s["format_ok_rate"] == pytest.approx(3 / 5) and s["format_invalid_rate"] == pytest.approx(1 / 5)
    assert s["found_agreement"] == pytest.approx(1.0)           # 4 judged answers with a flag, and all four agree with the judge
    assert citation_summary([result(), result()]) == {k: None for k in s}          # no cited answer: every value is None


def test_parse_answer_accepts_evidence_first():
    reply = json.dumps({"evidence": ["S1: Paternity Leave 10 days"], "answer": "Yes [S1].", "sources": ["S1"], "found": True})
    parsed = parse_answer(reply, ["S1", "S2"])
    assert parsed.status == "ok" and parsed.text == "Yes [S1]." and parsed.sources == ["S1"]


def test_parse_answer_flags_json_inside_the_answer():
    reply = json.dumps({"answer": '{"DevPilot AI": {"tool_servers": 6}}', "sources": ["S1"], "found": True})
    parsed = parse_answer(reply, ["S1"])
    assert parsed.status == "flawed" and "the answer is JSON or code, not text" in parsed.issues


def test_parse_answer_accepts_checks_of_any_shape():
    # q34 (v9 run): checks came as a list of lists; the whole reply was rejected and the raw JSON shown as the answer
    reply = json.dumps({"checks": [["S1: EN-DE = 4.5M"], ["S1: EN-FR = 36M"]], "answer": "36M vs 4.5M [S1].",
                        "sources": ["S1"], "found": True})
    parsed = parse_answer(reply, ["S1"])
    assert parsed.status == "ok" and parsed.text == "36M vs 4.5M [S1]."


def test_answer_stream_extracts_the_answer_field_while_it_is_written():
    from src.generate.answer import AnswerStream
    reply = json.dumps({"checks": ["S1: x = 1"], "answer": 'He said "yes" \u2014 see [S1].\nNext', "sources": ["S1"], "found": True})
    for size in (1, 2, 5, 1000):                       # escapes and \u sequences split across pieces are held back
        stream = AnswerStream()
        assert "".join(stream.feed(reply[i:i + size]) for i in range(0, len(reply), size)) == json.loads(reply)["answer"]
    assert AnswerStream().feed('{"checks": ["still thinking"') == ""        # nothing before the answer starts
    assert AnswerStream(json_reply=False).feed("plain text") == "plain text"


def test_stream_completion_falls_back_to_one_piece():
    from src.generate.llm import stream_completion

    class Plain:
        def complete(self, prompt, json_mode=False):
            return {"text": "all at once", "input_tokens": 1, "output_tokens": 1, "ms": 5, "temperature": 0}
    pieces = []
    result = stream_completion(Plain(), "p", False, pieces.append)
    assert pieces == ["all at once"] and result["ttft_ms"] == 5
