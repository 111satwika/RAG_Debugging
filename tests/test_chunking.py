"""Chunking tests. A word tokenizer and a hashing embedder keep them fast and deterministic."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from eval.chunk_diagnostics import diagnose, hub_chunks
from eval.chunking_experiments import Experiment, pick_winner, run_experiments
from eval.metrics import chunk_body, contains, label_relevant
from eval.run_eval import run_evaluation
from src.ingest.chunk import chunk_document
from src.ingest.chunkers import (ChunkSpec, _merge_small, build_sections, doc_title, make_children, make_pieces,
                                 merge_small_sections, split_fixed, split_recursive, split_table)


class WordTokenizer:
    name = "words"
    max_tokens = None

    def offsets(self, text):
        return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]

    def count(self, text):
        return len(self.offsets(text))


TOK = WordTokenizer()


def make_doc(*pages, doc_type="spec", doc_id="d"):
    return {"doc_id": doc_id, "doc_type": doc_type, "source": f"{doc_id}.pdf", "error": None,
            "pages": [{"page": i + 1, "text": text} for i, text in enumerate(pages)]}


def chunks_of(doc, **spec):
    return chunk_document(doc, TOK, spec=ChunkSpec(**spec))


# ---------------------------------------------------------------- spec

def test_chunk_spec_validation_and_label():
    assert ChunkSpec("structure", 500, headers=True, keep_tables=True).label == "structure-500-headers-tables"
    assert ChunkSpec("fixed", 300, overlap=50).label == "fixed-300-overlap50"
    assert ChunkSpec("recursive", 500, overlap=50).label == "recursive-500-overlap50"
    assert ChunkSpec("structure", 500, overlap=50, headers=True, keep_tables=True, child_size=150).label == \
        "structure-500-overlap50-headers-tables-child150"
    for bad in (dict(strategy="nope"), dict(overlap=500), dict(strategy="fixed", headers=True),
                dict(strategy="recursive", keep_tables=True), dict(strategy="fixed", min_tokens=10),
                dict(strategy="structure", min_tokens=500), dict(child_size=500), dict(child_size=5)):
        with pytest.raises(ValueError):
            ChunkSpec(**bad)
    assert ChunkSpec("structure", 500, overlap=50, headers=True, keep_tables=True, min_tokens=50).label == \
        "structure-500-overlap50-headers-tables-min50"


# ---------------------------------------------------------------- fixed and recursive

def test_fixed_windows_with_and_without_overlap():
    assert split_fixed("a b c d e f g", TOK, 3) == ["a b c", "d e f", "g"]
    assert split_fixed("a b c d e f g", TOK, 3, overlap=1) == ["a b c", "c d e", "e f g"]
    assert split_fixed("a b c d e", TOK, 3, overlap=1) == ["a b c", "c d e"]       # no tiny duplicate tail
    assert split_fixed("", TOK, 3) == []


def test_overlap_keeps_a_fact_that_a_plain_cut_splits():
    text = "alpha beta gamma delta MRR 0.831 epsilon zeta eta"
    plain = split_fixed(text, TOK, 5)
    assert not any(contains(piece, "MRR 0.831") for piece in plain)
    assert any(contains(piece, "MRR 0.831") for piece in split_fixed(text, TOK, 5, overlap=2))


def test_recursive_overlap_carries_whole_sentences_into_the_next_chunk():
    sentences = "One two three. Four five six. Seven eight nine."
    assert split_recursive(sentences, TOK, 6, overlap=3) == ["One two three. Four five six.", "Four five six. Seven eight nine."]
    assert split_recursive(sentences, TOK, 6, overlap=2) == ["One two three. Four five six.", "Seven eight nine."]


def test_a_tiny_tail_joins_the_chunk_before_it_instead_of_standing_alone():
    run = " ".join(f"w{i}" for i in range(31))                                   # one unbroken run of 31 words
    assert [TOK.count(p) for p in split_recursive(run, TOK, 30)] == [30, 1]
    merged = split_recursive(run, TOK, 30, min_tokens=5)
    assert len(merged) == 1 and merged[0] == run                                  # the words are not glued together


def test_small_pieces_merge_with_a_neighbour_in_the_same_section():
    long_piece = " ".join(f"w{i}" for i in range(30))
    merged = _merge_small(["a b c", long_piece, "x y"], TOK, size=30, min_tokens=5)
    assert len(merged) == 1 and merged[0].startswith("a b c") and merged[0].endswith("x y")
    big = " ".join(f"v{i}" for i in range(50))                                     # a whole table already over the budget
    assert len(_merge_small([big, "x y"], TOK, size=30, min_tokens=5)) == 1
    assert len(_merge_small(["a b c", "d e f g h i"], TOK, size=4, min_tokens=2)) == 2   # nothing small enough to merge


def test_overlap_stays_inside_a_section_and_every_piece_keeps_its_header():
    sentences = " ".join(f"s{i} b c d." for i in range(8))                         # eight 4-word sentences
    doc = make_doc(f"# T\n\n## A\n{sentences}\n\n## B\nbravo one two")
    chunks = chunks_of(doc, strategy="structure", size=30, overlap=4, headers=True)
    a_pieces = [c for c in chunks if c["header"] == "T > A"]
    assert len(a_pieces) >= 2 and all(c["text"].startswith("T > A\n\n") for c in a_pieces)
    shared = " ".join(a_pieces[0]["text"].split()[-4:])                                   # the last sentence of piece one
    assert shared.endswith("b c d.") and a_pieces[1]["text"].split("\n\n", 1)[1].startswith(shared)   # opens piece two
    b_pieces = [c for c in chunks if c["header"] == "T > B"]
    assert len(b_pieces) == 1 and "s7" not in b_pieces[0]["text"]                 # never overlaps across sections


def test_tiny_subsections_merge_with_their_parent_or_siblings_but_not_across_top_level_sections():
    text = "# Me\n\n## EXPERIENCE\nJob row\n\n### Proj A\nalpha text here\n\n### Proj B\nbeta text here\n\n## EDUCATION\nSchool text"
    doc = make_doc(text, doc_type="resume")
    chunks = chunks_of(doc, strategy="structure", size=60, headers=True, min_tokens=10)
    assert [c["header"] for c in chunks] == ["Me > EXPERIENCE", "Me > EDUCATION"]
    assert all(word in chunks[0]["text"] for word in ("Job row", "alpha", "beta")) and "School" not in chunks[0]["text"]
    without = chunks_of(doc, strategy="structure", size=60, headers=True)
    assert len(without) == 4                                                      # EXPERIENCE, Proj A, Proj B, EDUCATION


def test_sections_that_share_a_top_level_heading_are_not_merged_in_a_flat_document():
    text = "# Spec\n\n## 1. One\nalpha\n\n## 2. Two\nbeta"
    sections = build_sections(make_doc(text), 2)
    assert len(merge_small_sections(sections, TOK, "Spec", 60, 10)) == len(sections)


class LimitedTokenizer(WordTokenizer):
    max_tokens = 52          # like an embedding model that accepts 52 tokens: a whole table may use up to 50 of them


def test_a_whole_table_may_exceed_the_chunk_size_but_never_the_embedding_limit():
    table = long_table(14)                                                         # about 100 words
    doc = make_doc(f"## Data\nintro words here\n{table}")
    whole = chunk_document(doc, WordTokenizer(), spec=ChunkSpec("structure", 40, keep_tables=True))
    assert any(all(f"| r{i} aa | v{i} bb |" in c["text"] for i in range(14)) for c in whole)
    assert any(c["n_tokens"] > 40 for c in whole)                                  # over the budget, on purpose
    limited = chunk_document(doc, LimitedTokenizer(), spec=ChunkSpec("structure", 40, keep_tables=True))
    assert max(c["n_tokens"] for c in limited) <= 50 and len([c for c in limited if "| name x |" in c["text"]]) >= 2


def test_children_repeat_the_parent_header_and_point_back_to_it():
    body = " ".join(f"w{i}" for i in range(40))
    parents = chunks_of(make_doc(f"# T\n\n## Sec\n{body}"), strategy="structure", size=100, headers=True)
    assert len(parents) == 1
    children = make_children(parents, TOK, ChunkSpec("structure", 100, headers=True, child_size=20))
    assert len(children) > 1
    assert [c["chunk_id"] for c in children] == [f"d_c0_k{i}" for i in range(len(children))]
    assert all(c["parent_id"] == "d_c0" and c["text"].startswith("T > Sec\n\n") and c["n_tokens"] <= 20 for c in children)


def test_recursive_prefers_paragraph_then_sentence_boundaries():
    paragraphs = "one two three four\n\nfive six seven eight"
    assert split_recursive(paragraphs, TOK, 6) == ["one two three four", "five six seven eight"]
    sentences = "One two three. Four five six. Seven eight nine."
    assert split_recursive(sentences, TOK, 6) == ["One two three. Four five six.", "Seven eight nine."]
    unbroken = " ".join(f"w{i}" for i in range(20))
    assert all(TOK.count(p) <= 6 for p in split_recursive(unbroken, TOK, 6))


# ---------------------------------------------------------------- structure

SPEC_TEXT = "# My Spec\n\n## 1. Intro\nhello world here\n\n## 2. Rules\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n## 3. End\nbye"


def test_structure_cuts_at_headings_and_merges_a_heading_with_no_text_forward():
    chunks = chunks_of(make_doc(SPEC_TEXT), strategy="structure", size=50)
    assert [c["text"].split("\n")[0] for c in chunks] == ["# My Spec", "## 2. Rules", "## 3. End"]
    assert "## 1. Intro" in chunks[0]["text"] and "hello world here" in chunks[0]["text"]


def test_structure_headers_name_the_document_and_section():
    chunks = chunks_of(make_doc(SPEC_TEXT), strategy="structure", size=50, headers=True)
    assert [c["header"] for c in chunks] == ["My Spec > 1. Intro", "My Spec > 2. Rules", "My Spec > 3. End"]
    assert all(c["text"].startswith(c["header"] + "\n\n") for c in chunks)
    assert chunk_body(chunks[1]).lstrip().startswith("## 2. Rules")


def test_a_section_keeps_the_page_where_it_starts_and_the_title_is_read_from_the_text():
    doc = make_doc("# Title\n\n## A\nfirst page text", "continues here\n\n## B\nsecond", doc_id="my_doc")
    assert doc_title(doc) == "Title" and doc_title(make_doc("no heading", doc_id="my_doc")) == "my doc"
    sections = build_sections(doc, max_level=2)
    assert [s["page"] for s in sections] == [1, 2]
    assert "continues here" in "\n".join(sections[0]["lines"])


def test_every_chunk_fits_the_budget_header_included():
    body = " ".join(f"w{i}" for i in range(80))
    doc = make_doc(f"# T\n\n## Big section\n{body}")
    for chunk in chunks_of(doc, strategy="structure", size=30, headers=True):
        assert chunk["n_tokens"] <= 30


def long_table(rows=12):
    return "\n".join(["| name x | value y |", "|---|---|"] + [f"| r{i} aa | v{i} bb |" for i in range(rows)])


def test_keep_tables_repeats_the_header_row_when_a_table_beyond_the_embedding_limit_must_be_cut():
    intro = " ".join(f"intro{i}" for i in range(30))
    doc = make_doc(f"## Data\n{intro}\n{long_table()}")
    plain = chunks_of(doc, strategy="structure", size=40)
    kept = chunk_document(doc, LimitedTokenizer(), spec=ChunkSpec("structure", 40, keep_tables=True))
    assert any(c["text"].lstrip().startswith("| r") for c in plain[1:])                  # a part with no header row
    table_parts = [c for c in kept if "| name x | value y |" in c["text"]]
    assert len(table_parts) >= 2 and all(c["text"].count("|---|---|") == 1 for c in table_parts)


def test_a_table_that_fits_stays_whole_even_next_to_a_paragraph():
    intro = " ".join(f"intro{i}" for i in range(30))
    doc = make_doc(f"## Data\n{intro}\n{long_table(3)}")
    kept = chunks_of(doc, strategy="structure", size=40, keep_tables=True)
    assert any(all(f"| r{i} aa | v{i} bb |" in c["text"] for i in range(3)) for c in kept)


def test_split_table_parts_start_with_the_header():
    parts = split_table(long_table(8), TOK, 30)
    assert len(parts) > 1 and all(p.startswith("| name x | value y |\n|---|---|") for p in parts)


def test_chunk_document_ids_pages_and_metadata():
    chunks = chunks_of(make_doc(SPEC_TEXT, "## 4. More\nlater text"), strategy="structure", size=50, headers=True)
    assert [c["chunk_id"] for c in chunks] == [f"d_c{i}" for i in range(len(chunks))]
    assert chunks[-1]["page"] == 2 and chunks[-1]["strategy"] == "structure-50-headers" and chunks[0]["tokenizer"] == "words"
    assert len(make_pieces(make_doc("a b c d e f"), TOK, ChunkSpec("recursive", 3))) == 2


# ---------------------------------------------------------------- matching and labels

def test_a_word_matches_its_plural_but_numbers_stay_exact():
    assert contains("extremely small gradients^4 .", "gradient")
    assert contains("the heads", "head") and not contains("a headline", "head")
    assert not contains("16 heads", "6") and not contains("6s", "6")


def test_a_header_cannot_create_a_relevant_chunk():
    question = {"id": "q", "type": "factual", "source_docs": ["d.pdf"], "must_include": ["Leave Policy"]}
    chunks = [{"chunk_id": "d_c0", "doc_id": "d", "header": "Spec > 5. Leave Policy", "text": "Spec > 5. Leave Policy\n\nunrelated words"}]
    assert label_relevant(question, chunks)["ids"] == []


# ---------------------------------------------------------------- diagnostics

def chunk(cid, text, doc="d", header=""):
    return {"chunk_id": cid, "doc_id": doc, "text": header + text if header else text, "header": header, "n_tokens": len(text.split())}


def test_diagnose_flags_each_chunking_problem():
    chunks = [
        chunk("d_c0", "## One\ntext\n## Two\nmore text that is cut and"),            # two sections; cut mid-sentence
        chunk("d_c1", "continues here.\n| a | b |\n| 1 | 2 |"),                      # ends inside a table
        chunk("d_c2", "3 | 4 |\nOther text\n## Heading at the end"),                  # row fragment; orphan heading
        chunk("d_c3", "body of the heading", header="Doc > Section"),
    ]
    questions = [{"id": "q1", "type": "factual", "source_docs": ["d.pdf"], "must_include": ["cut", "heading"]},
                 {"id": "q2", "type": "multi_section", "source_docs": ["d.pdf"], "must_include": ["cut", "heading"]}]
    d = diagnose(chunks, questions)
    assert d["mixed_topic_chunks"] == ["d_c0"]
    assert d["orphan_headings"] == ["d_c2"]
    assert d["split_tables"] == [("d_c1", "d_c2")]
    assert ("d_c0", "d_c1") in d["mid_sentence_cuts"]
    assert d["context_free_chunks"] == ["d_c1"]                                      # no heading line and no header
    assert d["split_answers"] == ["q1"]                                              # multi_section is split by design


def test_a_table_cut_is_fine_when_the_next_chunk_repeats_the_header_row():
    header = "| name | value |\n|---|---|"
    repeated = [chunk("d_c0", f"{header}\n| a | 1 |"), chunk("d_c1", f"{header}\n| b | 2 |")]
    assert diagnose(repeated, [])["split_tables"] == []
    rows_only = [chunk("d_c0", f"{header}\n| a | 1 |"), chunk("d_c1", "| b | 2 |")]
    assert diagnose(rows_only, [])["split_tables"] == [("d_c0", "d_c1")]


def test_hub_chunks_need_several_questions():
    def result(top, relevant, ranked):
        return {"retrieved": [{"chunk_id": c} for c in ranked], "relevant_ids": [relevant]}

    results = [result("hub", f"r{i}", ["hub", f"r{i}"]) for i in range(3)] + [result("other", "x", ["other", "x"])]
    assert hub_chunks(results) == {"hub": 3}


# ---------------------------------------------------------------- experiments

class HashEmbedder:
    name = "hash"
    dim = 16

    def _vec(self, text):
        vec = [0.0] * self.dim
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vec[hash(word) % self.dim] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_queries(self, texts):
        return [self._vec(t) for t in texts]


def test_run_evaluation_can_score_an_alternative_index(tmp_path):
    golden = [{"id": "q1", "question": "where is the cat", "expected_answer": "on the mat", "source_docs": ["d.pdf"],
               "source_section": "s", "type": "factual", "must_include": ["mat"]}]
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(json.dumps(golden), encoding="utf-8")
    chunks = [{"chunk_id": "d_c0", "doc_id": "d", "text": "the cat sat on the mat", "header": ""},
              {"chunk_id": "d_c1", "doc_id": "d", "text": "nothing relevant here", "header": ""}]

    def searcher(query, k):
        return [{"rank": i, "chunk_id": c["chunk_id"], "doc_id": "d", "page": 1, "score": 0.9 - i / 10, "text": c["text"]}
                for i, c in enumerate(chunks[:k], start=1)]

    out = run_evaluation("alt", retrieval_only=True, use_judge=False, chunks=chunks, searcher=searcher,
                         config={"version": "exp", "chunk_strategy": "fixed"}, results_dir=tmp_path, golden_path=golden_path)
    assert out["summary"]["mrr"] == 1.0 and out["summary"]["recall"] == 1.0
    assert json.loads((tmp_path / "alt" / "config.json").read_text(encoding="utf-8"))["config"]["version"] == "exp"


def test_experiments_run_report_and_pick_a_winner(tmp_path):
    docs = [make_doc("# Cats\n\n## Pets\nthe cat sat on the mat\n\n## Food\nfish and milk are good", doc_id="cats")]
    golden = [{"id": "q1", "question": "where did the cat sit", "expected_answer": "mat", "source_docs": ["cats.pdf"],
               "source_section": "Pets", "type": "factual", "must_include": ["mat"]},
              {"id": "q2", "question": "what food is good", "expected_answer": "fish", "source_docs": ["cats.pdf"],
               "source_section": "Food", "type": "factual", "must_include": ["fish"]}]
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(json.dumps(golden), encoding="utf-8")
    experiments = [Experiment(ChunkSpec("fixed", 8), "control"),
                   Experiment(ChunkSpec("structure", 40, headers=True), "structure")]

    rows = run_experiments(k=2, results_dir=tmp_path, golden_path=golden_path, experiments=experiments,
                           tokenizer=TOK, embedder=HashEmbedder(), docs=docs)

    assert [r["experiment"].name for r in rows] == ["p3-fixed-8", "p3-structure-40-headers"]
    for name in ("p3-fixed-8", "p3-structure-40-headers"):
        config = json.loads((tmp_path / name / "config.json").read_text(encoding="utf-8"))["config"]
        assert config["version"] == "exp"
    assert json.loads((tmp_path / "p3-structure-40-headers" / "config.json").read_text(encoding="utf-8"))["config"]["chunk_headers"] is True
    report = (tmp_path / "chunking_experiments.md").read_text(encoding="utf-8")
    assert "Winner" in report and "| Cause | Symptom | In your baseline | With the winner |" in report
    assert pick_winner(rows) in rows


def test_parent_child_search_returns_each_parent_once_with_its_best_childs_score():
    import chromadb

    from src.ingest.index import build_index
    from src.retrieve.dense import retrieve

    embedder = HashEmbedder()
    parents = [{"chunk_id": "p0", "doc_id": "d", "doc_type": "spec", "page": 1, "text": "cats purr and cats nap and cats play", "header": ""},
               {"chunk_id": "p1", "doc_id": "d", "doc_type": "spec", "page": 2, "text": "money grows and money buys goods", "header": ""}]
    children = [{**p, "chunk_id": f"{p['chunk_id']}_k{i}", "parent_id": p["chunk_id"], "text": part, "n_tokens": 4, "tokenizer": "w"}
                for p in parents for i, part in enumerate([p["text"][:18], p["text"][18:]])]
    collection = build_index(children, embedder, chromadb.EphemeralClient(), parents=parents)

    hits = retrieve("cats purr", 2, embedder, collection, {p["chunk_id"]: p for p in parents})
    assert [h["chunk_id"] for h in hits] == ["p0", "p1"] or [h["chunk_id"] for h in hits][0] == "p0"
    assert hits[0]["text"] == parents[0]["text"] and hits[0]["rank"] == 1
    assert len({h["chunk_id"] for h in hits}) == len(hits)                    # a parent appears once however many children matched
    assert hits[0]["matched_child"].startswith("p0_k")


class FakeOpenAI:
    """Returns embeddings out of order, like a busy API might, and records every call."""
    def __init__(self):
        self.calls = []
        self.embeddings = self

    def create(self, model, input):
        from types import SimpleNamespace as NS
        self.calls.append(list(input))
        items = [NS(index=i, embedding=[float(len(text)), 1.0]) for i, text in enumerate(input)]
        return NS(data=list(reversed(items)))


def test_openai_embedder_batches_keeps_order_and_handles_blank_text():
    from src.ingest.embed import OpenAIEmbedder

    embedder = object.__new__(OpenAIEmbedder)
    embedder.name, embedder.dim, embedder.BATCH, embedder._client = "m", 2, 2, FakeOpenAI()
    vectors = embedder.embed_documents(["a", "bb", "", "dddd", "e"])
    assert [v[0] for v in vectors] == [1.0, 2.0, 1.0, 4.0, 1.0]               # the blank text was sent as one space
    assert [len(call) for call in embedder._client.calls] == [2, 2, 1]


def test_the_openai_experiment_set_covers_the_plan_and_the_combined_setup():
    from eval.chunking_experiments import COMBINED, LADDER, OPENAI_EXPERIMENTS, STACKS

    names = [e.name for e in OPENAI_EXPERIMENTS]
    assert len(names) == len(set(names)) and all(n.startswith("p3o-") for n in names)
    labels = {e.spec.label for e in OPENAI_EXPERIMENTS}
    assert {"fixed-300", "fixed-500-overlap50", "recursive-500-overlap50"} <= labels
    assert not any("semantic" in label or "800" in label for label in labels)          # both were removed
    assert COMBINED.label == "structure-500-overlap50-headers-tables-min50" and COMBINED.label in labels
    assert f"{COMBINED.label}-child150" in labels
    assert len(LADDER) == 5 and LADDER[-1] == f"p3o-{COMBINED.label}"
    assert STACKS["openai"].tokenizer[1] == "cl100k_base" and STACKS["openai"].embedder[1] == "text-embedding-3-small"


def test_the_v2_profile_matches_the_combined_experiment():
    out = subprocess.run([sys.executable, "-c", "from src.ingest.chunk import current_spec; print(current_spec().label)"],
                         cwd=Path(__file__).resolve().parents[1], env={**os.environ, "RAG_PIPELINE_VERSION": "v2"},
                         capture_output=True, text=True, timeout=60)
    assert out.stdout.strip() == "structure-500-overlap50-headers-tables-min50"


def test_a_parent_child_experiment_is_scored_on_the_parents(tmp_path):
    docs = [make_doc("# Cats\n\n## Pets\nthe cat sat on the mat and purred loudly all day long\n\n## Food\nfish and milk are good for a cat", doc_id="cats")]
    golden = [{"id": "q1", "question": "where did the cat sit", "expected_answer": "mat", "source_docs": ["cats.pdf"],
               "source_section": "Pets", "type": "factual", "must_include": ["mat"]}]
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(json.dumps(golden), encoding="utf-8")
    experiment = Experiment(ChunkSpec("structure", 40, headers=True, child_size=20), "parent-child")

    rows = run_experiments(k=2, results_dir=tmp_path, golden_path=golden_path, experiments=[experiment],
                           tokenizer=TOK, embedder=HashEmbedder(), docs=docs)

    results = [json.loads(line) for line in (tmp_path / experiment.name / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    retrieved = [h["chunk_id"] for h in results[0]["retrieved"]]
    assert all(cid.startswith("cats_c") and "_k" not in cid for cid in retrieved)       # parents, never children
    assert json.loads((tmp_path / experiment.name / "config.json").read_text(encoding="utf-8"))["config"]["child_size"] == 20
    assert rows[0]["summary"]["n"] == 1


def test_normalize_repairs_parsed_tables():
    from src.ingest.normalize import normalize_text
    table = ("||BL|EU|Training C|ost (FLOPs)|\n|---|---|---|---|---|\n|Model|||||\n||EN-DE|EN-FR|EN-DE|EN-FR|\n"
             "|MoE[32]|26.03|40.56|2_.0·10^19|1.2·_10^20|\n|Transformer (big)|28.4|41.8|2_.3·_|10^19|")
    lines = normalize_text(table).split("\n")
    assert lines[0] == "|Model|BLEU EN-DE|BLEU EN-FR|Training Cost (FLOPs) EN-DE|Training Cost (FLOPs) EN-FR|"
    assert lines[2] == "|MoE[32]|26.03|40.56|2.0·10^19|1.2·10^20|"                 # no underscores left in numbers
    assert lines[3] == "|Transformer (big)|28.4|41.8|2.3·10^19||"                     # the split value is one cell again
    assert normalize_text("||N|d_model|_d_ff|_h|params _×_10^6|") == "||N|d_model|d_ff|h|params ×10^6|"
    assert normalize_text("the warmup_steps value") == "the warmup_steps value"          # underscores in words stay


def test_table_chunks_are_headed_by_their_caption():
    from src.ingest.chunkers import table_caption
    rows = "|a|b|\n|---|---|\n|1|2|\n|3|4|"
    # caption and rows in the same piece (the paper's Tables 1 and 2)
    assert table_caption(f"Some text.\nTable 2: BLEU scores of the models. More caption.\n{rows}", None)[0] == "Table 2: BLEU scores of the models."
    # caption at the end of one piece, rows in the next (Tables 3 and 4)
    caption, carried = table_caption("Text before.\nTable 3: Variations on the architecture.", None)
    assert caption is None and carried == "Table 3: Variations on the architecture."
    assert table_caption(rows, carried)[0] == "Table 3: Variations on the architecture."
    # no caption, or no table: the section header stays
    assert table_caption(rows, None)[0] is None and table_caption("plain text", None) == (None, None)
