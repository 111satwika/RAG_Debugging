"""End-to-end query flow with tracing: embed_query -> retrieve -> validate -> build_prompt -> llm_call.

The validate step (Phase 6, versions with VALIDATE) can stop the query before the model is called: the answer is then "not found"."""
import argparse
import time

from src.config import PROMPT_VERSION, REFINE, REFINE_WITH_CONTEXT, RELEVANCE_HIGH, RELEVANCE_LOW, RETRIES, TOP_K, VALIDATE
from src.generate.answer import NOT_FOUND, AnswerStream, ParsedAnswer, parse_answer
from src.generate.generate import build_prompt, is_cited, source_ids
from src.generate.llm import LLM, load_llm, stream_completion
from src.ingest.embed import load_embedder
from src.observability.logger import get_logger
from src.observability.tracer import TRACES_PATH, Trace
from src.retrieve.dense import embed_query, get_collection, load_question, search
from src.retrieve.retriever import configured_search
from src.validate.gate import validate_with_retry
from src.validate.refine import refine_chunks
from src.validate.threshold import grade_chunks

log = get_logger("pipeline")
PREVIEW_CHARS = 150


def run_query(question: str, k: int = TOP_K, llm: LLM | None = None, version: str = PROMPT_VERSION,
              trace_path=TRACES_PATH, search_fn=None, config: dict | None = None, validate: bool = VALIDATE,
              retries: int | None = None, refine: bool = REFINE, on_event=None) -> dict:
    """Answer one question and save its trace. Returns the answer and the trace.

    `search_fn(question, k) -> hits` replaces the search of the active version (used to evaluate an experimental index
    or retrieval); `config` is then recorded in the trace instead of the current settings. Without it, a version that
    uses hybrid search or reranking searches through src/retrieve/retriever.py, and a dense version embeds the question
    and searches the index directly.

    `on_event(event)` streams the query: {"type": "stage", "stage": ...} when a step starts and {"type": "token", "text": ...}
    for each piece of the answer as the model writes it. The trace then records ttft_ms, the time from the question to the
    first answer character. Without it the answer is generated in one call, as in the evaluations.
    """
    trace = Trace(question, config)
    emit = on_event or (lambda event: None)
    log.info("trace %s: %s", trace.trace_id, question)
    answer = error = None
    chunks: list[dict] = []
    context = None
    try:
        llm = llm or load_llm()
        if search_fn is None:
            search_fn = configured_search()

        if search_fn is None:
            embedder = load_embedder()
            with trace.span("embed_query") as s:
                vector = embed_query(question, embedder)
                s["model"] = embedder.name
                s["dim"] = len(vector)

        emit({"type": "stage", "stage": "retrieve"})
        with trace.span("retrieve") as s:
            chunks = search_fn(question, k) if search_fn else search(vector, k, embedder, get_collection(embedder))
            s["results"] = [
                {"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "page": c["page"], "score": c["score"],
                 "preview": c["text"][:PREVIEW_CHARS], **({"query": c["query"]} if c.get("query") else {})}
                for c in chunks
            ]
            transform = next((c["transform"] for c in chunks if c.get("transform")), None)
            if transform:                       # a hard question: why it was transformed and what the model planned
                s["transform"] = transform
        log.info("retrieved %d chunks, best score %.3f", len(chunks), chunks[0]["score"] if chunks else 0)

        cited = is_cited(version)
        verdict = None
        if validate:
            emit({"type": "stage", "stage": "validate"})
            with trace.span("validate") as s:
                verdict, chunks = validate_with_retry(question, chunks, llm, search_fn, k, RELEVANCE_HIGH, RELEVANCE_LOW,
                                                      RETRIES if retries is None else retries)
                s.update(verdict)
            log.info("validation: %s, %s, %d retries", verdict["grade"], verdict["action"], len(verdict["retries"]))
            # the CRAG grade is known before the answer is written: send it so the chat can show it straight away
            emit({"type": "validation", "validation": verdict})

        # The CRAG grade of every retrieved chunk (the context grade above only looks at the best one)
        context = grade_chunks(chunks, RELEVANCE_HIGH, RELEVANCE_LOW) if validate else None

        if refine and len(chunks) > 1 and not (verdict and verdict["action"] == "abstain"):
            # CRAG knowledge refinement: the model reads only the chunks a grading call keeps (src/validate/refine.py)
            emit({"type": "stage", "stage": "refine"})
            with trace.span("refine") as s:
                chunks, record = refine_chunks(question, chunks, llm, with_context=REFINE_WITH_CONTEXT)
                s.update(record)
            log.info("refinement: %d graded, %d dropped", record["graded"], len(record["dropped"]))

        if context is not None:
            read = {c["chunk_id"] for c in chunks}
            abstained = bool(verdict and verdict["action"] == "abstain")
            for c in context:                       # did the answer model read it (not filtered out, not blocked)?
                c["used"] = not abstained and c["chunk_id"] in read
            trace.data["context_grades"] = context
            emit({"type": "context", "context": context})

        if verdict and verdict["action"] == "abstain":
            # Nothing is generated: the fallback is the same "not found" the cited prompt gives, with no sources.
            parsed = ParsedAnswer(text=NOT_FOUND, found=False, sources=[], status="ok")
            answer = parsed.text
            result = {"input_tokens": 0, "output_tokens": 0, "ms": 0}
            if on_event:
                trace.data["ttft_ms"] = trace.elapsed_ms()
                emit({"type": "token", "text": NOT_FOUND})
        else:
            with trace.span("build_prompt") as s:
                prompt = build_prompt(question, chunks, version)
                s["chars"] = len(prompt)
                s["final_prompt"] = prompt

            emit({"type": "stage", "stage": "generate"})
            with trace.span("llm_call") as s:
                if on_event:
                    streamer, called = AnswerStream(json_reply=cited), time.perf_counter()

                    def on_text(piece: str) -> None:
                        text = streamer.feed(piece)
                        if text:
                            if "ttft_ms" not in trace.data:      # the first answer character the user sees
                                trace.data["ttft_ms"] = trace.elapsed_ms()
                                s["answer_ttft_ms"] = round((time.perf_counter() - called) * 1000)
                            emit({"type": "token", "text": text})

                    result = stream_completion(llm, prompt, cited, on_text)
                    s["ttft_ms"] = result["ttft_ms"]                  # the model's first token, answer or not
                else:
                    result = llm.complete(prompt, json_mode=True) if cited else llm.complete(prompt)
                s.update(model=llm.name, input_tokens=result["input_tokens"],
                         output_tokens=result["output_tokens"], temperature=result["temperature"])
                if cited:                           # the structured answer: validate it against the sources in the prompt
                    parsed = parse_answer(result["text"], source_ids(chunks))
                    s["structured"] = {"status": parsed.status, "found": parsed.found, "sources": parsed.sources,
                                       "issues": parsed.issues}
                    s["raw_reply"] = result["text"]
            answer = parsed.text if cited else result["text"]
        log.info("answered in %d ms (%s in / %s out tokens)", result["ms"], result["input_tokens"], result["output_tokens"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        log.exception("trace %s failed", trace.trace_id)
        raise
    finally:
        trace.finish(answer=answer, error=error)
        trace.save(trace_path)
    out = {"trace_id": trace.trace_id, "answer": answer, "chunks": chunks, "trace": trace.data}
    if context is not None:
        out["context"] = context
    if verdict:
        out["validation"] = verdict
    if cited:
        by_id = dict(zip(source_ids(chunks), chunks))
        out.update(found=parsed.found, answer_status=parsed.status,
                   sources=[{"source_id": sid, "chunk_id": by_id[sid]["chunk_id"]} for sid in parsed.sources])
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask a question and save a trace")
    parser.add_argument("query", help="a question, or a golden-set id such as q11")
    parser.add_argument("-k", type=int, default=TOP_K)
    args = parser.parse_args()

    question, golden = load_question(args.query)
    out = run_query(question, args.k)
    print(f"Q: {question}")
    if golden:
        print(f"expected: {golden['expected_answer']}")
    print(f"\nANSWER:\n{out['answer']}")
    print(f"\ntrace_id: {out['trace_id']}  ({out['trace']['total_ms']} ms)  ->  python -m scripts.show_trace {out['trace_id']}")
