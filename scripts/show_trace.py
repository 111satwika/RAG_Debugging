"""Pretty-print a saved trace: config, timings, retrieved chunks, final prompt, answer.

Usage:  python -m scripts.show_trace <trace_id>
        python -m scripts.show_trace            (latest trace)
        python -m scripts.show_trace --no-prompt
"""
import argparse

from src.observability.tracer import load_trace


def show(trace: dict, with_prompt: bool = True) -> None:
    spans = trace["spans"]
    print(f"TRACE {trace['trace_id']}  {trace['timestamp']}  total {trace['total_ms']} ms")
    print(f"QUERY: {trace['query']}")
    print("CONFIG:", ", ".join(f"{k}={v}" for k, v in trace["config"].items()))
    if trace["error"]:
        print(f"ERROR: {trace['error']}")

    print("\nTIMINGS (ms):", "  ".join(f"{name}={s['ms']}" for name, s in spans.items()))

    if "retrieve" in spans:
        print("\nRETRIEVED CHUNKS:")
        for rank, r in enumerate(spans["retrieve"].get("results", []), start=1):
            preview = r["preview"].replace("\n", " ")
            print(f"  #{rank} score={r['score']:.3f}  {r['chunk_id']}  (page {r['page']})\n      {preview}...")

    if "llm_call" in spans:
        llm = spans["llm_call"]
        print(f"\nLLM: {llm.get('model')}  in={llm.get('input_tokens')} out={llm.get('output_tokens')} "
              f"temperature={llm.get('temperature')}")

    if with_prompt and "build_prompt" in spans:
        prompt = spans["build_prompt"]["final_prompt"]
        print(f"\n---------- FINAL PROMPT ({len(prompt)} chars) ----------\n{prompt}\n----------")

    print(f"\nANSWER:\n{trace['answer']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_id", nargs="?", help="trace id (default: latest)")
    parser.add_argument("--no-prompt", action="store_true", help="hide the final prompt")
    args = parser.parse_args()
    show(load_trace(args.trace_id), with_prompt=not args.no_prompt)
