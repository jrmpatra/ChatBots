# llm_hf_traced.py
import time
import os
from typing import Any, Dict
from tracer import LocalTracer
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

# --- configure model (same as your snippet) ---
model = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
llm = ChatHuggingFace(llm=model, temperature=1.1)

# --- tracer setup ---
tracer = LocalTracer(metadata={"integration": "langchain_hf", "model": "Llama-3.1-8B-Instruct"})

def call_llm_and_trace(prompt: str) -> str:
    """
    Calls the provided llm.invoke(...) and records a trace with start/end events.
    Returns the textual content (response) for the caller.
    """
    run_id = tracer.start_run("llm.invoke", {"prompt": prompt, "model": "Llama-3.1-8B-Instruct"})
    start_ts = time.time()
    try:
        # call the LLM (this returns whatever ChatHuggingFace.invoke returns)
        resp = llm.invoke(prompt)

        elapsed = time.time() - start_ts

        # normalize the response into a textual content and raw payload
        # many langchain wrappers put result text in `.content` or `.text` or return dicts
        resp_text = None
        resp_raw = None

        if hasattr(resp, "content"):
            resp_text = resp.content
            resp_raw = None
        elif isinstance(resp, dict):
            # try common keys
            resp_text = resp.get("content") or resp.get("text") or str(resp)
            resp_raw = resp
        else:
            resp_text = str(resp)
            resp_raw = None

        tracer.end_run(
            run_id,
            {
                "response": resp_text,
                "raw_response": resp_raw,
                "elapsed_s": elapsed
            },
            success=True
        )
        return resp_text
    except Exception as e:
        elapsed = time.time() - start_ts
        tracer.end_run(
            run_id,
            {"error": str(e), "elapsed_s": elapsed},
            success=False
        )
        # re-raise so caller sees the failure too
        raise

# --- small CLI demo when run directly ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", "-p", type=str, default="Hello, world!", help="Prompt to send to the LLM")
    args = parser.parse_args()

    print("Calling LLM (this will also write a trace into traces.jsonl)...")
    try:
        out = call_llm_and_trace(args.prompt)
        print("\n=== LLM Response ===\n")
        print(out)
        print("\nTrace saved. Open your viewer (python viewer.py) at http://localhost:8080 to inspect the trace.")
    except Exception as exc:
        print("LLM call failed:", exc)
        print("A failed run was also recorded in traces.jsonl.")
