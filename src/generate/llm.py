"""LLM clients. Swap the model by changing LLM_PROVIDER / LLM_MODEL in config.

To add a provider (Groq, ...), write a class with `name` and `complete()`,
then register it in PROVIDERS.
"""
import json
import os
import time
from pathlib import Path
from typing import Protocol

import requests

from src.config import LLM_CONTEXT_WINDOW, LLM_MODEL, LLM_PROVIDER, LLM_TEMPERATURE, OLLAMA_URL

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class LLM(Protocol):
    name: str

    def complete(self, prompt: str, json_mode: bool = False) -> dict:
        """Return {text, input_tokens, output_tokens, ms, temperature}."""


def stream_completion(llm, prompt: str, json_mode: bool, on_text) -> dict:
    """Call the model and pass each piece of text to `on_text` as it arrives. Returns what complete() returns, plus
    ttft_ms: the time from the call to the first piece of text. A client without stream() answers in one piece."""
    if hasattr(llm, "stream"):
        return llm.stream(prompt, json_mode, on_text)
    result = llm.complete(prompt, json_mode=json_mode)
    on_text(result["text"])
    return {**result, "ttft_ms": result["ms"]}


class OllamaLLM:
    def __init__(self, name: str, temperature: float = LLM_TEMPERATURE, url: str = OLLAMA_URL):
        self.name = name
        self.temperature = temperature
        self.url = url

    def complete(self, prompt: str, json_mode: bool = False) -> dict:
        start = time.perf_counter()
        payload = {
            "model": self.name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": self.temperature, "num_ctx": LLM_CONTEXT_WINDOW},
        }
        if json_mode:
            payload["format"] = "json"
        resp = requests.post(f"{self.url}/api/chat", json=payload, timeout=300)
        resp.raise_for_status()
        body = resp.json()
        return {
            "text": body["message"]["content"].strip(),
            "input_tokens": body.get("prompt_eval_count"),
            "output_tokens": body.get("eval_count"),
            "ms": round((time.perf_counter() - start) * 1000),
            "temperature": self.temperature,
        }

    def stream(self, prompt: str, json_mode: bool, on_text) -> dict:
        """complete() with each piece of text passed to `on_text` as Ollama writes it (one JSON object per line)."""
        start, first, parts, last = time.perf_counter(), None, [], {}
        payload = {"model": self.name, "messages": [{"role": "user", "content": prompt}], "stream": True,
                   "options": {"temperature": self.temperature, "num_ctx": LLM_CONTEXT_WINDOW}}
        if json_mode:
            payload["format"] = "json"
        with requests.post(f"{self.url}/api/chat", json=payload, timeout=300, stream=True) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                last = json.loads(line)
                piece = (last.get("message") or {}).get("content") or ""
                if piece:
                    first = first or time.perf_counter()
                    parts.append(piece)
                    on_text(piece)
        end = time.perf_counter()
        return {"text": "".join(parts).strip(), "input_tokens": last.get("prompt_eval_count"),
                "output_tokens": last.get("eval_count"), "ms": round((end - start) * 1000),
                "ttft_ms": round(((first or end) - start) * 1000), "temperature": self.temperature}


def _load_env() -> None:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")


def api_key_available(provider: str) -> bool:
    """True when the provider needs no key, or its key is set in the environment / .env."""
    if provider != "openai":
        return True
    _load_env()
    return bool(os.getenv("OPENAI_API_KEY"))


class OpenAILLM:
    def __init__(self, name: str, temperature: float = 0):
        _load_env()
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set. Put it in a .env file in the project root.")
        from openai import OpenAI

        self.name = name
        self.temperature = temperature
        self._client = OpenAI()

    def complete(self, prompt: str, json_mode: bool = False) -> dict:
        start = time.perf_counter()
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        resp = self._client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            **kwargs,
        )
        return {
            "text": (resp.choices[0].message.content or "").strip(),
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
            "ms": round((time.perf_counter() - start) * 1000),
            "temperature": self.temperature,
        }

    def stream(self, prompt: str, json_mode: bool, on_text) -> dict:
        """complete() with each piece of text passed to `on_text` as the model writes it. Token counts come in the last
        chunk (stream_options include_usage)."""
        start, first, parts, usage = time.perf_counter(), None, [], None
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        stream = self._client.chat.completions.create(
            model=self.name, messages=[{"role": "user", "content": prompt}], temperature=self.temperature,
            stream=True, stream_options={"include_usage": True}, **kwargs)
        for chunk in stream:
            if chunk.usage:
                usage = chunk.usage
            piece = chunk.choices[0].delta.content if chunk.choices else None
            if piece:
                first = first or time.perf_counter()
                parts.append(piece)
                on_text(piece)
        end = time.perf_counter()
        return {"text": "".join(parts).strip(), "input_tokens": usage.prompt_tokens if usage else None,
                "output_tokens": usage.completion_tokens if usage else None, "ms": round((end - start) * 1000),
                "ttft_ms": round(((first or end) - start) * 1000), "temperature": self.temperature}


PROVIDERS = {"ollama": OllamaLLM, "openai": OpenAILLM}


def load_llm(provider: str = LLM_PROVIDER, name: str = LLM_MODEL) -> LLM:
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown LLM provider '{provider}'. Choose from {list(PROVIDERS)}")
    return PROVIDERS[provider](name)
