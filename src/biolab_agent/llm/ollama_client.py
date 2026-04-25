"""Thin wrapper around the Ollama REST API."""

from __future__ import annotations

from typing import Any

from ollama import Client

from biolab_agent.config import load_settings


class OllamaChatClient:
    def __init__(self) -> None:
        settings = load_settings()
        self._client = Client(host=settings.ollama_host)

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        opts = dict(options or {})
        # Pull optional top-level knobs out of `options` so callers don't need
        # Ollama-specific plumbing. `format="json"` forces valid JSON output,
        # which is what the agent loop relies on (Gemma otherwise likes to
        # emit ```tool_code fences with Python-style calls).
        fmt = opts.pop("format", "json")
        keep_alive = opts.pop("keep_alive", None)
        extra: dict[str, Any] = {}
        if keep_alive is not None:
            extra["keep_alive"] = keep_alive
        return self._client.chat(
            model=model,
            messages=messages,
            options=opts,
            format=fmt,
            **extra,
        )
