"""Minimal chat-client protocol + factory."""

from __future__ import annotations

import os
from typing import Any, Protocol


class ChatClient(Protocol):
    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a dict of shape ``{"message": {"content": "<text>"}}``."""
        ...


def get_client() -> ChatClient:
    backend = os.getenv("BIOLAB_LLM_BACKEND", "ollama").lower()
    if backend == "hf":
        from biolab_agent.llm.hf_client import HFChatClient

        return HFChatClient()
    if backend == "ollama":
        from biolab_agent.llm.ollama_client import OllamaChatClient

        return OllamaChatClient()
    raise ValueError(f"Unknown BIOLAB_LLM_BACKEND={backend!r}. Use 'ollama' or 'hf'.")
