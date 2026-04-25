"""LLM chat clients used by the agent.

Two backends implement the same minimal contract (``chat(messages, options) -> {"message": {"content": str}}``):

- :mod:`biolab_agent.llm.ollama_client`  -  default, talks to a running Ollama.
- :mod:`biolab_agent.llm.hf_client`  -  in-process HuggingFace + PEFT, used when
  a LoRA adapter needs to be applied on top of MedGemma-4B.

Selected at runtime via the ``BIOLAB_LLM_BACKEND`` env var (``ollama`` | ``hf``).
"""

from biolab_agent.llm.base import ChatClient, get_client

__all__ = ["ChatClient", "get_client"]
