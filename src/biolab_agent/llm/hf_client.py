"""In-process HuggingFace chat client with optional PEFT LoRA adapter.

Selected via ``BIOLAB_LLM_BACKEND=hf``. Honours:
    BIOLAB_LLM_MODEL       Ollama-style model tag (used by the Ollama backend)
    BIOLAB_HF_MODEL        explicit HF repo id for the HF backend; if unset,
                           we fall back to the bnb-4bit MedGemma below
    BIOLAB_LORA_ADAPTER    optional path/HF repo of a PEFT adapter
    BIOLAB_DEVICE          cuda | cpu
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

import torch

from biolab_agent.config import load_settings

log = logging.getLogger(__name__)


# Default for laptops (8 GB VRAM): pre-quantized 4-bit weights via bitsandbytes.
# Workstations with >=12 GB can override BIOLAB_HF_MODEL to the non-quantized
# `unsloth/medgemma-4b-it` to skip the bitsandbytes / CUDA wiring entirely.
_BASE_MODEL_FALLBACK = "unsloth/medgemma-4b-it-bnb-4bit"


@lru_cache(maxsize=1)
def _load_model() -> tuple[Any, Any, str]:
    """Return (model, tokenizer, device). Cached so load cost is paid once."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    settings = load_settings()
    explicit = os.getenv("BIOLAB_HF_MODEL")
    if explicit:
        model_id = explicit
    else:
        model_id = settings.biolab_llm_model
        # Ollama tags (e.g. "medgemma:4b") are not HF repo ids; fall back.
        if ":" in model_id or "/" not in model_id:
            log.warning(
                "BIOLAB_LLM_MODEL=%r is not an HF repo id; using %s for HF backend.",
                model_id,
                _BASE_MODEL_FALLBACK,
            )
            model_id = _BASE_MODEL_FALLBACK

    device = "cuda" if (settings.biolab_device == "cuda" and torch.cuda.is_available()) else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # unsloth/medgemma-4b-it-bnb-4bit is already 4-bit quantized; loading it
    # implies bitsandbytes is available at runtime.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else {"": "cpu"},
        attn_implementation="eager",
    )

    adapter = settings.biolab_lora_adapter
    if adapter:
        from peft import PeftModel

        log.info("Loading LoRA adapter from %s", adapter)
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    return model, tokenizer, device


def _messages_to_prompt(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    # Gemma3's chat template only knows user/assistant/system; fold tool
    # observations into 'user' turns before templating.
    cleaned: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "tool":
            cleaned.append({"role": "user", "content": f"[tool result]\n{content}"})
        else:
            cleaned.append({"role": role, "content": content})
    return tokenizer.apply_chat_template(
        cleaned,
        tokenize=False,
        add_generation_prompt=True,
    )


class HFChatClient:
    def __init__(self, adapter_path: str | None = None) -> None:
        # Allow callers to override the env-driven adapter choice so we can
        # keep one "base-only" client and one "adapter-on" client in memory
        # concurrently without fighting over `BIOLAB_LORA_ADAPTER`.
        self._adapter_override = adapter_path

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        opts = options or {}
        max_new_tokens = int(opts.get("num_predict", 512))
        temperature = float(opts.get("temperature", 0.1))
        top_p = float(opts.get("top_p", 0.9))

        if self._adapter_override is not None:
            # Temporarily point the env at the explicit adapter; cleared after load.
            settings = load_settings()
            settings.biolab_lora_adapter = self._adapter_override
        hf_model, tokenizer, device = _load_model()
        prompt = _messages_to_prompt(tokenizer, messages)

        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.inference_mode():
            out = hf_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        # Decode only the generated continuation, not the prompt itself.
        gen_tokens = out[0, inputs["input_ids"].shape[-1] :]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        return {"message": {"content": text}}
