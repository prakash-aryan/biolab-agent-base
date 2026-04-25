"""Resolve ``BIOLAB_AGENT_CLASS`` to a concrete ``BaseAgent`` subclass."""

from __future__ import annotations

import importlib
import logging
import os
from typing import cast

from biolab_agent.agent.base import AgentConfig, BaseAgent
from biolab_agent.config import load_settings

logger = logging.getLogger(__name__)

DEFAULT_AGENT_ENV = "BIOLAB_AGENT_CLASS"
STUB_TARGET = "biolab_agent.agent.stub:StubAgent"


def load_agent(config: AgentConfig | None = None) -> BaseAgent:
    target = os.getenv(DEFAULT_AGENT_ENV, STUB_TARGET)
    if ":" not in target:
        raise ValueError(f"{DEFAULT_AGENT_ENV} must be 'module.path:ClassName' (got {target!r})")
    module_path, _, class_name = target.partition(":")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(f"Could not import agent module {module_path!r}: {exc}") from exc

    cls = getattr(module, class_name, None)
    if cls is None:
        raise AttributeError(f"Module {module_path!r} has no attribute {class_name!r}")
    if not isinstance(cls, type) or not issubclass(cls, BaseAgent):
        raise TypeError(f"{target} does not subclass biolab_agent.agent.base.BaseAgent")

    effective_config = config or load_settings().to_agent_config()
    logger.info(
        "Loaded agent %s with model=%s device=%s",
        target,
        effective_config.llm_model,
        effective_config.device,
    )
    return cast(BaseAgent, cls(effective_config))
