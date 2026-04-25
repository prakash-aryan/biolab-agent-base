"""BaseAgent  -  the contract concrete agents implement.

The harness resolves the concrete class at runtime via the
``BIOLAB_AGENT_CLASS`` env var (``module.path:ClassName``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from biolab_agent.schemas import AgentResult

__all__ = ["AgentConfig", "AgentResult", "BaseAgent"]


@dataclass(frozen=True, slots=True)
class AgentConfig:
    llm_model: str
    embed_model: str
    ollama_host: str
    qdrant_url: str
    data_dir: str
    artifact_dir: str
    device: str = "cuda"
    lora_adapter: str | None = None


class BaseAgent(ABC):
    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    # Startup/shutdown are awaited by the FastAPI server and by the harness
    # before the first query; no-op defaults let simple agents skip them.
    async def startup(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    @abstractmethod
    def run(
        self,
        query: str,
        image_ids: list[str] | None = None,
    ) -> AgentResult:
        """Process one query. ``image_ids`` resolve under ``config.data_dir/images``."""
        raise NotImplementedError
