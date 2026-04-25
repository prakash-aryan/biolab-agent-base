"""Fallback agent used when ``BIOLAB_AGENT_CLASS`` is unset.

Lets the harness, CI, and server boot end-to-end before a real
implementation is wired in.
"""

from __future__ import annotations

import time

from biolab_agent.agent.base import BaseAgent
from biolab_agent.schemas import AgentResult


class StubAgent(BaseAgent):
    def run(
        self,
        query: str,
        image_ids: list[str] | None = None,
    ) -> AgentResult:
        start = time.perf_counter()
        answer = (
            "StubAgent: no implementation is configured. "
            "Set BIOLAB_AGENT_CLASS to your agent class, "
            "e.g. 'biolab_agent.agent.solution:MyAgent'."
        )
        return AgentResult(
            query=query,
            answer=answer,
            structured=None,
            trace=[],
            model=self.config.llm_model,
            adapter=self.config.lora_adapter,
            elapsed_ms=(time.perf_counter() - start) * 1000.0,
            citations=[],
        )
