"""biolab_agent  -  starter package for an autonomous lab agent.

Downstream code subclasses ``BaseAgent`` from ``biolab_agent.agent.base``
and implements ``run``.
"""

from biolab_agent.agent.base import AgentResult, BaseAgent
from biolab_agent.schemas import (
    BoxPrompt,
    GridPrompt,
    PointPrompt,
    ProtocolHit,
    ReagentRecord,
    ToolTrace,
    WellMasks,
)

__version__ = "0.1.0"

__all__ = [
    "AgentResult",
    "BaseAgent",
    "BoxPrompt",
    "GridPrompt",
    "PointPrompt",
    "ProtocolHit",
    "ReagentRecord",
    "ToolTrace",
    "WellMasks",
    "__version__",
]
