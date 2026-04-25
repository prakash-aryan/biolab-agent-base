"""Typed I/O contracts between the agent, its tools, and the eval harness."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    # extra="forbid" makes the harness fail fast if a field is renamed
    # instead of populated.
    model_config = ConfigDict(extra="forbid")


class PointPrompt(_StrictModel):
    kind: Literal["point"] = "point"
    x: Annotated[float, Field(ge=0)]
    y: Annotated[float, Field(ge=0)]
    label: Literal["foreground", "background"] = "foreground"


class BoxPrompt(_StrictModel):
    kind: Literal["box"] = "box"
    x0: Annotated[float, Field(ge=0)]
    y0: Annotated[float, Field(ge=0)]
    x1: Annotated[float, Field(gt=0)]
    y1: Annotated[float, Field(gt=0)]


class GridPrompt(_StrictModel):
    """Regular plate grid  -  rows x cols of wells inferred from image size."""

    kind: Literal["grid"] = "grid"
    rows: Annotated[int, Field(ge=1, le=32)]
    cols: Annotated[int, Field(ge=1, le=48)]
    inset_px: Annotated[int, Field(ge=0)] = 0


Prompt = PointPrompt | BoxPrompt | GridPrompt


class WellMask(_StrictModel):
    well_id: str
    # COCO-style RLE so agent traces stay readable when logged.
    rle: str
    height: int
    width: int
    bbox: tuple[int, int, int, int]
    area_px: int
    confluency_pct: Annotated[float, Field(ge=0.0, le=100.0)]
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    # Cell count derived from connected components in the mask. Optional
    # because not every backend / image type exposes a meaningful count.
    cell_count: int | None = None


class WellMasks(_StrictModel):
    image_id: str
    prompt_kind: Literal["point", "box", "grid"]
    masks: list[WellMask]
    elapsed_ms: Annotated[float, Field(ge=0)]


class ProtocolHit(_StrictModel):
    doc_id: str
    chunk_id: str
    title: str
    source_url: str | None = None
    text: str
    score: Annotated[float, Field(ge=0.0)]


class ReagentRecord(_StrictModel):
    name: str
    cas: str | None = None
    vendor: str | None = None
    sku: str | None = None
    concentration: str | None = None
    hazard: str | None = None
    notes: str | None = None


class ToolTrace(_StrictModel):
    step: Annotated[int, Field(ge=0)]
    tool: str
    args: dict[str, Any]
    ok: bool
    observation: Any = None
    error: str | None = None
    elapsed_ms: Annotated[float, Field(ge=0)] = 0.0


class AgentResult(_StrictModel):
    query: str
    answer: str
    structured: dict[str, Any] | None = None
    trace: list[ToolTrace] = Field(default_factory=list)
    model: str
    adapter: str | None = None
    elapsed_ms: Annotated[float, Field(ge=0)]
    # (doc_id, chunk_id) pairs that grounded the answer.
    citations: list[tuple[str, str]] = Field(default_factory=list)
