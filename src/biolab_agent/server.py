"""FastAPI server. Endpoints:

GET  /healthz           liveness
GET  /readyz            readiness (Ollama + Qdrant reachable)
GET  /metrics           Prometheus metrics
GET  /version           package + model info
POST /ask               run the configured agent on a query
POST /segment           placeholder; wire your segmentation tool here
GET  /finetune/status   placeholder; wire your training-run status here
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from biolab_agent import __version__
from biolab_agent.agent import load_agent
from biolab_agent.agent.base import BaseAgent
from biolab_agent.config import load_settings
from biolab_agent.logging import configure_logging, get_logger
from biolab_agent.schemas import AgentResult

log = get_logger(__name__)

# --- Metrics ---
REQ_TOTAL = Counter(
    "biolab_requests_total",
    "Total requests by endpoint + status",
    ["endpoint", "status"],
)
REQ_LATENCY = Histogram(
    "biolab_request_seconds",
    "End-to-end request latency in seconds",
    ["endpoint"],
)


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    image_ids: list[str] | None = None


class ReadinessResponse(BaseModel):
    ok: bool
    ollama: bool
    qdrant: bool
    detail: dict[str, str] = Field(default_factory=dict)


# --- App lifespan ------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the agent once at process start; close it at shutdown."""
    configure_logging()
    settings = load_settings()
    agent: BaseAgent = load_agent(settings.to_agent_config())
    await agent.startup()
    app.state.agent = agent
    app.state.settings = settings
    log.info(
        "agent_loaded",
        llm=settings.biolab_llm_model,
        embed=settings.biolab_embed_model,
        device=settings.biolab_device,
        adapter=settings.biolab_lora_adapter,
    )
    try:
        yield
    finally:
        await agent.shutdown()


app = FastAPI(
    title="biolab-agent",
    version=__version__,
    summary="Autonomous lab agent starter (segmentation + RAG + LLM).",
    lifespan=lifespan,
)


# --- Endpoints ---------------------------------------------------------------


@app.get("/healthz", response_class=PlainTextResponse)
async def healthz() -> str:
    REQ_TOTAL.labels(endpoint="/healthz", status="200").inc()
    return "ok"


@app.get("/readyz", response_model=ReadinessResponse)
async def readyz() -> ReadinessResponse:
    """Check that Ollama and Qdrant are reachable from the app container."""
    settings = app.state.settings
    detail: dict[str, str] = {}
    ollama_ok = False
    qdrant_ok = False
    async with httpx.AsyncClient(timeout=2.0) as client:
        try:
            r = await client.get(f"{settings.ollama_host}/api/tags")
            ollama_ok = r.status_code == 200
            if not ollama_ok:
                detail["ollama"] = f"HTTP {r.status_code}"
        except Exception as exc:
            detail["ollama"] = str(exc)

        try:
            r = await client.get(f"{settings.qdrant_url}/readyz")
            qdrant_ok = r.status_code == 200
            if not qdrant_ok:
                detail["qdrant"] = f"HTTP {r.status_code}"
        except Exception as exc:
            detail["qdrant"] = str(exc)

    resp = ReadinessResponse(
        ok=ollama_ok and qdrant_ok,
        ollama=ollama_ok,
        qdrant=qdrant_ok,
        detail=detail,
    )
    REQ_TOTAL.labels(endpoint="/readyz", status="200" if resp.ok else "503").inc()
    return resp


@app.get("/version")
async def version() -> dict[str, str | None]:
    s = app.state.settings
    return {
        "package_version": __version__,
        "llm_model": s.biolab_llm_model,
        "embed_model": s.biolab_embed_model,
        "lora_adapter": s.biolab_lora_adapter,
        "device": s.biolab_device,
    }


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/ask", response_model=AgentResult)
async def ask(req: AskRequest) -> AgentResult:
    start = time.perf_counter()
    agent: BaseAgent = app.state.agent
    try:
        result = agent.run(req.query, image_ids=req.image_ids)
    except Exception as exc:
        REQ_TOTAL.labels(endpoint="/ask", status="500").inc()
        log.exception("agent_run_failed", query=req.query)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    elapsed = time.perf_counter() - start
    REQ_LATENCY.labels(endpoint="/ask").observe(elapsed)
    REQ_TOTAL.labels(endpoint="/ask", status="200").inc()
    return result


@app.post("/segment")
async def segment() -> JSONResponse:
    return JSONResponse(
        status_code=501,
        content={"detail": "Not implemented. Wire your segmentation tool here."},
    )


@app.get("/finetune/status")
async def finetune_status() -> JSONResponse:
    return JSONResponse(
        status_code=501,
        content={"detail": "Not implemented. Expose your training-run status here."},
    )


# --- Entrypoint --------------------------------------------------------------


def main() -> None:
    """Console entrypoint used by the ``biolab-serve`` script."""
    import uvicorn  # noqa: PLC0415  -  keep heavy import inside the entrypoint

    configure_logging()
    settings = load_settings()
    uvicorn.run(
        "biolab_agent.server:app",
        host="0.0.0.0",
        port=8000,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
