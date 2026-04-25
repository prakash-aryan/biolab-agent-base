"""Environment sanity checks.

Run with::

    pytest -q tests/test_env.py

The ``gpu``, ``ollama`` and ``qdrant`` marks make it easy to skip tests that
require external services::

    pytest -q tests/test_env.py -m 'not gpu and not ollama and not qdrant'
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


# ---------------------------------------------------------------------------
# Package imports  -  cheap, always run
# ---------------------------------------------------------------------------


def test_package_importable() -> None:
    pkg = importlib.import_module("biolab_agent")
    assert pkg.__version__
    assert hasattr(pkg, "BaseAgent")
    assert hasattr(pkg, "AgentResult")


def test_schemas_roundtrip() -> None:
    from biolab_agent.schemas import AgentResult

    result = AgentResult(
        query="hello",
        answer="world",
        structured={"confluency": {"P001_A01": 4.5}},
        trace=[],
        model="medgemma:4b",
        elapsed_ms=1.0,
    )
    data = json.loads(result.model_dump_json())
    restored = AgentResult.model_validate(data)
    assert restored.query == "hello"
    assert restored.structured == {"confluency": {"P001_A01": 4.5}}


# ---------------------------------------------------------------------------
# Data artifacts  -  require scripts/fetch_data.sh to have been run
# ---------------------------------------------------------------------------


@pytest.fixture
def ground_truth() -> dict:
    path = DATA_DIR / "images" / "ground_truth.json"
    if not path.exists():
        pytest.skip(f"Ground truth missing: {path}. Run scripts/fetch_data.sh.")
    return json.loads(path.read_text())


def test_ground_truth_has_20_images(ground_truth: dict) -> None:
    assert len(ground_truth["images"]) == 20
    ids = [img["image_id"] for img in ground_truth["images"]]
    assert all(i.startswith("P001_") for i in ids)


def test_ground_truth_files_exist(ground_truth: dict) -> None:
    for img in ground_truth["images"]:
        path = DATA_DIR / "images" / f"{img['image_id']}.png"
        assert path.exists(), f"Missing image file: {path}"
        assert path.stat().st_size > 1000, f"Image is suspiciously small: {path}"


def test_protocols_corpus() -> None:
    path = DATA_DIR / "protocols" / "opentrons.jsonl"
    if not path.exists():
        pytest.skip("Protocol corpus missing  -  run scripts/fetch_data.sh")
    docs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(docs) >= 50, f"Corpus too small: {len(docs)}"
    required_fields = {"doc_id", "title", "source_url", "text"}
    for d in docs:
        assert required_fields.issubset(d.keys()), f"Bad doc: {d.get('doc_id')}"


def test_reagent_catalog() -> None:
    import csv

    path = DATA_DIR / "reagents" / "catalog.csv"
    if not path.exists():
        pytest.skip("Reagent catalog missing  -  run scripts/fetch_data.sh")
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) >= 50
    assert {"name", "cas", "vendor", "sku", "concentration", "hazard", "notes"}.issubset(
        rows[0].keys()
    )


def test_finetune_splits() -> None:
    for name, min_rows in [("train.jsonl", 200), ("eval.jsonl", 20)]:
        path = DATA_DIR / "finetune" / name
        if not path.exists():
            pytest.skip(f"Fine-tune split missing ({name})  -  run scripts/fetch_data.sh")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert len(rows) >= min_rows, f"{name} has only {len(rows)} rows"
        assert {"instruction", "output", "source_protocol"}.issubset(rows[0].keys())


def test_queries_public_valid() -> None:
    import yaml

    path = DATA_DIR / "queries_public.yaml"
    doc = yaml.safe_load(path.read_text())
    assert doc["version"] == 1
    queries = doc["queries"]
    assert len(queries) >= 5
    kinds = {q["kind"] for q in queries}
    assert kinds.issubset(
        {
            "confluency",
            "cell_count",
            "rag_retrieval",
            "structured_protocol",
            "answer_contains",
            "tool_order",
            "composite",
        }
    )


# ---------------------------------------------------------------------------
# Stub agent  -  proves the loader path works without external services
# ---------------------------------------------------------------------------


def test_stub_agent_loads_and_runs() -> None:
    # Clear any agent class set by the surrounding shell.
    os.environ.pop("BIOLAB_AGENT_CLASS", None)
    from biolab_agent.agent import load_agent

    agent = load_agent()
    result = agent.run("hello", image_ids=None)
    assert result.query == "hello"
    assert "StubAgent" in result.answer
    assert result.model  # propagated from config


def test_harness_runs_with_stub_agent(tmp_path: Path) -> None:
    """Harness must survive any result shape  -  including the stub's empties."""
    os.environ.pop("BIOLAB_AGENT_CLASS", None)
    # Clear the cached settings so harness sees our env overrides
    from biolab_agent.config import load_settings

    load_settings.cache_clear()

    from eval.harness import run_benchmark

    queries = DATA_DIR / "queries_public.yaml"
    if not queries.exists():
        pytest.skip("queries_public.yaml not yet materialized")
    report = run_benchmark(queries)
    assert report["total"] >= 5
    # Stub returns no tool calls / no structured content  -  score should be low but not crash
    assert 0.0 <= report["overall_score"] <= 1.0
    assert all("task_id" in row for row in report["per_task"])


# ---------------------------------------------------------------------------
# External services  -  marked so they can be skipped in CI
# ---------------------------------------------------------------------------


@pytest.mark.ollama
def test_ollama_reachable() -> None:
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    try:
        r = httpx.get(f"{host}/api/tags", timeout=3.0)
    except httpx.HTTPError as exc:
        pytest.skip(f"Ollama not reachable at {host}: {exc}")
    assert r.status_code == 200, r.text


@pytest.mark.qdrant
def test_qdrant_reachable() -> None:
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    try:
        r = httpx.get(f"{url}/readyz", timeout=3.0)
    except httpx.HTTPError as exc:
        pytest.skip(f"Qdrant not reachable at {url}: {exc}")
    assert r.status_code == 200, r.text


@pytest.mark.gpu
def test_cuda_available() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    assert torch.cuda.device_count() >= 1
    # Minimal real op to prove it actually runs
    x = torch.randn(8, 8, device="cuda")
    y = x @ x.T
    assert y.shape == (8, 8)
    assert y.device.type == "cuda"
