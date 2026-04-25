"""Gradio UI for the biolab agent.

Launches on ``0.0.0.0:7860``. Expects the stack (Ollama + Qdrant) to be
reachable via the same env vars the server uses.

Run::

    python ui/app.py
    # or inside the container:
    docker compose exec app python ui/app.py
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr
import yaml
from eval.harness import Task, grade

from biolab_agent.agent import load_agent
from biolab_agent.config import load_settings
from biolab_agent.logging import configure_logging
from biolab_agent.segmentation.visualize import render_segmentation_overlay

configure_logging()
log = logging.getLogger("biolab.ui")

QUERIES_PATH = Path("data/queries_public.yaml")
NO_TASK = "(none  -  free-form query)"

_agent = None


def _get_agent():
    global _agent  # noqa: PLW0603  -  UI process holds a single agent instance
    if _agent is None:
        _agent = load_agent()
    return _agent


def _list_images() -> list[str]:
    settings = load_settings()
    img_dir = Path(settings.biolab_data_dir) / "images"
    if not img_dir.exists():
        return []
    return sorted(p.stem for p in img_dir.glob("*.png"))


def _resolve_image_paths(image_ids: list[str]) -> list[str]:
    settings = load_settings()
    img_dir = Path(settings.biolab_data_dir) / "images"
    return [str(img_dir / f"{i}.png") for i in image_ids if (img_dir / f"{i}.png").exists()]


def _load_tasks() -> dict[str, dict[str, Any]]:
    """Map task_id → raw YAML entry. Empty dict if the file is missing."""
    if not QUERIES_PATH.exists():
        return {}
    doc = yaml.safe_load(QUERIES_PATH.read_text(encoding="utf-8"))
    return {q["id"]: q for q in doc.get("queries", [])}


def _task_to_obj(entry: dict[str, Any]) -> Task:
    return Task(
        id=entry["id"],
        kind=entry["kind"],
        query=entry["query"],
        image_ids=entry.get("image_ids", []),
        scoring=entry.get("scoring", {}),
        pass_threshold=float(entry.get("pass_threshold", 0.7)),
        weight=float(entry.get("weight", 1.0)),
    )


def _build_segmentation_overlays(
    result,
    settings,
    only_image_ids: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Render segmentation overlays from the agent trace.

    If ``only_image_ids`` is provided, restrict the gallery to those IDs;
    otherwise emit every successful segment_wells call.
    """
    img_dir = Path(settings.biolab_data_dir) / "images"
    out_dir = Path(tempfile.gettempdir()) / "biolab_overlays"
    out_dir.mkdir(exist_ok=True)
    allow = set(only_image_ids) if only_image_ids else None
    items: list[tuple[str, str]] = []
    for step in result.trace:
        if step.tool != "segment_wells" or not step.ok or not isinstance(step.observation, dict):
            continue
        image_id = step.args.get("image_id") or step.observation.get("image_id")
        if not image_id:
            continue
        if allow is not None and image_id not in allow:
            continue
        img_path = img_dir / f"{image_id}.png"
        if not img_path.exists():
            continue
        for mask in step.observation.get("masks", []):
            rle = mask.get("rle") or ""
            try:
                overlay = render_segmentation_overlay(
                    img_path, rle, int(mask["height"]), int(mask["width"])
                )
            except Exception:
                continue
            cells = mask.get("cell_count", "?")
            confl = mask.get("confluency_pct", "?")
            out_path = out_dir / f"{image_id}_step{step.step}.png"
            overlay.save(out_path)
            label = f"{image_id}  -  cells={cells}, confluency={confl}%"
            items.append((str(out_path), label))
    return items


def _status_banner(result, has_error: bool) -> str:
    """Compact summary header above the result panels."""
    n_calls = len(result.trace)
    n_ok = sum(1 for t in result.trace if t.ok)
    n_err = n_calls - n_ok
    n_cites = len(result.citations)
    has_struct = bool(result.structured)
    elapsed_s = result.elapsed_ms / 1000.0

    if has_error or (n_calls == 0 and not result.answer):
        emoji, color = "⚠️", "#a33"
    elif n_err == 0:
        emoji, color = "✅", "#2a7"
    else:
        emoji, color = "🟡", "#b80"
    parts = [
        f"{emoji} done in {elapsed_s:.1f} s",
        f"{n_calls} tool call{'s' if n_calls != 1 else ''} ({n_ok} ok / {n_err} err)",
        f"{n_cites} citation{'s' if n_cites != 1 else ''}",
        ("structured ✓" if has_struct else "no structured"),
    ]
    return f"<div style='color:{color};font-weight:600'>{' · '.join(parts)}</div>"


def _score_banner(task_id: str, result) -> str:
    """Run the harness scoring for a single task and return a PASS/FAIL panel."""
    if task_id == NO_TASK or not task_id:
        return "<div style='color:#777'>No benchmark task selected.</div>"
    tasks = _load_tasks()
    entry = tasks.get(task_id)
    if entry is None:
        return f"<div style='color:#a33'>Task {task_id!r} not found in queries_public.yaml</div>"
    task = _task_to_obj(entry)
    score, details = grade(task, result)
    passed = score >= task.pass_threshold
    badge = "PASS" if passed else "FAIL"
    color = "#2a7" if passed else "#a33"
    summary = (
        f"<div style='color:{color};font-weight:700;font-size:1.1em'>"
        f"{badge}  -  score {score:.2f} (threshold {task.pass_threshold:.2f}, "
        f"weight {task.weight}, kind={task.kind})</div>"
    )
    detail_block = (
        "<details><summary>scoring details</summary>"
        f"<pre>{json.dumps(details, indent=2)[:3000]}</pre>"
        "</details>"
    )
    return summary + detail_block


def _on_task_change(task_id: str):  # type: ignore[no-untyped-def]
    """When the user picks a benchmark task, autofill query + image_ids."""
    if task_id == NO_TASK or not task_id:
        return gr.update(), gr.update()
    tasks = _load_tasks()
    entry = tasks.get(task_id)
    if entry is None:
        return gr.update(), gr.update()
    return gr.update(value=entry["query"]), gr.update(value=entry.get("image_ids", []))


def _run(
    query: str,
    image_ids: list[str],
    show_all_overlays: bool,
    task_id: str,
):  # type: ignore[no-untyped-def]
    if not query.strip():
        return (
            "<div style='color:#a33'>Enter a query (or pick a benchmark task).</div>",
            "Enter a query.",
            "",
            "",
            [],
            [],
            "",
        )

    agent = _get_agent()
    try:
        result = agent.run(query=query, image_ids=image_ids or None)
        had_error = False
    except Exception as exc:
        log.exception("agent_run_failed")
        return (
            f"<div style='color:#a33;font-weight:600'>⚠️ Agent crashed: {exc}</div>",
            f"(crashed: {exc})",
            "",
            "",
            [],
            [],
            _score_banner(task_id, _empty_result(query)),
        )
    settings = load_settings()

    structured_str = json.dumps(result.structured, indent=2) if result.structured else "(none)"
    trace_rows = [
        f"{i + 1}. {t.tool}({json.dumps(t.args)[:120]})  "
        f"→ {'OK' if t.ok else 'ERR: ' + (t.error or '')}  "
        f"({t.elapsed_ms:.0f} ms)"
        for i, t in enumerate(result.trace)
    ]
    trace_str = "\n".join(trace_rows) or "(no tool calls)"
    cites_str = "\n".join(f"- {doc}:{chunk}" for doc, chunk in result.citations) or "(no citations)"
    overlays = _build_segmentation_overlays(
        result,
        settings,
        only_image_ids=None if show_all_overlays else (image_ids or None),
    )
    return (
        _status_banner(result, has_error=had_error),
        result.answer,
        structured_str,
        f"{trace_str}\n\nCitations:\n{cites_str}",
        _resolve_image_paths(image_ids),
        overlays,
        _score_banner(task_id, result),
    )


def _empty_result(query: str):
    """Placeholder AgentResult-shaped dict for the score banner on a crash."""
    from biolab_agent.schemas import AgentResult

    return AgentResult(query=query, answer="", model="", elapsed_ms=0.0)


def _build_interface() -> gr.Blocks:
    image_choices = _list_images()
    task_choices = [NO_TASK, *sorted(_load_tasks().keys())]

    with gr.Blocks(title="Biolab Agent") as ui:
        gr.Markdown("# Biolab Agent")
        gr.Markdown("Pick a benchmark task to autofill query + images, or write a free-form query.")

        with gr.Row():
            task_picker = gr.Dropdown(
                choices=task_choices,
                value=NO_TASK,
                label="Benchmark task (optional  -  autofills query and shows PASS/FAIL)",
            )

        with gr.Row():
            query = gr.Textbox(
                label="Query",
                lines=4,
                placeholder="e.g. Count cells in wells P001_A01..A05",
            )
        with gr.Row():
            images = gr.Dropdown(
                choices=image_choices,
                multiselect=True,
                label="Image IDs (optional)",
                value=[],
            )
        with gr.Row():
            show_all = gr.Checkbox(
                label="Show overlays for every image the agent segments "
                "(not just the ones I picked)",
                value=False,
            )
        with gr.Row():
            submit = gr.Button("Run agent", variant="primary")

        status = gr.Markdown(value="")
        score_panel = gr.Markdown(value="")

        with gr.Row():
            gallery = gr.Gallery(label="Selected images", columns=5, rows=1, height=220)
        with gr.Row():
            seg_gallery = gr.Gallery(
                label="Segmentation overlays (orange = mask, label = count + confluency)",
                columns=5,
                rows=1,
                height=240,
            )

        with gr.Row():
            answer = gr.Markdown(label="Answer", value="")
        with gr.Row():
            structured = gr.Code(label="Structured output", language="json", value="")
        with gr.Row():
            trace = gr.Textbox(label="Trace + citations", lines=12, interactive=False)

        task_picker.change(_on_task_change, inputs=[task_picker], outputs=[query, images])

        submit.click(
            _run,
            inputs=[query, images, show_all, task_picker],
            outputs=[
                status,
                answer,
                structured,
                trace,
                gallery,
                seg_gallery,
                score_panel,
            ],
        )
    return ui


def main() -> None:
    host = os.getenv("UI_HOST", "0.0.0.0")
    port = int(os.getenv("UI_PORT", "7860"))
    ui = _build_interface()
    ui.launch(server_name=host, server_port=port, show_error=True)


if __name__ == "__main__":
    main()
