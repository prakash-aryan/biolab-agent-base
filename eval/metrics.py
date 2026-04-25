"""Scoring functions for the benchmark harness.

Each function takes an :class:`~biolab_agent.schemas.AgentResult` plus
configuration and returns ``(score, details)``.
"""

from __future__ import annotations

import json
import re
from typing import Any

from biolab_agent.schemas import AgentResult


def confluency_score(
    result: AgentResult,
    target: dict[str, float],
    tolerance_pct: float,
) -> tuple[float, dict[str, Any]]:
    """Score ``result.structured["confluency"]`` against ``target``.

    Pass a well when the reported confluency is within ``tolerance_pct``
    absolute points of the target. Score = hits / total.
    """
    if result.structured is None or "confluency" not in result.structured:
        return 0.0, {
            "error": "result.structured['confluency'] missing",
            "target": target,
        }
    reported = result.structured["confluency"]
    if not isinstance(reported, dict):
        return 0.0, {
            "error": "confluency must be a dict[str, float]",
            "got": type(reported).__name__,
        }

    hits = 0
    details: dict[str, dict[str, float | bool]] = {}
    for well, gt in target.items():
        pred_raw = reported.get(well)
        try:
            pred = float(pred_raw) if pred_raw is not None else None
        except (TypeError, ValueError):
            pred = None
        passed = pred is not None and abs(pred - gt) <= tolerance_pct
        details[well] = {
            "target_pct": gt,
            "predicted_pct": pred if pred is not None else -1.0,
            "passed": passed,
        }
        hits += int(passed)

    score = hits / max(1, len(target))
    return score, {
        "per_well": details,
        "tolerance_pct": tolerance_pct,
        "hits": hits,
        "total": len(target),
    }


def cell_count_score(
    result: AgentResult,
    target: dict[str, int],
    rel_tolerance: float = 0.30,
    abs_tolerance: int = 10,
) -> tuple[float, dict[str, Any]]:
    """Score ``result.structured["cell_count"]`` against BBBC ground-truth counts.

    Per-well passes when the predicted count is within either ``rel_tolerance``
    (fractional, e.g. 0.30 = 30 %) OR ``abs_tolerance`` (cells) of target  -
    whichever is looser, so we don't punish tiny targets for small absolute
    errors nor large targets for tiny relative errors.
    """
    if result.structured is None or "cell_count" not in result.structured:
        return 0.0, {
            "error": "result.structured['cell_count'] missing",
            "target": target,
        }
    reported = result.structured["cell_count"]
    if not isinstance(reported, dict):
        return 0.0, {"error": "cell_count must be a dict[str, int]", "got": type(reported).__name__}

    hits = 0
    details: dict[str, dict[str, float | bool | int]] = {}
    for well, gt in target.items():
        raw = reported.get(well)
        try:
            pred = int(raw) if raw is not None else None
        except (TypeError, ValueError):
            pred = None
        if pred is None:
            passed = False
        else:
            err = abs(pred - gt)
            rel = err / max(1, gt)
            passed = rel <= rel_tolerance or err <= abs_tolerance
        details[well] = {
            "target": gt,
            "predicted": pred if pred is not None else -1,
            "passed": bool(passed),
        }
        hits += int(passed)

    score = hits / max(1, len(target))
    return score, {
        "per_well": details,
        "rel_tolerance": rel_tolerance,
        "abs_tolerance": abs_tolerance,
        "hits": hits,
        "total": len(target),
    }


def retrieval_score(
    result: AgentResult,
    expected_doc_ids: list[str],
    require_all: bool = False,
) -> tuple[float, dict[str, Any]]:
    """Score how many expected ``doc_id``s appear in ``result.citations``."""
    cited = {doc_id for doc_id, _chunk in result.citations}
    found = [d for d in expected_doc_ids if d in cited]
    if not expected_doc_ids:
        return 1.0, {"cited": list(cited), "expected": [], "found": []}

    coverage = len(found) / len(expected_doc_ids)
    if require_all and coverage < 1.0:
        coverage = 0.0
    return coverage, {
        "cited": list(cited),
        "expected": expected_doc_ids,
        "found": found,
        "require_all": require_all,
    }


_REQUIRED_STRUCTURED_FIELDS = ("title", "labware", "pipettes", "reagents")


def structured_protocol_score(
    result: AgentResult,
    min_labware: int = 1,
    min_pipettes: int = 1,
    min_reagents: int = 0,
) -> tuple[float, dict[str, Any]]:
    """Check that ``result.structured`` has the required fields and minimum counts."""
    structured = result.structured
    if not isinstance(structured, dict):
        return 0.0, {"error": "result.structured is not a dict"}

    checks: dict[str, bool] = {}
    for field in _REQUIRED_STRUCTURED_FIELDS:
        checks[f"has_{field}"] = field in structured

    labware = structured.get("labware") or []
    pipettes = structured.get("pipettes") or []
    reagents = structured.get("reagents") or []
    checks["labware_count_ok"] = isinstance(labware, list) and len(labware) >= min_labware
    checks["pipettes_count_ok"] = isinstance(pipettes, list) and len(pipettes) >= min_pipettes
    checks["reagents_count_ok"] = isinstance(reagents, list) and len(reagents) >= min_reagents

    score = sum(checks.values()) / len(checks)
    return score, {
        "checks": checks,
        "thresholds": {
            "min_labware": min_labware,
            "min_pipettes": min_pipettes,
            "min_reagents": min_reagents,
        },
    }


def answer_contains_score(
    result: AgentResult,
    expected_substrings: list[str],
    case_sensitive: bool = False,
) -> tuple[float, dict[str, Any]]:
    haystack = result.answer if case_sensitive else result.answer.lower()
    needles = expected_substrings if case_sensitive else [s.lower() for s in expected_substrings]
    hits = [s for s in needles if s and s in haystack]
    score = len(hits) / max(1, len(needles))
    return score, {
        "expected": expected_substrings,
        "found": hits,
        "case_sensitive": case_sensitive,
    }


def tool_order_score(
    result: AgentResult,
    required_tools: list[str],
    ordered: bool = False,
) -> tuple[float, dict[str, Any]]:
    """Check the agent invoked the required tools; honor ordering if asked."""
    called = [t.tool for t in result.trace if t.ok]
    if not required_tools:
        return 1.0, {"called": called, "required": []}

    if ordered:
        # Longest prefix of required_tools that appears as a subsequence of `called`.
        i = 0
        for t in called:
            if i < len(required_tools) and t == required_tools[i]:
                i += 1
        score = i / len(required_tools)
    else:
        present = [t for t in required_tools if t in called]
        score = len(present) / len(required_tools)

    return score, {"called": called, "required": required_tools, "ordered": ordered}


def extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort: pull a JSON object out of free-form text.

    Used when the agent inlines JSON in ``answer`` rather than populating
    ``structured``. Populating ``structured`` is still expected.
    """
    match = re.search(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
