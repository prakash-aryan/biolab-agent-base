#!/usr/bin/env bash
# Run the benchmark harness against the currently configured agent.
#
# Looks at BIOLAB_AGENT_CLASS to decide which agent to instantiate; falls back
# to the stub agent if unset. The report is written to artifacts/bench_report.json.
set -euo pipefail

QUERIES="${QUERIES:-data/queries_public.yaml}"
REPORT="${REPORT:-artifacts/bench_report.json}"

mkdir -p "$(dirname "$REPORT")"

if docker compose ps app --status=running -q | grep -q .; then
    docker compose exec -T app biolab-bench --queries "$QUERIES" --report "$REPORT"
else
    echo "[bench] app container not running; running locally with uv."
    uv run biolab-bench --queries "$QUERIES" --report "$REPORT"
fi
