#!/usr/bin/env bash
# One-shot setup: brings up the stack, pulls models, fetches data.
# Safe to re-run  -  every step is idempotent.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

bold() { printf "\033[1m%s\033[0m\n" "$*"; }
log()  { printf "[setup] %s\n" "$*"; }

# --- 1. Sanity ----------------------------------------------------------------
command -v docker >/dev/null || { echo "docker is required"; exit 1; }
if ! docker compose version >/dev/null 2>&1; then
    echo "'docker compose' plugin is required (Docker 20.10+)." >&2
    exit 1
fi

# --- 2. Data ------------------------------------------------------------------
bold "==> Fetching BBBC images and OpenTrons protocols"
bash scripts/fetch_data.sh

# --- 3. Stack -----------------------------------------------------------------
bold "==> Building app image"
docker compose build app

bold "==> Starting ollama + qdrant + app"
docker compose up -d

# --- 4. Models ----------------------------------------------------------------
bold "==> Waiting for Ollama to be ready"
for _ in $(seq 1 60); do
    if docker compose exec -T ollama ollama list >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

bold "==> Pulling LLM + embedding models"
bash scripts/pull_models.sh

# --- 5. Index corpus ----------------------------------------------------------
bold "==> Indexing the protocol corpus into Qdrant"
docker compose exec -T app biolab-index || {
    log "Index step failed  -  try 'docker compose logs app' to debug."
    exit 1
}

bold "==> Done. Try:  docker compose exec app biolab-bench"
