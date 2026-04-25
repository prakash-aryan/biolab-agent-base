#!/usr/bin/env bash
# Pull the LLM + embedding models into the Ollama container.
#
# Models pulled (Ollama public registry):
#   - medgemma:4b              Google MedGemma 4B  -  medical LLM
#   - nomic-embed-text         768-dim general-purpose embedder
#
# Override with env vars:
#   BIOLAB_LLM_MODEL=medgemma:27b BIOLAB_EMBED_MODEL=bge-large bash scripts/pull_models.sh

set -euo pipefail

LLM_MODEL="${BIOLAB_LLM_MODEL:-medgemma:4b}"
EMBED_MODEL="${BIOLAB_EMBED_MODEL:-nomic-embed-text}"

pull() {
    local tag="$1"
    echo "[pull_models] Pulling $tag"
    if docker compose ps ollama --status=running -q | grep -q .; then
        docker compose exec -T ollama ollama pull "$tag"
    elif command -v ollama >/dev/null 2>&1; then
        OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}" ollama pull "$tag"
    else
        echo "[pull_models] ERROR: Ollama not running in compose and no local ollama binary."
        exit 1
    fi
}

pull "$LLM_MODEL"
pull "$EMBED_MODEL"

echo "[pull_models] All models pulled:"
if docker compose ps ollama --status=running -q | grep -q .; then
    docker compose exec -T ollama ollama list
fi
