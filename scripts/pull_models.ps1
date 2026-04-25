# Windows PowerShell equivalent of scripts/pull_models.sh.
# Pulls the LLM + embedding models into the Ollama container.
#
# Usage (from the repo root, with Docker Desktop running):
#   powershell -ExecutionPolicy Bypass -File .\scripts\pull_models.ps1
#
# Override with env vars:
#   $env:BIOLAB_LLM_MODEL = "medgemma:27b"
#   .\scripts\pull_models.ps1

$ErrorActionPreference = "Stop"

$LlmModel   = if ($env:BIOLAB_LLM_MODEL)   { $env:BIOLAB_LLM_MODEL }   else { "medgemma:4b" }
$EmbedModel = if ($env:BIOLAB_EMBED_MODEL) { $env:BIOLAB_EMBED_MODEL } else { "nomic-embed-text" }

function Pull-Model([string]$Tag) {
    Write-Host "[pull_models] Pulling $Tag"
    $running = docker compose ps ollama --status=running -q 2>$null
    if ($running) {
        docker compose exec -T ollama ollama pull $Tag
    }
    else {
        $localOllama = Get-Command ollama -ErrorAction SilentlyContinue
        if ($null -ne $localOllama) {
            if (-not $env:OLLAMA_HOST) { $env:OLLAMA_HOST = "http://localhost:11434" }
            ollama pull $Tag
        }
        else {
            Write-Error "Ollama is not running in compose and no local 'ollama' binary was found. Start the stack with 'docker compose up -d' first."
        }
    }
}

Pull-Model $LlmModel
Pull-Model $EmbedModel

Write-Host "[pull_models] All models pulled:"
$running = docker compose ps ollama --status=running -q 2>$null
if ($running) {
    docker compose exec -T ollama ollama list
}
