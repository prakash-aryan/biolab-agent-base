# Windows PowerShell equivalent of scripts/setup.sh.
# Brings up the stack, pulls models, fetches data.
#
# Usage (from the repo root, with Docker Desktop running):
#   powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
#
# Note: Fetching BBBC + OpenTrons data requires a Python installation on the
# host. Most Windows installs of Docker Desktop do not include Python, so we
# recommend installing it from https://www.python.org/ first, or running the
# data-fetching step inside the container (see README → "Windows users").

$ErrorActionPreference = "Stop"

function Info([string]$Msg) { Write-Host "[setup] $Msg" -ForegroundColor Cyan }

# --- 1. Sanity ---------------------------------------------------------------
Get-Command docker -ErrorAction Stop | Out-Null
docker compose version | Out-Null

# --- 2. Data -----------------------------------------------------------------
Info "Fetching BBBC images and OpenTrons protocols"
$py = Get-Command python -ErrorAction SilentlyContinue
if ($null -eq $py) { $py = Get-Command python3 -ErrorAction SilentlyContinue }

if ($null -eq $py) {
    Write-Warning "Python not found on host. Running data fetch inside the container instead."
    docker compose build app | Out-Host
    docker compose up -d ollama qdrant
    docker compose run --rm -v "${PWD}:/work" -w /work app bash scripts/fetch_data.sh
}
else {
    bash scripts/fetch_data.sh
}

# --- 3. Stack ----------------------------------------------------------------
Info "Building app image"
docker compose build app

Info "Starting ollama + qdrant + app"
docker compose up -d

# --- 4. Models ---------------------------------------------------------------
Info "Waiting for Ollama to be ready"
$ready = $false
for ($i = 0; $i -lt 60; $i++) {
    try {
        docker compose exec -T ollama ollama list | Out-Null
        $ready = $true
        break
    }
    catch { Start-Sleep -Seconds 1 }
}
if (-not $ready) { Write-Error "Ollama did not come up in time" }

Info "Pulling LLM + embedding models"
& "$PSScriptRoot\pull_models.ps1"

# --- 5. Index corpus ---------------------------------------------------------
Info "Indexing the protocol corpus into Qdrant"
docker compose exec -T app biolab-index

Info "Done. Try:  docker compose exec app biolab-bench"
