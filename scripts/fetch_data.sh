#!/usr/bin/env bash
# Fetch the datasets used by the starter:
#
#   * BBBC002 v1  -  50 Drosophila Kc167 cell microscopy images + per-image counts
#     https://bbbc.broadinstitute.org/BBBC002  (public domain  -  Broad / BBBC)
#   * Opentrons/Protocols  -  OT-2 protocols
#     https://github.com/Opentrons/Protocols  (Apache-2.0)
#
# Outputs are written under `data/` and then processed into the final
# corpus + fine-tuning dataset by `scripts/build_dataset.py`.
#
# Idempotent  -  safe to re-run; downloads are cached under `data/.cache/`.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

CACHE="data/.cache"
mkdir -p "$CACHE"

echo "[fetch] Cache directory: $CACHE"

# --- 1. BBBC002 images + counts ----------------------------------------------
BBBC_IMG_URL="https://data.broadinstitute.org/bbbc/BBBC002/BBBC002_v1_images.zip"
BBBC_CNT_URL="https://data.broadinstitute.org/bbbc/BBBC002/BBBC002_v1_counts.txt"

if [ ! -f "$CACHE/bbbc002_images.zip" ]; then
    echo "[fetch] Downloading BBBC002 images (~3.5 MB)"
    curl -fL --retry 3 --retry-delay 2 -o "$CACHE/bbbc002_images.zip" "$BBBC_IMG_URL"
else
    echo "[fetch] BBBC002 images cached"
fi

if [ ! -f "$CACHE/bbbc002_counts.txt" ]; then
    echo "[fetch] Downloading BBBC002 counts"
    curl -fL --retry 3 --retry-delay 2 -o "$CACHE/bbbc002_counts.txt" "$BBBC_CNT_URL"
else
    echo "[fetch] BBBC002 counts cached"
fi

if [ ! -d "$CACHE/bbbc002_images" ]; then
    echo "[fetch] Extracting BBBC002 images"
    mkdir -p "$CACHE/bbbc002_images"
    unzip -q -o "$CACHE/bbbc002_images.zip" -d "$CACHE/bbbc002_images"
fi

# --- 2. Opentrons Protocols --------------------------------------------------
if [ ! -d "$CACHE/opentrons_protocols/.git" ]; then
    echo "[fetch] Cloning Opentrons/Protocols (shallow)"
    rm -rf "$CACHE/opentrons_protocols"
    git clone --depth 1 \
        https://github.com/Opentrons/Protocols.git \
        "$CACHE/opentrons_protocols"
else
    echo "[fetch] Opentrons/Protocols cached (running git pull)"
    git -C "$CACHE/opentrons_protocols" pull --quiet || true
fi

# --- 3. Build final dataset artifacts ----------------------------------------
echo "[fetch] Building data/images, data/protocols, data/reagents, data/finetune"
python3 scripts/build_dataset.py

# --- 4. Write LICENSE attribution --------------------------------------------
cat > data/DATA_SOURCES.md <<'EOF'
# Data Sources and Licenses

The datasets shipped in this starter are **not modified** from their upstream
sources other than the selection and renaming performed by
`scripts/build_dataset.py`.

## Cell microscopy images  -  `data/images/`

* **Source**: Broad Bioimage Benchmark Collection, [BBBC002 v1](https://bbbc.broadinstitute.org/BBBC002)
* **Description**: Drosophila Kc167 cells, DAPI-stained, 512×512 TIFF (converted to PNG)
* **License**: Public domain (Creative Commons CC0)
* **Reference**: Carpenter AE et al., *Genome Biology* (2006).

Ground-truth cell counts are taken verbatim from `BBBC002_v1_counts.txt`.

## Lab protocols  -  `data/protocols/`

* **Source**: [Opentrons/Protocols](https://github.com/Opentrons/Protocols) GitHub repository
* **License**: Apache License 2.0
* **Notes**: Each entry is an OpenTrons Protocol API v2 protocol from the
  lab-automation community. The upstream `README.md` description,
  labware list, pipettes, and categories are preserved, along with the file
  path so the original can be inspected.

## Reagent catalog  -  `data/reagents/catalog.csv`

Built from labware and reagent references appearing in the downloaded
OpenTrons protocols. No invented entries.

## Fine-tuning dataset  -  `data/finetune/*.jsonl`

Derived 1-to-1 from OpenTrons protocols. Each training pair maps the
protocol's own description to its own structured metadata.
EOF

echo "[fetch] Done."
ls -la data/images data/protocols data/reagents data/finetune 2>/dev/null | head -30
