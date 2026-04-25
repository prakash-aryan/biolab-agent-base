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
  lab-automation community. The upstream `README.md` description, labware
  list, pipettes, and categories are preserved, along with the file path so
  the original can be inspected.

## Reagent catalog  -  `data/reagents/catalog.csv`

Built from labware and reagent references appearing in the downloaded
OpenTrons protocols.

## Fine-tuning dataset  -  `data/finetune/*.jsonl`

Derived 1-to-1 from OpenTrons protocols. Each training pair maps the
protocol's own description to its own structured metadata.
