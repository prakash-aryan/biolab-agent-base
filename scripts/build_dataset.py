"""Build the starter dataset from cached BBBC002 + Opentrons/Protocols.

Called by ``scripts/fetch_data.sh``. Reads from ``data/.cache/`` and writes:

    data/images/<plate>_<well>.png
    data/images/ground_truth.json
    data/protocols/opentrons.jsonl
    data/reagents/catalog.csv
    data/finetune/train.jsonl
    data/finetune/eval.jsonl

Dependencies: Pillow + numpy (starter image ships with them).
"""

from __future__ import annotations

import csv
import json
import logging
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

logging.basicConfig(format="[build] %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "data" / ".cache"
OUT_IMAGES = REPO / "data" / "images"
OUT_PROTOCOLS = REPO / "data" / "protocols"
OUT_REAGENTS = REPO / "data" / "reagents"
OUT_FINETUNE = REPO / "data" / "finetune"

# Deterministic so the starter produces the same 20 images / same split on every run.
SEED = 20260424
N_IMAGES = 20
N_PROTOCOLS = 200  # 160 for corpus/train, ~40 held out for eval
N_FINETUNE_TRAIN = 500
N_FINETUNE_EVAL = 50


# ---------------------------------------------------------------------------
# 1. Images
# ---------------------------------------------------------------------------


@dataclass
class ImageGT:
    image_id: str  # e.g. P001_A01
    source_file: str  # original BBBC filename
    cell_count: int  # BBBC-published count
    foreground_pct: float  # Otsu-threshold approximation used as a scoring proxy


def _compute_foreground_pct(img_path: Path) -> float:
    """Approximate foreground % from a DAPI image via Otsu threshold.

    Used as a scoring proxy because BBBC002 does not publish pixel masks.
    """
    # Lazy import so the build script works even in minimal envs
    import numpy as np  # type: ignore[import-not-found]
    from PIL import Image  # type: ignore[import-not-found]

    img = Image.open(img_path).convert("L")
    arr = np.asarray(img, dtype=np.uint8)

    # Otsu threshold  -  reference implementation, no external deps
    hist, _ = np.histogram(arr.ravel(), bins=256, range=(0, 256))
    total = arr.size
    sum_all = float((np.arange(256) * hist).sum())

    sumB, wB, maximum = 0.0, 0, -1.0
    threshold = 0
    for t in range(256):
        wB += int(hist[t])
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += t * int(hist[t])
        mB = sumB / wB
        mF = (sum_all - sumB) / wF
        between = wB * wF * (mB - mF) ** 2
        if between > maximum:
            maximum = between
            threshold = t

    fg = (arr > threshold).sum()
    return round(100.0 * fg / arr.size, 2)


def _parse_bbbc_counts(counts_txt: Path) -> dict[str, int]:
    """Parse BBBC002_v1_counts.txt into {image_stem: mean_count}."""
    counts: dict[str, int] = {}
    for line in counts_txt.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Format: "<image_basename> <count1> <count2> ..."
        parts = line.split()
        if len(parts) < 2:
            continue
        stem = parts[0]
        vals: list[int] = []
        for v in parts[1:]:
            try:
                vals.append(int(v))
            except ValueError:
                continue
        if vals:
            counts[stem] = round(sum(vals) / len(vals))
    return counts


def build_images() -> None:
    src_dir = CACHE / "bbbc002_images"
    counts_txt = CACHE / "bbbc002_counts.txt"
    if not src_dir.exists():
        raise RuntimeError(f"BBBC002 cache missing: {src_dir}. Run scripts/fetch_data.sh first.")

    tifs = sorted(src_dir.rglob("*.TIF")) + sorted(src_dir.rglob("*.tif"))
    if not tifs:
        raise RuntimeError(f"No TIFF images found under {src_dir}")

    counts_map = _parse_bbbc_counts(counts_txt) if counts_txt.exists() else {}

    # Deterministic selection
    rng = random.Random(SEED)
    selected = rng.sample(tifs, k=min(N_IMAGES, len(tifs)))

    OUT_IMAGES.mkdir(parents=True, exist_ok=True)

    # Stamp each as a "plate / well" id: P001_A01..A20
    from PIL import Image  # type: ignore[import-not-found]

    gts: list[ImageGT] = []
    for i, src in enumerate(selected):
        row = chr(ord("A") + i // 5)
        col = (i % 5) + 1
        image_id = f"P001_{row}{col:02d}"

        img = Image.open(src)
        dst_png = OUT_IMAGES / f"{image_id}.png"
        img.convert("L").save(dst_png, optimize=True)

        stem = src.stem
        count = counts_map.get(stem, -1)
        fg_pct = _compute_foreground_pct(dst_png)
        gts.append(
            ImageGT(
                image_id=image_id,
                source_file=src.name,
                cell_count=count,
                foreground_pct=fg_pct,
            )
        )
        log.info("image %s <- %s (count=%s, fg=%.2f%%)", image_id, src.name, count, fg_pct)

    (OUT_IMAGES / "ground_truth.json").write_text(
        json.dumps(
            {
                "source": "BBBC002 v1 (Broad Bioimage Benchmark Collection)",
                "notes": (
                    "cell_count is the published per-image mean across "
                    "annotators; foreground_pct is an Otsu-threshold proxy "
                    "computed locally  -  used only as a loose bound for "
                    "segmentation scoring, not a pixel-wise mask."
                ),
                "images": [asdict(g) for g in gts],
            },
            indent=2,
        )
    )


# ---------------------------------------------------------------------------
# 2. Protocols corpus
# ---------------------------------------------------------------------------


_LABWARE_RE = re.compile(r"^\s*\*\s+\[([^\]]+)\]\(([^)]+)\)", re.MULTILINE)
_HEADER_RE = re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE)
_DESC_RE = re.compile(r"##\s+Description\s*\n(.+?)(?:\n##|\n###|\Z)", re.DOTALL)


@dataclass
class Protocol:
    doc_id: str
    title: str
    source_url: str
    description: str
    categories: list[str]
    labware: list[str]
    pipettes: list[str]
    reagents: list[str]
    text: str  # for RAG ingestion (concatenated fields)


def _extract_section(readme: str, header: str) -> str:
    pattern = re.compile(
        rf"###\s+{re.escape(header)}\s*\n(.+?)(?:\n##|\n###|\Z)",
        re.DOTALL,
    )
    m = pattern.search(readme)
    return m.group(1).strip() if m else ""


def _extract_bulleted(readme: str, header: str) -> list[str]:
    section = _extract_section(readme, header)
    items: list[str] = []
    for line in section.splitlines():
        line = line.strip()
        m = re.match(r"^\*\s+(?:\[([^\]]+)\].*|(.+))$", line)
        if m:
            items.append((m.group(1) or m.group(2) or "").strip())
    return [x for x in items if x]


def _parse_protocol(proto_dir: Path) -> Protocol | None:
    readme = proto_dir / "README.md"
    if not readme.exists():
        return None
    text = readme.read_text(encoding="utf-8", errors="replace")

    title_match = re.match(r"^#\s+(.+?)\s*$", text, re.MULTILINE)
    if not title_match:
        return None
    title = title_match.group(1).strip()

    desc_match = _DESC_RE.search(text)
    description = desc_match.group(1).strip() if desc_match else ""

    categories = _extract_bulleted(text, "Categories")
    labware = _extract_bulleted(text, "Labware")
    pipettes = _extract_bulleted(text, "Pipettes")
    reagents = _extract_bulleted(text, "Reagents")

    combined = "\n\n".join(
        [
            f"# {title}",
            description,
            "Categories: " + ", ".join(categories),
            "Labware: " + ", ".join(labware),
            "Pipettes: " + ", ".join(pipettes),
            "Reagents: " + ", ".join(reagents),
        ]
    )

    return Protocol(
        doc_id=proto_dir.name,
        title=title,
        source_url=f"https://github.com/Opentrons/Protocols/tree/master/protocols/{proto_dir.name}",
        description=description,
        categories=categories,
        labware=labware,
        pipettes=pipettes,
        reagents=reagents,
        text=combined,
    )


def build_protocols() -> list[Protocol]:
    src = CACHE / "opentrons_protocols" / "protocols"
    if not src.exists():
        raise RuntimeError(f"Opentrons cache missing: {src}")

    all_protos: list[Protocol] = []
    for sub in sorted(src.iterdir()):
        if not sub.is_dir():
            continue
        proto = _parse_protocol(sub)
        if proto is None:
            continue
        # Skip too-short descriptions
        if len(proto.description) < 50:
            continue
        all_protos.append(proto)

    log.info("parsed %d usable OpenTrons protocols from %s", len(all_protos), src)

    rng = random.Random(SEED + 1)
    rng.shuffle(all_protos)
    selected = all_protos[:N_PROTOCOLS]

    OUT_PROTOCOLS.mkdir(parents=True, exist_ok=True)
    jsonl_path = OUT_PROTOCOLS / "opentrons.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for p in selected:
            fh.write(
                json.dumps(
                    {
                        "doc_id": p.doc_id,
                        "title": p.title,
                        "source_url": p.source_url,
                        "text": p.text,
                        "meta": {
                            "categories": p.categories,
                            "labware": p.labware,
                            "pipettes": p.pipettes,
                            "reagents": p.reagents,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    log.info("wrote %d protocols to %s", len(selected), jsonl_path)
    return selected


# ---------------------------------------------------------------------------
# 3. Reagent catalog
# ---------------------------------------------------------------------------


def build_reagents(protocols: list[Protocol]) -> None:
    """Collect every reagent/labware entry seen in the protocols into a catalog."""
    OUT_REAGENTS.mkdir(parents=True, exist_ok=True)
    entries: dict[str, dict[str, str]] = {}

    for p in protocols:
        for name in p.reagents:
            entries.setdefault(
                name,
                {
                    "name": name,
                    "cas": "",
                    "vendor": "",
                    "sku": "",
                    "concentration": "",
                    "hazard": "",
                    "notes": f"Used in protocol {p.doc_id} ({p.title})",
                },
            )
        # Include labware, since agents will lookup both
        for name in p.labware:
            entries.setdefault(
                name,
                {
                    "name": name,
                    "cas": "",
                    "vendor": "Opentrons",
                    "sku": "",
                    "concentration": "",
                    "hazard": "",
                    "notes": f"Labware referenced by protocol {p.doc_id}",
                },
            )

    catalog_path = OUT_REAGENTS / "catalog.csv"
    fields = ["name", "cas", "vendor", "sku", "concentration", "hazard", "notes"]
    with catalog_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for name in sorted(entries):
            writer.writerow(entries[name])
    log.info("wrote %d reagent/labware entries to %s", len(entries), catalog_path)


# ---------------------------------------------------------------------------
# 4. Fine-tuning dataset
# ---------------------------------------------------------------------------


_PROMPT_TEMPLATES = [
    "Design a protocol for the following experiment:\n{description}",
    "I need a structured OT-2 protocol for this task:\n{description}",
    "Given the description below, produce the protocol metadata (labware, pipettes, reagents, categories) as JSON.\n{description}",
    "Please convert this experiment brief into a structured protocol definition.\n{description}",
    "Lab assistant: extract the required labware, pipettes, and reagents from the protocol description below and return JSON.\n{description}",
]


def build_finetune(protocols: list[Protocol]) -> None:
    OUT_FINETUNE.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED + 2)

    # Need MORE training examples than len(protocols) → template-multiply each protocol
    # Each pair is REAL (same protocol  -  different prompt template). No synthesis of
    # fake protocols.
    usable = [p for p in protocols if p.labware or p.reagents or p.pipettes]
    log.info("building fine-tune pairs from %d protocols", len(usable))

    pairs: list[dict[str, object]] = []
    for p in usable:
        for template in _PROMPT_TEMPLATES:
            target = {
                "title": p.title,
                "categories": p.categories,
                "labware": p.labware,
                "pipettes": p.pipettes,
                "reagents": p.reagents,
            }
            pairs.append(
                {
                    "instruction": template.format(description=p.description),
                    "input": "",
                    "output": json.dumps(target, ensure_ascii=False, indent=2),
                    "source_protocol": p.doc_id,
                }
            )

    rng.shuffle(pairs)

    train_n = min(N_FINETUNE_TRAIN, max(0, len(pairs) - N_FINETUNE_EVAL))
    train = pairs[:train_n]
    eval_set = pairs[train_n : train_n + N_FINETUNE_EVAL]

    def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    _write_jsonl(OUT_FINETUNE / "train.jsonl", train)
    _write_jsonl(OUT_FINETUNE / "eval.jsonl", eval_set)
    log.info("fine-tune dataset: %d train, %d eval pairs", len(train), len(eval_set))


# ---------------------------------------------------------------------------


def main() -> None:
    # Images + protocols are independent; run both
    build_images()
    protocols = build_protocols()
    build_reagents(protocols)
    build_finetune(protocols)
    log.info("done.")


if __name__ == "__main__":
    main()
