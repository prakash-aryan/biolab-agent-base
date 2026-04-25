"""SAM-based segmentation backend using the ``mask-generation`` pipeline.

Uses ``facebook/sam-vit-base`` as a stand-in for EfficientSAM3. The
transformers mask-generation pipeline grids prompts over the image and
runs NMS, producing per-cell masks suitable for counting dense
fluorescence microscopy fields.

Swap in EfficientSAM3 by reimplementing ``_mask_generator`` with weights
from ``Simon7108528/EfficientSAM3``  -  keep the WellMasks return shape.
"""

from __future__ import annotations

import functools
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from biolab_agent.config import load_settings
from biolab_agent.schemas import WellMask, WellMasks

_DEFAULT_MODEL = "facebook/sam-vit-base"


@functools.lru_cache(maxsize=1)
def _mask_generator(model_id: str = _DEFAULT_MODEL):  # type: ignore[no-untyped-def]
    from transformers import pipeline as hf_pipeline

    settings = load_settings()
    device = 0 if (settings.biolab_device == "cuda" and torch.cuda.is_available()) else -1
    # fp32 throughout: the mask-generation pipeline's NMS step crashes with
    # "dets should have the same type as scores" when the model runs in fp16.
    return hf_pipeline(
        "mask-generation",
        model=model_id,
        device=device,
        torch_dtype=torch.float32,
    )


def _encode_rle(mask: np.ndarray) -> str:
    flat = mask.astype(np.uint8).ravel()
    if flat.size == 0:
        return ""
    runs: list[int] = []
    cur = 0
    run_len = 0
    for v in flat:
        if int(v) == cur:
            run_len += 1
        else:
            runs.append(run_len)
            cur = int(v)
            run_len = 1
    runs.append(run_len)
    return ",".join(str(r) for r in runs)


# Cell-sized filter: (min_px, max_px). Derived from BBBC002 cell scales at
# 40× magnification  -  typical nuclei cover 200-3000 pixels at 512×512.
_CELL_MIN_AREA = 120
_CELL_MAX_AREA = 6000


def segment_wells_sam(image_id: str, prompt: dict[str, Any] | None = None) -> WellMasks:
    start = time.perf_counter()
    settings = load_settings()
    img_path = Path(settings.biolab_data_dir) / "images" / f"{image_id}.png"
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    gen = _mask_generator()
    # `points_per_side=16` → 256 prompts; NMS in the pipeline drops overlaps.
    result = gen(
        img,
        points_per_side=16,
        points_per_batch=64,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.85,
    )

    masks = result.get("masks", [])
    # Filter to cell-sized components, union for confluency.
    kept: list[np.ndarray] = []
    union = np.zeros((h, w), dtype=bool)
    for m in masks:
        arr = np.asarray(m, dtype=bool)
        area = int(arr.sum())
        if _CELL_MIN_AREA <= area <= _CELL_MAX_AREA:
            kept.append(arr)
            union |= arr

    area = int(union.sum())
    confluency_pct = round(100.0 * area / max(1, h * w), 2)
    cell_count = len(kept)

    prompt_kind = "grid"
    if isinstance(prompt, dict) and prompt.get("kind") in ("point", "box", "grid"):
        prompt_kind = str(prompt["kind"])

    well = WellMask(
        well_id=image_id,
        rle=_encode_rle(union),
        height=h,
        width=w,
        bbox=(0, 0, w, h),
        area_px=area,
        confluency_pct=confluency_pct,
        confidence=0.85,
        cell_count=cell_count,
    )
    return WellMasks(
        image_id=image_id,
        prompt_kind=prompt_kind,
        masks=[well],
        elapsed_ms=round((time.perf_counter() - start) * 1000.0, 2),
    )
