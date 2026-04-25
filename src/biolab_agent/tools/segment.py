"""segment_wells tool.

Single backend: promptable SAM (``facebook/sam-vit-base``) via Transformers.
It exposes the same point/box/grid prompt interface that EfficientSAM3 uses,
so the code in :mod:`biolab_agent.segmentation.sam_backend` can be swapped
for an EfficientSAM3 wrapper without touching this file.

How to use EfficientSAM3 instead:

    git clone https://github.com/SimonZeng7108/efficientsam3 /opt/efficientsam3
    huggingface-cli download Simon7108528/EfficientSAM3 --local-dir /opt/esam3
    # Re-implement ``segment_wells_sam`` in sam_backend.py using the repo's
    # ``stage1/inference`` utilities; keep the WellMasks return shape.
"""

from __future__ import annotations

from typing import Any

from biolab_agent.schemas import WellMasks


def segment_wells(image_id: str, prompt: dict[str, Any] | None = None) -> WellMasks:
    # Import lazily so pure-Python tooling (tests, linters) doesn't pay for
    # torch + HF model loading just to inspect the module.
    from biolab_agent.segmentation.sam_backend import segment_wells_sam

    return segment_wells_sam(image_id, prompt)


segment_wells_spec = {
    "type": "function",
    "function": {
        "name": "segment_wells",
        "description": (
            "Segment a microscopy image of a well and return foreground masks "
            "plus a confluency percentage (0-100) and a cell count (connected "
            "components). One mask per well. For a single-well field-of-view, "
            "call with prompt {'kind':'grid','rows':1,'cols':1}. Use the "
            "returned `confluency_pct` and `cell_count` verbatim as the "
            "measurement to report."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_id": {
                    "type": "string",
                    "description": "ID of the image, e.g. 'P001_A01'.",
                },
                "prompt": {
                    "type": "object",
                    "description": "Segmentation prompt (point/box/grid).",
                    "properties": {
                        "kind": {"type": "string", "enum": ["point", "box", "grid"]},
                        "rows": {"type": "integer"},
                        "cols": {"type": "integer"},
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "x0": {"type": "number"},
                        "y0": {"type": "number"},
                        "x1": {"type": "number"},
                        "y1": {"type": "number"},
                    },
                },
            },
            "required": ["image_id"],
        },
    },
}
