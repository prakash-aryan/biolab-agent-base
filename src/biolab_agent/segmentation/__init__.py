"""Segmentation package (empty by design).

Add an ``efficientsam3_backend.py`` (or similar) and expose a
``segment_wells(image_id, prompt)`` callable returning
``biolab_agent.schemas.WellMasks``.

Hints:

- EfficientSAM3: https://github.com/SimonZeng7108/efficientsam3
- Grid prompts over a microplate need per-well centers derived from image
  dimensions + rows + cols + inset.
- Confluency per well = foreground_pixels / well_bbox_area * 100.
- Use RLE (pycocotools-style) when serializing masks into the agent trace
  so logs stay reasonably sized.
"""
