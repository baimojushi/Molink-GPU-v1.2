---
name: semantic-polygon-coordinate-space
description: Semantic polygon coordinates must stay in original image space — _remap_objects_to_processed_space must NOT scale polygon back to processed coordinates, or hanging pipeline will render boxes in wrong position
source: auto-skill
extracted_at: '2026-06-27T00:30:00.000Z'
---

## Bug pattern

In `semantic_mvp/service.py`, `_remap_objects_to_processed_space()` was calling:

```python
obj2["polygon"] = scale_polygon(poly_orig, sx, sy)
```

This scaled polygon from original image coordinates (e.g. 0..3648, 0..2736) back to processed coordinates (e.g. 0..2048, 0..1536), causing:

- Semantic wall debug images show boxes concentrated in upper-left corner
- Hanging pipeline uses wrong coordinates for mask generation and quad projection
- existing_art detection fails because bbox is misaligned with actual wall content

## Fix

Keep polygon in original image coordinates — do NOT scale it:

```python
if poly_orig:
    # Keep polygon in original image coordinates — hanging pipeline
    # works in original space and will remap if needed.
    obj2["polygon"] = poly_orig
    obj2["polygon_original"] = poly_orig
```

The `bbox_xyxy_processed` field should still be scaled (for metric3d alignment), but `polygon` must remain in original space because the hanging pipeline operates in original image coordinates.

## Root cause chain

1. VLM returns bbox in `sent_image_pixels` space (e.g. 0..2048, 0..1536)
2. `_normalize_vlm_bbox_to_original_pixels` correctly scales to original (e.g. 0..3648, 0..2736)
3. `_build_vlm_objects_in_original_space` creates `polygon` from normalized bbox — correct
4. `_remap_objects_to_processed_space` then calls `scale_polygon(poly_orig, sx, sy)` — **BUG**: this undoes the normalization, shrinking polygon back to processed space
5. Hanging pipeline reads `polygon` which is now in wrong coordinate space

## Verification

Check semantics.json output:
- `bbox_xyxy_original` should be in original pixels (values up to 3648, 2736)
- `bbox_xyxy_processed` should be in processed pixels (values up to 2048, 1536)
- `polygon` max coordinates should match `bbox_xyxy_original` scale, NOT `bbox_xyxy_processed` scale

If `polygon` max coords are ≤ processed_size, the bug is present.
