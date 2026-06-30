---
name: vlm-dynamic-size-inference
description: VLM models may internally resize images differently from sent_size — use dynamic coordinate range inference to detect actual pixel space
source: auto-skill
extracted_at: '2026-06-26T14:28:32.259Z'
---

## Problem

VLM models (e.g. qwen3-vl-flash) may internally resize images to a different size than what the client sends. For example:
- Original image: 4608×3456
- Client sends: 2048×1536 (`sent_size`)
- VLM internally processes: 1536×1152

If VLM returns bbox coordinates in the 1536×1152 space but code assumes 2048×1536, the scale factor is wrong (2.25x vs actual 3.0x), causing all wall coordinates, masks, and quads to be misaligned.

## Detection Logic (in `semantic_mvp/service.py`)

The `_infer_vlm_bbox_coordinate_space()` function now includes a dynamic inference guard after all static guards fail:

```python
# If normalized_1000_like fails and boxes don't match sent_size,
# infer VLM's actual internal pixel size from bbox coordinate ranges
inferred_vlm_w = max(1, int(round(max_x)))
inferred_vlm_h = max(1, int(round(max_y)))

# Check if coords match original aspect ratio and are between 1000 and original
aspect_match = abs(aspect_inferred - aspect_orig) < 0.10
coords_between = max_coord > 1024.0 and max_coord < min(original_w, original_h) * 0.95

if aspect_match and coords_between:
    # VLM used an internal pixel space
    scale_x = original_w / inferred_vlm_w
    scale_y = original_h / inferred_vlm_h
    return 'sent_image_pixels', debug_with_inferred_size
```

## How It Works

1. Extract max coordinate values from all VLM bbox coordinates
2. Compute inferred VLM internal size as (round(max_x), round(max_y))
3. Check if inferred aspect ratio matches original image within 10%
4. Check if max coord is between 1024 and 95% of min(original dimensions)
5. If both conditions met: compute actual scale from inferred size to original

## When This Triggers

- VLM internally resizes images (common for efficiency)
- sent_size differs from VLM's actual processing size
- Coordinates don't match 0..1000 normalized grid
- Coordinates don't match sent_size pixel space

## Debug Output

Look for `reason: dynamic_inferred_vlm_internal_pixel_space_<W>x<H>` in `vlm_bbox_coordinate_debug` to confirm this path was used.
