---
name: vlm-coordinate-space-detection
description: Diagnose and fix VLM bounding box coordinate space mis-detection when 0..999 normalized grid is mistaken for original pixel coordinates on high-resolution images
source: auto-skill
extracted_at: '2026-06-25T14:16:53.048Z'
---

## Symptom

`/analyze_hanging_mvp` returns `no_safe_wall` with zero candidates. All walls rejected at `insufficient_safe_wall_support` with `solid_wall_ratio = 0.0`. Input image is high-resolution (e.g., 3648×2736 from phone camera).

## Root cause

`_infer_vlm_bbox_coordinate_space()` in `semantic_mvp/service.py` fails to detect that VLM returned boxes in **0..999 normalized grid** when the original image is tall (>980px).

The detection logic requires one of two guards to fire:
- `y_exceeds_original`: `max_y > original_h * 1.03 AND original_h < 980` — fails for tall images
- `looks_1000_tall`: `original_h < 900 AND max_y >= 900` — fails for tall images

Both guards were designed for short images. For a 3648×2736 image with max_coord=999:
- `normalized_1000_like = True` (coords fit 0..1024)
- Both guards = False (2736 > 980, 2736 > 900)
- Falls through to `original_pixels` — **wrong**

Result: wall bboxes at [120, 43, 325, 208] are treated as pixels on a 3648×2736 image, covering only ~3% of the area. Wall masks become tiny → zero solid coverage → all walls rejected.

## Diagnosis

```bash
# 1. Find the run directory
ls -lt /root/autodl-fs/hanging_runs/ | head -5

# 2. Check semantic output for coordinate space
python3 -c "
import json
with open('<run_dir>/03_semantic/semantics.json') as f:
    d = json.load(f)
vcd = d.get('vlm_bbox_coordinate_debug', {})
print('source_coordinate_space:', vcd.get('source_coordinate_space'))
print('raw_max_x:', vcd.get('raw_max_x'))
print('raw_max_y:', vcd.get('raw_max_y'))
print('original_size:', vcd.get('original_size'))
print('_vlm_image_meta present:', '_vlm_image_meta' in str(d.get('objects', '')))
"
```

Key indicators of this bug:
- `source_coordinate_space = "original_pixels"` but `raw_max_x <= 1000` and `raw_max_y <= 1000`
- `original_size` has a dimension > 1500px
- No `_vlm_image_meta` in output (pre-patch code)
- Wall polygon bboxes all have max values < 1600 on a large image

## Fix

Add a ratio-based guard to `_infer_vlm_bbox_coordinate_space()` that fires when max_coord is far smaller than image dimensions, regardless of height:

```python
# Add after the normalized_1000_like check, before the final fallthrough:
ratio_threshold = 0.5  # max_coord should be at least 50% of min dimension if truly in pixel space
max_coord_fits_image = max_coord >= min(float(original_w), float(original_h)) * ratio_threshold
if normalized_1000_like and not max_coord_fits_image:
    debug.update({
        'source_coordinate_space': 'normalized_0_1000',
        'reason': 'max_coord_too_small_for_pixel_space_on_high_res_image',
        'scale_x_to_original': float(original_w) / 1000.0,
        'scale_y_to_original': float(original_h) / 1000.0,
    })
    return 'normalized_0_1000', debug
```

## Why the coordinate_and_max_rect_fix_v2 patch does not fully fix this

The patch adds `high_res_image_1000_grid` which checks:
```python
(float(original_w) > 1400.0 or float(original_h) > 1400.0) and (sent_w > 1100 or sent_h > 1100)
```

This only catches the **sent-image-pixels** path (when VLM client resized before sending). It does **not** catch the case where the VLM provider returns 0..999 normalized coordinates regardless of the sent image size — which is what many VLM providers do.

## Immediate workaround (no code change)

For affected runs, the VLM prompt currently says: `"Coordinate system: pixels of the image attached in this request."`

If the VLM provider always returns 0..999 regardless of image size, the prompt should explicitly ask for 0..1000 coordinates and the detection should trust that when max_coord ≤ 1000 and image is large.
