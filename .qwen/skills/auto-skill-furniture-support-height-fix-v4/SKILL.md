---
name: furniture-support-height-fix-v4
description: V4 fixes furniture support detection when SAM mask is unavailable — uses 2D bbox inner-region fallback and unlabeled geometry inference for wall-adjacent counters/cabinets
source: auto-skill
extracted_at: '2026-07-17T00:22:00.000Z'
---

## Problem

`job_5b71f65a` selected `floor_center_152` (1.52m) although furniture/counter geometry existed below the artwork. The semantic JSON omitted SAM masks because the worker pipeline did not pass metric3d point clouds to semantic service.

## V4 contract

- A semantic support with a SAM mask keeps the strict current-wall association gate
- When only a processed 2D bbox is available, sample its conservative inner region and still require metric depth association with the current wall
- When no semantic furniture is associated, inspect unlabeled geometry near the candidate axis, excluding both the wall plane and distant foreground
- A dense, broad height band in the plausible furniture range may provide a generic wall-adjacent support top
- Furniture context never lowers the trusted floor default — it may only raise the center so artwork bottom keeps at least 0.15m clearance above support

## Key code changes

### `_object_sampling_mask()` — mask fallback

When `obj.get("mask")` is unavailable, create a conservative inner rectangle from `bbox_xyxy_processed`:

```python
def _object_sampling_mask(obj, shape_hw):
    mask = obj.get("mask")
    if mask is not None:
        return np.asarray(mask), "semantic_mask"
    
    bbox = obj.get("bbox_xyxy_processed")
    # inset by 12% on each side to reduce wall/floor contamination
    inset_x = max(1.0, 0.12 * (x2 - x1))
    inset_y = max(1.0, 0.12 * (y2 - y1))
    fallback = np.zeros((h, w), dtype=np.uint8)
    fallback[iy1:iy2, ix1:ix2] = 255
    return fallback, "processed_bbox_inner_fallback"
```

### `_infer_wall_adjacent_geometry_support()` — unlabeled geometry fallback

When no semantic furniture is associated, scan unlabeled point cloud for dense height bands in plausible furniture range (typically 0.6-1.2m above floor):

- Excludes wall plane itself
- Excludes distant foreground furniture (wall_distance_p20_m threshold)
- Requires broad, dense point band to avoid false positives from noise

### `_object_points_near_axis()` — now returns point source

Changed return type from `np.ndarray` to `tuple[np.ndarray, str]` so callers know whether points came from SAM mask or bbox fallback.

## How to apply

When investigating incorrect hanging heights (e.g., 1.52m floor fallback when furniture exists):

1. Check if semantic result has SAM masks for furniture objects
2. If masks are missing but `bbox_xyxy_processed` exists, V4 bbox fallback will be used
3. Check `_infer_wall_adjacent_geometry_support` debug output for unlabeled geometry detection
4. Verify furniture support was not rejected as distant foreground

## Related skills

- `furniture-support-requires-sam-mask` — V3 and earlier behavior (requires SAM mask)
- `height-override-diagnosis` — tracing height calculation through support context
