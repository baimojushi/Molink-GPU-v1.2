---
name: floor-plane-ransac-rejection
description: floor_wall_estimation wall_floor_plane_intersection attempt fails with floor_plane_inlier_ratio_below_threshold when RANSAC inlier ratio < ~0.6 — typical ratio ~0.52 causes fallback to less reliable methods
source: auto-skill
extracted_at: '2026-07-14T15:36:21.953Z'
---

## Problem

`floor_wall_estimation.debug.attempts[0]` (wall_floor_plane_intersection method) is rejected with `floor_plane_inlier_ratio_below_threshold`. This forces fallback to `solid_wall_mask_bottom_percentile`, the least reliable floor estimation method.

## Root cause

RANSAC floor plane fitting requires a minimum inlier ratio (~0.6). Observed ratios of ~0.52 are below threshold, causing rejection. This happens when the floor point cloud contains significant noise or non-floor geometry (furniture, shadows) that prevents clean plane fitting.

## Symptoms

In `rejected_walls.json`, walls with `trusted_wall_floor_intersection_required` rejection contain:

```json
"floor_wall_estimation": {
  "debug": {
    "attempts": [
      {
        "source": "wall_floor_plane_intersection",
        "floor_ransac_inlier_ratio": 0.52,
        "reject_reasons": ["floor_plane_inlier_ratio_below_threshold"]
      },
      {
        "source": "floor_mask_projected_to_current_wall_plane",
        "rejected_reason": "floor_points_too_far_from_current_wall_plane",
        "signed_distance_to_wall_p50_m": 0.67,
        "max_floor_wall_projection_distance_m": 0.45
      },
      {
        "source": "solid_wall_mask_bottom_percentile"
      }
    ],
    "selected_source": "solid_wall_mask_bottom_percentile"
  }
}
```

## Attempt priority chain

1. **wall_floor_plane_intersection** — RANSAC plane fitting on floor point cloud (requires inlier_ratio >= threshold)
2. **floor_mask_projected_to_current_wall_plane** — Project floor mask points onto wall plane (requires signed_distance_p50 <= max_floor_wall_projection_distance_m, default 0.45m)
3. **solid_wall_mask_bottom_percentile** — Bottom percentile of solid wall mask (fallback, least reliable)
4. **visible_wall_bottom_percentile_fallback** — Last resort

## Fix approaches

- Lower the RANSAC inlier ratio threshold if 0.52 is acceptable for the scene geometry
- Increase `max_floor_wall_projection_distance_m` (default 0.45) if floor points are valid but too far from wall plane
- Investigate why floor point cloud quality is poor — may indicate Metric3D depth/normal issues for that specific room image
- **Critical**: When fallback to `solid_wall_mask_bottom_percentile` occurs, also check `floor_height_abs_m` sign — if negative, g_hat direction is flipped, causing artwork to hang too high. See `g-hat-direction-floor-height-negative` skill.

## Location

`hanging_mvp/support_context.py` — `_estimate_floor_wall_intersection()` function

## Related skills

- `g-hat-direction-floor-height-negative` — negative floor_height_abs_m indicates g_hat flip, causing artwork to hang too high
