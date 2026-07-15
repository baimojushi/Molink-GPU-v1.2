---
name: g-hat-direction-floor-height-negative
description: floor_height_abs_m negative indicates g_hat direction flip — gravity vector points down instead of up, causing artwork center t coordinate to be systematically elevated (hung too high)
source: auto-skill
extracted_at: '2026-07-15T21:04:38.304Z'
---

# Negative floor_height_abs_m from g_hat Direction Error

## Problem

Artwork hangs noticeably higher than intended on the wall. Debug output shows `floor_height_abs_m` is **negative** (e.g., -1.34m, -1.82m), meaning the estimated floor point is above the camera origin in gravity space.

## Root cause chain

1. `g_hat` (camera-up gravity vector) direction is not explicitly enforced in `estimate_g_hat_from_floor_normals()`
2. The mean floor normal is returned as-is, but Metric3D floor normals can point in either direction along the surface normal line
3. `floor_height_abs_m = dot(floor_point_on_wall, g_hat)` — if g_hat points **down** (positive camera-y) instead of **up** (negative camera-y), the dot product flips sign
4. Negative `floor_height_abs_m` causes `center_height_above_floor_m = center_height_abs_m - floor_height_abs_m` to be **larger than intended**
5. Artwork center t coordinate is pushed upward on the wall

## Symptoms

In `hanging_plan.json` or `rejected_walls.json`:

```json
"floor_wall_estimation": {
  "source": "solid_wall_mask_bottom_percentile",
  "t_floor_m": -0.918,
  "floor_height_abs_m": -1.342,
  "floor_point_on_wall_m": [0.053, 0.788, 4.341]
}
```

Key indicator: `floor_height_abs_m < 0` when the floor should be below the camera (typical room photography).

## Diagnosis steps

1. Check `floor_height_abs_m` in rejected_walls.json or hanging_plan.json — if negative, g_hat direction is suspect
2. Trace gravity source: check `gravity_source` in service logs — `metric3d_global_floor_like_normals` fallback is more prone to direction errors than semantic floor masks
3. Verify `g_hat` y-component: should be **negative** in camera coordinates (camera-y positive = downward in Metric3D convention)
4. If RANSAC floor plane failed (inlier_ratio < 0.60), the fallback to `solid_wall_mask_bottom_percentile` compounds the problem — the t_floor value is an approximation, and combined with wrong g_hat direction, produces large height errors

## Fix approaches

### Immediate: Enforce g_hat direction (CONFIRMED FIX)

In `estimate_g_hat_from_floor_normals()` (geometry.py ~L108), after computing mean normal:

```python
# Enforce g_hat points toward camera-up (negative camera-y)
if g_hat[1] > 0:
    g_hat = -g_hat
```

Same fix needed in:
- `estimate_g_hat_from_horizontal_reference_normals()` — already does sign alignment for roof/table
- `estimate_g_hat_from_global_floor_like_normals()` — currently missing explicit flip

**Why this works:** `estimate_g_hat_from_horizontal_reference_normals()` (L168) already flips normals toward `default_up = [0, -1, 0]` before averaging. The floor path lacks this step. Adding the flip makes both paths consistent.

### Secondary: Lower RANSAC rejection threshold

If `floor_ransac_inlier_ratio` hovers around 0.52 (observed), consider lowering `min_floor_ransac_inlier_ratio` from 0.60 to 0.50 to accept more wall_floor_plane_intersection results, which are more reliable than solid_wall_mask fallback.

### Tertiary: Add floor_height sanity check

In `_estimate_floor_wall_intersection()` (support_context.py), add a guard:

```python
if floor_height_abs_m > 0.0:
    # Floor should be below camera in typical room shots
    # Flag as suspicious — may indicate g_hat direction error
    logger.warning(f"Suspicious floor_height_abs_m={floor_height_abs_m:.2f} > 0, possible g_hat flip")
```

## Verified data (2026-07-15, order b355cfb8)

| Wall | floor_ransac_inlier | floor_point_proj_dist | source | t_floor_m | floor_height_abs_m |
|---|---|---|---|---|---|
| wall_front_01_00 | 0.29 (REJECTED) | 1.93m (REJECTED) | solid_wall_mask | -0.918 | **-1.342** |
| wall_right_01_02 | 0.29 (REJECTED) | 0.79m (REJECTED) | solid_wall_mask | -1.193 | **-1.816** |

Both walls rejected RANSAC (inlier 0.29 < 0.60) and floor projection (distance > 0.45m). Fallback to `solid_wall_mask_bottom_percentile` produced negative `floor_height_abs_m`, confirming g_hat direction error.

## Affected code

- `hanging_mvp/geometry.py` — `estimate_g_hat_from_floor_normals()`, `estimate_g_hat_from_global_floor_like_normals()`
- `hanging_mvp/support_context.py` — `_estimate_floor_wall_intersection()`, `_finish()`
- `hanging_mvp/service.py` — gravity estimation block (~L1312-1400), candidate height calculation (~L2305-2351)

## Related skills

- `floor-plane-ransac-rejection` — RANSAC failure forces fallback to less reliable methods
- `gravity-reference-fallback-chain` — full gravity estimation priority chain
