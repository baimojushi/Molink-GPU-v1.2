---
name: gravity-reference-fallback-chain
description: Hanging gravity estimation uses a fallback chain: semantic floor → roof → table → Metric3D global floor-like normals; roof/ceiling labels are canonicalized from ceiling/ceilings
source: auto-skill
extracted_at: '2026-06-26T01:31:28.930Z'
---

# Gravity Reference Fallback Chain

## Problem

Camera-up (g_hat) estimation previously relied solely on semantic floor masks. When no floor is visible (e.g., camera pointed upward at ceiling, or floor occluded by furniture), gravity estimation failed, causing incorrect wall basis and hanging positions.

## Solution

A fallback chain now tries multiple horizontal reference surfaces in priority order:

1. **Semantic floor mask** — first floor object, uses `estimate_g_hat_from_floor_normals`
2. **Semantic roof/ceiling mask** — union of all roof objects, uses `estimate_g_hat_from_horizontal_reference_normals` (sign-aligned normals)
3. **Semantic table mask** — union of all table objects, same horizontal reference logic
4. **Metric3D global floor-like fallback** — scans entire normal map for floor-like normals (y < -0.75, |z| < 0.45), prioritizing lower image region

## Key details

- **Normal sign resolution**: Roof/ceiling normals may point in either direction. The horizontal reference function flips normals toward camera-up (negative camera-y) before averaging.
- **Floor normals need explicit flip too**: `estimate_g_hat_from_floor_normals()` does NOT enforce g_hat direction — floor normals can point either way. If g_hat[1] > 0 (points down), floor_height_abs_m becomes negative, causing artwork to hang too high. Fix: `if g_hat[1] > 0: g_hat = -g_hat`.
- **Label canonicalization**: `ceiling`, `ceilings`, `roof ceiling` → `roof` (in both `hanging_mvp/masks.py` and `semantic_mvp/service.py`)
- **VLM prompt**: System prompt now requires `label=roof, role=support` for every visible ceiling region
- **Height anchor unchanged**: Floor mask is still the height anchor; roof/table are gravity references only, never replace floor for height estimation

## When to apply

- Debugging incorrect hanging height/position when floor is not visible in the input image
- Adding new horizontal reference surfaces (e.g., shelves, countertops) to the fallback chain
- Tuning `abs_y_min`, `abs_z_max`, `min_pixels` thresholds for normal filtering
