---
name: photometric-lock-removal
description: photometric_lock (wall_rgb_correction + adaptive relight + texture pasting) has been fully removed — render output from GPT Image is always used directly; QA runs for diagnostics only (project)
source: auto-skill
extracted_at: '2026-06-27T00:45:00.000Z'
---

## Rule

The `photometric_lock` post-processing pipeline has been **fully removed**. Render output from GPT Image (`03_flux2_render.png`) is always used directly as the final output.

## Background

Previous flow:
1. GPT Image renders the scene
2. `_apply_final_photometric_lock` crops the locked texture, repositions it, and pastes it back
3. Adaptive relight adjusts artwork brightness to match environment
4. Wall RGB correction aligns wall colors

Problems:
- Cropping and pasting back caused misalignment artifacts
- `wall_rgb_only` mode skipped adaptive relight entirely, leaving artwork brightness mismatched with environment
- QA failures triggered photometric_lock which often made things worse

## Current flow

1. GPT Image renders the scene → `03_flux2_render.png`
2. QA runs for diagnostics only (logs SSIM, forbidden outer guard, etc.)
3. Render output is used as-is — no post-processing

## wall_rgb_only mode

If `MOLINK_FINAL_POSTPROCESS_MODE=wall_rgb_only` is still configured, the pipeline applies:
1. Wall RGB correction (aligns wall colors between hard-paste and render)
2. Adaptive relight with anchor relocation (finds actual rendered artwork position, then applies brightness compensation to that region)

## Env vars no longer effective

These env vars are now ignored:
- `MOLINK_FINAL_PHOTOMETRIC_LOCK`
- `MOLINK_FINAL_POSTPROCESS_MODE`
- `MOLINK_FINAL_PHOTOMETRIC_LOCK_MODE`
- `MOLINK_FULLSPACE_FINAL_POSTPROCESS_MODE`
- `MOLINK_FINAL_PHOTOMETRIC_LOCK_BLUR_PX`
- `MOLINK_FINAL_PHOTOMETRIC_LOCK_MAX_DELTA`
- `MOLINK_FINAL_PHOTOMETRIC_LOCK_STRENGTH`
- `MOLINK_FINAL_PHOTOMETRIC_LOCK_FEATHER_PX`
- `MOLINK_FINAL_RELOCATION_SEARCH_PX` (still used in wall_rgb_only relight)
- `MOLINK_FINAL_OBJECT_RELIGHT_MODE` (still used in wall_rgb_only relight)
- `MOLINK_FINAL_OBJECT_BASE_MAX_BRIGHTEN` (still used in wall_rgb_only relight)
- `MOLINK_FINAL_SELF_GLOW_MARGIN` (still used in wall_rgb_only relight)
