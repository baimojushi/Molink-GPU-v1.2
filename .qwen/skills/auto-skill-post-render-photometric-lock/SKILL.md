---
name: post-render-photometric-lock
description: Apply a post-render photometric lock step to preserve provider shadows while locking artwork identity to the hard-paste source (project)
source: auto-skill
extracted_at: '2026-06-24T22:30:00.000Z'
---

## Context

Generative models (FLUX.2, GPT Image 2 VIP) tend to drift artwork intrinsic colors,
restyle frames, or hallucinate content inside the painted area. A post-render
photometric lock step runs after the provider produces its render to prevent drift
while preserving contact/projection shadows the model generated outside the artwork.

## Pattern

```
provider render → photometric lock → final output
```

The lock step:
1. Takes hard-paste (source of truth), provider render, object mask, and optional edit mask.
2. Computes a low-frequency achromatic luma delta between hard-paste and render.
3. Applies that delta to the hard-paste — only brightness/lightness changes, no color shift.
4. Composites: object pixels come from adjusted hard-paste; shadow pixels come from render; background comes from original hard-paste.

## Implementation

**Low-frequency luma delta** (`_low_frequency_luma_delta_image`):
- Gaussian blur both images at `blur_px` radius.
- Convert to luma using Rec. 709 weights: `[0.2126, 0.7152, 0.0722]`.
- Delta = gen_luma - orig_luma, clamped to `max_delta`, scaled by `strength`.
- Result is a single-channel achromatic adjustment, not RGB color shift.

**Shadow preservation** (`_apply_final_photometric_lock`):
- `object_alpha`: feathered object mask normalized to [0, 1].
- `edit_alpha`: feathered edit mask normalized to [0, 1].
- `shadow_alpha = clip(edit_alpha * (1 - object_alpha), 0, 1)` — the region outside
  the artwork but inside the edit mask where provider shadows should be kept.
- `out = hardpaste * (1 - shadow_alpha) + render * shadow_alpha` — preserve shadows.
- `out = out * (1 - object_alpha) + adjusted_object * object_alpha` — lock object.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `MOLINK_FINAL_PHOTOMETRIC_LOCK` | `"1"` (enabled) | Toggle the post-render lock step |
| `MOLINK_FINAL_PHOTOMETRIC_LOCK_BLUR_PX` | `"18"` | Gaussian blur radius for luma extraction |
| `MOLINK_FINAL_PHOTOMETRIC_LOCK_MAX_DELTA` | `"32.0"` | Maximum per-channel luma delta |
| `MOLINK_FINAL_PHOTOMETRIC_LOCK_STRENGTH` | `"1.0"` | Strength multiplier for luma adjustment |
| `MOLINK_FINAL_PHOTOMETRIC_LOCK_FEATHER_PX` | `"1"` | Object mask feather radius |
| `MOLINK_FINAL_SHADOW_FEATHER_PX` | `"2"` | Edit mask shadow feather radius |

## Mask resolution

- `render_object_mask` is resolved from `mount_mask_path` if available in the manifest,
  otherwise falls back to `art_region_mask`. This mask defines the artwork object boundary.
- `render_mask` (sent to provider) is selected via `_select_render_mask_path` based on
  `--render-mask-scope`: `art_only`, `front_only`, or `edit_region` (default).
- These are separate: the provider receives one mask; the photometric lock uses the object mask.

## Provider prompt requirements

The prompt sent to the model must reinforce:
- **INTRINSIC COLOR LOCK**: do not recolor painting, frame, border, or mat.
- **CONSERVATIVE EDIT PROFILE**: photometric compositing, not repainting.
- Only low-frequency achromatic light/shadow/brightness modulation is allowed.

## Where to find in codebase

- `scripts/run_bfl_hanging_flux_pipeline.py`: `_apply_final_photometric_lock`, `_low_frequency_luma_delta_image`
- `hanging_mvp/flux_export.py`: prompt segments with INTRINSIC COLOR LOCK, CONSERVATIVE EDIT PROFILE
- `image_editing/providers/apiyi_gpt_image2.py`: `_low_frequency_luma_delta` (provider-side analog), Chinese prompt with 固有色 constraints
