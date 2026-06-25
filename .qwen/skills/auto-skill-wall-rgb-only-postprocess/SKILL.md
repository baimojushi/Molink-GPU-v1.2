---
name: wall-rgb-only-postprocess
description: Preserve GPT Image render pixels and only correct surrounding wall RGB — avoids locked-texture repaste, frame stacking, and geometry mismatch artifacts
source: auto-skill
extracted_at: '2026-06-25T05:33:00Z'
---

## Problem

When the GPT Image provider returns a render that already looks correct for the artwork object/frame but has wall color mismatch around it, running the full photometric lock pipeline (similarity transform + locked texture warp + shadow composite) can:
- Cover generated contact shadows with the hard-paste locked texture
- Create frame stacking when render geometry shifts or scales differently than the hard-paste anchor
- Repaint pixels that were already correct

## Solution: wall_rgb_only mode

`wall_rgb_only` mode preserves the provider-rendered crop as the visual source of truth and only applies a robust median RGB delta correction to wall pixels outside the object mask.

### Key behaviors
- **No relocation**: Does not estimate translation or scale between hard-paste and render
- **No locked texture warp**: Does not paste or re-apply the hard-paste texture
- **No frame/core repaste**: Artwork, frame, mat, and object pixels remain exactly as the provider returned them
- **Wall-only correction**: Applies median wall color delta only to wall pixels inside the edit mask, outside the object mask

### Affected functions
- `apiyi_gpt_image2.py`: `_correct_generated_wall_rgb(..., apply_to_wall_only=True)`, `_composite_wall_rgb_only_crop()`
- `run_bfl_hanging_flux_pipeline.py`: `_correct_render_wall_rgb_full(..., apply_to_wall_only=True)`, `_apply_final_photometric_lock()` with `postprocess_mode="wall_rgb_only"`

## Configuration

### Worker env vars (`/etc/molink/gpu.env`)

```bash
# Mask edit mode: use wall_rgb_only instead of local_composite or photometric_composite
APIYI_GPT_IMAGE_MASK_EDIT_MODE=wall_rgb_only

# Final postprocess: enable wall-only correction, skip full photometric lock
MOLINK_FINAL_PHOTOMETRIC_LOCK=1
MOLINK_FINAL_POSTPROCESS_MODE=wall_rgb_only
```

### Aliases recognized

The following values are normalized to the same mode: `wall_rgb_only`, `wall_rgb_correction_only`, `render_wall_rgb_only` (hyphens converted to underscores).

### Output naming

- `wall_rgb_only` mode: output file named `03_flux2_render_wall_rgb_corrected.png` (v7 sequence) or `02_flux2_render_wall_rgb_corrected.png` (other sequences)
- `photometric_lock` mode: output file named `03_flux2_render_photometric_locked.png` or `02_flux2_render_photometric_locked.png`

## When to use

- Provider render looks correct for artwork object/frame but wall color doesn't match the original room
- Full photometric lock causes frame stacking, shadow occlusion, or geometry mismatch
- Quick iteration on wall color correction without touching artwork pixels

## When NOT to use

- Provider render has wrong artwork geometry, scale, or frame design — use `photometric_composite` with similarity transform instead
- You need hard-paste pixel-perfect artwork identity lock — use full `photometric_lock` pipeline

## Deployment

After deploying patches that add wall_rgb_only support:
1. Add env vars to `/etc/molink/gpu.env`
2. Restart worker: `pkill -f 'worker.main|worker/start_worker.sh' || true && bash /opt/molink/hanging/worker/start_worker.sh &`
3. Verify: `cat /proc/$(pgrep -f 'worker.main')/environ | tr '\0' '\n' | grep -E 'MASK_EDIT_MODE|POSTPROCESS_MODE'`
