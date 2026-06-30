---
name: fullspace-render-pipeline
description: Full-space render pipeline (textured walls) must skip wall RGB correction and only generate visible side faces — not all four edges
source: auto-skill
extracted_at: '2026-06-26T18:30:00.000Z'
---

# Full-Space Render Pipeline Fixes

## Bug 1: Wall RGB correction corrupts full-space renders

**Symptom:** When textured-wall triggers full-space rendering (entire room image sent to GPT Image with remote coordinate matrix), the post-render wall RGB correction step corrupts the wall appearance — the wall looks "painted over" or "smeared."

**Root cause:** `wall_rgb_only` postprocess corrects surrounding wall RGB to match the source room. But in full-space mode, the ENTIRE image is the render target — there is no "surrounding wall" to correct. The correction step overwrites the model's rendered wall pixels with source room pixels, destroying the render quality.

**Fix:** New env var `MOLINK_FULLSPACE_FINAL_POSTPROCESS_MODE`:

| Value | Behavior |
|---|---|
| `photometric_lock_no_wall_rgb` (default for full-space) | Object lock + shadow preservation, NO wall RGB correction |
| `fullspace_object_lock` | Same as above (alias) |
| `full_space_object_lock` | Same as above (alias) |
| `wall_rgb_only` | Re-enables wall RGB correction (for local-crop mode only) |

**Diagnosis:** Check `render_aggregate.json` for `final_postprocess_mode`. If full-space render used `wall_rgb_only`, the wall was corrupted.

## Bug 2: Side faces generated on all four edges instead of visible ones only

**Symptom:** The artwork mount policy generates frame side faces on ALL four edges (left, right, top, bottom), creating a visible border around the entire artwork. But in reality, only 2 sides are visible (e.g., left + right for a portrait artwork on a front-facing wall).

**Root cause:** `build_render_prompt` in `flux_export.py` did not filter side faces by visibility. The prompt told the model to preserve "side faces" without specifying which ones, so the model preserved all four.

**Fix:** `flux_export.py` now reads `visible_side_faces_original` from mount policy and constructs explicit visibility instructions:

```python
visible_side_text = "left, right"  # only visible faces
hidden_side_text = "top, bottom"   # hidden faces
prompt = f"Only these side faces are visible: {visible_side_text}. " \
         f"Do not invent a border on hidden edges such as {hidden_side_text}."
```

**Diagnosis:** Check `frame_side_mask.png` in flux_inputs. If it shows a 4-edge border instead of 2-side thickness, the visible face filter is not active.

## Bug 3: narration_bundle missing light analysis in LLM output

**Symptom:** Worker sends `narration_bundle` in progress messages and result payload, but the LLM output does not include fine-grained light analysis (rho, lambda, risk band, etc.).

**Root cause:** `worker/result_builder.py` did not properly serialize `narration_bundle` with `light_components` and `facts.light_analysis`. The bundle was assembled but the light fields were dropped during serialization.

**Fix:** `result_builder.py` now correctly propagates `light_components` into `narration_bundle.facts.light_analysis` when `MOLINK_CAUSAL_LIGHT_IN_NARRATION=1`.

**Diagnosis:** Check `result_payload.json` in 99_result/ for `narration_bundle.facts.light_analysis`. If absent or empty, the serialization fix is not active.

## Relevant files

| File | Responsibility |
|---|---|
| `scripts/run_bfl_hanging_flux_pipeline.py` | `no_wall_rgb_modes` check, `MOLINK_FULLSPACE_FINAL_POSTPROCESS_MODE` |
| `hanging_mvp/flux_export.py` | `visible_side_faces_original` prompt construction |
| `hanging_mvp/service.py` | mount policy `visible_side_faces_original` computation |
| `worker/result_builder.py` | `narration_bundle` serialization with light analysis |
| `image_editing/providers/apiyi_gpt_image2.py` | `photometric_lock_no_wall_rgb` postprocess mode |

## Verification after deployment

1. Check `render_aggregate.json` → `final_postprocess_mode` should be `photometric_lock_no_wall_rgb` for full-space renders
2. Check `frame_side_mask.png` → should show 2 sides (not 4 edges)
3. Check `result_payload.json` → `narration_bundle.facts.light_analysis` should contain rho, lambda, risk_band
