---
name: semantic-polygon-coordinate-space
description: Semantic polygon coordinates must match the declared coordinate_space — if coordinate_space=processed, polygon must be in processed pixels, not original; polygon_original must preserve original coords; hanging masks.py must fall back to bbox_xyxy_processed when polygon is inconsistent
source: auto-skill
extracted_at: '2026-07-04T00:33:00.000Z'
---

## Contract

`semantics.json` declares `coordinate_space` (e.g. `"processed"`). ALL `polygon` fields MUST be in that coordinate space. `polygon_original` MUST always preserve original image coordinates. `bbox_xyxy_processed` and `bbox_xyxy_original` must be maintained in parallel.

## Bug pattern (job_ef2c7d32, 2026-07-03)

Semantic output declared `"coordinate_space": "processed"` with `processed_size: [1536, 1152]`, but polygon coordinates for some walls were still in original image space (e.g. `[2819, 235]` for right wall, max coord > 3648).

**Symptom chain:**
1. `coordinate_space: processed` but `polygon` contains original coords (e.g. 0..3648, not 0..1536)
2. `hanging_mvp/masks.py::load_semantic_masks()` reads `polygon` directly, generates mask in processed_size
3. Right wall polygon `[2819, 235, 3647, 1400]` is entirely out of bounds in 1536×1152 → mask is empty
4. Left wall polygon `[0, 0, 554, 2733]` is partially in bounds → mask is cropped/enlarged incorrectly
5. Front wall polygon happens to be in processed coords → correct
6. Semantic multiwall coverage computed from bbox: 0.87, max_right=1535 → "vlm_multiwall_preserved"
7. Hanging stage recomputes from polygon mask: coverage=0.77, max_right=1188 → "low_horizontal_coverage"
8. Geometry fallback replaces 3 semantic walls with 7 auto_wall normal-cluster fragments
9. All auto_walls rejected → `PARTIAL_ALL_REJECTED_WITH_REASONS`, `candidates: []`

## Fix applied (gpu_coordinate_contract_hotfix)

### 1. `semantic_mvp/service.py` — `_remap_objects_to_processed_space()`

- `polygon` always set to processed coordinates (scaled from original)
- `polygon_original` always preserved as original coordinates
- `bbox_xyxy_processed` / `bbox_xyxy_original` maintained in parallel
- `_sanitize_wall_bboxes_against_edge_structures()` updates `polygon_original` when trimming

### 2. `hanging_mvp/masks.py` — `load_semantic_masks()` coordinate tolerance

- If polygon is valid and consistent with `bbox_xyxy_processed` → use polygon
- If polygon is out-of-bounds, produces empty mask, or clearly disagrees with bbox → fall back to `bbox_xyxy_processed`
- Can scale `polygon_original` from original_size → processed_size when needed
- Outputs `mask_source` and `mask_debug` fields for diagnosis

### 3. `hanging_mvp/service.py` — geometry fallback safety

- Computes BOTH mask coverage AND bbox coverage for semantic walls
- If bbox coverage passes multiwall threshold but mask coverage is abnormally low → reason = `semantic_multiwall_preserved_bbox_mask_coordinate_mismatch_suspected`
- Does NOT replace semantic walls with auto_wall in this case

## Verification checklist

Check semantics.json output:
- `coordinate_space` field present and has a value (e.g. "processed")
- `polygon` max coordinates match `coordinate_space` scale (≤ processed_size if "processed")
- `polygon_original` max coordinates match `original_size` scale
- `bbox_xyxy_processed` values ≤ processed_size
- `bbox_xyxy_original` values ≤ original_size

**Red flag:** if `coordinate_space: processed` but any `polygon` has values > processed_size, the contract is broken.

## Historical note

Prior to 2026-07-04, the fix was "keep polygon in original image coordinates" because the hanging pipeline was assumed to work in original space. The hotfix changed the contract: polygon now follows `coordinate_space` declaration, and masks.py defends against mismatches by falling back to bbox_xyxy_processed.
