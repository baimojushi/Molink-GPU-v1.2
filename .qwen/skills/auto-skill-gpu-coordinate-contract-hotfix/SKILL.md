---
name: gpu-coordinate-contract-hotfix
description: GPU semantic coordinate_space contract fix — polygon must match declared coordinate_space; masks.py falls back to bbox_xyxy_processed on mismatch; geometry fallback protected against coordinate artifacts
source: auto-skill
extracted_at: '2026-07-04T00:33:00.000Z'
---

## Trigger

Hanging pipeline returns `status: "no_safe_wall"`, `candidates: []`, but semantic output shows adequate wall coverage (e.g. `semantic_wall_coverage_debug.coverage_ratio >= 0.87`). Diagnosis reveals `coordinate_space` declaration disagrees with actual `polygon` values.

## Root cause

`semantics.json` declares `"coordinate_space": "processed"` but some objects have `polygon` in original image coordinates. `hanging_mvp/masks.py::load_semantic_masks()` reads polygon directly without checking coordinate consistency, producing wrong masks (empty or cropped). This cascades into geometry fallback replacing good semantic walls with unusable auto_wall fragments.

## Applied fix (3 files)

### semantic_mvp/service.py
- `_remap_objects_to_processed_space()`: polygon always in processed coords; polygon_original always in original coords
- `_sanitize_wall_bboxes_against_edge_structures()`: syncs polygon_original when trimming bboxes

### hanging_mvp/masks.py
- `load_semantic_masks()`: defensive coordinate validation
  - Validates polygon against processed_size bounds
  - Falls back to bbox_xyxy_processed when polygon is out-of-bounds, produces empty mask, or disagrees with bbox
  - Can scale polygon_original → processed coordinates when needed
  - Outputs mask_source and mask_debug for diagnosis

### hanging_mvp/service.py
- Geometry fallback computes both mask coverage AND bbox coverage
- If bbox passes but mask fails → reason `semantic_multiwall_preserved_bbox_mask_coordinate_mismatch_suspected`, preserves semantic walls

## Deployment

```bash
cd /opt/molink/hanging
patch -p1 < gpu_coordinate_contract_hotfix.patch
python3 -m py_compile semantic_mvp/service.py hanging_mvp/masks.py hanging_mvp/service.py
# Restart semantic and hanging services
```

## Verification

After applying, check next job's semantics.json:
- All `polygon` max values ≤ processed_size (if coordinate_space=processed)
- All `polygon_original` max values ≤ original_size
- `auto_multiwall_debug.semantic_wall_reason` should NOT be "low_horizontal_coverage_or_missing_right_side" for rooms with adequate walls

## Related skill

- `semantic-polygon-coordinate-space` — detailed contract and historical context
