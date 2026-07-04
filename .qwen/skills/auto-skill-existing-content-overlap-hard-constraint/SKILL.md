---
name: existing-content-overlap-hard-constraint
description: Hanging pipeline hard_constraints must check artwork overlap with existing_content objects (small_shelf, cabinet, painting, etc.) — currently missing, causing artwork to be placed on furniture
source: auto-skill
extracted_at: '2026-07-04T08:26:23.974Z'
---

## Problem

Hanging pipeline `hard_constraints` checks `wall_inside_ratio` (artwork within wall bounds) and `solid_wall_ratio` (solid wall support), but **does NOT check whether artwork overlaps with `existing_content` objects** detected by VLM (e.g., `small_shelf`, `cabinet`, `wall_label`, `existing_art`).

### Observed in job_42ee5bae

- VLM detected `small_shelf_11` (bbox `[732,170,1113,244]`) and `small_shelf_12` (bbox `[732,282,1113,352]`) as `existing_content`
- Selected candidate `cand_002` on `wall_front_01_01` projected artwork at center `[875,294]` with quad `[[830,253],[920,253],[918,334],[828,334]]`
- Artwork y-range `[253,334]` directly overlaps `small_shelf_12` y-range `[282,352]`
- `hard_constraints.passed = true` because only `wall_inside_ratio=0.984` was checked
- Result: artwork rendered ON TOP of the shelf/cabinet

## Fix location

`hanging_mvp/service.py` — in the `hard_constraints` check for each candidate, add an `existing_content_overlap` gate:

1. Get all `existing_content` objects from semantics: `role in {existing_content, foreground}` or `label in {small_shelf, cabinet, wall_label, existing_art, person}`
2. For each candidate's projected `quad_stage` polygon, compute IoU or containment with each existing_content polygon/bbox
3. If IoU > threshold (e.g., 0.05) OR artwork center inside existing_content bbox → fail hard constraint
4. Add `existing_content_overlap` to `reject_reasons` if failed

### Pseudocode

```python
def _check_existing_content_overlap(candidate_quad: np.ndarray, semantic_objects: list, threshold: float = 0.05) -> tuple[bool, list[str]]:
    reasons = []
    for obj in semantic_objects:
        if obj.get("role") in ("existing_content", "foreground"):
            obj_poly = np.array(obj.get("polygon", []), dtype=np.int32)
            if len(obj_poly) < 3:
                bbox = obj.get("bbox_xyxy_processed", [])
                if len(bbox) == 4:
                    obj_poly = np.array([[bbox[0],bbox[1]],[bbox[2],bbox[1]],[bbox[2],bbox[3]],[bbox[0],bbox[3]]], dtype=np.int32)
            if len(obj_poly) >= 3:
                iou = _compute_polygon_iou(candidate_quad, obj_poly)
                if iou > threshold:
                    reasons.append(f"existing_content_overlap:{obj.get('object_id')}:{obj.get('label')}:iou={iou:.2f}")
    return len(reasons) == 0, reasons
```

## Why this is critical

- VLM reliably detects furniture (`small_shelf`, `table`, `cabinet`, `door`, `window`)
- Without this check, candidates can pass all hard gates but still place artwork on furniture
- The `existing_content` objects are already used for cleanup (erase mask) but NOT for candidate validation
- This is a silent failure — no error, just wrong placement

## Related

- `flux2_input_policy.cleanup_labels` already includes `["existing_art", "person", "small_shelf", "wall_label"]` for render cleanup
- `bfl_v7_export.remove_mask_dilation_policy` already protects art_region during erase
- The missing piece is **candidate validation** — these objects should block candidate placement, not just be cleaned up in render
