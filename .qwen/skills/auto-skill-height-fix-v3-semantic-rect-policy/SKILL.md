---
name: height-fix-v3-semantic-rect-policy
description: semantic_clear_rect_vertical_fit changed from hard gate to diagnostic-only in HEIGHT-FIX-V3 — candidates are no longer rejected for metric height outside semantic rectangle (project)
source: auto-skill
extracted_at: '2026-07-16T18:55:00.000Z'
---

## Semantic Clear Rectangle Policy Change (V3)

### V2 Behavior (Removed)

In HEIGHT-FIX-V2, `evaluate_semantic_clear_rect_vertical_fit` was a **hard gate**:
- If the metric rule height (e.g., `table_center_135` = 1.35m off floor) fell outside the semantic clear rectangle's t-range, the candidate was **rejected**
- Reason: `metric_height_outside_semantic_clear_rectangle`
- Problem: Valid candidates like `cand_004` in job_461299b7 were rejected even though all support ratios were 1.0 (perfect)

### V3 Behavior (Current)

In HEIGHT-FIX-V3, `evaluate_semantic_clear_rect_vertical_fit` is **diagnostic only**:
- `enforced: False` (never gates)
- `passed: True` (always passes)
- `policy: "diagnostic_only_exact_projected_footprint_is_authoritative"`
- The rectangle is only a **horizontal candidate seed** — its vertical interval comes from whichever empty rectangle happened to be largest and is not evidence that the resolved metric-height footprint is obstructed

### Why This Changed

**Observed failure (job_461299b7):**
- `cand_004` had perfect support ratios: wall=1.0, edge=1.0, corner=1.0
- Rejected solely because `center_t_m=0.1296` was below `component_t_range_m=[0.2526, 1.2188]`
- The rule height (`floor_center_152` = 1.52m) was in a clear area but outside the semantic rectangle
- The semantic rectangle only covered the upper part of the wall (where the table/sofa was detected)
- Result: no candidates passed, pipeline returned `PARTIAL_REVIEW_CANDIDATES_ONLY`

### Code Changes

**height_fit_contract.py:**
- `enforced` → `evaluated` (semantic intent changed)
- `enforced: False` hardcoded (never gates)
- `passed: True` hardcoded (always passes)
- `blocking: False` added
- `geometric_fit` field added (shows actual fit without gating)
- `height_contract_gate_reasons()` function added — only checks `metric_height_rule_consistent`

**service.py:**
- `HANGING_PIPELINE_BUILD_ID` changed to `2026-07-16-metric-height-exact-footprint-v3`
- Rejection logic changed from inline checks to `height_contract_gate_reasons()` call
- `metric_height_outside_semantic_clear_rectangle` removed from reject reasons
- `candidate_reject_stage` changed from `height_contract_gate` to geometry-based stages

### Safety Model Under V3

Safety is still enforced, but at a different stage:
- **Before V3:** Safety checked at height selection time (semantic rectangle fit)
- **After V3:** Safety checked at projection time (wall support, edges/corners, structures, blocking objects)

The exact projected artwork footprint at the authoritative height is what determines safety, not whether the height happens to fall inside an arbitrary empty rectangle.

### How to Apply

When analyzing candidate rejection:
1. If `semantic_clear_rect_vertical_fit.passed` is False, this is **not** why the candidate was rejected under V3
2. Check `reject_reasons` for actual rejection causes
3. If all candidates rejected but near-miss exists with good support ratios, investigate non-height gates (existing content overlap, edge/corner support)
4. The `semantic_clear_rect_vertical_fit` data is still useful for debugging but should never be treated as a gate
