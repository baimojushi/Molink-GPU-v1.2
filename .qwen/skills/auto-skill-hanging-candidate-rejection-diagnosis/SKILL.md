---
name: hanging-candidate-rejection-diagnosis
description: Diagnose why hanging pipeline produces zero candidates — trace through worker logs, client_result.json, reject summaries, and edge/corner support thresholds
source: auto-skill
extracted_at: '2026-07-03T15:42:28.073Z'
---

# Hanging Candidate Rejection Diagnosis

When a hanging job returns `status=no_safe_wall` with `exit_code=PARTIAL_ALL_REJECTED_WITH_REASONS` and `candidates=0`, all wall candidates failed hard gates.

## Quick triage path

1. **Worker log**: grep job_id in `logs/worker.log` — look for `send_result` line showing `status=no_safe_wall exit_code=PARTIAL_ALL_REJECTED_WITH_REASONS candidates=0`
2. **Diagnostics zip**: job artifacts live in `/root/autodl-fs/hanging_runs/<timestamp>_<job_id>.zip`
3. **client_result.json**: inside zip at `04_hanging/client_result.json` — check `not_recommended[]` for per-wall rejection stages
4. **rejected_walls.json**: inside zip at `04_hanging/extracted/rejected_walls.json` — shows `blocked_reason` and per-candidate `reject_reasons`
5. **reject_summary.json**: per wall at `rejected_wall_assets/<wall_id>/reject_summary.json` — detailed hard constraint values

## FIRST CHECK: coordinate space contract mismatch (job_ef2c7d32 pattern, 2026-07-03)

Before analyzing rejection thresholds, verify the semantic-hanging coordinate contract:

1. Check `03_semantic/semantics.json`:
   - `coordinate_space` field value (e.g. "processed")
   - `processed_size` vs `original_size`
2. For each wall object, compare `polygon` max coords against `processed_size`:
   - If `coordinate_space: processed` but any `polygon` has values > `processed_size` → **contract broken**
   - Check `bbox_xyxy_processed` values — if these are correct but `polygon` is wrong, masks will be wrong
3. Check `04_hanging/extracted/hanging_plan.json` debug section:
   - If `auto_multiwall_debug.reason` = `geometry_fallback_replaced_low_coverage_semantic_walls`
   - AND `semantic_wall_reason` = `low_horizontal_coverage_or_missing_right_side`
   - BUT semantic stage reported adequate coverage → **this is the coordinate mismatch cascade**

**Why this matters first:** If the coordinate contract is broken, all downstream rejection reasons are artifacts of wrong wall masks. Fix the coordinate issue before touching thresholds. See skill `gpu-coordinate-contract-hotfix` for the fix and `semantic-polygon-coordinate-space` for the contract.

## Rejection reason taxonomy

| Reason | Meaning | Threshold |
|---|---|---|
| `insufficient_safe_wall_support` | solid_wall_ratio below `hard_reject_threshold` (default 0.85) | `hanging_mvp/candidate_safety.py` |
| `artwork_edge_support_below_hard_reject_threshold` | min edge support ratio < 0.50 | `hanging_mvp/service.py:1340` |
| `artwork_corner_support_below_hard_reject_threshold` | min corner support ratio < 0.50 | `hanging_mvp/service.py:1341` |
| `no_safe_axis_and_insufficient_solid_component` | geometry gate — semantic axis safe coverage < 0.50 | `candidate_safety.py:semantic_axis_policy()` |
| `projected_artwork_too_close_to_image_boundary` | artwork projection touches image border | `min_projected_artwork_border_margin_px_original` |
| `component_edge_margin_below_threshold` | artwork edge clearance < required half-width | `edge_margin_debug` |

## Threshold defaults (overridable via rules_overrides)

- `hard_reject_artwork_edge_support_ratio`: 0.50
- `hard_reject_artwork_corner_support_ratio`: 0.50
- `min_artwork_edge_support_ratio`: 0.80 (auto-pass threshold)
- `min_artwork_corner_support_ratio`: 0.75 (auto-pass threshold)
- `hard_reject_threshold` (wall support): 0.85
- `auto_pass_threshold` (wall support): 0.97

## Key files in diagnostics zip

| Path | Purpose |
|---|---|
| `00_job/job_payload.json` | Input artwork dimensions, rules_overrides |
| `03_semantic/semantics.json` | Detected walls, objects, scene_profile |
| `04_hanging/client_result.json` | Final exit_code, not_recommended[], best_plan |
| `04_hanging/extracted/rejected_walls.json` | Per-wall block reason + per-candidate reject_reasons |
| `04_hanging/extracted/hanging_plan.json` | selected_candidate (null when all rejected), ranked_candidates, passed_walls |
| `04_hanging/rules_overrides.json` | Any threshold overrides from the job |
| `rejected_wall_assets/<wall_id>/reject_summary.json` | Detailed edge/corner support ratios, wall_inside_ratio, hard_gate_reasons |
| `rejected_wall_assets/<wall_id>/debug_overlay.png` | Visual overlay of artwork projection on wall |

## Common patterns

**solid_wall_ratio=0.0**: wall fully covered by openings (doors/windows), furniture, or existing_art — no safe hanging area exists.

**solid_wall_ratio=0.85-0.97**: wall passes hard_reject but fails auto_pass, enters "review" zone. Still rejected if edge/corner support falls below 0.50.

**edge/corner support = 0.0 on specific edge/corner index**: artwork projection lands exactly on a doorway, window, or furniture edge. The edge index maps to the 4 sides of the artwork quad (0=top, 1=right, 2=bottom, 3=left in projection order).

**geometry_gate (auto_wall_03/06 type)**: semantic axis policy fails — safe_axis_wall_mask covers < 50% of semantic wall area. Typically happens on partial walls or walls with heavy structural overlap.
