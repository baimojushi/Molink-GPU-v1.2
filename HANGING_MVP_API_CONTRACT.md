# Hanging MVP API Contract Notes

Compatibility contract for `POST /analyze_hanging_mvp`. This file is intentionally kept alongside `[CONTRACT].hanging-mvp-api.md` because the existing test suite and downstream scripts reference the unbracketed filename.

## Endpoint stability

`POST /analyze_hanging_mvp` remains unchanged. Candidate and manifest fields are append-only; downstream consumers must tolerate unknown nested keys.

## Preserved extension fields

The following pre-existing extension fields remain available:

- `risk_level`
- `soft_risk_reasons`
- `requires_hardpaste_vlm_review`
- `structural_clearance`
- `cleanup_scope`
- `flux_visual_refinement`
- `requires_flux_visual_height_refinement`
- `metric_height_finalized`
- `height_is_provisional`

## P1 safe-wall support additions

Candidates now expose lightweight required fields for production decisions and trace review:

- `solid_wall_inside_ratio`
- `safe_wall_support_gate_ratio`
- `wall_support_gate`
- `candidate_reject_stage`
- `candidate_reject_reason`
- `hard_gate_reasons`
- `opening_or_recess_overlap_ratio`
- `opening_recess_used_in_solid_mask`
- `edge_support_debug`
- `corner_support_debug`
- `center_t_normalized`

`opening_or_recess_overlap_ratio` is an explanation field when openings have already been excluded from `solid_wall_mask`. The primary hard decision is the single safe-wall-support gate. Existing art or wall-attached clutter may contribute support only after current-wall / near-plane classification; foreground people never contribute wall support.

Semantic axes are dropped as a set when `semantic_axis_safe_mask_coverage` is below threshold. The service does not clip the semantic `s_range`; safe connected-component geometry axes remain eligible as a fallback. A wall may exit with `wall_has_no_safe_candidate=true` instead of being forced into the Flux manifest.

## P3 Flux additions

The safe render handoff adds:

- `flux2_input_policy`
- `requested_flux2_input_mode`
- `flux2_input_mode`
- `render_size_qa_status`
- `primary_render_size_qa`
- `selected_render_size_qa`
- `render_fallback`
- `exit_code`
- `production_usable`

Raw original-room dual references are prohibited. Safe dual mode uses `dual_erased_room_plus_target`, where Image 1 is the authoritative hard-paste target and Image 2 is the already-erased clean-room reference. Person cleanup and existing-art cleanup use single-target mode.

Render drift QA is mask-anchored: `art_region_mask` defines the deterministic geometry anchor. The pipeline checks core masked SSIM, core color delta and changes in a forbidden outer guard band. It does not use SAM/VLM bbox re-detection for pass/fail.

Failed Flux QA must never be silently published as `OK_SELECTED`. Failed items return `PARTIAL_HARDPASTE_REVIEW` and are written under `partial_review/`, not `final_hd/`.

## P2 contract-only additions

`room_corner_floor_fit` is specification-only and remains disabled in production until validated. Its minimum contract requires:

- at least two valid wall planes;
- wall-normal angle greater than `15°`;
- per-wall RANSAC inlier ratio greater than `0.60`;
- wall-corner intersection direction within `15°` of `g_hat_cam`;
- floor-side extent sampling from wall-plane RANSAC inliers at p3;
- explicit single-wall downgrade.

When a metric floor anchor is not trusted, the client result must expose a provisional preview height rather than a finalized installation height.
