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

Positive-wall recovery v7 adds three fail-closed recovery paths without weakening the final projected-footprint gates:

- instance-style VLM wall labels are canonicalized to class `wall` for SAM refinement, while position stays in `object_id`;
- when an unconstrained fit selects a horizontal surface inside a coarse wall proposal, a gravity-constrained vertical RANSAC fallback must meet explicit inlier ratio/count thresholds;
- if the floor is fully occluded, a plane-associated existing artwork on the same wall may supply a direct center-height anchor. It is never reported as a floor estimate.

The max-empty-rectangle path may also emit a limited raise-only obstruction-aware height seed. It cannot lower a metric placement, is capped by `candidate_generation.max_obstruction_aware_height_shift_m`, and remains subject to unchanged exact wall-support, edge/corner, structural-fixture, blocking-content and image-border gates.

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

For `gpt-image-2-vip` textured-wall requests, the remote endpoint cannot accept
a native edit mask and receives a full-space coordinate prompt. The production
output must therefore pass a deterministic local identity/scope lock before QA:

- candidate artwork-front and mount pixels are restored at the planned geometry;
- all pixels outside `flux2_edit_mask` are restored from the cleaned hard-paste;
- provider shadow pixels are admitted only when raw artwork identity passed QA;
  otherwise a soft bounded contact shadow is projected from the exact mount mask
  and no provider pixel is admitted;
- QA runs on this composite, while `raw_provider_render_size_qa` retains the raw
  model result for audit. Thresholds are not relaxed.

`result.status=render_review` is a completed review state, not a worker failure.
The terminal progress event is `stage=r2_upload`, `stage_state=done`, `pct=92`
with partial/final counts attached; only actual fatal result statuses emit
`stage=failed`, `stage_state=failed`.

## P2 contract-only additions

`room_corner_floor_fit` is specification-only and remains disabled in production until validated. Its minimum contract requires:

- at least two valid wall planes;
- wall-normal angle greater than `15°`;
- per-wall RANSAC inlier ratio greater than `0.60`;
- wall-corner intersection direction within `15°` of `g_hat_cam`;
- floor-side extent sampling from wall-plane RANSAC inliers at p3;
- explicit single-wall downgrade.

When a metric floor anchor is not trusted, the client result must expose a provisional preview height rather than a finalized installation height.



## 可选扩展：挂画因果层 narration bundle 与光线评分

本扩展为 append-only 字段。旧消费方可忽略未知键。

环境变量：

- `MOLINK_CAUSAL_ENABLED=1`：启用因果层计算、worker `narration_bundle` 输出与候选 `light_components` 影子记录。默认 `0`。
- `HANGING_CAUSAL_DECISION=1`：把 `light_components.delta` 回灌到候选 `score`，参与既有排序；同时追加 `light_caution` / `light_concern` / `light_deprioritized` 等光线标签到软风险与原因标签。默认 `0`。
- `MOLINK_CAUSAL_VLM_FIELDS=1`：semantic 单次 VLM 调用额外输出顶层 `light` 对象。默认 `0`。
- `MOLINK_CAUSAL_LIGHT_IN_NARRATION=1`：worker bundle 的 `facts.light_analysis` 写入光线分量明细，供业务端 LLM 正文解释排序原因。默认 `0`。
- `MOLINK_FULLSPACE_FINAL_POSTPROCESS_MODE=photometric_lock_no_wall_rgb`：当 textured-wall 触发"整图送模型 + 远端坐标矩阵"时，默认改用对象锁 + 阴影保留，但不做墙面 RGB 校正。可选值：`photometric_lock_no_wall_rgb` / `fullspace_object_lock` / `full_space_object_lock`；若显式改回 `wall_rgb_only` 会重新启用墙面 RGB 校正。

新增字段：

- progress 消息：`narration_bundle?: { task, char_limit, state, pct, facts, fired, copy_hints, light_components, constraints }`
- result payload：`narration_bundle?: object`
- candidate/plan：`light_components?: object`、`light_risk_band?: "none"|"caution"|"concern"`、`light_penalty_band?: "none"|"caution"|"concern"`、`light_semantics?: object`、`light_facts?: object`、`artwork_analysis?: object`

`HANGING_CAUSAL_DECISION=0` 时不改变候选 `score`、排序、`risk_level`、交付门。`light_components` 只作为叙事与影子观测字段。`light_risk_band` 表示 R_light 风险，`light_penalty_band` 表示 P_backdrop/P_view 等降序惩罚，二者均不抬升 `_risk_level`。

## 可选扩展：自适应画作明度融合与解释型 QA

本扩展针对最终渲染里“画作像自发光”的问题，目标不是禁止提亮，而是让提亮有证据：原始作品暗拍、周围环境同步变亮、render 打光方向一致时允许提亮；只有无法解释的提亮才触发 self-glow 风险。

新增环境变量：

- `MOLINK_FINAL_OBJECT_RELIGHT_MODE=adaptive`：最终对象锁后使用自适应 relight。可选 `adaptive` / `legacy` / `off`，默认 `adaptive`。
- `MOLINK_FINAL_OBJECT_BASE_MAX_BRIGHTEN=5`：基础可提亮预算。
- `MOLINK_FINAL_OBJECT_UNDEREXPOSED_EXTRA_BRIGHTEN=16`：作品暗拍评分带来的额外提亮预算。
- `MOLINK_FINAL_OBJECT_SCENE_LIGHT_EXTRA_BRIGHTEN=10`：周围环境同步变亮带来的额外提亮预算。
- `MOLINK_FINAL_OBJECT_RENDER_EXTRA_BRIGHTEN=8`：render 打光可信时带来的额外提亮预算。
- `MOLINK_FINAL_OBJECT_MAX_BRIGHTEN_CAP=30`：提亮预算硬上限。
- `MOLINK_FINAL_OBJECT_MAX_DARKEN=34`：暗化硬上限。
- `MOLINK_FINAL_SELF_GLOW_MARGIN=8`：超过可解释预算多少后触发 soft fallback。

QA 从“必须像 hard-paste”升级为“明度变化解释得通”：

- `MOLINK_FLUX_QA_ADAPTIVE_RELIGHT=1`：启用解释型 relight QA，默认 `1`。
- `MOLINK_FLUX_QA_ADAPTIVE_AMBIENT_INNER_PX=6` / `MOLINK_FLUX_QA_ADAPTIVE_AMBIENT_OUTER_PX=32`：环境环采样范围。
- `MOLINK_FLUX_QA_ADAPTIVE_SELF_GLOW_MARGIN=8`：不可解释提亮失败阈值。
- `MOLINK_FLUX_QA_ADAPTIVE_MAX_CORE_CHROMA_DELTA=16`：允许作为明度变化解释的最大核心色度残差。
- `MOLINK_FLUX_QA_ADAPTIVE_MIN_IDENTITY_HF_SSIM=0.55`：高频结构保真下限。低频明度变化可被解释，高频重绘仍失败。

新增 trace 字段：

- `selected_render_trace.post_render_photometric_lock.adaptive_relight`：记录 `underexposed_score`、`ambient_delta`、`render_coherence_score`、`max_brighten`、`allowed_brightening`、`self_glow_guard_triggered` 等。
- `selected_render_size_qa.adaptive_relight`：记录 `core_luma_delta`、`ambient_luma_delta`、`underexposed_score`、`render_coherence_score`、`unexplained_brightening`、`self_glow_risk`、`core_luma_change_explained` 等。
