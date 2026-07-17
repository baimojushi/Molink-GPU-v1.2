from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from hanging_mvp.candidate_safety import (
    choose_flux2_input_policy,
    evaluate_wall_support_gate,
    semantic_axis_policy,
)
from hanging_mvp.render_qa import enforce_identity_locked_shadow_composite, evaluate_mask_anchored_render_qa
from hanging_mvp.hardpaste import warp_artwork_to_quad


def test_wall_support_gate_hard_rejects_opening_hang_without_duplicate_opening_gate():
    gate = evaluate_wall_support_gate(
        solid_wall_inside_ratio=0.808,
        effective_wall_inside_ratio=0.808,
        removable_existing_content_overlap_ratio=0.0,
        opening_or_recess_overlap_ratio=0.075,
        hard_reject_threshold=0.90,
        auto_pass_threshold=0.97,
        opening_recess_used_in_solid_mask=True,
    )
    assert gate["status"] == "hard_reject"
    assert gate["opening_or_recess_is_explanation_only"] is True
    assert "insufficient_safe_wall_support" in gate["hard_gate_reasons"]
    assert "overlaps_opening_or_recess" not in gate["hard_gate_reasons"]
    assert "overlaps_opening_or_recess" in gate["non_solid_reason_breakdown"]


def test_wall_support_gate_borderline_is_review_not_hard_reject():
    gate = evaluate_wall_support_gate(
        solid_wall_inside_ratio=0.893,
        effective_wall_inside_ratio=0.893,
        removable_existing_content_overlap_ratio=0.0,
        opening_or_recess_overlap_ratio=0.0,
        hard_reject_threshold=0.85,
        auto_pass_threshold=0.97,
        opening_recess_used_in_solid_mask=True,
    )
    assert gate["status"] == "review"
    assert "insufficient_safe_wall_support" not in gate["hard_gate_reasons"]
    assert "safe_wall_support_below_auto_pass_threshold" in gate["review_gate_reasons"]


def test_semantic_axis_policy_drops_entire_axis_set_instead_of_clipping():
    wall = np.ones((10, 10), np.uint8) * 255
    safe = np.zeros((10, 10), np.uint8)
    safe[:, :4] = 255
    policy = semantic_axis_policy(semantic_wall_mask=wall, safe_axis_wall_mask=safe, min_safe_coverage=0.50)
    assert policy["allowed"] is False
    assert policy["status"] == "dropped_low_safe_coverage"
    assert "do_not_clip_s_range" in policy["policy"]


def test_class_aware_dual_policy_uses_erased_room_only_for_non_art_wall_attachment():
    allowed = choose_flux2_input_policy(cleanup_labels=["hook"], has_person_cleanup=False, has_existing_cleanup=True)
    assert allowed["flux2_input_mode"] == "dual_erased_room_plus_target"
    assert allowed["raw_original_dual_allowed"] is False
    old_art = choose_flux2_input_policy(cleanup_labels=["hook", "existing_art"], has_person_cleanup=False, has_existing_cleanup=True)
    assert old_art["flux2_input_mode"] == "single_after_erase_hardpaste"
    aliased_old_art = choose_flux2_input_policy(cleanup_labels=["wall art"], has_person_cleanup=False, has_existing_cleanup=True)
    assert aliased_old_art["flux2_input_mode"] == "single_after_erase_hardpaste"
    assert aliased_old_art["cleanup_labels"] == ["existing_art"]
    person = choose_flux2_input_policy(cleanup_labels=["hook"], has_person_cleanup=True, has_existing_cleanup=True)
    assert person["flux2_input_mode"] == "single_after_erase_hardpaste"


def _save(path: Path, arr: np.ndarray) -> Path:
    Image.fromarray(arr.astype(np.uint8)).save(path)
    return path


def test_mask_anchored_render_qa_passes_noop_and_rejects_core_rewrite(tmp_path: Path):
    target = np.zeros((64, 64, 3), np.uint8) + 180
    target[20:44, 22:42] = [220, 30, 30]
    mask = np.zeros((64, 64), np.uint8)
    mask[20:44, 22:42] = 255
    target_path = _save(tmp_path / "target.png", target)
    mask_path = _save(tmp_path / "mask.png", mask)
    noop_path = _save(tmp_path / "noop.png", target.copy())
    qa_ok = evaluate_mask_anchored_render_qa(hard_paste_path=target_path, render_path=noop_path, art_region_mask_path=mask_path)
    assert qa_ok["passed"] is True
    rewritten = target.copy()
    rewritten[20:44, 22:42] = [20, 20, 240]
    rewritten_path = _save(tmp_path / "rewritten.png", rewritten)
    qa_bad = evaluate_mask_anchored_render_qa(hard_paste_path=target_path, render_path=rewritten_path, art_region_mask_path=mask_path)
    assert qa_bad["passed"] is False
    assert "core_masked_ssim_below_threshold" in qa_bad["failures"] or "core_color_delta_above_threshold" in qa_bad["failures"]


def test_full_space_identity_lock_restores_core_and_scope_but_keeps_neutral_shadow(tmp_path: Path):
    hard = np.zeros((80, 96, 3), np.uint8) + 180
    hard[28:52, 36:60] = [220, 35, 45]
    front = np.zeros((80, 96), np.uint8)
    front[28:52, 36:60] = 255
    mount = front.copy()
    edit = np.zeros((80, 96), np.uint8)
    edit[20:60, 28:68] = 255

    provider = np.clip(hard.astype(np.int16) + 12, 0, 255).astype(np.uint8)
    # Repainted core and a displaced colored ghost must not enter production.
    provider[28:52, 36:60] = [30, 50, 235]
    provider[24:28, 32:56] = [15, 180, 30]
    # Connected neutral darkening outside the mount is a valid contact shadow.
    provider[54:58, 36:60] = np.clip(hard[54:58, 36:60].astype(np.int16) - 18, 0, 255)

    hard_path = _save(tmp_path / "hard.png", hard)
    provider_path = _save(tmp_path / "provider.png", provider)
    front_path = _save(tmp_path / "front.png", front)
    mount_path = _save(tmp_path / "mount.png", mount)
    edit_path = _save(tmp_path / "edit.png", edit)
    output_path = tmp_path / "identity_locked.png"
    trace = enforce_identity_locked_shadow_composite(
        hard_paste_path=hard_path,
        provider_render_path=provider_path,
        artwork_front_mask_path=front_path,
        mount_mask_path=mount_path,
        edit_mask_path=edit_path,
        output_path=output_path,
        protect_dilate_px=1,
        shadow_blur_px=0,
        provider_identity_trusted=False,
    )

    output = np.asarray(Image.open(output_path).convert("RGB"), dtype=np.uint8)
    assert np.array_equal(output[front > 0], hard[front > 0])
    assert np.array_equal(output[edit == 0], hard[edit == 0])
    assert np.array_equal(output[25, 40], hard[25, 40])
    assert int(output[53, 40, 0]) < int(hard[53, 40, 0])
    assert trace["scope_enforced_locally"] is True
    assert trace["provider_contribution_present"] is False
    assert trace["shadow_source"] == "mask_projected_contact_shadow"

    qa = evaluate_mask_anchored_render_qa(
        hard_paste_path=hard_path,
        render_path=output_path,
        art_region_mask_path=front_path,
    )
    assert qa["passed"] is True
    assert qa["core_masked_ssim"] == 1.0
    assert qa["core_color_delta"] == 0.0


def test_supersampled_hardpaste_supports_tiny_artwork_without_empty_mask():
    room = np.zeros((64, 64, 3), np.uint8) + 200
    art = np.zeros((32, 32, 4), np.uint8)
    art[:, :, :3] = [255, 0, 0]
    art[:, :, 3] = 255
    quad = [[20, 20], [27, 20], [27, 31], [20, 31]]
    pasted, mask = warp_artwork_to_quad(room, art, quad, supersample_scale=4)
    assert pasted.shape == room.shape
    assert mask.shape == room.shape[:2]
    assert np.count_nonzero(mask) > 0


def test_adaptive_relight_qa_accepts_explainable_luma_boost(tmp_path: Path):
    target = np.zeros((80, 80, 3), np.uint8) + 150
    target[25:55, 25:55] = [120, 120, 120]
    mask = np.zeros((80, 80), np.uint8)
    mask[25:55, 25:55] = 255
    render = target.copy().astype(np.int16)
    # Nearby environment brightens, and the originally dark neutral artwork is
    # brightened in a low-frequency, chroma-preserving way.
    render[18:62, 18:62] = np.clip(render[18:62, 18:62] + 16, 0, 255)
    render[25:55, 25:55] = np.clip(target[25:55, 25:55].astype(np.int16) + 26, 0, 255)
    target_path = _save(tmp_path / "target.png", target)
    render_path = _save(tmp_path / "render.png", render.astype(np.uint8))
    mask_path = _save(tmp_path / "mask.png", mask)
    qa = evaluate_mask_anchored_render_qa(
        hard_paste_path=target_path,
        render_path=render_path,
        art_region_mask_path=mask_path,
        max_core_color_delta=10,
        min_core_masked_ssim=0.96,
    )
    assert qa["passed"] is True
    assert qa["adaptive_relight"]["core_luma_change_explained"] is True
    assert qa["adaptive_relight"]["self_glow_risk"] is False


def test_adaptive_relight_qa_rejects_unexplained_self_glow(tmp_path: Path):
    target = np.zeros((80, 80, 3), np.uint8) + 150
    target[25:55, 25:55] = [185, 185, 185]
    mask = np.zeros((80, 80), np.uint8)
    mask[25:55, 25:55] = 255
    render = target.copy().astype(np.int16)
    # Artwork brightens dramatically by itself; surrounding wall is unchanged.
    render[25:55, 25:55] = np.clip(target[25:55, 25:55].astype(np.int16) + 45, 0, 255)
    target_path = _save(tmp_path / "target.png", target)
    render_path = _save(tmp_path / "render.png", render.astype(np.uint8))
    mask_path = _save(tmp_path / "mask.png", mask)
    qa = evaluate_mask_anchored_render_qa(
        hard_paste_path=target_path,
        render_path=render_path,
        art_region_mask_path=mask_path,
        max_core_color_delta=10,
        min_core_masked_ssim=0.96,
    )
    assert qa["passed"] is False
    assert "adaptive_self_glow_unexplained_brightening" in qa["failures"]
    assert qa["adaptive_relight"]["self_glow_risk"] is True
