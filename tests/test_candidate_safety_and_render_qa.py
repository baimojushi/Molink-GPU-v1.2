from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from hanging_mvp.candidate_safety import (
    choose_flux2_input_policy,
    evaluate_wall_support_gate,
    semantic_axis_policy,
)
from hanging_mvp.render_qa import evaluate_mask_anchored_render_qa
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
