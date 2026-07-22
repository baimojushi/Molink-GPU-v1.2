from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hanging_mvp.candidate_safety import (
    evaluate_wall_mask_lineage,
    geometry_gate_review_reasons,
)
from hanging_mvp.exit_review import review_hardpaste_exit
from hanging_mvp.client_result import build_client_result
from hanging_mvp.masks import load_semantic_masks
from semantic_mvp.service import _semantic_multiwall_postprocess, _wall_bbox_curtain_guard
from semantic_mvp.service import _build_vlm_objects_in_original_space


def _sam_wall(object_id: str, box: list[float]) -> dict:
    x1, y1, x2, y2 = box
    return {
        "object_id": object_id,
        "label": object_id,
        "role": "wall",
        "bbox_xyxy_processed": box,
        "bbox_xyxy_original": box,
        "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        "source": "vlm_bbox_sam_polygon",
        "sam": {"enabled": True, "fallback_to_bbox": False, "score": 0.91},
    }


def _forbidden(object_id: str, label: str, box: list[float]) -> dict:
    x1, y1, x2, y2 = box
    return {
        "object_id": object_id,
        "label": label,
        "role": "forbidden",
        "bbox_xyxy_processed": box,
        "bbox_xyxy_original": box,
        "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        "source": "vlm_bbox_sam_polygon",
        "sam": {"enabled": True, "fallback_to_bbox": False, "score": 0.93},
    }


def test_regression_window_subtraction_drops_curtain_width_residual():
    # Reproduces the diagnosed right-side relation: the window covers roughly
    # 79% of the VLM wall width and subtraction leaves only the curtain strip.
    objects = [
        _sam_wall("wall_front_01", [300, 120, 650, 700]),
        _sam_wall("wall_right_01", [676, 100, 1150, 700]),
        _forbidden("window_right_01", "window", [777, 100, 1150, 700]),
    ]

    out, debug = _semantic_multiwall_postprocess(objects, 1280, 800, 1.0, 1.0, b"")

    assert not any(item["object_id"] == "wall_right_01" for item in out)
    dropped = next(item for item in debug["edge_structural_trim"] if item["wall_id"] == "wall_right_01")
    assert dropped["status"] == "dropped"
    assert "high_coverage_opening_left_narrow_wall_fragment" in dropped["wall_mask_lineage"]["reject_reasons"]
    assert debug["geometry_augmentation_suppressed"] is True


def test_structural_residual_without_sam_lineage_is_dropped():
    wall = _sam_wall("wall_left_01", [0, 0, 400, 700])
    wall["source"] = "vlm_bbox_polygon"
    wall.pop("sam")
    objects = [
        wall,
        _sam_wall("wall_front_01", [430, 0, 900, 700]),
        _forbidden("door_left_01", "door", [0, 0, 90, 700]),
    ]

    out, debug = _semantic_multiwall_postprocess(objects, 1000, 700, 1.0, 1.0, b"")

    assert not any(item["object_id"] == "wall_left_01" for item in out)
    dropped = next(item for item in debug["edge_structural_trim"] if item["wall_id"] == "wall_left_01")
    assert "structural_residual_missing_trusted_sam_wall_lineage" in dropped["wall_mask_lineage"]["reject_reasons"]


def test_retained_structural_component_requires_fail_closed_material_review(monkeypatch):
    objects = [
        _sam_wall("wall_left_01", [0, 0, 400, 700]),
        _sam_wall("wall_front_01", [430, 0, 900, 700]),
        _forbidden("door_left_01", "door", [0, 0, 90, 700]),
    ]
    out, _ = _semantic_multiwall_postprocess(objects, 1000, 700, 1.0, 1.0, b"")
    loaded = load_semantic_masks({"objects": out, "processed_size": [1000, 700]}, (700, 1000))
    wall = loaded["wall_left_01"]
    gate = evaluate_wall_mask_lineage(wall)
    assert gate["passed"] is True
    assert gate["requires_visual_material_review"] is True

    monkeypatch.delenv("VLM_API_URL", raising=False)
    monkeypatch.delenv("VLM_API_KEY", raising=False)
    monkeypatch.delenv("VLM_MODEL", raising=False)
    review = review_hardpaste_exit(
        np.zeros((32, 32, 3), dtype=np.uint8),
        {
            "candidate_id": "cand_fragment",
            "requires_hardpaste_vlm_review": True,
            "hardpaste_vlm_review_fail_closed": True,
            "wall_mask_lineage_gate": gate,
        },
        {
            "exit_review": {
                "hardpaste_vlm_mode": "auto",
                # The structural-fragment contract cannot be bypassed even if
                # a caller relaxes the generic review setting.
                "block_unreviewed_required_vlm": False,
            }
        },
    )
    assert review["approved_for_flux_render"] is False
    assert review["fail_closed_required"] is True
    assert review["status"] == "pending_vlm_review_not_configured"


def test_support_edge_and_corner_review_intervals_cannot_auto_approve():
    reasons = geometry_gate_review_reasons(
        wall_support_gate={"status": "review"},
        edge_corner_support_debug={
            "status": "ok",
            "min_edge_support_ratio": 0.5161,
            "min_corner_support_ratio": 0.5918,
        },
        auto_pass_edge_support_ratio=0.80,
        auto_pass_corner_support_ratio=0.75,
    )
    assert reasons == [
        "safe_wall_support_between_hard_reject_and_auto_pass",
        "artwork_edge_support_between_hard_reject_and_auto_pass",
        "artwork_corner_support_between_hard_reject_and_auto_pass",
    ]


def test_tampered_structural_fragment_lineage_is_hard_rejected():
    gate = evaluate_wall_mask_lineage({
        "wall_mask_lineage": {
            "structural_subtraction_applied": True,
            "trusted_sam_wall_evidence": False,
            "final_component_source_wall_coverage_ratio": 0.60,
            "gate_status": "review_required",
        }
    })
    assert gate["passed"] is False
    assert "structural_wall_fragment_missing_trusted_sam_lineage" in gate["hard_reasons"]
    assert "structural_wall_fragment_not_covered_by_source_sam_wall" in gate["hard_reasons"]


def test_client_result_exposes_only_final_manifest_approved_candidates():
    candidates = [
        {"candidate_id": "curtain", "wall_id": "left", "score": 0.99, "approved_for_flux_render": False},
        {"candidate_id": "front", "wall_id": "front", "score": 0.90, "approved_for_flux_render": True},
    ]
    result = build_client_result(
        {
            "status": "ok",
            "exit_code": "OK_SELECTED",
            "image_space": {"original_size": [1280, 800]},
            "final_hardpaste_assets": [{"candidate_id": "front", "wall_id": "front"}],
        },
        candidates,
        [],
    )
    assert result["best_plan"]["candidate_id"] == "front"
    assert result["alternative_plans"] == []
    assert result["public_candidate_policy"] == "final_manifest_approved_for_flux_render_only"


def test_direct_wall_label_on_narrow_edge_curtain_is_rejected_without_curtain_label():
    # The provider sometimes returns the curtain itself as role=wall and omits
    # both ``curtain`` and ``window``.  A SAM success must not turn that wrong
    # class into trusted hanging surface evidence.
    wall = _sam_wall("wall_right_01", [900, 120, 1000, 700])
    image = np.zeros((700, 1000, 3), dtype=np.uint8)
    for x in range(900, 1000, 8):
        image[:, x:x + 3] = 220
    guard = _wall_bbox_curtain_guard(wall, [wall], image, 1000, 700)
    assert guard["hard_reject"] is True
    assert "vertical_fold_texture_on_narrow_edge_wall_bbox" in guard["reasons"]


def test_window_adjacent_residual_is_rejected_before_sam_lineage_can_pass():
    wall = _sam_wall("wall_right_01", [676, 100, 1150, 700])
    window = _forbidden("window_right_01", "window", [777, 100, 1150, 700])
    guard = _wall_bbox_curtain_guard(wall, [wall, window], None, 1280, 800)
    assert guard["hard_reject"] is True
    assert "wall_bbox_mostly_window_covered_curtain_residual" in guard["reasons"]


def test_free_text_curtain_evidence_overrides_misleading_wall_role():
    out = _build_vlm_objects_in_original_space(
        {"objects": [{
            "object_id": "wall_right_01",
            "label": "wall",
            "role": "wall",
            "bbox_xyxy": [800, 100, 999, 900],
            "notes": "dark sheer curtain covering the window",
        }]},
        1000, 1000, "original_pixels",
    )
    assert out[0]["label"] == "curtain"
    assert out[0]["role"] == "forbidden"
