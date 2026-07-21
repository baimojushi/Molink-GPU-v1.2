from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_mvp.service import (
    _build_vlm_response_binding,
    _infer_vlm_bbox_coordinate_space,
    _normalize_vlm_bbox_to_original_pixels,
    _semantic_multiwall_postprocess,
)


def test_edge_door_is_trimmed_from_wall_bbox_without_metric_zip():
    objects = [
        {
            "object_id": "wall_left_01",
            "label": "wall_left_01",
            "role": "wall",
            "bbox_xyxy_processed": [0, 0, 320, 700],
            "bbox_xyxy_original": [0, 0, 320, 700],
            "polygon": [[0, 0], [320, 0], [320, 700], [0, 700]],
            "source": "vlm_bbox_sam_polygon",
            "sam": {"enabled": True, "fallback_to_bbox": False},
        },
        {
            "object_id": "door_01",
            "label": "door",
            "role": "forbidden",
            "bbox_xyxy_processed": [230, 0, 320, 700],
            "bbox_xyxy_original": [230, 0, 320, 700],
            "polygon": [[230, 0], [320, 0], [320, 700], [230, 700]],
        },
        {
            "object_id": "wall_front_01",
            "label": "wall_front_01",
            "role": "wall",
            "bbox_xyxy_processed": [350, 0, 800, 700],
            "bbox_xyxy_original": [350, 0, 800, 700],
            "polygon": [[350, 0], [800, 0], [800, 700], [350, 700]],
        },
    ]
    out, debug = _semantic_multiwall_postprocess(objects, 1000, 700, 1.0, 1.0, b"")
    left = next(o for o in out if o["object_id"] == "wall_left_01")
    assert left["bbox_xyxy_processed"][2] < 230
    assert left["source"].endswith("+edge_structural_mask_subtract")
    assert left["wall_mask_lineage"]["trusted_sam_wall_evidence"] is True
    assert left["wall_mask_lineage"]["gate_status"] == "review_required"
    assert left["requires_wall_material_vlm_review"] is True
    assert debug.get("edge_structural_trim")


def test_low_horizontal_wall_coverage_requests_full_scene_augmentation():
    objects = [
        {
            "object_id": "wall_left_01",
            "label": "wall_left_01",
            "role": "wall",
            "bbox_xyxy_processed": [0, 0, 200, 700],
            "bbox_xyxy_original": [0, 0, 200, 700],
            "polygon": [[0, 0], [200, 0], [200, 700], [0, 700]],
        },
        {
            "object_id": "wall_front_01",
            "label": "wall_front_01",
            "role": "wall",
            "bbox_xyxy_processed": [250, 0, 500, 700],
            "bbox_xyxy_original": [250, 0, 500, 700],
            "polygon": [[250, 0], [500, 0], [500, 700], [250, 700]],
        },
    ]
    _, debug = _semantic_multiwall_postprocess(objects, 1000, 700, 1.0, 1.0, b"")
    assert debug["semantic_wall_coverage_debug"]["coverage_ratio"] < 0.70
    assert debug.get("full_scene_geometry_augment_debug", {}).get("enabled") is True


def test_vlm_0_1000_bbox_space_is_detected_and_scaled_to_original_pixels():
    objects = [
        {"label": "wall_left", "bbox_xyxy": [0, 0, 344, 998]},
        {"label": "wall_front", "bbox_xyxy": [345, 0, 642, 976]},
        {"label": "wall_right", "bbox_xyxy": [643, 0, 958, 820]},
        {"label": "door", "bbox_xyxy": [250, 0, 344, 998]},
    ]
    mode, debug = _infer_vlm_bbox_coordinate_space(objects, 1664, 768)
    assert mode == "normalized_0_1000"
    assert debug["source_coordinate_space"] == "normalized_0_1000"

    right_wall = _normalize_vlm_bbox_to_original_pixels([643, 0, 958, 820], 1664, 768, mode)
    assert abs(right_wall[0] - 1069.95) < 1.0
    assert abs(right_wall[2] - 1594.11) < 1.0
    assert abs(right_wall[3] - 629.76) < 1.0

    door = _normalize_vlm_bbox_to_original_pixels([250, 0, 344, 998], 1664, 768, mode)
    assert abs(door[0] - 416.0) < 1.0
    assert abs(door[2] - 572.42) < 1.0
    assert abs(door[3] - 766.46) < 1.0


def test_regression_b7d44210_mixed_x_grid_and_pixel_y_are_scaled_per_axis():
    # The provider used a 0..1000 x grid, but emitted the floor's y2 in the
    # requested 1080-pixel height.  A single four-coordinate hypothesis used to
    # classify everything as original pixels and compressed all VLM walls into
    # the left 52% of this wide scene.
    objects = [
        {"label": "wall_left", "role": "wall", "bbox_xyxy": [0, 0, 345, 999]},
        {"label": "wall_front", "role": "wall", "bbox_xyxy": [346, 0, 728, 999]},
        {"label": "wall_right", "role": "wall", "bbox_xyxy": [728, 0, 1000, 999]},
        {"label": "floor", "role": "support", "bbox_xyxy": [0, 820, 1000, 1080]},
    ]

    mode, debug = _infer_vlm_bbox_coordinate_space(objects, 1920, 1080)

    assert mode == "mixed_x_1000_y_original"
    assert debug["coordinate_hypothesis_scores"]["x_prefers_1000_grid"] is True
    assert debug["coordinate_hypothesis_scores"]["y_prefers_1000_grid"] is False
    front = _normalize_vlm_bbox_to_original_pixels([346, 0, 728, 999], 1920, 1080, mode)
    assert abs(front[0] - 664.32) < 0.01
    assert abs(front[2] - 1397.76) < 0.01
    assert front[3] == 999


def test_regression_83f8bf2e_internal_1000_grid_with_original_edge_sentinels():
    # Internal objects use the provider's 0..1000 grid. Objects touching the
    # image boundary copy exact original dimensions (1280/1708), which used to
    # make the entire response look like original pixels.
    objects = [
        {"label": "wall", "role": "wall", "bbox_xyxy": [0, 220, 341, 640]},
        {"label": "wall", "role": "wall", "bbox_xyxy": [342, 257, 600, 487]},
        {"label": "wall", "role": "wall", "bbox_xyxy": [600, 210, 1000, 650]},
        {"label": "window", "role": "forbidden", "bbox_xyxy": [600, 205, 1280, 690]},
        {"label": "floor", "role": "support", "bbox_xyxy": [0, 620, 1280, 1708]},
        {"label": "roof", "role": "support", "bbox_xyxy": [0, 0, 1280, 280]},
    ]
    mode, debug = _infer_vlm_bbox_coordinate_space(objects, 1280, 1708)
    assert mode == "normalized_0_1000_with_original_boundary_sentinels"
    assert debug["preserve_original_boundary_sentinels"] is True

    center = _normalize_vlm_bbox_to_original_pixels([342, 257, 600, 487], 1280, 1708, mode)
    assert all(abs(a - b) < 1e-6 for a, b in zip(center, [437.76, 438.956, 768.0, 831.796]))
    floor = _normalize_vlm_bbox_to_original_pixels([0, 620, 1280, 1708], 1280, 1708, mode)
    assert all(abs(a - b) < 1e-6 for a, b in zip(floor, [0.0, 1058.96, 1279.0, 1707.0]))


def test_v13_joint_hypothesis_accepts_grid_xy_when_axis_x_alone_is_weak():
    # Production family: ordinary values are 0..1000 grid values, while boxes
    # touching the right/bottom edge copy 1280/1708.  The v12 x-only scorer can
    # be much less decisive than y; the complete 2-D scene must decide once.
    objects = [
        {"label": "wall", "role": "wall", "bbox_xyxy": [0, 220, 341, 640]},
        {"label": "wall", "role": "wall", "bbox_xyxy": [342, 257, 600, 487]},
        {"label": "wall", "role": "wall", "bbox_xyxy": [600, 210, 1000, 650]},
        {"label": "window", "role": "forbidden", "bbox_xyxy": [600, 205, 1280, 690]},
        {"label": "floor", "role": "support", "bbox_xyxy": [0, 620, 1280, 1708]},
        {"label": "roof", "role": "support", "bbox_xyxy": [0, 0, 1280, 280]},
        {"label": "door", "role": "forbidden", "bbox_xyxy": [45, 260, 190, 650]},
        {"label": "opening", "role": "forbidden", "bbox_xyxy": [600, 230, 1280, 640]},
        {"label": "painting", "role": "existing_content", "bbox_xyxy": [405, 315, 465, 405]},
        {"label": "painting", "role": "existing_content", "bbox_xyxy": [480, 320, 540, 410]},
        {"label": "person", "role": "foreground", "bbox_xyxy": [250, 410, 330, 720]},
        {"label": "person", "role": "foreground", "bbox_xyxy": [720, 405, 810, 715]},
        {"label": "lamp", "role": "forbidden", "bbox_xyxy": [290, 110, 330, 190]},
        {"label": "lamp", "role": "forbidden", "bbox_xyxy": [650, 100, 700, 185]},
        {"label": "cabinet", "role": "support", "bbox_xyxy": [370, 500, 580, 670]},
        {"label": "chair", "role": "foreground", "bbox_xyxy": [200, 520, 300, 720]},
        {"label": "curtain", "role": "forbidden", "bbox_xyxy": [900, 220, 1280, 690]},
        {"label": "track_lighting", "role": "forbidden", "bbox_xyxy": [200, 80, 800, 145]},
    ]

    mode, debug = _infer_vlm_bbox_coordinate_space(objects, 1280, 1708)

    assert mode == "normalized_0_1000_with_original_boundary_sentinels"
    joint = debug["joint_coordinate_inference"]
    assert joint["accepted"] is True
    assert joint["selected"] == "grid_boundary_x__grid_boundary_y"
    assert joint["score_margin"] >= joint["minimum_score_margin"]
    assert joint["policy"] == "joint_2d_pixel_grid_hypothesis_with_scene_evidence_v4"


def test_v13_ambiguous_joint_hypothesis_exposes_no_selected_coordinate_space():
    objects = [
        {"label": "wall", "role": "wall", "bbox_xyxy": [0, 200, 500, 600]},
        {"label": "floor", "role": "support", "bbox_xyxy": [0, 600, 1280, 1708]},
    ]

    mode, debug = _infer_vlm_bbox_coordinate_space(objects, 1280, 1708)

    assert mode is None
    assert debug["source_coordinate_space"] is None
    joint = debug["joint_coordinate_inference"]
    assert joint["accepted"] is False
    assert joint["selected"] is None
    assert joint["fail_closed"] is True
    # Per-axis diagnostics also must not invent a selected mode when their own
    # acceptance threshold was not met.
    for axis in ("x", "y"):
        item = debug["dynamic_axis_coordinate_inference"].get(axis)
        if item is not None and not item["accepted"]:
            assert item["selected"] is None


def test_regression_mixed_original_x_and_low_extent_grid_y_with_bottom_sentinel():
    # Production regression: x is already in 1280-pixel space, while ordinary
    # y values use a 0..1000 grid and only bottom-touching boxes emit y2=1708.
    # The old fixed internal_max>=850 test missed this because internal y stops
    # near 650, compressing all semantic boxes into the upper image region.
    objects = [
        {"label": "wall_left", "role": "wall", "bbox_xyxy": [0, 220, 430, 640]},
        {"label": "wall_front", "role": "wall", "bbox_xyxy": [430, 257, 770, 487]},
        {"label": "wall_right", "role": "wall", "bbox_xyxy": [770, 210, 1280, 650]},
        {"label": "window", "role": "forbidden", "bbox_xyxy": [820, 205, 1280, 690]},
        {"label": "floor", "role": "support", "bbox_xyxy": [0, 620, 1280, 1708]},
        {"label": "roof", "role": "support", "bbox_xyxy": [0, 0, 1280, 280]},
    ]

    mode, debug = _infer_vlm_bbox_coordinate_space(objects, 1280, 1708)

    assert mode == "mixed_x_original_y_1000_boundary"
    dynamic = debug["dynamic_axis_coordinate_inference"]
    assert dynamic["x"]["selected"] == "original"
    assert dynamic["y"]["selected"] == "grid_boundary"
    assert dynamic["y"]["internal_max"] < 850
    front = _normalize_vlm_bbox_to_original_pixels([430, 257, 770, 487], 1280, 1708, mode)
    assert all(abs(a - b) < 1e-6 for a, b in zip(front, [430.0, 438.956, 770.0, 831.796]))
    floor = _normalize_vlm_bbox_to_original_pixels([0, 620, 1280, 1708], 1280, 1708, mode)
    assert all(abs(a - b) < 1e-6 for a, b in zip(floor, [0.0, 1058.96, 1279.0, 1707.0]))


def test_dynamic_axis_scoring_does_not_rescale_true_original_pixel_boxes():
    objects = [
        {"label": "wall_left", "role": "wall", "bbox_xyxy": [0, 280, 430, 1120]},
        {"label": "wall_front", "role": "wall", "bbox_xyxy": [430, 300, 770, 1050]},
        {"label": "wall_right", "role": "wall", "bbox_xyxy": [770, 260, 1280, 1110]},
        {"label": "floor", "role": "support", "bbox_xyxy": [0, 1040, 1280, 1708]},
        {"label": "roof", "role": "support", "bbox_xyxy": [0, 0, 1280, 330]},
    ]

    mode, debug = _infer_vlm_bbox_coordinate_space(objects, 1280, 1708)

    assert mode == "original_pixels"
    assert debug["dynamic_axis_coordinate_inference"]["x"]["selected"] == "original"
    # y has values >1024, so it is not eligible for the grid-sentinel family.
    assert debug["dynamic_axis_coordinate_inference"]["y"] is None


def test_same_call_vlm_binding_preserves_exact_object_snapshot_and_hashes():
    raw = {
        "coordinate_space": "pixels",
        "image_size_px": [1280, 1708],
        "objects": [{"label": "wall", "bbox_xyxy": [1, 2, 3, 4]}],
        "_vlm_call_audit": {
            "provider": "openai_compatible",
            "model": "vision-model",
            "provider_request_id": "req-123",
            "prompt_sha256": "prompt-hash",
        },
    }
    binding = _build_vlm_response_binding(raw, b"same-image")
    assert binding["object_count"] == 1
    assert binding["provider_request_id"] == "req-123"
    assert binding["raw_response_snapshot"]["objects"] == raw["objects"]
    assert "_vlm_call_audit" not in binding["raw_response_snapshot"]
    assert len(binding["image_sha256"]) == 64
    assert len(binding["response_sha256"]) == 64

from semantic_mvp.service import _canonical_label_and_role, _build_object_inventory


def test_semantic_label_aliases_promote_art_and_people_to_cleanup_inventory():
    assert _canonical_label_and_role("wall_front_01", "wall") == ("wall", "wall")
    assert _canonical_label_and_role("painting", "wall") == ("painting", "existing_content")
    assert _canonical_label_and_role("painting_01", "existing_content") == ("painting", "existing_content")
    assert _canonical_label_and_role("framed_art_02", "existing_content") == ("framed_art", "existing_content")
    assert _canonical_label_and_role("people", None) == ("person", "foreground")
    assert _canonical_label_and_role("person_02", "foreground") == ("person", "foreground")
    assert _canonical_label_and_role("sheer_curtains", None) == ("curtain", "forbidden")
    objects = [
        {"object_id": "paint_1", "label": "painting", "role": "existing_content", "bbox_xyxy_original": [1, 2, 3, 4]},
        {"object_id": "person_1", "label": "person", "role": "foreground", "bbox_xyxy_original": [5, 6, 7, 8]},
    ]
    inventory = _build_object_inventory(objects)
    assert inventory["status"] == "ok"
    assert inventory["counts_by_role"]["existing_content"] == 1
    assert inventory["counts_by_role"]["foreground"] == 1
    assert inventory["per_wall_high_res_crop_used"] is False
