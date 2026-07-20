from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_mvp.service import (
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

from semantic_mvp.service import _canonical_label_and_role, _build_object_inventory


def test_semantic_label_aliases_promote_art_and_people_to_cleanup_inventory():
    assert _canonical_label_and_role("wall_front_01", "wall") == ("wall", "wall")
    assert _canonical_label_and_role("painting", "wall") == ("painting", "existing_content")
    assert _canonical_label_and_role("painting_01", "existing_content") == ("painting", "existing_content")
    assert _canonical_label_and_role("framed_art_02", "existing_content") == ("framed_art", "existing_content")
    assert _canonical_label_and_role("people", None) == ("person", "foreground")
    assert _canonical_label_and_role("person_02", "foreground") == ("person", "foreground")
    objects = [
        {"object_id": "paint_1", "label": "painting", "role": "existing_content", "bbox_xyxy_original": [1, 2, 3, 4]},
        {"object_id": "person_1", "label": "person", "role": "foreground", "bbox_xyxy_original": [5, 6, 7, 8]},
    ]
    inventory = _build_object_inventory(objects)
    assert inventory["status"] == "ok"
    assert inventory["counts_by_role"]["existing_content"] == 1
    assert inventory["counts_by_role"]["foreground"] == 1
    assert inventory["per_wall_high_res_crop_used"] is False
