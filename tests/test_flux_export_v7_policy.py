from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hanging_mvp.flux_export import (
    build_render_prompt,
    select_flux_manifest_items,
    select_public_candidate_after_exit_gate,
)
from hanging_mvp.masks import load_semantic_masks, masks_by_label, masks_by_role


def _entry(asset_id: str, wall_id: str, score: float, x0: int) -> dict:
    return {
        "asset_id": asset_id,
        "candidate_id": asset_id,
        "wall_id": wall_id,
        "score": score,
        "bbox_px": [x0, 10, x0 + 100, 80],
    }


def test_select_flux_manifest_items_caps_per_wall_and_global():
    entries = [
        _entry("r0", "right", 0.99, 100),
        _entry("r1", "right", 0.98, 400),
        _entry("r2", "right", 0.97, 700),
        _entry("f0", "front", 0.96, 200),
        _entry("f1", "front", 0.95, 500),
    ]
    selected, summary = select_flux_manifest_items(entries, per_wall_limit=2, global_limit=4, nms_center_distance_fraction=0.1)
    assert len(selected) == 4
    assert sum(1 for x in selected if x["wall_id"] == "right") == 2
    assert sum(1 for x in selected if x["wall_id"] == "front") == 2
    assert summary["dropped_count"] == 1


def test_public_candidate_never_falls_back_to_unapproved_wall():
    candidates = [
        {"candidate_id": "curtain", "wall_id": "left", "score": 0.99, "approved_for_flux_render": False},
        {"candidate_id": "front", "wall_id": "front", "score": 0.90, "approved_for_flux_render": True},
    ]
    final_items = [{"candidate_id": "front", "wall_id": "front", "asset_id": "front_asset"}]
    selected = select_public_candidate_after_exit_gate(candidates, final_items)
    assert selected["candidate_id"] == "front"
    assert select_public_candidate_after_exit_gate(candidates, []) is None




def test_select_flux_manifest_items_drops_same_wall_polygon_overlap_even_when_centers_pass():
    entries = [
        _entry("a0", "right", 0.99, 100),
        _entry("a1", "right", 0.98, 135),
        _entry("a2", "right", 0.97, 320),
    ]
    selected, summary = select_flux_manifest_items(
        entries,
        per_wall_limit=3,
        global_limit=3,
        nms_center_distance_fraction=0.1,
        nms_polygon_iou_threshold=0.25,
    )
    assert [x["asset_id"] for x in selected] == ["a0", "a2"]
    assert summary["dropped_count"] == 1
    dropped = summary["dropped"][0]
    assert dropped["status"] == "dropped_by_same_wall_nms"
    assert dropped["drop_reason"] == "projected_polygon_iou"
    assert dropped["projected_polygon_iou"] > 0.25

def test_render_prompt_is_conservative_and_injects_bbox_polygon():
    prompt = build_render_prompt(
        {
            "candidate_id": "cand_001",
            "wall_id": "wall_a",
            "risk_level": "low",
            "artwork_has_frame": True,
            "projection": {"projected_quad_px_original": [[10.2, 20.1], [80.0, 20.0], [80.0, 60.0], [10.0, 60.0]]},
        },
        flux2_input_mode="dual_erased_room_plus_target",
    )
    assert "bbox_px" in prompt
    assert "polygon_px" in prompt
    assert "Image 1 is the cleaned hard-paste target" in prompt
    assert "Image 2 is the already-erased clean room reference" in prompt
    assert "STRICT SIZE LOCK" in prompt
    assert "Do not change the painting core size" in prompt
    assert "add mat board" in prompt
    assert "Return a clean high-resolution product delivery image" not in prompt
    assert "Add realistic frame thickness" not in prompt
    assert "Do not add a frame" in prompt or "Keep that frame exactly" in prompt
    assert "FINAL SIZE LOCK AFTER FRAME RULE" in prompt



def test_rules_document_remove_mask_dilation_stack():
    import json
    rules = json.loads((Path(__file__).resolve().parents[1] / "configs" / "rules_v1.json").read_text(encoding="utf-8"))
    cleanup = rules["cleanup_policy"]
    assert cleanup["remove_mask_dilate_px"] == 6
    assert cleanup["bfl_erase_dilate_pixels_default"] == 10
    policy = cleanup["remove_mask_dilation_policy"]
    assert policy["effective_total_dilate_px_approx"] == 16
    assert "two-stage" in policy["intent"]


def test_hanging_masks_canonicalize_vlm_instance_labels_for_cleanup():
    semantics = {
        "objects": [
            {
                "object_id": "painting_01",
                "label": "painting_01",
                "role": "existing_content",
                "polygon": [[10, 10], [30, 10], [30, 30], [10, 30]],
            },
            {
                "object_id": "window_02",
                "label": "window_02",
                "role": "forbidden",
                "polygon": [[40, 10], [55, 10], [55, 25], [40, 25]],
            },
        ]
    }
    objects = load_semantic_masks(semantics, (64, 64))
    assert objects["painting_01"]["label"] == "painting"
    assert objects["window_02"]["label"] == "window"
    assert len(masks_by_role(objects, "existing_content")) == 1
    assert len(masks_by_label(objects, {"painting"})) == 1
    assert len(masks_by_label(objects, {"window"})) == 1
