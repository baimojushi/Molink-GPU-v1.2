import math

import numpy as np

from hanging_mvp.flux_export import build_remove_mask_for_hardpaste
from hanging_mvp.geometry import make_wall_basis, refine_wall_orientation_evidence
from hanging_mvp.schemas import WallPlane
from hanging_mvp.support_context import _estimate_same_wall_art_height_anchor


def _scene(shape=(120, 160)):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    xyz = np.stack([
        (xx - w / 2.0) / 80.0,
        (yy - h / 2.0) / 80.0,
        np.full((h, w), 5.0),
    ], axis=-1).astype(np.float32)
    return xyz


def test_detected_art_is_peeled_from_orientation_but_kept_for_anchor_and_cleanup():
    h, w = 120, 160
    gravity = np.array([0.0, -1.0, 0.0], np.float32)
    wall_normal = np.array([0.0, 0.0, 1.0], np.float32)
    art_normal = np.array([1.0, 0.0, 0.0], np.float32)
    plane = WallPlane((0.0, 0.0, 1.0), -5.0)
    xyz = _scene((h, w))

    art_mask = np.zeros((h, w), np.uint8)
    art_mask[40:80, 62:98] = 255
    # A slightly wider normal/depth halo models frame sides and a leaning work
    # that are not fully covered by the raw semantic mask.
    art_halo = np.zeros_like(art_mask)
    art_halo[36:84, 58:102] = 255
    xyz[art_halo > 0, 2] = 4.90

    normals = np.broadcast_to(wall_normal, (h, w, 3)).copy()
    normals[art_halo > 0] = art_normal
    evidence = np.ones((h, w), np.uint8) * 255
    evidence[art_mask > 0] = 0  # service already subtracts the exact object mask

    refined, debug = refine_wall_orientation_evidence(
        normals,
        xyz,
        evidence,
        plane,
        gravity,
        semantic_art_mask=art_mask,
        art_mask_dilate_px=6,
        whole_depth_inlier_ratio=0.90,
        segmented_depth_yaw=math.pi / 2.0,
        segmented_depth_confident=True,
        min_retained_pixels=1000,
    )

    assert debug["raw_normal_consensus"]["reliable"] is True
    assert debug["reason"] == "semantic_clean_evidence_already_reliable"
    assert np.count_nonzero(refined[art_halo > 0]) == 0
    assert debug["artwork_retained_for_height_anchor"] is True
    assert debug["artwork_retained_for_candidate_cleanup"] is True

    anchors = xyz[np.ones((h, w), dtype=bool)].reshape(-1, 3)
    basis = make_wall_basis(plane, anchors, gravity)
    objects = {
        "artwork_01": {
            "object_id": "artwork_01",
            "label": "existing_art",
            "role": "existing_content",
            "mask": art_mask.copy(),
        }
    }
    height_anchor = _estimate_same_wall_art_height_anchor(
        objects=objects,
        xyz=xyz,
        plane=plane,
        basis=basis,
        s_range_m=[-2.0, 2.0],
        rules={
            "existing_art_height_anchor": {
                "enabled": True,
                "labels": ["existing_art"],
                "min_points": 30,
                "max_plane_distance_m": 0.22,
                "min_near_plane_ratio": 0.30,
                "s_margin_m": 0.20,
                "max_anchor_spread_m": 0.45,
            }
        },
    )
    assert height_anchor is not None
    assert height_anchor["source"] == "same_wall_existing_art_center"
    assert height_anchor["anchors"][0]["object_id"] == "artwork_01"

    remove = build_remove_mask_for_hardpaste(
        art_region_mask=np.ones((h, w), np.uint8) * 255,
        current_wall_mask=np.ones((h, w), np.uint8) * 255,
        existing_masks=list(objects.values()),
        foreground_masks=[],
        original_size_wh=(w, h),
        remove_foreground_global=False,
        dilate_px=0,
        protect_art_region=False,
    )
    assert np.count_nonzero(remove & art_mask) == np.count_nonzero(art_mask)


def test_bimodal_wall_normals_recover_only_the_depth_supported_material_mode():
    h, w = 120, 160
    gravity = np.array([0.0, -1.0, 0.0], np.float32)
    wall_normal = np.array([0.0, 0.0, 1.0], np.float32)
    adjacent_plane_normal = np.array([1.0, 0.0, 0.0], np.float32)
    plane = WallPlane((0.0, 0.0, 1.0), -5.0)
    xyz = _scene((h, w))

    split = 48  # 30% coherent adjacent plane, 70% target wall
    xyz[:, :split, 2] = 3.0
    normals = np.broadcast_to(wall_normal, (h, w, 3)).copy()
    normals[:, :split] = adjacent_plane_normal
    evidence = np.ones((h, w), np.uint8) * 255

    refined, debug = refine_wall_orientation_evidence(
        normals,
        xyz,
        evidence,
        plane,
        gravity,
        whole_depth_inlier_ratio=0.67,
        segmented_depth_yaw=math.pi / 2.0,
        segmented_depth_confident=True,
        min_retained_pixels=1000,
        min_component_pixels=50,
    )

    raw = debug["raw_normal_consensus"]
    assert raw["angular_spread_p50_deg"] < 1.0
    assert raw["angular_spread_p90_deg"] > 80.0
    assert debug["recovered"] is True
    assert debug["reason"] == "depth_supported_wall_material_mode_recovered"
    assert debug["refined_normal_consensus"]["reliable"] is True
    assert np.count_nonzero(refined[:, :split]) == 0
    assert np.count_nonzero(refined[:, split:]) > 0.95 * h * (w - split)


def test_bimodal_recovery_fails_closed_without_independent_segmented_depth_agreement():
    h, w = 100, 140
    gravity = np.array([0.0, -1.0, 0.0], np.float32)
    plane = WallPlane((0.0, 0.0, 1.0), -5.0)
    xyz = _scene((h, w))
    xyz[:, :45, 2] = 3.0
    normals = np.zeros((h, w, 3), np.float32)
    normals[..., 2] = 1.0
    normals[:, :45] = np.array([1.0, 0.0, 0.0], np.float32)
    evidence = np.ones((h, w), np.uint8) * 255

    refined, debug = refine_wall_orientation_evidence(
        normals,
        xyz,
        evidence,
        plane,
        gravity,
        whole_depth_inlier_ratio=0.70,
        segmented_depth_yaw=0.0,  # 90 degrees away from the whole-wall depth yaw
        segmented_depth_confident=True,
        min_retained_pixels=1000,
    )

    assert debug["recovered"] is False
    assert debug["reason"] == "depth_supported_material_mode_recovery_gates_failed"
    assert debug["gates"]["whole_and_segmented_depth_agree"] is False
    assert np.array_equal(refined, evidence)
