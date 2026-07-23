import math

import numpy as np

from hanging_mvp.geometry import (
    backproject,
    refine_wall_orientation_evidence,
    refit_semantic_art_to_wall_plane,
    resolve_wall_projection_plane,
)
from hanging_mvp.schemas import Intrinsics, WallPlane


def _plane_xyz(h: int, w: int, z: float = 5.0):
    intr = Intrinsics(fx=180.0, fy=180.0, cx=w / 2.0, cy=h / 2.0)
    depth = np.full((h, w), z, np.float32)
    return intr, backproject(depth, intr)


def _yaw_normal(yaw_deg: float):
    yaw = math.radians(yaw_deg)
    return np.array([math.cos(yaw), 0.0, math.sin(yaw)], np.float32)


def test_structural_opening_bimodality_recovers_only_continuous_tight_normal_island():
    h, w = 120, 180
    gravity = np.array([0.0, -1.0, 0.0], np.float32)
    intr, xyz = _plane_xyz(h, w)
    del intr
    plane = WallPlane((0.0, 0.0, 1.0), -5.0)

    normals = np.zeros((h, w, 3), np.float32)
    normals[..., 2] = 1.0
    # Two window/reveal bands remain depth-connected to the semantic wall but
    # carry orthogonal local normals.  The central real-wall island is 60%.
    normals[:, :36] = np.array([1.0, 0.0, 0.0], np.float32)
    normals[:, 144:] = np.array([1.0, 0.0, 0.0], np.float32)
    evidence = np.ones((h, w), np.uint8) * 255

    refined, debug = refine_wall_orientation_evidence(
        normals,
        xyz,
        evidence,
        plane,
        gravity,
        whole_depth_inlier_ratio=0.985,
        segmented_depth_yaw=math.pi / 2.0,
        segmented_depth_confident=True,
        min_retained_pixels=1000,
        localized_min_component_pixels=1000,
    )

    assert debug["refined_normal_consensus"]["angular_spread_p50_deg"] < 1.0
    assert debug["refined_normal_consensus"]["angular_spread_p90_deg"] > 80.0
    assert debug["reason"] == "localized_coherent_normal_support_recovered"
    assert debug["segmented_depth_consensus_authorized_for_yaw"] is True
    local = debug["localized_normal_support_recovery"]
    assert local["authorized"] is True
    assert local["strict_recheck"]["angular_spread_p90_deg"] < 4.0
    assert np.count_nonzero(refined[:, :36]) == 0
    assert np.count_nonzero(refined[:, 144:]) == 0
    assert np.count_nonzero(refined[:, 36:144]) > 0.95 * h * 108


def test_fragmented_tight_normals_do_not_open_the_localized_route():
    h, w = 120, 180
    gravity = np.array([0.0, -1.0, 0.0], np.float32)
    _, xyz = _plane_xyz(h, w)
    plane = WallPlane((0.0, 0.0, 1.0), -5.0)
    normals = np.zeros((h, w, 3), np.float32)
    normals[..., 0] = 1.0

    # More than half the pixels have the target normal, but only as many small
    # disconnected patches.  This must not authorize a hangable wall mode.
    target = np.zeros((h, w), bool)
    for y in range(0, h, 20):
        for x in range(0, w, 20):
            target[y:y + 15, x:x + 15] = True
    normals[target] = np.array([0.0, 0.0, 1.0], np.float32)
    evidence = np.ones((h, w), np.uint8) * 255

    refined, debug = refine_wall_orientation_evidence(
        normals,
        xyz,
        evidence,
        plane,
        gravity,
        whole_depth_inlier_ratio=0.98,
        segmented_depth_yaw=math.pi / 2.0,
        segmented_depth_confident=True,
        min_retained_pixels=1000,
        localized_min_component_pixels=1000,
        localized_min_component_fraction=0.08,
        localized_morphology_close_px=0,
    )

    assert debug["recovered"] is False
    assert debug["reason"] == "refined_material_mode_still_fails_direct_normal_quality_gate"
    assert debug["localized_normal_support_recovery"]["authorized"] is False
    assert np.array_equal(refined, evidence)


def test_localized_route_uses_segmented_depth_yaw_not_direct_normal_yaw():
    h, w = 100, 140
    gravity = np.array([0.0, -1.0, 0.0], np.float32)
    depth_yaw_deg = 38.0
    segmented_yaw_deg = 41.72
    direct_yaw_deg = 62.0
    depth_n = _yaw_normal(depth_yaw_deg)
    direct_n = _yaw_normal(direct_yaw_deg)
    plane = WallPlane(tuple(float(x) for x in depth_n), -5.0)
    normals = np.broadcast_to(direct_n, (h, w, 3)).copy()
    evidence = np.ones((h, w), np.uint8) * 255
    anchors = np.broadcast_to(np.array([0.0, 0.0, 5.0], np.float32), (3000, 3)).copy()

    projection_plane, debug = resolve_wall_projection_plane(
        plane,
        anchors,
        normals,
        evidence,
        gravity,
        horizontal_structure_enabled=False,
        segmented_depth_plane_yaw=math.radians(segmented_yaw_deg),
        segmented_depth_plane_confident=True,
        localized_depth_consensus_authorized=True,
        localized_depth_max_yaw_disagreement_deg=8.0,
    )

    actual = np.asarray(projection_plane.n, np.float32)
    actual_yaw = math.degrees(math.atan2(float(actual[2]), float(actual[0])))
    # Axial headings can differ by 180 degrees.
    error = abs((actual_yaw - segmented_yaw_deg + 90.0) % 180.0 - 90.0)
    assert debug["hard_reject"] is False
    assert debug["projection_orientation_source"] == "segmented_depth_yaw_on_localized_coherent_normal_support"
    assert debug["direct_normal_used_for_yaw"] is False
    assert debug["normal_yaw_conflict_requires_structure_quorum"] is False
    assert error < 0.1


def test_tilted_artwork_is_refit_to_underlying_wall_only_with_surrounding_support():
    h, w = 120, 180
    intr, xyz = _plane_xyz(h, w)
    plane = WallPlane((0.0, 0.0, 1.0), -5.0)
    art = np.zeros((h, w), np.uint8)
    art[35:85, 65:115] = 255
    xyz[art > 0, 2] = 4.65  # observed tilted/protruding artwork face
    wall = np.ones((h, w), np.uint8) * 255
    support = wall.copy()
    support[art > 0] = 0

    refit_mask, refit_xyz, debug = refit_semantic_art_to_wall_plane(
        art,
        wall,
        support,
        xyz,
        intr,
        plane,
        ring_dilate_px=10,
        min_component_pixels=100,
        min_ring_wall_fraction=0.5,
        min_ring_support_fraction=0.5,
        min_supported_sides=3,
        min_side_support_fraction=0.5,
    )

    assert debug["accepted_pixel_count"] == np.count_nonzero(art)
    assert debug["artwork_observed_depth_used"] is False
    assert debug["artwork_observed_normal_used"] is False
    assert np.array_equal(refit_mask > 0, art > 0)
    distances = np.abs(refit_xyz[refit_mask > 0] @ np.array(plane.n) + plane.d)
    assert float(np.max(distances)) < 1e-4
    assert abs(float(np.median(xyz[art > 0, 2])) - 4.65) < 1e-5
    assert float(np.median(refit_xyz[art > 0, 2])) == 5.0


def test_artwork_refit_fails_when_it_touches_structural_forbidden_region():
    h, w = 100, 160
    intr, xyz = _plane_xyz(h, w)
    plane = WallPlane((0.0, 0.0, 1.0), -5.0)
    art = np.zeros((h, w), np.uint8)
    art[30:70, 60:100] = 255
    wall = np.ones((h, w), np.uint8) * 255
    support = wall.copy()
    support[art > 0] = 0
    forbidden = np.zeros_like(art)
    forbidden[:, 90:] = 255

    refit_mask, _, debug = refit_semantic_art_to_wall_plane(
        art,
        wall,
        support,
        xyz,
        intr,
        plane,
        structural_forbidden_mask=forbidden,
    )

    assert np.count_nonzero(refit_mask) == 0
    assert debug["components"][0]["reason"] == "component_overlaps_structural_forbidden_mask"
