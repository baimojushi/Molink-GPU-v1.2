"""Regression coverage for V20 projection-shape feedback arbitration."""

import numpy as np

from hanging_mvp.geometry import (
    evaluate_projection_orientation_shape_feedback,
    resolve_wall_projection_plane,
)
from hanging_mvp.schemas import Intrinsics, WallPlane


DIAGNOSTIC_DEPTH_NORMAL = np.array(
    [0.8388601690987286, 0.15779406765668144, -0.5209747104342122],
    dtype=np.float32,
)
DIAGNOSTIC_DIRECT_NORMAL = np.array(
    [0.6051228046417236, 0.21059982478618622, -0.7677721381187439],
    dtype=np.float32,
)
DIAGNOSTIC_ANCHOR = np.array(
    [-0.5091041922569275, -1.6276906728744507, 10.648569107055664],
    dtype=np.float32,
)
DIAGNOSTIC_GRAVITY = np.array(
    [0.05921376869082451, -0.9778359532356262, -0.2008247971534729],
    dtype=np.float32,
)
INTRINSICS = Intrinsics(fx=996.969, fy=996.969, cx=575.5, cy=768.0)


def test_v19_depth_heading_is_replaced_by_safe_physical_direct_plane():
    selected, normal, debug = evaluate_projection_orientation_shape_feedback(
        DIAGNOSTIC_DEPTH_NORMAL,
        DIAGNOSTIC_DIRECT_NORMAL,
        DIAGNOSTIC_ANCHOR,
        DIAGNOSTIC_GRAVITY,
        INTRINSICS,
    )

    assert selected is True
    assert normal is not None
    assert debug["depth_hierarchy_probe"]["distortion_ratio"] > 1.70
    assert debug["direct_normal_probe"]["distortion_ratio"] < 1.25
    assert debug["distortion_improvement"] > 0.50
    assert debug["failed_gates"] == []
    assert debug["policy"] == "reproject_two_physical_3d_plane_hypotheses_no_2d_width_recovery"


def test_feedback_does_not_select_an_alternative_that_still_needs_review():
    selected, normal, debug = evaluate_projection_orientation_shape_feedback(
        DIAGNOSTIC_DEPTH_NORMAL,
        DIAGNOSTIC_DEPTH_NORMAL,
        DIAGNOSTIC_ANCHOR,
        DIAGNOSTIC_GRAVITY,
        INTRINSICS,
    )

    assert selected is False
    assert normal is None
    assert "direct_projection_inside_safe_region" in debug["failed_gates"]
    assert "distortion_improvement" in debug["failed_gates"]


def test_feedback_fails_closed_above_the_sensor_conflict_hard_limit():
    far_normal = np.array([-0.10, 0.0, -0.995], dtype=np.float32)
    selected, normal, debug = evaluate_projection_orientation_shape_feedback(
        DIAGNOSTIC_DEPTH_NORMAL,
        far_normal,
        DIAGNOSTIC_ANCHOR,
        DIAGNOSTIC_GRAVITY,
        INTRINSICS,
        max_depth_direct_yaw_disagreement_deg=15.0,
    )

    assert selected is False
    assert normal is None
    assert "yaw_conflict_below_hard_limit" in debug["failed_gates"]


def test_feedback_never_mutates_or_rectifies_a_projected_quad():
    selected, _normal, debug = evaluate_projection_orientation_shape_feedback(
        DIAGNOSTIC_DEPTH_NORMAL,
        DIAGNOSTIC_DIRECT_NORMAL,
        DIAGNOSTIC_ANCHOR,
        DIAGNOSTIC_GRAVITY,
        INTRINSICS,
    )

    assert selected is True
    assert "no_2d_width_recovery" in debug["policy"]
    assert "projected_quad" not in debug


def test_resolver_runs_v19_hierarchy_then_v20_feedback_on_recorded_geometry():
    raw_depth_normal = np.array(
        [0.8340790271759033, 0.24945688247680664, -0.4920198619365692],
        dtype=np.float32,
    )
    normal_field = np.tile(DIAGNOSTIC_DIRECT_NORMAL, (40, 40, 1))
    evidence_mask = np.full((40, 40), 255, dtype=np.uint8)
    depth_plane = WallPlane(
        tuple(float(x) for x in raw_depth_normal),
        -float(np.dot(raw_depth_normal, DIAGNOSTIC_ANCHOR)),
    )

    projection_plane, debug = resolve_wall_projection_plane(
        depth_plane,
        np.tile(DIAGNOSTIC_ANCHOR, (200, 1)),
        normal_field,
        evidence_mask,
        DIAGNOSTIC_GRAVITY,
        intr=INTRINSICS,
        normal_min_samples=100,
        horizontal_structure_enabled=False,
        segmented_depth_plane_yaw=np.radians(-32.3453361564),
        segmented_depth_plane_confident=True,
        segmented_depth_plane_debug={
            "cluster_coverage": 1.0,
            "cluster_p90_deg": 4.04,
            "cluster_size": 16,
        },
        whole_depth_plane_inlier_ratio=0.98812,
        hierarchical_depth_consensus_enabled=True,
        projection_shape_feedback_enabled=True,
    )

    assert debug["hard_reject"] is False
    assert debug["hierarchical_depth_consensus"]["recovered"] is True
    assert debug["projection_shape_feedback"]["selected"] is True
    assert debug["policy"] == "projection_shape_feedback_orientation_recovery"
    assert debug["projection_orientation_source"] == "direct_normal_shape_feedback_after_depth_hierarchy"
    assert debug["direct_normal_used_for_yaw"] is True
    assert np.allclose(np.asarray(projection_plane.n), DIAGNOSTIC_DIRECT_NORMAL, atol=0.04)
