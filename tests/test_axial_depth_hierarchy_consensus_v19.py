"""Regression coverage for the V19 axial depth hierarchy policy."""

import math

import numpy as np
import pytest

import hanging_mvp.geometry as geometry
from hanging_mvp.geometry import (
    estimate_segmented_depth_plane_yaw,
    estimate_wall_horizontal_structure_direction,
    evaluate_hierarchical_depth_consensus,
)
from hanging_mvp.schemas import Intrinsics, WallPlane


def _axial_error_deg(actual: float, expected: float) -> float:
    delta = abs(float(actual) - float(expected)) % 180.0
    return min(delta, 180.0 - delta)


def test_segmented_depth_clusters_opposite_normal_signs_as_one_axis(monkeypatch):
    """+146 and -30 degree normals are one wall axis, not two clusters."""
    calls = {"count": 0}

    def fake_fit(_points, iterations=300, threshold=0.06):
        yaw_deg = 146.0 if calls["count"] % 2 == 0 else -30.0
        calls["count"] += 1
        yaw = math.radians(yaw_deg)
        return WallPlane((math.cos(yaw), 0.0, math.sin(yaw)), -3.0), 0.92

    monkeypatch.setattr(geometry, "fit_plane_ransac", fake_fit)
    xyz = np.zeros((40, 40, 3), dtype=np.float32)
    xyz[..., 0] = np.linspace(-1.0, 1.0, 40)[None, :]
    xyz[..., 1] = np.linspace(-1.0, 1.0, 40)[:, None]
    xyz[..., 2] = 3.0
    mask = np.full((40, 40), 255, dtype=np.uint8)

    yaw, reliable, debug = estimate_segmented_depth_plane_yaw(
        xyz,
        mask,
        np.array([0.0, -1.0, 0.0], dtype=np.float32),
        grid_size=4,
        min_points_per_cell=20,
        cluster_angle_deg=6.0,
        min_cluster_cells=8,
        min_cluster_coverage=0.75,
    )

    assert reliable is True
    assert yaw is not None
    assert debug["cluster_size"] == 16
    assert debug["cluster_coverage"] == pytest.approx(1.0)
    assert debug["orientation_period_deg"] == 180.0
    assert debug["normal_sign_invariant"] is True
    assert _axial_error_deg(math.degrees(yaw), -32.0) < 0.25


def test_inside_wall_vanishing_point_never_gets_structure_vote(monkeypatch):
    """Line count cannot rescue a texture family whose VP is in the wall."""
    synthetic_lines = np.asarray(
        [
            [[20, 20, 50, 50]],
            [[20, 35, 50, 50]],
            [[50, 50, 80, 70]],
            [[35, 20, 50, 50]],
            [[50, 50, 70, 80]],
            [[30, 30, 70, 70]],
            [[30, 40, 70, 60]],
            [[40, 30, 60, 70]],
        ],
        dtype=np.int32,
    )
    monkeypatch.setattr(geometry.cv2, "HoughLinesP", lambda *args, **kwargs: synthetic_lines)
    image = np.tile(np.arange(100, dtype=np.uint8), (100, 1))
    mask = np.full((100, 100), 255, dtype=np.uint8)
    intr = Intrinsics(fx=50.0, fy=50.0, cx=50.0, cy=50.0)

    _direction, debug = estimate_wall_horizontal_structure_direction(
        image,
        mask,
        intr,
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, -1.0, 0.0], dtype=np.float32),
        prefilter_angle_deg=30.0,
        min_unique_lines=4,
    )

    assert debug["vanishing_point_inside_wall_body"] is True
    assert debug["vp_penalty_applied"] is True
    assert debug["reliable"] is False
    assert debug["reason"] == "vanishing_point_inside_wall_body_texture_family"


def test_hierarchical_depth_recovers_only_after_every_gate_passes():
    recovered, consensus_yaw, debug = evaluate_hierarchical_depth_consensus(
        whole_depth_yaw=math.radians(147.2),
        whole_depth_inlier_ratio=0.88,
        whole_depth_vertical_tilt_deg=6.1,
        segmented_depth_yaw=math.radians(-32.36),
        segmented_depth_confident=True,
        segmented_depth_debug={
            "cluster_coverage": 1.0,
            "cluster_p90_deg": 4.03,
            "cluster_size": 16,
        },
        direct_normal_material_reliable=True,
    )

    assert recovered is True
    assert consensus_yaw is not None
    assert debug["failed_gates"] == []
    assert debug["direct_normal_used_for_yaw"] is False
    assert _axial_error_deg(math.degrees(consensus_yaw), -32.58) < 0.5


@pytest.mark.parametrize(
    "overrides,failed_gate",
    [
        ({"whole_depth_inlier_ratio": 0.50}, "whole_depth_inlier_ratio"),
        ({"whole_depth_vertical_tilt_deg": 16.0}, "whole_depth_vertical_tilt"),
        ({"segmented_depth_confident": False}, "segmented_depth_confident"),
        ({"segmented_depth_debug": {"cluster_coverage": 0.5, "cluster_p90_deg": 4.0, "cluster_size": 16}}, "segmented_cluster_coverage"),
        ({"segmented_depth_debug": {"cluster_coverage": 1.0, "cluster_p90_deg": 9.0, "cluster_size": 16}}, "segmented_cluster_p90"),
        ({"segmented_depth_yaw": math.radians(-10.0)}, "depth_hierarchy_yaw_agreement"),
        ({"direct_normal_material_reliable": False}, "direct_normal_material_stability"),
    ],
)
def test_hierarchical_depth_fails_closed_when_any_gate_fails(overrides, failed_gate):
    values = {
        "whole_depth_yaw": math.radians(147.2),
        "whole_depth_inlier_ratio": 0.88,
        "whole_depth_vertical_tilt_deg": 6.1,
        "segmented_depth_yaw": math.radians(-32.36),
        "segmented_depth_confident": True,
        "segmented_depth_debug": {
            "cluster_coverage": 1.0,
            "cluster_p90_deg": 4.03,
            "cluster_size": 16,
        },
        "direct_normal_material_reliable": True,
    }
    values.update(overrides)

    recovered, consensus_yaw, debug = evaluate_hierarchical_depth_consensus(**values)

    assert recovered is False
    assert consensus_yaw is None
    assert failed_gate in debug["failed_gates"]
    assert debug["reason"] == "hierarchical_depth_quality_gate_failed"
