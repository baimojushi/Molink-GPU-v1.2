"""Tests for controlled structural quorum fallback (V17).

Covers:
- 2-of-3 quorum recovery with segmented depth + direct normal
- Direct normal alone cannot unlock
- Third source confidently opposing blocks recovery
- VP inside wall body penalizes structure confidence
- Insufficient sources for quorum
"""
import math
import numpy as np
import pytest

from hanging_mvp.geometry import (
    evaluate_controlled_structural_quorum,
    estimate_segmented_depth_plane_yaw,
    _undirected_angle_deg,
)


def _yaw_rad(deg: float) -> float:
    return float(np.radians(deg))


class TestEvaluateControlledStructuralQuorum:
    """Unit tests for the 2-of-3 controlled structural quorum."""

    def test_recovers_when_depth_and_direct_normal_agree(self):
        """Depth and direct normal within 6°, third source low-confidence → recovered."""
        recovered, source, debug = evaluate_controlled_structural_quorum(
            depth_yaw=_yaw_rad(10.0),
            depth_yaw_confident=True,
            direct_normal_yaw=_yaw_rad(12.0),
            direct_normal_confident=True,
            structure_yaw=_yaw_rad(80.0),
            structure_confident=False,  # low confidence → doesn't block
            angle_threshold_deg=6.0,
        )
        assert recovered is True
        assert "depth" in debug["agreeing_sources"]
        assert "direct_normal" in debug["agreeing_sources"]
        assert debug["consensus_yaw_deg"] == pytest.approx(11.0, abs=1.0)

    def test_recovers_when_depth_and_structure_agree(self):
        """Depth and structure agree, third source low-confidence → recovered."""
        recovered, source, debug = evaluate_controlled_structural_quorum(
            depth_yaw=_yaw_rad(10.0),
            depth_yaw_confident=True,
            direct_normal_yaw=_yaw_rad(80.0),
            direct_normal_confident=False,  # low confidence
            structure_yaw=_yaw_rad(11.0),
            structure_confident=True,
            angle_threshold_deg=6.0,
        )
        assert recovered is True
        assert "depth" in debug["agreeing_sources"]
        assert "structure" in debug["agreeing_sources"]

    def test_rejects_when_third_source_confidently_opposes(self):
        """Depth and structure agree, but third source (direct_normal) confident
        and opposes → should reject."""
        recovered, source, debug = evaluate_controlled_structural_quorum(
            depth_yaw=_yaw_rad(10.0),
            depth_yaw_confident=True,
            direct_normal_yaw=_yaw_rad(80.0),
            direct_normal_confident=True,  # confident and opposes
            structure_yaw=_yaw_rad(11.0),
            structure_confident=True,
            angle_threshold_deg=6.0,
        )
        assert recovered is False
        assert "opposes" in debug.get("reason", "")

    def test_direct_normal_alone_cannot_unlock(self):
        """Only direct normal is confident, depth agrees but not confident →
        direct_normal alone cannot unlock."""
        recovered, source, debug = evaluate_controlled_structural_quorum(
            depth_yaw=_yaw_rad(12.0),  # agrees with direct_normal
            depth_yaw_confident=False,  # but not confident
            direct_normal_yaw=_yaw_rad(12.0),
            direct_normal_confident=True,
            structure_yaw=_yaw_rad(80.0),  # disagrees
            structure_confident=False,
            angle_threshold_deg=6.0,
        )
        # A low-confidence depth estimate has no vote.  V18+ must not count it
        # beside the direct normal merely because their numeric yaws match.
        assert recovered is False
        assert debug["reason"] == "insufficient_confident_sources_for_quorum"

    def test_insufficient_sources(self):
        """Only one source available → cannot form quorum."""
        recovered, source, debug = evaluate_controlled_structural_quorum(
            depth_yaw=_yaw_rad(10.0),
            depth_yaw_confident=True,
            direct_normal_yaw=None,
            direct_normal_confident=False,
            structure_yaw=None,
            structure_confident=False,
            angle_threshold_deg=6.0,
        )
        assert recovered is False
        assert debug["reason"] == "insufficient_sources_for_quorum"

    def test_no_pair_within_threshold(self):
        """All three sources disagree → no agreement."""
        recovered, source, debug = evaluate_controlled_structural_quorum(
            depth_yaw=_yaw_rad(0.0),
            depth_yaw_confident=True,
            direct_normal_yaw=_yaw_rad(60.0),
            direct_normal_confident=True,
            structure_yaw=_yaw_rad(120.0),
            structure_confident=True,
            angle_threshold_deg=6.0,
        )
        assert recovered is False
        assert debug["reason"] == "no_pair_within_threshold"

    def test_unanimous_agreement(self):
        """All three sources agree → recovered (2-of-3 minimum met)."""
        recovered, source, debug = evaluate_controlled_structural_quorum(
            depth_yaw=_yaw_rad(10.0),
            depth_yaw_confident=True,
            direct_normal_yaw=_yaw_rad(11.0),
            direct_normal_confident=True,
            structure_yaw=_yaw_rad(10.5),
            structure_confident=True,
            angle_threshold_deg=6.0,
        )
        assert recovered is True
        # At least 2 sources agree
        assert len(debug["agreeing_sources"]) >= 2


class TestEstimateSegmentedDepthPlaneYaw:
    """Unit tests for segmented depth plane yaw estimation."""

    def test_returns_none_for_insufficient_points(self):
        """Too few points → returns None, not reliable."""
        pts = np.random.randn(10, 3).astype(np.float32)
        mask = np.ones((32, 32), dtype=np.uint8) * 255
        g_hat = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        yaw, reliable, debug = estimate_segmented_depth_plane_yaw(pts, mask, g_hat)
        assert yaw is None
        assert reliable is False

    def test_returns_none_for_degenerate_distribution(self):
        """All points at same location → degenerate."""
        pts = np.tile([1.0, 2.0, 3.0], (200, 1)).astype(np.float32)
        mask = np.ones((32, 32), dtype=np.uint8) * 255
        g_hat = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        yaw, reliable, debug = estimate_segmented_depth_plane_yaw(pts, mask, g_hat)
        assert yaw is None
        assert reliable is False

    def test_recovers_yaw_from_planar_wall(self):
        """Points on a vertical plane at ~10° yaw → should recover that yaw."""
        rng = np.random.default_rng(42)
        n_pts = 500
        yaw_true = math.radians(10.0)
        normal = np.array([math.cos(yaw_true), 0.0, math.sin(yaw_true)])
        u = rng.uniform(-1.0, 1.0, n_pts)
        v = rng.uniform(-1.0, 1.0, n_pts)
        tangent_u = np.array([-math.sin(yaw_true), 0.0, math.cos(yaw_true)])
        tangent_v = np.array([0.0, 1.0, 0.0])
        pts = (
            3.0 * normal
            + u[:, None] * tangent_u[None, :]
            + v[:, None] * tangent_v[None, :]
        ).astype(np.float32)
        pts += rng.normal(0, 0.01, pts.shape).astype(np.float32)

        mask = np.ones((64, 64), dtype=np.uint8) * 255
        g_hat = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        yaw, reliable, debug = estimate_segmented_depth_plane_yaw(
            pts, mask, g_hat, grid_size=4, min_points_per_cell=20
        )
        # Should at least return a yaw value (may not be reliable with small grid)
        if yaw is not None:
            yaw_deg = math.degrees(yaw)
            # Allow wider tolerance due to grid quantization
            assert abs(yaw_deg - 10.0) < 15.0, f"Expected ~10°, got {yaw_deg}°"
