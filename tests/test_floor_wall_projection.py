from __future__ import annotations

import numpy as np

from hanging_mvp.schemas import WallBasis, WallPlane
from hanging_mvp.support_context import _estimate_floor_wall_intersection


def test_floor_projection_uses_near_wall_subset_not_global_distance_median():
    h, w = 20, 20
    xyz = np.zeros((h, w, 3), np.float32)
    xyz[:, :, 0] = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    xyz[:, :, 1] = 1.5
    xyz[:, :, 2] = np.linspace(2.0, 4.9, h, dtype=np.float32)[:, None]
    floor_mask = np.full((h, w), 255, np.uint8)
    plane = WallPlane(n=(0.0, 0.0, -1.0), d=5.0)
    basis = WallBasis(
        origin=(0.0, 0.0, 5.0),
        u=(1.0, 0.0, 0.0),
        v=(0.0, -1.0, 0.0),
        n=(0.0, 0.0, -1.0),
        g_hat_cam=(0.0, -1.0, 0.0),
    )
    rules = {
        "room_corner_floor_fit": {"enabled": False},
        "floor_wall_intersection": {
            "min_floor_points": 20,
            "max_floor_points": 1000,
            "max_floor_wall_projection_distance_m": 0.45,
            "floor_projected_t_percentile": 50.0,
        },
    }

    result = _estimate_floor_wall_intersection(
        floor_mask, xyz, plane, basis, (-1.0, 1.0), np.array([], np.float32), rules=rules
    )

    attempt = result["debug"]["attempts"][1]
    assert attempt["signed_distance_to_wall_p50_m"] > 0.45
    assert attempt["kept_after_wall_distance_filter"] >= 20
    assert result["source"] == "floor_mask_projected_to_current_wall_plane"
    assert result["metric_trusted"] is True
    assert result["floor_below_camera"] is True
    assert np.isclose(result["floor_height_signed_m"], -1.5, atol=1e-5)


def test_floor_plane_fit_rejects_vertical_semantic_mask_contamination_by_normals():
    h, w = 40, 40
    xyz = np.zeros((h, w, 3), np.float32)
    xyz[:, :, 0] = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    # Most of the semantic floor mask is actually a vertical foreground plane.
    xyz[:30, :, 1] = np.linspace(-1.0, 1.0, 30, dtype=np.float32)[:, None]
    xyz[:30, :, 2] = 3.0
    # The gravity-aligned subset is the real floor and reaches the target wall.
    xyz[30:, :, 1] = 1.5
    xyz[30:, :, 2] = np.linspace(4.6, 5.0, 10, dtype=np.float32)[:, None]
    normal = np.zeros((h, w, 3), np.float32)
    normal[:30, :, 2] = -1.0
    normal[30:, :, 1] = -1.0
    floor_mask = np.full((h, w), 255, np.uint8)
    plane = WallPlane(n=(0.0, 0.0, -1.0), d=5.0)
    basis = WallBasis(
        origin=(0.0, 0.0, 5.0),
        u=(1.0, 0.0, 0.0),
        v=(0.0, -1.0, 0.0),
        n=(0.0, 0.0, -1.0),
        g_hat_cam=(0.0, -1.0, 0.0),
    )
    rules = {
        "room_corner_floor_fit": {
            "enabled": True,
            "min_floor_points": 100,
            "floor_ransac_iterations": 200,
            "floor_ransac_threshold_m": 0.02,
        },
        "floor_wall_intersection": {
            "min_floor_points": 100,
            "max_floor_points": 5000,
        },
    }

    result = _estimate_floor_wall_intersection(
        floor_mask,
        xyz,
        plane,
        basis,
        (-1.0, 1.0),
        np.array([], np.float32),
        normal=normal,
        rules=rules,
    )

    assert result["source"] == "wall_floor_plane_intersection"
    assert result["metric_trusted"] is True
    assert result["debug"]["floor_normal_prefilter"]["used"] is True
    assert result["debug"]["floor_normal_prefilter"]["floor_like_points"] == 400
    assert np.isclose(result["floor_height_signed_m"], -1.5, atol=1e-4)
