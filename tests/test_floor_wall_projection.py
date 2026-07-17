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
