from __future__ import annotations

import numpy as np

from hanging_mvp.schemas import WallBasis


def test_max_empty_rect_generates_horizontal_axis_without_height_override():
    from hanging_mvp.service import _semantic_solid_max_empty_rect_axes

    h, w = 80, 100
    safe = np.full((h, w), 255, np.uint8)
    xyz = np.zeros((h, w, 3), np.float32)
    xyz[:, :, 0] = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    xyz[:, :, 1] = np.linspace(0.0, -1.2, h, dtype=np.float32)[:, None]
    xyz[:, :, 2] = 5.0
    basis = WallBasis(
        origin=(0.0, 0.0, 5.0),
        u=(1.0, 0.0, 0.0),
        v=(0.0, -1.0, 0.0),
        n=(0.0, 0.0, -1.0),
        g_hat_cam=(0.0, -1.0, 0.0),
    )

    axes, debug = _semantic_solid_max_empty_rect_axes(
        semantic_clear_wall_mask=safe,
        solid_wall_mask=safe,
        xyz=xyz,
        basis=basis,
        artwork_w_m=0.5,
        artwork_h_m=0.4,
        min_edge_margin_m=0.08,
    )

    assert debug["status"] == "available"
    assert axes
    assert "center_t_m_override" not in axes[0]
    assert axes[0]["source"] == "semantic_clear_subregion_max_empty_rect_horizontal_axis"
    assert axes[0]["vertical_policy"] == "diagnostic_only_exact_projected_footprint_is_authoritative"
