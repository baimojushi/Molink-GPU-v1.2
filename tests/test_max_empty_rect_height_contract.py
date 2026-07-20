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


def test_regression_03a00a51_subcentimeter_metric_noise_keeps_candidate_seed():
    from hanging_mvp.service import _semantic_solid_max_empty_rect_axes

    h, w = 100, 100
    safe = np.full((h, w), 255, np.uint8)
    xyz = np.zeros((h, w, 3), np.float32)
    xyz[:, :, 0] = np.arange(w, dtype=np.float32)[None, :] * 0.012
    # 5th..95th percentile span is about 0.953 m: just 7 mm below the
    # 0.960 m seed requirement seen in the diagnostic job.
    xyz[:, :, 1] = np.arange(h, dtype=np.float32)[:, None] * 0.0107
    xyz[:, :, 2] = 5.0
    basis = WallBasis(
        origin=(0.0, 0.0, 5.0),
        u=(1.0, 0.0, 0.0),
        v=(0.0, 1.0, 0.0),
        n=(0.0, 0.0, -1.0),
        g_hat_cam=(0.0, -1.0, 0.0),
    )

    axes, debug = _semantic_solid_max_empty_rect_axes(
        semantic_clear_wall_mask=safe,
        solid_wall_mask=safe,
        xyz=xyz,
        basis=basis,
        artwork_w_m=0.6,
        artwork_h_m=0.8,
        min_edge_margin_m=0.08,
        metric_seed_tolerance_m=0.02,
    )

    assert axes
    assert debug["components"][0]["height_m"] < debug["required_height_m"]
    assert debug["components"][0]["accepted_with_metric_seed_tolerance"] is True
    assert axes[0]["vertical_policy"] == "diagnostic_only_exact_projected_footprint_is_authoritative"
