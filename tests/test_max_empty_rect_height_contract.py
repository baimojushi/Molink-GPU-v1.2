from __future__ import annotations

import numpy as np

from hanging_mvp.schemas import WallBasis


def test_max_empty_rect_generates_distinct_limited_raise_seed():
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
    assert abs(axes[0]["center_t_m_override"] - 0.6) < 1e-5
    assert axes[0]["metric_obstruction_aware_height"] is True
    assert axes[0]["source"] == "semantic_clear_subregion_max_empty_rect_horizontal_axis"
    assert axes[0]["vertical_policy"] == "limited_raise_only_then_exact_projected_footprint_authoritative"


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
        artwork_h_m=0.96,
        min_edge_margin_m=0.08,
        metric_seed_tolerance_m=0.02,
    )

    assert axes
    assert debug["components"][0]["height_m"] < debug["required_height_m"]
    assert debug["components"][0]["accepted_with_metric_seed_tolerance"] is True
    assert axes[0]["vertical_policy"] == "limited_raise_only_then_exact_projected_footprint_authoritative"


def test_regression_f5b1c6c1_clear_rect_is_not_pre_rejected_for_vertical_margin():
    from hanging_mvp.service import _semantic_solid_max_empty_rect_axes

    h, w = 100, 160
    safe = np.full((h, w), 255, np.uint8)
    xyz = np.zeros((h, w, 3), np.float32)
    xyz[:, :, 0] = np.linspace(-0.95, 0.95, w, dtype=np.float32)[None, :]
    # 5th..95th span is 0.961 m: the exact diagnostic clear-rectangle
    # height, 3.9 cm under the 1.0 m artwork due to Metric3D noise.
    xyz[:, :, 1] = np.linspace(0.0, 1.0677778, h, dtype=np.float32)[:, None]
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
        artwork_w_m=0.8,
        artwork_h_m=1.0,
        min_edge_margin_m=0.08,
        metric_seed_tolerance_m=0.05,
    )

    assert axes
    assert abs(debug["components"][0]["height_m"] - 0.961) < 0.01
    assert debug["required_height_m"] == 1.0
    assert debug["components"][0]["accepted_with_metric_seed_tolerance"] is True
