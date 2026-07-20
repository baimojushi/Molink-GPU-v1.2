from __future__ import annotations

import numpy as np

from hanging_mvp.geometry import fit_plane_ransac, fit_vertical_plane_ransac
from hanging_mvp.schemas import WallBasis, WallPlane
from hanging_mvp.support_context import _estimate_same_wall_art_height_anchor


def test_horizontal_dominant_coarse_box_recovers_supported_vertical_wall():
    rng = np.random.default_rng(7)
    # A coarse wall proposal contains more ceiling points than wall points.
    ceiling = np.column_stack([
        rng.uniform(-2.0, 2.0, 4500),
        rng.normal(-1.5, 0.008, 4500),
        rng.uniform(3.0, 6.0, 4500),
    ]).astype(np.float32)
    wall = np.column_stack([
        rng.uniform(-1.2, 1.2, 2600),
        rng.uniform(-1.3, 1.0, 2600),
        rng.normal(5.0, 0.008, 2600),
    ]).astype(np.float32)
    points = np.concatenate([ceiling, wall], axis=0)
    gravity = np.array([0.0, -1.0, 0.0], np.float32)

    unconstrained, _ = fit_plane_ransac(points, iterations=500, threshold=0.04)
    constrained, ratio = fit_vertical_plane_ransac(points, gravity, iterations=700, threshold=0.04)

    assert abs(float(np.dot(np.asarray(unconstrained.n), gravity))) > 0.8
    assert abs(float(np.dot(np.asarray(constrained.n), gravity))) < 1e-5
    assert ratio > 0.30


def test_same_wall_existing_art_supplies_direct_height_without_fake_floor():
    h, w = 80, 100
    xyz = np.zeros((h, w, 3), np.float32)
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    yy = np.linspace(0.8, -0.8, h, dtype=np.float32)
    xyz[:, :, 0] = xx[None, :]
    xyz[:, :, 1] = yy[:, None]
    xyz[:, :, 2] = 5.0
    mask = np.zeros((h, w), np.uint8)
    mask[20:40, 35:65] = 255
    objects = {
        "framed_art_01": {
            "object_id": "framed_art_01",
            "label": "framed_art",
            "role": "existing_content",
            "mask": mask,
        }
    }
    basis = WallBasis(
        origin=(0.0, 0.0, 5.0),
        u=(1.0, 0.0, 0.0),
        v=(0.0, -1.0, 0.0),
        n=(0.0, 0.0, -1.0),
        g_hat_cam=(0.0, -1.0, 0.0),
    )
    anchor = _estimate_same_wall_art_height_anchor(
        objects=objects,
        xyz=xyz,
        plane=WallPlane((0.0, 0.0, -1.0), 5.0),
        basis=basis,
        s_range_m=(-1.0, 1.0),
        rules={},
    )

    assert anchor is not None
    assert anchor["source"] == "same_wall_existing_art_center"
    assert anchor["metric_trusted"] is True
    assert anchor["anchor_count"] == 1
    assert anchor["policy"] == "direct_same_wall_height_anchor_never_relabel_as_floor"
