"""Regression coverage for V26 structure voting authority.

The job_42b054d1 failure contained a numerically tight five-line VP family, but
four inliers were fragments/reflections of one top band and the remaining vote
was a marble vein.  Whole-wall and segmented depth independently agreed, yet
that weak visual family prevented the existing hierarchy recovery from running.
"""

import numpy as np

import hanging_mvp.geometry as geometry
from hanging_mvp.geometry import normalize, resolve_wall_projection_plane
from hanging_mvp.schemas import Intrinsics, WallPlane


DEPTH_NORMAL = np.array(
    [0.8340790271759033, 0.24945688247680664, -0.4920198619365692],
    dtype=np.float32,
)
DIRECT_NORMAL = np.array(
    [0.6051228046417236, 0.21059982478618622, -0.7677721381187439],
    dtype=np.float32,
)
ANCHOR = np.array(
    [-0.5091041922569275, -1.6276906728744507, 10.648569107055664],
    dtype=np.float32,
)
GRAVITY = np.array(
    [0.05921376869082451, -0.9778359532356262, -0.2008247971534729],
    dtype=np.float32,
)
INTRINSICS = Intrinsics(fx=996.969, fy=996.969, cx=575.5, cy=768.0)


def _line_item(x1, y1, x2, y2, length):
    p1 = np.array([x1, y1, 1.0], dtype=np.float64)
    p2 = np.array([x2, y2, 1.0], dtype=np.float64)
    line = np.cross(p1, p2)
    line /= np.linalg.norm(line[:2])
    return {
        "line": line,
        "midpoint": np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float64),
        "length_px": float(length),
    }


def test_recorded_hough_fragments_collapse_to_two_physical_support_bands():
    # Five accepted lines from the diagnostic.  Four are one reflected/top
    # physical band; only the short diagonal marble vein is spatially separate.
    lines = [
        _line_item(400, 479, 569, 513, 172.386),
        _line_item(460, 496, 567, 506, 107.466),
        _line_item(489, 514, 568, 510, 79.101),
        _line_item(425, 642, 490, 618, 69.289),
        _line_item(399, 478, 465, 488, 66.753),
    ]
    center = np.array([525.5, 603.5], dtype=np.float64)
    predicted_vp = np.array([1426.798, 546.316], dtype=np.float64)
    expected_direction = normalize(predicted_vp - center)
    cross_axis = np.array([-expected_direction[1], expected_direction[0]], dtype=np.float64)

    debug = geometry._parallel_line_support_band_stats(
        lines,
        np.ones(len(lines), dtype=bool),
        center,
        cross_axis,
        box_height_px=254.0,
    )

    assert debug["independent_band_count"] == 2
    assert debug["dominant_band_length_fraction"] > 0.84
    assert sorted(band["line_count"] for band in debug["bands"]) == [1, 4]


def test_non_authoritative_structure_cannot_block_strict_depth_hierarchy(monkeypatch):
    depth_vertical = DEPTH_NORMAL - float(np.dot(DEPTH_NORMAL, GRAVITY)) * GRAVITY
    direct_vertical = DIRECT_NORMAL - float(np.dot(DIRECT_NORMAL, GRAVITY)) * GRAVITY
    depth_u = normalize(np.cross(GRAVITY, normalize(depth_vertical)))
    direct_u = normalize(np.cross(GRAVITY, normalize(direct_vertical)))
    false_texture_direction = normalize(depth_u + direct_u)

    def fake_structure(*_args, **_kwargs):
        return false_texture_direction, {
            "reliable": True,
            "voting_authoritative": False,
            "authority_reason": "numerically_coherent_but_not_independent_structural_evidence",
            "voting_authority_failed_gates": [
                "vanishing_point_too_close_to_wall_body",
                "insufficient_independent_structural_bands",
                "single_physical_band_dominates_line_vote",
            ],
            "correction_confident": False,
        }

    monkeypatch.setattr(geometry, "estimate_wall_horizontal_structure_direction", fake_structure)
    normal_field = np.tile(DIRECT_NORMAL, (40, 40, 1))
    evidence_mask = np.full((40, 40), 255, dtype=np.uint8)
    depth_plane = WallPlane(
        tuple(float(x) for x in DEPTH_NORMAL),
        -float(np.dot(DEPTH_NORMAL, ANCHOR)),
    )

    _projection_plane, debug = resolve_wall_projection_plane(
        depth_plane,
        np.tile(ANCHOR, (200, 1)),
        normal_field,
        evidence_mask,
        GRAVITY,
        image_rgb=np.zeros((40, 40, 3), dtype=np.uint8),
        wall_mask=evidence_mask,
        intr=INTRINSICS,
        normal_min_samples=100,
        segmented_depth_plane_yaw=np.radians(-32.3453361564),
        segmented_depth_plane_confident=True,
        segmented_depth_plane_debug={
            "cluster_coverage": 1.0,
            "cluster_p90_deg": 4.04,
            "cluster_size": 16,
        },
        whole_depth_plane_inlier_ratio=0.98792,
        hierarchical_depth_consensus_enabled=True,
        projection_shape_feedback_enabled=True,
    )

    assert debug["hard_reject"] is False
    assert debug["horizontal_structure"]["resolver_voting_authoritative"] is False
    assert debug["hierarchical_depth_consensus"]["recovered"] is True
    assert debug["projection_orientation_source"] in {
        "whole_and_segmented_depth_hierarchy_consensus",
        "direct_normal_shape_feedback_after_depth_hierarchy",
    }


def test_non_authoritative_structure_does_not_unlock_without_depth_hierarchy(monkeypatch):
    depth_vertical = DEPTH_NORMAL - float(np.dot(DEPTH_NORMAL, GRAVITY)) * GRAVITY
    direct_vertical = DIRECT_NORMAL - float(np.dot(DIRECT_NORMAL, GRAVITY)) * GRAVITY
    texture_direction = normalize(
        normalize(np.cross(GRAVITY, normalize(depth_vertical)))
        + normalize(np.cross(GRAVITY, normalize(direct_vertical)))
    )
    monkeypatch.setattr(
        geometry,
        "estimate_wall_horizontal_structure_direction",
        lambda *_args, **_kwargs: (
            texture_direction,
            {"reliable": True, "voting_authoritative": False, "correction_confident": False},
        ),
    )
    normal_field = np.tile(DIRECT_NORMAL, (40, 40, 1))
    evidence_mask = np.full((40, 40), 255, dtype=np.uint8)
    depth_plane = WallPlane(
        tuple(float(x) for x in DEPTH_NORMAL),
        -float(np.dot(DEPTH_NORMAL, ANCHOR)),
    )

    _projection_plane, debug = resolve_wall_projection_plane(
        depth_plane,
        np.tile(ANCHOR, (200, 1)),
        normal_field,
        evidence_mask,
        GRAVITY,
        image_rgb=np.zeros((40, 40, 3), dtype=np.uint8),
        wall_mask=evidence_mask,
        intr=INTRINSICS,
        normal_min_samples=100,
        hierarchical_depth_consensus_enabled=False,
        controlled_structural_quorum_enabled=False,
    )

    assert debug["hard_reject"] is True
    assert debug["horizontal_structure"]["resolver_voting_authoritative"] is False
