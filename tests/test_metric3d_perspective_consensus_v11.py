import math

import cv2
import numpy as np

from hanging_mvp.geometry import (
    estimate_wall_horizontal_structure_direction,
    make_wall_basis,
    normalize,
    project_quad,
    resolve_wall_projection_plane,
)
from hanging_mvp.schemas import Intrinsics, WallPlane


G = np.array([0.0627687052, -0.9777769446, -0.2000307292], np.float32)
DEPTH_N = np.array([0.83401864, 0.24909153, -0.49230714], np.float32)
DIRECT_N = np.array([0.60521966, 0.21141924, -0.76747054], np.float32)


def _coherent_normal_field(normal, shape=(80, 100)):
    field = np.broadcast_to(np.asarray(normal, np.float32), (*shape, 3)).copy()
    mask = np.ones(shape, np.uint8) * 255
    return field, mask


def _verticalized_depth_normal(depth_normal=DEPTH_N, gravity=G):
    result = np.asarray(depth_normal, np.float32) - float(np.dot(depth_normal, gravity)) * np.asarray(gravity, np.float32)
    return normalize(result)


def test_direct_normal_conflict_does_not_replace_depth_yaw():
    normal_xyz, evidence = _coherent_normal_field(DIRECT_N)
    rng = np.random.default_rng(7)
    anchors = np.array([-0.45, -1.59, 10.76], np.float32) + rng.normal(0, 0.02, size=(3000, 3))
    depth_plane = WallPlane(tuple(float(x) for x in DEPTH_N), 6.07)

    projection_plane, debug = resolve_wall_projection_plane(
        depth_plane,
        anchors,
        normal_xyz,
        evidence,
        G,
        horizontal_structure_enabled=False,
    )

    expected = _verticalized_depth_normal()
    actual = np.asarray(projection_plane.n, np.float32)
    if float(np.dot(actual, expected)) < 0.0:
        actual = -actual
    assert debug["hard_reject"] is False
    assert debug["projection_orientation_source"] == "depth_plane_gravity_verticalized"
    assert debug["direct_normal_used_for_yaw"] is False
    assert debug["depth_horizontal_heading_preserved"] is True
    assert 20.0 < debug["depth_normal_disagreement_deg"] < 22.0
    assert debug["normal_consensus"]["reliable"] is True
    assert np.allclose(actual, expected, atol=1e-6)
    assert abs(float(np.dot(actual, G))) < 1e-6


def test_depth_yaw_preservation_only_removes_vertical_shear():
    normal_xyz, evidence = _coherent_normal_field(DIRECT_N)
    center = np.array([-0.41957, -1.35803, 10.93997], np.float32)
    anchors = np.broadcast_to(center, (3000, 3)).copy()
    depth_plane = WallPlane(tuple(float(x) for x in DEPTH_N), -float(np.dot(DEPTH_N, center)))
    projection_plane, debug = resolve_wall_projection_plane(
        depth_plane,
        anchors,
        normal_xyz,
        evidence,
        G,
        horizontal_structure_enabled=False,
    )
    assert debug["hard_reject"] is False
    basis = make_wall_basis(projection_plane, anchors, G)
    intr = Intrinsics(fx=1330.2150202, fy=1330.2150202, cx=575.5, cy=768.0)
    quad = project_quad(basis.origin, basis.u, basis.v, intr, 0.0, 0.0, 0.6, 0.8)
    left_dx_dy = float((quad[3, 0] - quad[0, 0]) / (quad[3, 1] - quad[0, 1]))

    # The vertical edge is gravity-corrected while the original depth yaw and
    # its foreshortening remain intact.
    assert -0.07 < left_dx_dy < -0.03
    assert 40.0 < float(np.linalg.norm(quad[1] - quad[0])) < 43.0


def test_unstable_direct_normals_fail_material_validation_with_enough_samples():
    rng = np.random.default_rng(11)
    random_normals = rng.normal(size=(80, 100, 3)).astype(np.float32)
    random_normals /= np.maximum(np.linalg.norm(random_normals, axis=-1, keepdims=True), 1e-8)
    evidence = np.ones((80, 100), np.uint8) * 255
    anchors = np.broadcast_to(np.array([-0.4, -1.4, 11.0], np.float32), (3000, 3)).copy()
    depth_plane = WallPlane(tuple(float(x) for x in DEPTH_N), 6.07)

    _, debug = resolve_wall_projection_plane(
        depth_plane,
        anchors,
        random_normals,
        evidence,
        G,
        horizontal_structure_enabled=False,
    )

    assert debug["hard_reject"] is True
    assert debug["reason"] == "direct_normal_wall_validation_failed"
    assert debug["direct_normal_used_for_yaw"] is False
    assert debug["normal_consensus"]["reliable"] is False


def _draw_converging_lines(image, intr, direction, y_values):
    K = np.array([[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy], [0.0, 0.0, 1.0]], np.float64)
    vp = K @ np.asarray(direction, np.float64)
    vp_xy = vp[:2] / vp[2]
    for y0 in y_values:
        p0 = np.array([170.0, float(y0)])
        x1 = 500.0
        scale = (x1 - p0[0]) / (vp_xy[0] - p0[0])
        y1 = p0[1] + scale * (vp_xy[1] - p0[1])
        cv2.line(image, tuple(np.rint(p0).astype(int)), (int(round(x1)), int(round(y1))), (245, 245, 245), 2, cv2.LINE_AA)


def test_calibrated_structure_lines_estimate_vanishing_direction():
    intr = Intrinsics(fx=800.0, fy=800.0, cx=320.0, cy=240.0)
    gravity = np.array([0.0, -1.0, 0.0], np.float32)
    expected_u = np.array([-1.0, 0.0, 0.0], np.float32)
    theta = math.radians(10.0)
    observed_u = normalize(np.array([-math.cos(theta), 0.0, -math.sin(theta)], np.float32))
    image = np.zeros((480, 640, 3), np.uint8)
    mask = np.zeros((480, 640), np.uint8)
    cv2.rectangle(mask, (180, 150), (470, 390), 255, -1)
    _draw_converging_lines(image, intr, observed_u, [150, 190, 230, 270, 310, 350, 390])

    estimated, debug = estimate_wall_horizontal_structure_direction(
        image,
        mask,
        intr,
        expected_u,
        gravity,
        prefilter_angle_deg=30.0,
        min_unique_lines=4,
    )

    assert estimated is not None
    assert debug["reliable"] is True
    assert debug["inlier_count"] >= 4
    assert debug["residual_p90_deg"] <= 2.5
    assert 7.0 < debug["direction_angle_to_depth_basis_u_deg"] < 13.0
    assert abs(float(np.dot(estimated, gravity))) < 1e-6


def test_confident_structure_mismatch_performs_bounded_yaw_correction():
    intr = Intrinsics(fx=800.0, fy=800.0, cx=320.0, cy=240.0)
    gravity = np.array([0.0, -1.0, 0.0], np.float32)
    depth_n = np.array([0.0, 0.0, 1.0], np.float32)
    expected_u = normalize(np.cross(gravity, depth_n))
    theta = math.radians(10.0)
    observed_u = normalize(expected_u * math.cos(theta) + depth_n * math.sin(theta))
    observed_n = normalize(np.cross(observed_u, gravity))
    image = np.zeros((480, 640, 3), np.uint8)
    wall_mask = np.zeros((480, 640), np.uint8)
    cv2.rectangle(wall_mask, (180, 150), (470, 390), 255, -1)
    _draw_converging_lines(image, intr, observed_u, [150, 190, 230, 270, 310, 350, 390])
    normal_xyz, evidence = _coherent_normal_field(depth_n, shape=wall_mask.shape)
    evidence = wall_mask.copy()
    anchors = np.broadcast_to(np.array([0.0, 0.0, 5.0], np.float32), (3000, 3)).copy()
    plane = WallPlane(tuple(float(x) for x in depth_n), -5.0)

    projection_plane, debug = resolve_wall_projection_plane(
        plane,
        anchors,
        normal_xyz,
        evidence,
        gravity,
        image_rgb=image,
        wall_mask=wall_mask,
        intr=intr,
        horizontal_structure_prefilter_angle_deg=30.0,
    )

    actual_n = np.asarray(projection_plane.n, np.float32)
    if float(np.dot(actual_n, observed_n)) < 0.0:
        actual_n = -actual_n
    assert debug["hard_reject"] is False
    assert debug["projection_orientation_source"] == "depth_yaw_corrected_by_horizontal_structure_vanishing_point"
    assert debug["horizontal_structure"]["action"] == "corrected_horizontal_heading"
    assert np.allclose(actual_n, observed_n, atol=0.04)


def test_wall_basis_origin_is_projected_onto_resolved_plane():
    normal = _verticalized_depth_normal(DIRECT_N, G)
    anchor = np.array([-0.45, -1.59, 10.76], np.float32)
    plane = WallPlane(tuple(float(x) for x in normal), -float(np.dot(normal, anchor)))
    points = np.stack((anchor + normal * 0.05, anchor - normal * 0.03, anchor + normal * 0.01))
    basis = make_wall_basis(plane, points, G)

    assert abs(float(np.dot(np.asarray(plane.n), np.asarray(basis.origin)) + plane.d)) < 1e-5
    assert float(np.dot(np.asarray(basis.v), G)) > 0.99999
