import cv2
import numpy as np

import hanging_mvp.geometry as geometry
from hanging_mvp.exit_review import review_hardpaste_exit
from hanging_mvp.geometry import estimate_wall_horizontal_structure_direction, normalize, resolve_wall_projection_plane
from hanging_mvp.schemas import Intrinsics, WallPlane


G = np.array([0.0627687052, -0.9777769446, -0.2000307292], np.float32)
DEPTH_N = np.array([0.83401864, 0.24909153, -0.49230714], np.float32)
DIRECT_N = np.array([0.60521966, 0.21141924, -0.76747054], np.float32)


def _field(normal, shape=(80, 100)):
    return (
        np.broadcast_to(np.asarray(normal, np.float32), (*shape, 3)).copy(),
        np.ones(shape, np.uint8) * 255,
    )


def _anchors():
    rng = np.random.default_rng(15)
    return np.array([-0.45, -1.59, 10.76], np.float32) + rng.normal(0, 0.02, size=(3000, 3))


def _verticalized(value):
    value = np.asarray(value, np.float32) - float(np.dot(value, G)) * G
    return normalize(value)


def test_ceiling_lines_outside_authoritative_wall_mask_cannot_vote():
    image = np.zeros((480, 640, 3), np.uint8)
    wall_mask = np.zeros((480, 640), np.uint8)
    cv2.rectangle(wall_mask, (180, 210), (470, 410), 255, -1)
    # Reproduce the v14 contamination mode: numerous long, coherent ceiling
    # slot edges inside the old expanded context rectangle but outside the wall.
    for y in (70, 90, 110, 130, 150, 170):
        cv2.line(image, (80, y), (560, y + 4), (255, 255, 255), 2, cv2.LINE_AA)
    direction, debug = estimate_wall_horizontal_structure_direction(
        image,
        wall_mask,
        Intrinsics(fx=800.0, fy=800.0, cx=320.0, cy=240.0),
        np.array([-1.0, 0.0, 0.0], np.float32),
        np.array([0.0, -1.0, 0.0], np.float32),
    )
    assert direction is None
    assert debug["reliable"] is False
    assert debug["reason"] in {
        "no_horizontal_structure_lines_detected",
        "insufficient_unique_horizontal_structure_lines",
    }


def test_reliable_wall_structure_can_uniquely_resolve_depth_normal_conflict(monkeypatch):
    normal_xyz, evidence = _field(DIRECT_N)
    direct_n = _verticalized(DIRECT_N)
    direct_u = normalize(np.cross(G, direct_n))

    def fake_structure(*_args, **_kwargs):
        return direct_u, {
            "reliable": True,
            "correction_confident": True,
            "evidence_policy": "authoritative_wall_mask_interior_only_no_ceiling_or_adjacent_trim",
            "post_gravity_projection_residual_p90_deg": 0.7,
            "inlier_wall_interior_length_fraction": 0.96,
        }

    monkeypatch.setattr(geometry, "estimate_wall_horizontal_structure_direction", fake_structure)
    plane = WallPlane(tuple(float(x) for x in DEPTH_N), 6.07)
    projection, debug = resolve_wall_projection_plane(
        plane,
        _anchors(),
        normal_xyz,
        evidence,
        G,
        image_rgb=np.zeros((80, 100, 3), np.uint8),
        wall_mask=evidence,
        intr=Intrinsics(fx=800.0, fy=800.0, cx=50.0, cy=40.0),
    )
    actual = np.asarray(projection.n, np.float32)
    if float(np.dot(actual, direct_n)) < 0.0:
        actual = -actual
    assert debug["hard_reject"] is False
    assert debug["projection_orientation_source"] == "wall_internal_structure_quorum_supports_direct_normal_heading"
    assert debug["horizontal_structure"]["supports_direct_normal_heading"] is True
    assert np.allclose(actual, direct_n, atol=1e-5)


def test_structure_that_does_not_uniquely_support_either_sensor_fails_closed(monkeypatch):
    normal_xyz, evidence = _field(DIRECT_N)
    depth_u = normalize(np.cross(G, _verticalized(DEPTH_N)))
    direct_u = normalize(np.cross(G, _verticalized(DIRECT_N)))
    midpoint_u = normalize(depth_u + direct_u)

    monkeypatch.setattr(
        geometry,
        "estimate_wall_horizontal_structure_direction",
        lambda *_args, **_kwargs: (midpoint_u, {"reliable": True, "correction_confident": True}),
    )
    plane = WallPlane(tuple(float(x) for x in DEPTH_N), 6.07)
    _, debug = resolve_wall_projection_plane(
        plane,
        _anchors(),
        normal_xyz,
        evidence,
        G,
        image_rgb=np.zeros((80, 100, 3), np.uint8),
        wall_mask=evidence,
        intr=Intrinsics(fx=800.0, fy=800.0, cx=50.0, cy=40.0),
    )
    assert debug["hard_reject"] is True
    assert debug["reason"] == "depth_direct_normal_yaw_conflict_not_uniquely_resolved_by_wall_structure"


def test_visual_review_cannot_override_numeric_projection_hard_reject(monkeypatch):
    monkeypatch.setenv("VLM_API_URL", "https://example.invalid/v1/chat/completions")
    monkeypatch.setenv("VLM_API_KEY", "configured")
    monkeypatch.setenv("VLM_MODEL", "configured")
    review = review_hardpaste_exit(
        np.zeros((8, 8, 3), np.uint8),
        {
            "candidate_id": "bad_perspective",
            "hard_constraints": {
                "passed": False,
                "hard_gate_reasons": ["projected_artwork_aspect_distortion_exceeds_hard_limit"],
                "projection_degenerate_guard": {
                    "projection_shape_gate": {
                        "status": "hard_reject",
                        "hard_reasons": ["projected_artwork_aspect_distortion_exceeds_hard_limit"],
                    }
                },
            },
        },
        {"exit_review": {"hardpaste_vlm_mode": "auto"}},
    )
    assert review["status"] == "blocked_by_numeric_geometry_hard_gate"
    assert review["approved_for_flux_render"] is False
    assert review["visual_review_can_override"] is False
