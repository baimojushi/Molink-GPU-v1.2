from __future__ import annotations

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from hanging_mvp.flux_export import build_render_prompt
from hanging_mvp.hardpaste import _edge_side_texture, warp_artwork_mount_to_geometry
from semantic_mvp.vlm_client import parse_artwork_frame_with_vlm


def test_artwork_frame_mock_forced_enables_precompose(monkeypatch):
    monkeypatch.setenv("ARTWORK_FRAME_MOCK_HAS_FRAME", "1")
    result = parse_artwork_frame_with_vlm(b"not-used-in-forced-mode", "mock", 100, 80)
    assert result["has_frame"] is True
    assert result["should_precompose_frame_sides"] is True
    assert result["source"] == "mock_forced"


def test_mount_hardpaste_outputs_front_side_and_union_masks():
    room = np.zeros((80, 100, 3), np.uint8) + 210
    art = np.zeros((30, 40, 4), np.uint8)
    art[:, :, :3] = [160, 30, 20]
    art[:, :, 3] = 255
    # Dark right and bottom frame-like edges should be carried into side faces.
    art[:, -6:, :3] = [40, 30, 20]
    art[-6:, :, :3] = [45, 32, 22]
    front = [[30, 20], [70, 20], [70, 55], [30, 55]]
    faces = [
        {"source_edge": "right", "quad": [[70, 20], [70, 55], [76, 58], [76, 23]], "area_px": 200},
        {"source_edge": "bottom", "quad": [[70, 55], [30, 55], [28, 61], [76, 58]], "area_px": 200},
    ]
    mounted = warp_artwork_mount_to_geometry(room, art, front, faces, edge_sample_px=6)
    assert mounted["image"].shape == room.shape
    assert int(np.count_nonzero(mounted["front_mask"])) > 0
    assert int(np.count_nonzero(mounted["side_mask"])) > 0
    assert int(np.count_nonzero(mounted["mount_mask"])) >= int(np.count_nonzero(mounted["front_mask"]))


def test_mount_aware_prompt_replaces_legacy_frame_rule():
    prompt = build_render_prompt(
        {
            "candidate_id": "cand_mount",
            "wall_id": "wall_right",
            "projection": {"projected_quad_px_original": [[10, 10], [50, 10], [50, 40], [10, 40]]},
            "artwork_mount_policy": {"precompose_frame_sides": True},
        },
        artwork_mount_policy={"precompose_frame_sides": True},
    )
    assert "deterministic 3D frame side faces" in prompt
    assert "Do not add, redraw, thicken" in prompt
    assert "The hard-paste target intentionally contains no generated frame" not in prompt


def test_artwork_frame_mock_env_does_not_override_openai_mode(monkeypatch):
    monkeypatch.setenv("ARTWORK_FRAME_MOCK_HAS_FRAME", "1")

    def fake_call(image_bytes, system_text, user_text, *, timeout=120):
        return {"has_frame": False, "confidence": 0.91, "frame_style": "none", "reason": "real provider result"}

    monkeypatch.setattr("semantic_mvp.vlm_client.call_openai_compatible_with_prompts", fake_call)
    result = parse_artwork_frame_with_vlm(b"image", "openai_compatible", 100, 80)
    assert result["source"] == "openai_compatible"
    assert result["has_frame"] is False
    assert result["should_precompose_frame_sides"] is False


def test_edge_side_texture_shading_is_front_adjacent_brightest():
    art = np.zeros((8, 10, 4), np.uint8)
    art[:, :, :3] = 100
    art[:, :, 3] = 255
    right = _edge_side_texture(art, "right", 4, side_shading_strength=0.25)
    left = _edge_side_texture(art, "left", 4, side_shading_strength=0.25)
    bottom = _edge_side_texture(art, "bottom", 4, side_shading_strength=0.25)
    top = _edge_side_texture(art, "top", 4, side_shading_strength=0.25)
    assert int(right[0, 0, 0]) > int(right[0, -1, 0])
    assert int(left[0, -1, 0]) > int(left[0, 0, 0])
    assert int(bottom[0, 0, 0]) > int(bottom[-1, 0, 0])
    assert int(top[-1, 0, 0]) > int(top[0, 0, 0])


def test_mount_policy_locks_visible_front_to_candidate_projection():
    from hanging_mvp.geometry import project_quad
    from hanging_mvp.schemas import Intrinsics, WallBasis
    from hanging_mvp.service import _compute_artwork_mount_policy_for_candidate

    basis = WallBasis(
        origin=(0.0, 0.0, 1.0),
        u=(1.0, 0.0, 0.0),
        v=(0.0, -1.0, 0.0),
        n=(0.0, 0.0, -1.0),
        g_hat_cam=(0.0, -1.0, 0.0),
    )
    intr = Intrinsics(fx=100.0, fy=100.0, cx=100.0, cy=100.0)
    expected_front = project_quad(basis.origin, basis.u, basis.v, intr, 0.0, 0.0, 0.4, 0.4)
    policy = _compute_artwork_mount_policy_for_candidate(
        rules={"artwork_mount": {"precompose_frame_sides_when_detected": True, "default_frame_depth_m": 0.025, "min_side_face_area_px": 0, "min_side_face_width_px": 0}},
        artwork_analysis={"has_frame": True, "confidence": 0.99},
        artwork_has_frame=True,
        artwork_has_frame_source="test",
        basis=basis,
        axis_s=0.0,
        center_t=0.0,
        artwork_w_m=0.4,
        artwork_h_m=0.4,
        intr=intr,
        scale_to_original=(1.0, 1.0),
        original_size_wh=(200, 200),
    )
    assert policy["precompose_frame_sides"] is True
    assert policy["front_anchor_mode"] == "candidate_projection_visual_front_locked"
    assert np.allclose(np.array(policy["front_quad_original"]), expected_front, atol=1e-4)
    # Back face recedes from the locked visual front; it must not be reused as the prompt/QA polygon.
    assert not np.allclose(np.array(policy["back_quad_original"]), expected_front, atol=1e-4)


def test_mount_policy_generates_sides_for_unframed_canvas_with_double_depth():
    from hanging_mvp.schemas import Intrinsics, WallBasis
    from hanging_mvp.service import _compute_artwork_mount_policy_for_candidate

    basis = WallBasis(
        origin=(0.0, 0.0, 1.0),
        u=(1.0, 0.0, 0.0),
        v=(0.0, -1.0, 0.0),
        n=(0.0, 0.0, -1.0),
        g_hat_cam=(0.0, -1.0, 0.0),
    )
    intr = Intrinsics(fx=100.0, fy=100.0, cx=100.0, cy=100.0)
    policy = _compute_artwork_mount_policy_for_candidate(
        rules={"artwork_mount": {"precompose_frame_sides_when_detected": True, "default_frame_depth_m": 0.025, "max_frame_depth_m": 0.08, "min_side_face_area_px": 0, "min_side_face_width_px": 0}},
        artwork_analysis={"has_frame": False, "confidence": 0.94, "artwork_surface_style": "rough_or_canvas", "side_thickness_policy": "double"},
        artwork_has_frame=False,
        artwork_has_frame_source="artwork_vlm",
        basis=basis,
        axis_s=0.0,
        center_t=0.0,
        artwork_w_m=0.4,
        artwork_h_m=0.4,
        intr=intr,
        scale_to_original=(1.0, 1.0),
        original_size_wh=(200, 200),
    )
    assert policy["precompose_frame_sides"] is True
    assert policy["side_thickness_policy"] == "double"
    assert policy["side_depth_multiplier"] == 2.0
    assert policy["frame_depth_m"] == 0.05


def test_mount_policy_skips_sides_for_chinese_hanging_scroll():
    from hanging_mvp.schemas import Intrinsics, WallBasis
    from hanging_mvp.service import _compute_artwork_mount_policy_for_candidate

    basis = WallBasis(
        origin=(0.0, 0.0, 1.0),
        u=(1.0, 0.0, 0.0),
        v=(0.0, -1.0, 0.0),
        n=(0.0, 0.0, -1.0),
        g_hat_cam=(0.0, -1.0, 0.0),
    )
    policy = _compute_artwork_mount_policy_for_candidate(
        rules={"artwork_mount": {"precompose_frame_sides_when_detected": True}},
        artwork_analysis={"artwork_surface_style": "chinese_hanging_scroll", "side_thickness_policy": "none"},
        artwork_has_frame=False,
        artwork_has_frame_source="artwork_vlm",
        basis=basis,
        axis_s=0.0,
        center_t=0.0,
        artwork_w_m=0.4,
        artwork_h_m=0.4,
        intr=Intrinsics(fx=100.0, fy=100.0, cx=100.0, cy=100.0),
        scale_to_original=(1.0, 1.0),
        original_size_wh=(200, 200),
    )
    assert policy["precompose_frame_sides"] is False
    assert policy["fallback_reason"] == "artwork_vlm_classified_as_chinese_hanging_scroll_no_side_faces"


def test_mount_aware_prompt_explicitly_locks_visible_side_edges_only():
    prompt = build_render_prompt(
        {
            "candidate_id": "cand_mount_edges",
            "wall_id": "wall_front",
            "projection": {"projected_quad_px_original": [[10, 10], [50, 10], [50, 40], [10, 40]]},
            "artwork_mount_policy": {
                "precompose_frame_sides": True,
                "visible_side_faces_original": [
                    {"source_edge": "left", "quad": [[10, 10], [10, 40], [8, 42], [8, 12]]},
                    {"source_edge": "bottom", "quad": [[10, 40], [50, 40], [52, 42], [8, 42]]},
                ],
            },
        },
        artwork_mount_policy={
            "precompose_frame_sides": True,
            "visible_side_faces_original": [
                {"source_edge": "left", "quad": [[10, 10], [10, 40], [8, 42], [8, 12]]},
                {"source_edge": "bottom", "quad": [[10, 40], [50, 40], [52, 42], [8, 42]]},
            ],
        },
    )
    assert "Only these side faces are visible and allowed to remain visible: left, bottom." in prompt
    assert "Do not invent a perimeter border or thickness strip on hidden edges such as right, top." in prompt
