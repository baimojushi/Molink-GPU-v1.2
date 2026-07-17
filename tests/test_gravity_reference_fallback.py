import numpy as np

from hanging_mvp.geometry import (
    estimate_g_hat_from_floor_normals,
    estimate_g_hat_from_global_floor_like_normals,
    estimate_g_hat_from_horizontal_reference_normals,
)
from hanging_mvp.masks import canonical_semantic_label
from semantic_mvp.prompts import system_prompt, user_prompt


def test_roof_ceiling_is_requested_and_canonicalized():
    prompt = system_prompt("gallery") + user_prompt("gallery", 1536, 1152)
    assert "roof" in prompt
    assert "label=roof" in prompt
    assert canonical_semantic_label("ceiling") == "roof"
    assert canonical_semantic_label("roof_01") == "roof"


def test_roof_normals_are_sign_aligned_to_camera_up():
    normal = np.zeros((20, 20, 3), np.float32)
    normal[..., 1] = 1.0  # ceiling normals may point down/opposite camera-up
    mask = np.full((20, 20), 255, np.uint8)

    g_hat, conf, debug = estimate_g_hat_from_horizontal_reference_normals(
        normal,
        mask,
        reference_label="roof",
        min_pixels=50,
        full_confidence_pixels=50,
    )

    assert conf > 0.99
    assert np.allclose(g_hat, [0.0, -1.0, 0.0], atol=1e-5)
    assert debug["reference_label"] == "roof"
    assert debug["flipped_normal_count"] == 400


def test_table_normals_can_be_used_as_horizontal_reference():
    normal = np.zeros((20, 20, 3), np.float32)
    normal[..., 1] = -1.0
    mask = np.full((20, 20), 255, np.uint8)

    g_hat, conf, debug = estimate_g_hat_from_horizontal_reference_normals(
        normal,
        mask,
        reference_label="table",
        min_pixels=50,
        full_confidence_pixels=50,
    )

    assert conf > 0.99
    assert np.allclose(g_hat, [0.0, -1.0, 0.0], atol=1e-5)
    assert debug["reference_label"] == "table"


def test_global_metric3d_floor_like_fallback_uses_lower_region_first():
    normal = np.zeros((40, 40, 3), np.float32)
    normal[..., 2] = 1.0
    normal[24:, :, :] = np.array([0.05, -0.99, 0.02], np.float32)

    g_hat, conf, debug = estimate_g_hat_from_global_floor_like_normals(
        normal,
        min_pixels=100,
        full_confidence_pixels=100,
        lower_image_start_ratio=0.55,
    )

    assert conf > 0.99
    assert g_hat[1] < -0.98
    assert debug["selected_region"] == "lower_image"


def test_floor_normal_sign_is_canonicalized_to_camera_up():
    normal = np.zeros((24, 24, 3), np.float32)
    normal[:, :, 1] = 1.0
    mask = np.full((24, 24), 255, np.uint8)

    g_hat, conf, debug = estimate_g_hat_from_floor_normals(
        normal, mask, min_pixels=100, full_confidence_pixels=100
    )

    assert conf > 0.0
    assert np.allclose(g_hat, [0.0, -1.0, 0.0], atol=1e-5)
    assert debug["flipped_normal_count"] == 24 * 24


def test_global_floor_like_fallback_accepts_opposite_normal_sign():
    normal = np.zeros((30, 20, 3), np.float32)
    normal[:, :, 1] = 1.0

    g_hat, conf, debug = estimate_g_hat_from_global_floor_like_normals(
        normal,
        min_pixels=100,
        full_confidence_pixels=100,
        lower_image_start_ratio=0.5,
    )

    assert conf > 0.0
    assert np.allclose(g_hat, [0.0, -1.0, 0.0], atol=1e-5)
    assert debug["flipped_normal_count"] == 15 * 20
