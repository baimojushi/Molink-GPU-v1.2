import numpy as np

from hanging_mvp.styling_metrics import (
    analyze_artwork_luminance,
    measure_candidate_wall_luminance,
    should_suggest_dark_wallpaper,
)
from worker.result_builder import _plan_to_candidate


def test_dark_or_high_contrast_artwork_on_light_wall_is_suggested():
    artwork = np.zeros((60, 80, 4), dtype=np.uint8)
    artwork[:, :, :3] = [32, 38, 52]
    artwork[:, :, 3] = 255
    artwork_metrics = analyze_artwork_luminance(artwork)
    wall_metrics = {"wall_mean_luminance": 0.82, "is_light_wall": True}

    assert artwork_metrics["artwork_mean_luminance"] < 0.42
    assert artwork_metrics["is_dark_or_high_contrast"] is True
    assert len(artwork_metrics["suggested_wall_tone_rgb"]) == 3
    assert should_suggest_dark_wallpaper(artwork_metrics, wall_metrics) is True


def test_light_low_contrast_artwork_or_dark_wall_is_not_suggested():
    artwork = np.full((40, 40, 4), 230, dtype=np.uint8)
    artwork[:, :, 3] = 255
    metrics = analyze_artwork_luminance(artwork)

    assert metrics["is_dark_or_high_contrast"] is False
    assert should_suggest_dark_wallpaper(metrics, {"is_light_wall": True}) is False
    assert should_suggest_dark_wallpaper({"is_dark_or_high_contrast": True}, {"is_light_wall": False}) is False


def test_wall_luminance_excludes_dilated_artwork_quad():
    room = np.full((100, 140, 3), 240, dtype=np.uint8)
    room[20:85, 30:110] = 8
    wall = np.full((100, 140), 255, dtype=np.uint8)
    quad = np.array([[30, 20], [110, 20], [110, 85], [30, 85]], dtype=np.float32)

    measured = measure_candidate_wall_luminance(room, wall, quad, exclusion_margin_px=2)

    assert measured["wall_mean_luminance"] > 0.90
    assert measured["is_light_wall"] is True
    assert measured["wall_luminance_sample_px"] < wall.size


def test_phase4_install_contract_reaches_result_candidate_without_render_request():
    plan = {
        "wall_id": "wall_a",
        "candidate_id": "candidate_1",
        "install_reference": {
            "wall_mean_luminance": 0.78,
            "suggest_dark_wallpaper": True,
            "suggested_wall_tone_rgb": [58, 62, 71],
        },
        "artwork_analysis": {
            "artwork_mean_luminance": 0.31,
            "artwork_contrast": 0.62,
        },
    }
    candidate = _plan_to_candidate(plan, 1, {"styling_status": "not_requested"})

    assert candidate["install"]["wall_mean_luminance"] == 0.78
    assert candidate["install"]["suggest_dark_wallpaper"] is True
    assert candidate["install"]["suggested_wall_tone_rgb"] == [58, 62, 71]
    assert candidate["artwork_analysis"]["artwork_contrast"] == 0.62
    assert candidate["styling_status"] == "not_requested"
