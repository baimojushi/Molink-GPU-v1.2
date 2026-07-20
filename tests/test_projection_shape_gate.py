from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hanging_mvp.geometry import select_gravity_consensus
from hanging_mvp.support_context import _quad_metrics_original, evaluate_projection_shape_gate


def test_landscape_artwork_projected_as_portrait_is_hard_rejected():
    # Regression shape from the curtain-side failure: an approximately 29x50
    # screen footprint for a physical 1.25:1 landscape work.
    quad = np.array([[0, 0], [29, 0], [29, 50], [0, 50]], np.float32)
    metrics = _quad_metrics_original(quad, (1.0, 1.0))
    gate = evaluate_projection_shape_gate(metrics, 1.0, 0.8)
    assert gate["status"] == "hard_reject"
    assert gate["orientation_flip"] is True
    assert "projected_artwork_orientation_flipped_by_foreshortening" in gate["hard_reasons"]


def test_moderate_perspective_keeps_landscape_orientation_and_passes():
    quad = np.array([[0, 0], [100, 4], [96, 80], [2, 78]], np.float32)
    metrics = _quad_metrics_original(quad, (1.0, 1.0))
    gate = evaluate_projection_shape_gate(metrics, 1.0, 0.8)
    assert gate["status"] == "pass"
    assert gate["orientation_flip"] is False


def test_gravity_consensus_preserves_camera_pitch_and_drops_outlier():
    pitched_up = np.array([0.04, -0.993, -0.11], np.float32)
    pitched_up /= np.linalg.norm(pitched_up)
    near = np.array([0.02, -0.996, -0.085], np.float32)
    near /= np.linalg.norm(near)
    bad = np.array([0.75, -0.20, -0.63], np.float32)
    bad /= np.linalg.norm(bad)
    fused, confidence, debug = select_gravity_consensus([
        {"reference_label": "floor", "g_hat": pitched_up, "confidence": 0.9},
        {"reference_label": "roof", "g_hat": near, "confidence": 0.8},
        {"reference_label": "contaminated_table", "g_hat": bad, "confidence": 0.6},
    ])
    assert float(np.degrees(np.arccos(np.clip(np.dot(fused, pitched_up), -1, 1)))) < 3.0
    assert abs(float(fused[2])) > 0.07  # pitch is preserved, not forced to [0,-1,0]
    assert confidence > 0.4
    contaminated = next(x for x in debug["candidates"] if x["reference_label"] == "contaminated_table")
    assert contaminated["included_in_consensus"] is False
    assert debug["gravity_consensus_unresolved"] is False


def test_two_strong_conflicting_gravity_references_fail_closed():
    up = np.array([0.0, -1.0, 0.0], np.float32)
    side = np.array([0.8, -0.2, -0.56], np.float32)
    side /= np.linalg.norm(side)
    _fused, _confidence, debug = select_gravity_consensus([
        {"reference_label": "floor", "g_hat": up, "confidence": 0.9},
        {"reference_label": "roof", "g_hat": side, "confidence": 0.8},
    ])
    assert debug["gravity_reference_conflict"] is True
    assert debug["gravity_consensus_unresolved"] is True
