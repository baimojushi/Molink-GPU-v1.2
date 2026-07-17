from __future__ import annotations

import numpy as np

from hanging_mvp.schemas import WallBasis, WallPlane
from hanging_mvp.support_context import _estimate_axis_support_context, _resolve_furniture_height


def _scene():
    h, w = 20, 20
    xyz = np.zeros((h, w, 3), np.float32)
    xyz[:, :, 0] = np.linspace(-0.2, 0.2, w, dtype=np.float32)[None, :]
    xyz[:, :, 1] = 0.8
    xyz[:10, :, 2] = 4.95  # counter touches the current z=5 wall
    xyz[10:, :, 2] = 2.0   # foreground table is three metres away
    basis = WallBasis(
        origin=(0.0, 0.0, 5.0),
        u=(1.0, 0.0, 0.0),
        v=(0.0, -1.0, 0.0),
        n=(0.0, 0.0, -1.0),
        g_hat_cam=(0.0, -1.0, 0.0),
    )
    plane = WallPlane(n=(0.0, 0.0, -1.0), d=5.0)
    rules = {
        "support_surface_constraints": {
            "min_valid_points": 20,
            "min_floor_valid_points": 20,
            "max_distance_to_wall_m": 0.10,
            "min_touch_ratio": 0.08,
        },
        "height_rules": {
            "counter": {"center_height_m": 1.35},
            "floor": {"center_height_above_floor_m": 1.52},
        },
    }
    return xyz, basis, plane, rules


def test_foreground_table_cannot_claim_support_context_of_current_wall():
    xyz, basis, plane, rules = _scene()
    counter_mask = np.zeros((20, 20), np.uint8)
    counter_mask[:10, :] = 255
    table_mask = np.zeros((20, 20), np.uint8)
    table_mask[10:, :] = 255
    objects = {
        "counter": {"object_id": "counter_06", "label": "counter", "mask": counter_mask},
        "table": {"object_id": "table_05", "label": "table", "mask": table_mask},
    }

    result = _estimate_axis_support_context(
        objects,
        xyz,
        basis,
        0.0,
        0.5,
        0.4,
        rules,
        -1.6,
        plane=plane,
        floor_source_trusted=True,
    )

    assert result["label"] == "counter"
    assert result["rule_id"] == "counter_floor_default_min"
    assert result["center_height_above_floor_m"] == 1.52
    table_probe = next(x for x in result["probe_debug"]["probes"] if x["label"] == "table")
    assert table_probe["rejected_reason"] == "support_not_associated_with_current_wall"


def test_trusted_floor_rule_is_used_when_only_support_is_foreground():
    xyz, basis, plane, rules = _scene()
    xyz[:, :, 2] = 2.0  # no unlabeled geometry close enough to the current wall
    table_mask = np.zeros((20, 20), np.uint8)
    table_mask[10:, :] = 255
    objects = {"table": {"object_id": "table_05", "label": "table", "mask": table_mask}}

    result = _estimate_axis_support_context(
        objects,
        xyz,
        basis,
        0.0,
        0.5,
        0.4,
        rules,
        -1.6,
        plane=plane,
        floor_source_trusted=True,
    )

    assert result["label"] == "floor"
    assert result["source"] == "trusted_metric_floor_fallback"
    assert result["rule_id"] == "floor_center_152"
    assert result["center_height_above_floor_m"] == 1.52


def test_job_5b71f65a_unlabeled_wall_furniture_raises_height_for_bottom_clearance():
    xyz, basis, plane, rules = _scene()
    # An unlabeled cabinet face spans 0.70m..1.30m above floor. The highest
    # dense band is its support top; choosing the first histogram maximum would
    # reproduce the too-low fallback.
    xyz[:10, :, 1] = np.linspace(0.30, 0.90, 10, dtype=np.float32)[:, None]
    xyz[:10, :, 2] = 4.50
    xyz[10:, :, 2] = 2.0
    rules["support_surface_constraints"].update({
        "geometry_fallback_enabled": True,
        "geometry_min_front_distance_m": 0.12,
        "geometry_max_front_distance_m": 0.90,
        "geometry_min_support_height_above_floor_m": 0.55,
        "geometry_max_support_height_above_floor_m": 1.45,
        "geometry_height_bin_m": 0.04,
        "geometry_min_height_bin_points": 15,
        "geometry_min_support_span_m": 0.25,
        "furniture_min_bottom_gap_m": 0.15,
        "furniture_never_below_floor_default": True,
    })

    result = _estimate_axis_support_context(
        {}, xyz, basis, 0.0, 0.5, 0.4, rules, -1.6,
        plane=plane, floor_source_trusted=True,
    )

    assert result["label"] == "geometry_support"
    assert result["source"] == "geometry_wall_adjacent_support_probe"
    assert result["rule_id"] == "geometry_support_bottom_clearance_15cm"
    assert abs(result["support_top_abs_m"] - (-0.30)) < 0.03
    assert result["center_height_above_floor_m"] >= 1.64
    assert result["bottom_gap_to_support_m"] >= 0.15 - 1e-6


def test_processed_bbox_fallback_recovers_wall_adjacent_counter_without_mask():
    xyz, basis, plane, rules = _scene()
    xyz[:10, :, 1] = 0.30
    xyz[:10, :, 2] = 4.50
    xyz[10:, :, 2] = 2.0
    rules["support_surface_constraints"].update({
        "bbox_fallback_max_distance_to_wall_m": 0.90,
        "furniture_min_bottom_gap_m": 0.15,
        "furniture_never_below_floor_default": True,
    })
    objects = {
        "counter": {
            "object_id": "counter_bbox_only",
            "label": "counter",
            "bbox_xyxy_processed": [0.0, 0.0, 20.0, 10.0],
        }
    }

    result = _estimate_axis_support_context(
        objects, xyz, basis, 0.0, 0.5, 0.4, rules, -1.6,
        plane=plane, floor_source_trusted=True,
    )

    assert result["label"] == "counter"
    assert result["object_id"] == "counter_bbox_only"
    assert result["rule_id"] == "counter_bottom_clearance_15cm"
    probe = next(x for x in result["probe_debug"]["probes"] if x["object_id"] == "counter_bbox_only")
    assert probe["point_source"] == "processed_bbox_inner_fallback"
    assert probe["wall_association_passed"] is True


def test_processed_bbox_fallback_does_not_attach_distant_foreground_table_to_wall():
    xyz, basis, plane, rules = _scene()
    xyz[:, :, 2] = 2.0
    rules["support_surface_constraints"]["bbox_fallback_max_distance_to_wall_m"] = 0.90
    objects = {
        "table": {
            "object_id": "foreground_table_bbox_only",
            "label": "table",
            "bbox_xyxy_processed": [0.0, 0.0, 20.0, 20.0],
        }
    }

    result = _estimate_axis_support_context(
        objects, xyz, basis, 0.0, 0.5, 0.4, rules, -1.6,
        plane=plane, floor_source_trusted=True,
    )

    assert result["label"] == "floor"
    probe = next(x for x in result["probe_debug"]["probes"] if x["object_id"] == "foreground_table_bbox_only")
    assert probe["point_source"] == "processed_bbox_inner_fallback"
    assert probe["rejected_reason"] == "support_not_associated_with_current_wall"


def test_geometry_fallback_does_not_mistake_wall_plane_for_furniture():
    xyz, basis, plane, rules = _scene()
    xyz[:, :, 2] = 4.95
    rules["support_surface_constraints"].update({
        "geometry_fallback_enabled": True,
        "geometry_min_front_distance_m": 0.12,
        "geometry_max_front_distance_m": 0.90,
        "geometry_min_height_bin_points": 20,
    })

    result = _estimate_axis_support_context(
        {}, xyz, basis, 0.0, 0.5, 0.4, rules, -1.6,
        plane=plane, floor_source_trusted=True,
    )

    assert result["label"] == "floor"
    assert result["rule_id"] == "floor_center_152"


def test_v5_recognized_job_support_uses_cabinet_clearance_instead_of_default_152():
    floor_abs = -1.6403186321258545
    support_top_abs = -0.4720879793167114
    rules = {
        "support_surface_constraints": {
            "furniture_min_bottom_gap_m": 0.15,
            "furniture_never_below_floor_default": True,
        },
        "height_rules": {
            "floor": {"center_height_above_floor_m": 1.52},
            "cabinet": {
                "center_height_m": 1.35,
                "min_bottom_gap_to_support_m": 0.25,
            },
        },
    }

    result = _resolve_furniture_height(
        selected_label="geometry_support",
        support_top_abs_m=support_top_abs,
        floor_height_abs_m=floor_abs,
        artwork_height_m=0.40,
        rules=rules,
    )

    expected_center_above_floor = support_top_abs + 0.25 + 0.20 - floor_abs
    assert result["rule_id"] == "geometry_support_bottom_clearance_25cm"
    assert abs(result["center_height_above_floor_m"] - expected_center_above_floor) < 1e-12
    assert abs(result["center_height_above_floor_m"] - 1.618230652809143) < 1e-9
    assert abs(result["bottom_gap_to_support_m"] - 0.25) < 1e-12
    assert result["center_height_above_floor_m"] > 1.52
