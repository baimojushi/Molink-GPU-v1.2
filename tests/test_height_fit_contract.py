from __future__ import annotations

from hanging_mvp.height_fit_contract import floor_source_is_metric_trusted, room_corner_floor_fit_contract


def test_room_corner_floor_fit_is_spec_only_with_quantified_thresholds():
    contract = room_corner_floor_fit_contract({})
    assert contract["enabled"] is False
    assert contract["implementation_status"] == "spec_only_not_enabled_in_production"
    assert contract["min_valid_wall_planes"] == 2
    assert contract["min_wall_normal_angle_deg"] == 15.0
    assert contract["min_wall_ransac_inlier_ratio"] == 0.60
    assert contract["max_corner_gravity_angle_deg"] == 15.0
    assert contract["floor_side_extent_percentile"] == 3.0


def test_only_floor_mask_projection_is_metric_trusted_for_current_release():
    assert floor_source_is_metric_trusted("floor_mask_projected_to_current_wall_plane") is True
    assert floor_source_is_metric_trusted("solid_wall_mask_bottom_percentile") is False
    assert floor_source_is_metric_trusted("visible_wall_bottom_percentile_fallback") is False
