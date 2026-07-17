from __future__ import annotations

from hanging_mvp.height_fit_contract import (
    evaluate_semantic_clear_rect_vertical_fit,
    floor_source_is_metric_trusted,
    height_contract_gate_reasons,
    resolve_candidate_center_t,
    room_corner_floor_fit_contract,
)


def test_room_corner_floor_fit_is_production_with_quantified_thresholds():
    contract = room_corner_floor_fit_contract({})
    assert contract["enabled"] is True
    assert contract["implementation_status"] == "production"
    assert contract["min_valid_wall_planes"] == 2
    assert contract["min_wall_normal_angle_deg"] == 15.0
    assert contract["min_wall_ransac_inlier_ratio"] == 0.60
    assert contract["max_corner_gravity_angle_deg"] == 15.0
    assert contract["floor_side_extent_percentile"] == 3.0


def test_both_metric_floor_sources_are_trusted_for_current_release():
    assert floor_source_is_metric_trusted("wall_floor_plane_intersection") is True
    assert floor_source_is_metric_trusted("floor_mask_projected_to_current_wall_plane") is True
    assert floor_source_is_metric_trusted("solid_wall_mask_bottom_percentile") is False
    assert floor_source_is_metric_trusted("visible_wall_bottom_percentile_fallback") is False


def test_metric_height_rule_ignores_automatic_empty_rect_override():
    # Exact values from the regression: table_center_135 was silently rewritten
    # to 2.121 m by center_t_m_override=0.744.
    floor_height = -1.639449954032898
    rule_center_abs = floor_height + 1.35
    result = resolve_candidate_center_t(
        rule_center_height_abs_m=rule_center_abs,
        floor_height_abs_m=floor_height,
        wall_origin_dot_g=-0.2623492226881792,
        dot_v_g=0.9997621178627014,
        height_mode="metric_rule",
        axis_item={
            "center_t_m_override": 0.7440142892301083,
            "source": "semantic_clear_subregion_max_empty_rect_center",
        },
    )

    assert result["center_t_override_applied"] is False
    assert result["center_t_override_ignored_reason"] == "metric_height_rule_is_authoritative"
    assert abs(result["resolved_center_height_above_floor_m"] - 1.35) < 1e-9
    assert abs(result["resolved_center_t_m"] - (-0.027107179658552)) < 1e-9
    assert result["metric_height_rule_consistent"] is True


def test_provisional_height_may_use_empty_rect_as_visual_seed():
    result = resolve_candidate_center_t(
        rule_center_height_abs_m=-0.1,
        floor_height_abs_m=-1.6,
        wall_origin_dot_g=-0.2,
        dot_v_g=1.0,
        height_mode="provisional_visual_review_required",
        axis_item={"center_t_m_override": 0.5, "source": "visual_seed"},
    )

    assert result["center_t_override_applied"] is True
    assert result["resolved_center_t_m"] == 0.5


def test_regression_job_461299b7_metric_target_outside_high_empty_rect_is_diagnostic_only():
    result = evaluate_semantic_clear_rect_vertical_fit(
        center_t_m=0.129624654020198,
        artwork_height_m=0.40,
        min_edge_margin_m=0.08,
        component_t_range_m=[0.2526162639260292, 1.2188170552253723],
        axis_source="semantic_clear_subregion_max_empty_rect_horizontal_axis",
    )

    assert result["evaluated"] is True
    assert result["enforced"] is False
    assert result["geometric_fit"] is False
    assert result["blocking"] is False
    assert result["passed"] is True
    assert result["center_t_m"] == 0.129624654020198
    assert result["policy"] == "diagnostic_only_exact_projected_footprint_is_authoritative"


def test_non_max_empty_rect_axis_has_no_vertical_gate():
    result = evaluate_semantic_clear_rect_vertical_fit(
        center_t_m=0.13,
        artwork_height_m=0.40,
        min_edge_margin_m=0.08,
        component_t_range_m=None,
        axis_source="solid_component_center",
    )

    assert result["evaluated"] is False
    assert result["geometric_fit"] is None
    assert result["blocking"] is False
    assert result["passed"] is True


def test_job_461299b7_rectangle_mismatch_adds_no_height_gate_reason():
    relation = evaluate_semantic_clear_rect_vertical_fit(
        center_t_m=0.129624654020198,
        artwork_height_m=0.40,
        min_edge_margin_m=0.08,
        component_t_range_m=[0.2526162639260292, 1.2188170552253723],
        axis_source="semantic_clear_subregion_max_empty_rect_horizontal_axis",
    )
    reasons = height_contract_gate_reasons(
        height_contract={"metric_height_rule_consistent": True},
        semantic_clear_rect_relation=relation,
    )

    assert relation["geometric_fit"] is False
    assert reasons == []


def test_real_metric_height_deviation_remains_a_hard_gate():
    reasons = height_contract_gate_reasons(
        height_contract={"metric_height_rule_consistent": False},
        semantic_clear_rect_relation={"geometric_fit": True},
    )

    assert reasons == ["metric_height_rule_deviation_exceeds_threshold"]
