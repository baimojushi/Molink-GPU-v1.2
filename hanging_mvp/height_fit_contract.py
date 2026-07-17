from __future__ import annotations

from typing import Any, Mapping


def room_corner_floor_fit_contract(rules: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the production wall/floor plane-intersection contract."""
    rules = rules or {}
    cfg = dict(rules.get("room_corner_floor_fit", {}) or {})
    return {
        "implementation_status": "production",
        "enabled": bool(cfg.get("enabled", True)),
        "min_valid_wall_planes": int(cfg.get("min_valid_wall_planes", 2)),
        "min_wall_normal_angle_deg": float(cfg.get("min_wall_normal_angle_deg", 15.0)),
        "min_wall_ransac_inlier_ratio": float(cfg.get("min_wall_ransac_inlier_ratio", 0.60)),
        "max_corner_gravity_angle_deg": float(cfg.get("max_corner_gravity_angle_deg", 15.0)),
        "floor_side_extent_percentile": float(cfg.get("floor_side_extent_percentile", 3.0)),
        "single_wall_degrade": "height_band_base=unknown; metric_height_finalized=false; requires_visual_height_selection=true",
        "visual_center_band_base": "estimated_full_wall_height_m_only_never_visible_safe_mask_height",
        "notes": "RANSAC floor plane is intersected with each wall plane; untrusted fallbacks cannot auto-commit candidates.",
    }


def floor_source_is_metric_trusted(source: str | None) -> bool:
    return str(source or "") in {
        "wall_floor_plane_intersection",
        "floor_mask_projected_to_current_wall_plane",
    }


def resolve_candidate_center_t(
    *,
    rule_center_height_abs_m: float,
    floor_height_abs_m: float,
    wall_origin_dot_g: float,
    dot_v_g: float,
    height_mode: str,
    axis_item: Mapping[str, Any] | None = None,
    max_metric_rule_deviation_m: float = 0.10,
) -> dict[str, Any]:
    """Resolve vertical placement without letting auto geometry rewrite a metric rule.

    Empty-space discovery is allowed to choose a horizontal axis.  When a
    trusted metric rule exists, its center height remains authoritative; any
    automatic ``center_t_m_override`` is retained only as rejected diagnostic
    evidence.  Provisional/manual height modes may still use such a seed.
    """
    if abs(float(dot_v_g)) < 1e-8:
        raise ValueError("height_projection_unstable_dot_v_g")

    axis_item = dict(axis_item or {})
    rule_center_t_m = (
        float(rule_center_height_abs_m) - float(wall_origin_dot_g)
    ) / float(dot_v_g)
    requested = axis_item.get("center_t_m_override")
    requested_finite = False
    if requested is not None:
        try:
            requested = float(requested)
            requested_finite = bool(requested == requested and abs(requested) != float("inf"))
        except (TypeError, ValueError):
            requested_finite = False

    metric_rule = str(height_mode or "") == "metric_rule"
    override_applied = bool(requested_finite and not metric_rule)
    resolved_t = float(requested) if override_applied else float(rule_center_t_m)
    resolved_abs = float(wall_origin_dot_g) + float(dot_v_g) * resolved_t
    resolved_above_floor = resolved_abs - float(floor_height_abs_m)
    rule_above_floor = float(rule_center_height_abs_m) - float(floor_height_abs_m)
    deviation_m = abs(resolved_above_floor - rule_above_floor)
    consistent = (not metric_rule) or deviation_m <= float(max_metric_rule_deviation_m)

    return {
        "rule_center_t_m": float(rule_center_t_m),
        "resolved_center_t_m": float(resolved_t),
        "rule_center_height_abs_m": float(rule_center_height_abs_m),
        "resolved_center_height_abs_m": float(resolved_abs),
        "rule_center_height_above_floor_m": float(rule_above_floor),
        "resolved_center_height_above_floor_m": float(resolved_above_floor),
        "center_t_override_requested_m": float(requested) if requested_finite else None,
        "center_t_override_source": axis_item.get("source") if requested_finite else None,
        "center_t_override_applied": bool(override_applied),
        "center_t_override_ignored_reason": (
            "metric_height_rule_is_authoritative" if requested_finite and metric_rule else None
        ),
        "metric_height_rule_deviation_m": float(deviation_m),
        "max_metric_height_rule_deviation_m": float(max_metric_rule_deviation_m),
        "metric_height_rule_consistent": bool(consistent),
        "height_contract": "metric_height_authoritative_exact_footprint_v3",
    }


def evaluate_semantic_clear_rect_vertical_fit(
    *,
    center_t_m: float,
    artwork_height_m: float,
    min_edge_margin_m: float,
    component_t_range_m: Any,
    axis_source: str | None,
) -> dict[str, Any]:
    """Describe, but never gate on, a max-empty-rectangle vertical interval.

    The rectangle is only a horizontal candidate seed.  Its vertical interval
    comes from whichever empty rectangle happened to be largest and is not
    evidence that the resolved metric-height footprint is obstructed.  Safety
    is decided later from the artwork quad projected at the authoritative
    height (wall support, edges/corners, structures and blocking objects).
    """
    evaluated = bool(
        isinstance(component_t_range_m, (list, tuple))
        and len(component_t_range_m) >= 2
        and str(axis_source or "") == "semantic_clear_subregion_max_empty_rect_horizontal_axis"
    )
    required_half_h = float(artwork_height_m) / 2.0 + float(min_edge_margin_m)
    result: dict[str, Any] = {
        "evaluated": evaluated,
        "enforced": False,
        "blocking": False,
        "source": axis_source,
        "component_t_range_m": component_t_range_m,
        "center_t_m": float(center_t_m),
        "required_half_height_with_margin_m": float(required_half_h),
        "passed": True,
        "policy": "diagnostic_only_exact_projected_footprint_is_authoritative",
    }
    if not evaluated:
        result["geometric_fit"] = None
        return result

    t_min, t_max = sorted((float(component_t_range_m[0]), float(component_t_range_m[1])))
    lower_clearance = float(center_t_m) - t_min
    upper_clearance = t_max - float(center_t_m)
    result.update({
        "component_t_range_m": [t_min, t_max],
        "lower_clearance_m": float(lower_clearance),
        "upper_clearance_m": float(upper_clearance),
        "geometric_fit": bool(lower_clearance >= required_half_h and upper_clearance >= required_half_h),
    })
    return result


def height_contract_gate_reasons(
    *,
    height_contract: Mapping[str, Any],
    semantic_clear_rect_relation: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return only genuine height-contract hard failures.

    ``semantic_clear_rect_relation`` is accepted deliberately so callers cannot
    accidentally confuse the diagnostic rectangle relation with a hard gate.
    Exact projected-footprint gates are evaluated separately by the service.
    """
    _ = semantic_clear_rect_relation
    reasons: list[str] = []
    if not bool(height_contract.get("metric_height_rule_consistent", True)):
        reasons.append("metric_height_rule_deviation_exceeds_threshold")
    return reasons
