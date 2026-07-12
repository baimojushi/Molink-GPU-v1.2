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
    return str(source or "") == "wall_floor_plane_intersection"
