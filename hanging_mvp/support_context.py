from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .geometry import points_from_mask
from .numeric_utils import _safe_percentile
from .rules_access import rule_value

def _estimate_floor_wall_intersection(
    floor_mask,
    xyz: np.ndarray,
    plane,
    basis,
    s_range_m,
    visible_wall_t_vals: np.ndarray,
    solid_wall_mask=None,
    rules: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Estimate the real floor-wall intersection for the *current* wall plane.

    Priority order is intentionally explicit:
      1) project semantic floor_mask point cloud onto the current wall plane;
      2) use the bottom percentile of the solid_wall_mask;
      3) fall back to the visible wall bottom percentile.

    This keeps floor height local to each candidate wall instead of using a global
    image-bottom heuristic, which is important in multi-wall scenes.
    """
    rules = rules or {}
    origin = np.array(basis.origin, np.float32)
    u = np.array(basis.u, np.float32)
    v = np.array(basis.v, np.float32)
    g_hat = np.array(basis.g_hat_cam, np.float32)
    n = np.array(plane.n, np.float32)
    d = float(plane.d)

    s_min, s_max = float(s_range_m[0]), float(s_range_m[1])
    s_margin_m = float(rule_value(rules, ["floor_wall_intersection", "s_margin_m"], 0.35))
    min_floor_points = int(rule_value(rules, ["floor_wall_intersection", "min_floor_points"], 150))
    max_floor_points = int(rule_value(rules, ["floor_wall_intersection", "max_floor_points"], 80000))
    floor_t_percentile = float(rule_value(rules, ["floor_wall_intersection", "floor_projected_t_percentile"], 50.0))
    solid_bottom_percentile = float(rule_value(rules, ["floor_wall_intersection", "solid_wall_bottom_percentile"], 2.0))
    visible_bottom_percentile = float(rule_value(rules, ["floor_wall_intersection", "visible_wall_bottom_percentile"], 2.0))
    max_floor_wall_projection_distance_m = float(rule_value(
        rules,
        ["floor_wall_intersection", "max_floor_wall_projection_distance_m"],
        0.45,
    ))

    debug: Dict[str, Any] = {
        "priority": [
            "floor_mask_projected_to_current_wall_plane",
            "solid_wall_mask_bottom_percentile",
            "visible_wall_bottom_percentile_fallback",
        ],
        "candidate_wall_s_range_m": [s_min, s_max],
        "s_margin_m": float(s_margin_m),
        "max_floor_wall_projection_distance_m": float(max_floor_wall_projection_distance_m),
        "attempts": [],
    }

    def _finish(source: str, t_floor_m: float, count: int, extra: Dict[str, Any]) -> Dict[str, Any]:
        floor_point = origin + v * float(t_floor_m)
        floor_height_abs_m = float(np.dot(floor_point, g_hat))
        result = {
            "source": source,
            "t_floor_m": float(t_floor_m),
            "floor_height_abs_m": floor_height_abs_m,
            "floor_point_on_wall_m": [float(x) for x in floor_point],
            "points_used": int(count),
            "debug": debug,
        }
        result.update(extra)
        debug["selected_source"] = source
        debug["selected_t_floor_m"] = float(t_floor_m)
        debug["selected_floor_height_abs_m"] = floor_height_abs_m
        return result

    if floor_mask is not None:
        floor_pts = points_from_mask(xyz, floor_mask, max_points=max_floor_points)
        floor_pts = floor_pts[np.isfinite(floor_pts).all(axis=1)] if floor_pts.size else floor_pts
        attempt = {
            "source": "floor_mask_projected_to_current_wall_plane",
            "raw_floor_points": int(len(floor_pts)),
            "min_floor_points": int(min_floor_points),
        }
        if len(floor_pts) > 0:
            signed_dist = floor_pts @ n.reshape(3) + d
            projected = floor_pts - signed_dist.reshape(-1, 1) * n.reshape(1, 3)
            rel = projected - origin.reshape(1, 3)
            s = rel @ u.reshape(3)
            t = rel @ v.reshape(3)
            keep = (
                np.isfinite(s)
                & np.isfinite(t)
                & (s >= s_min - s_margin_m)
                & (s <= s_max + s_margin_m)
            )
            t_kept = t[keep]
            dist_kept = np.abs(signed_dist[keep]) if np.any(keep) else np.array([], dtype=np.float32)
            dist_p50 = float(np.percentile(dist_kept, 50)) if dist_kept.size else None
            attempt.update({
                "kept_after_current_wall_s_filter": int(t_kept.size),
                "projected_s_range_m": [float(np.percentile(s[keep], 5)), float(np.percentile(s[keep], 95))] if np.any(keep) else None,
                "signed_distance_to_wall_p50_m": dist_p50,
                "max_floor_wall_projection_distance_m": float(max_floor_wall_projection_distance_m),
                "t_percentile_used": float(floor_t_percentile),
            })
            if t_kept.size >= min_floor_points and dist_p50 is not None and dist_p50 <= max_floor_wall_projection_distance_m:
                t_floor = float(np.percentile(t_kept, floor_t_percentile))
                debug["attempts"].append(attempt)
                return _finish(
                    "floor_mask_projected_to_current_wall_plane",
                    t_floor,
                    int(t_kept.size),
                    {
                        "floor_projected_t_percentile": float(floor_t_percentile),
                        "floor_wall_projection_distance_p50_m": float(dist_p50),
                    },
                )
            if t_kept.size >= min_floor_points:
                attempt["rejected_reason"] = "floor_points_too_far_from_current_wall_plane"
        debug["attempts"].append(attempt)
    else:
        debug["attempts"].append({
            "source": "floor_mask_projected_to_current_wall_plane",
            "skipped": True,
            "reason": "no_floor_mask",
        })

    if solid_wall_mask is not None and np.count_nonzero(solid_wall_mask) > 0:
        solid_pts = points_from_mask(xyz, solid_wall_mask, max_points=50000)
        solid_pts = solid_pts[np.isfinite(solid_pts).all(axis=1)] if solid_pts.size else solid_pts
        rel = solid_pts - origin.reshape(1, 3) if solid_pts.size else solid_pts
        solid_t = rel @ v.reshape(3) if solid_pts.size else np.array([], dtype=np.float32)
        solid_t = solid_t[np.isfinite(solid_t)]
        attempt = {
            "source": "solid_wall_mask_bottom_percentile",
            "solid_points": int(solid_t.size),
            "bottom_percentile": float(solid_bottom_percentile),
        }
        debug["attempts"].append(attempt)
        if solid_t.size > 0:
            t_floor = float(np.percentile(solid_t, solid_bottom_percentile))
            return _finish(
                "solid_wall_mask_bottom_percentile",
                t_floor,
                int(solid_t.size),
                {"solid_wall_bottom_percentile": float(solid_bottom_percentile)},
            )
    else:
        debug["attempts"].append({
            "source": "solid_wall_mask_bottom_percentile",
            "skipped": True,
            "reason": "no_solid_wall_mask",
        })

    visible_t = np.asarray(visible_wall_t_vals, dtype=np.float32)
    visible_t = visible_t[np.isfinite(visible_t)]
    if visible_t.size == 0:
        t_floor = 0.0
        debug["attempts"].append({
            "source": "visible_wall_bottom_percentile_fallback",
            "visible_wall_points": 0,
            "bottom_percentile": float(visible_bottom_percentile),
            "reason": "empty_visible_wall_t_vals_return_zero",
        })
    else:
        t_floor = float(np.percentile(visible_t, visible_bottom_percentile))
        debug["attempts"].append({
            "source": "visible_wall_bottom_percentile_fallback",
            "visible_wall_points": int(visible_t.size),
            "bottom_percentile": float(visible_bottom_percentile),
        })
    return _finish(
        "visible_wall_bottom_percentile_fallback",
        t_floor,
        int(visible_t.size),
        {"visible_wall_bottom_percentile": float(visible_bottom_percentile)},
    )


def _object_points_near_axis(obj: Dict[str, Any], xyz: np.ndarray, origin: np.ndarray, u: np.ndarray, axis_s_m: float, margin_m: float, max_points: int = 30000) -> np.ndarray:
    pts = points_from_mask(xyz, obj.get("mask"), max_points=max_points)
    if pts.size == 0:
        return pts
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.size == 0:
        return pts
    rel = pts - origin.reshape(1, 3)
    s_vals = rel @ u.reshape(3)
    return pts[np.abs(s_vals - float(axis_s_m)) <= float(margin_m)]


def _estimate_axis_support_context(objects: Dict[str, Dict[str, Any]], xyz: np.ndarray, basis, axis_s_m: float, artwork_w_m: float, artwork_h_m: float, rules: Dict[str, Any], floor_height_abs_m: float) -> Dict[str, Any]:
    origin = np.array(basis.origin, np.float32)
    u = np.array(basis.u, np.float32)
    g_hat = np.array(basis.g_hat_cam, np.float32)
    axis_extra = float(rule_value(rules, ["support_surface_constraints", "axis_neighborhood_extra_margin_m"], 0.20))
    margin_m = max(0.15, float(artwork_w_m) * 0.50 + axis_extra)
    min_valid_points = int(rule_value(rules, ["support_surface_constraints", "min_valid_points"], 200))
    min_floor_valid_points = int(rule_value(rules, ["support_surface_constraints", "min_floor_valid_points"], 50))
    labels_by_priority = ["sofa", "table", "cabinet", "floor"]
    probes: List[Dict[str, Any]] = []
    best_by_label: Dict[str, Dict[str, Any]] = {}
    for obj in objects.values():
        label = str(obj.get("label", ""))
        if label not in set(labels_by_priority):
            continue
        pts = _object_points_near_axis(obj, xyz, origin, u, axis_s_m, margin_m)
        count = int(len(pts))
        if count <= 0:
            probes.append({"object_id": obj.get("object_id"), "label": label, "valid_points_near_axis": 0})
            continue
        h_abs = pts @ g_hat.reshape(3)
        top_abs = _safe_percentile(h_abs, 85, None)
        bottom_abs = _safe_percentile(h_abs, 5, None)
        probe = {"object_id": obj.get("object_id"), "label": label, "valid_points_near_axis": count, "axis_margin_m": float(margin_m), "support_top_abs_m_p85": top_abs, "support_bottom_abs_m_p05": bottom_abs}
        probes.append(probe)
        required_points = min_floor_valid_points if label == "floor" else min_valid_points
        if count >= required_points:
            prev = best_by_label.get(label)
            if prev is None or count > int(prev.get("valid_points_near_axis", 0)):
                best_by_label[label] = probe
    selected_label = None; selected_probe = None
    for label in labels_by_priority:
        if label in best_by_label:
            selected_label = label; selected_probe = best_by_label[label]; break
    if selected_label is None:
        selected_label = "unknown"; selected_probe = None
    floor_abs = float(floor_height_abs_m)
    flux_visual_refinement: Optional[Dict[str, Any]] = None
    if selected_label == "sofa":
        cfg = rules.get("height_rules", {}).get("sofa", {})
        initial_center_height_m = float(cfg.get("initial_center_height_m", 1.35))
        max_gap = float(cfg.get("max_bottom_gap_to_support_m", 0.15))
        support_top_abs = selected_probe.get("support_top_abs_m_p85") if selected_probe else None
        center_abs = floor_abs + initial_center_height_m
        rule_id = "sofa_center_135"; bottom_gap = None
        if support_top_abs is not None:
            bottom_abs = center_abs - float(artwork_h_m) / 2.0
            bottom_gap = float(bottom_abs - float(support_top_abs))
            if bottom_gap > max_gap:
                center_abs = float(support_top_abs) + max_gap + float(artwork_h_m) / 2.0
                bottom_gap = max_gap
                rule_id = "sofa_bottom_gap_15"
        mode = "metric_rule"; center_height_above_floor_m = center_abs - floor_abs
    elif selected_label in {"table", "cabinet"}:
        cfg = rules.get("height_rules", {}).get(selected_label, {})
        center_height_m = float(cfg.get("center_height_m", 1.35))
        center_abs = floor_abs + center_height_m
        center_height_above_floor_m = center_height_m
        rule_id = f"{selected_label}_center_135"; bottom_gap = None; mode = "metric_rule"
    elif selected_label == "floor":
        cfg = rules.get("height_rules", {}).get("floor", {})
        center_height_m = float(cfg.get("center_height_above_floor_m", 1.52))
        center_abs = floor_abs + center_height_m
        center_height_above_floor_m = center_height_m
        rule_id = "floor_center_152"; bottom_gap = None; mode = "metric_rule"
    else:
        # Unknown support remains a provisional visual seed. FLUX must not decide
        # a new installation height, because the P2 room-corner floor fit is still
        # spec-only and web-image metric scale can be unreliable.
        cfg = rules.get("height_rules", {}).get("unknown", {})
        floor_cfg = rules.get("height_rules", {}).get("floor", {})
        fallback_h = float(cfg.get("provisional_center_height_above_floor_m", floor_cfg.get("center_height_above_floor_m", 1.52)))
        center_abs = floor_abs + fallback_h
        center_height_above_floor_m = fallback_h
        rule_id = "unknown_provisional_visual_review_required"; bottom_gap = None; mode = "provisional_visual_review_required"
        flux_visual_refinement = None
    metric_height_finalized = (mode == "metric_rule")
    requires_flux_refine = False
    return {
        "label": selected_label,
        "object_id": selected_probe.get("object_id") if selected_probe else None,
        "source": "semantic_support_probe" if selected_probe else "geometry_floor_fallback_for_flux_seed",
        "support_top_abs_m": selected_probe.get("support_top_abs_m_p85") if selected_probe else None,
        "support_bottom_abs_m": selected_probe.get("support_bottom_abs_m_p05") if selected_probe else None,
        "floor_height_abs_m": float(floor_abs),
        "center_height_abs_m": float(center_abs),
        "center_height_above_floor_m": float(center_height_above_floor_m),
        "bottom_gap_to_support_m": bottom_gap,
        "height_mode": mode,
        "rule_id": rule_id,
        "metric_height_finalized": metric_height_finalized,
        "height_is_provisional": not metric_height_finalized,
        "requires_frontend_visual_height_selection": not metric_height_finalized,
        "requires_visual_height_selection": not metric_height_finalized,
        "requires_flux_visual_height_refinement": requires_flux_refine,
        "height_band_base": "unknown" if not metric_height_finalized else "trusted_metric_floor_or_support",
        "height_warning": "web_image_metric_scale_or_floor_anchor_untrusted" if not metric_height_finalized else None,
        "flux_visual_refinement": flux_visual_refinement,
        "probe_debug": {"axis_s_m": float(axis_s_m), "axis_margin_m": float(margin_m), "min_valid_points": int(min_valid_points), "probes": probes[:24]},
    }


def _quad_metrics_original(quad_processed: np.ndarray, scale_xy) -> Dict[str, float]:
    sx, sy = float(scale_xy[0]), float(scale_xy[1])
    q = np.asarray(quad_processed, dtype=np.float32).copy()
    q[:, 0] *= sx; q[:, 1] *= sy
    edge = [float(np.linalg.norm(q[(i + 1) % 4] - q[i])) for i in range(4)]
    width_px = 0.5 * (edge[0] + edge[2])
    height_px = 0.5 * (edge[1] + edge[3])
    short_side = max(1e-6, min(width_px, height_px)); long_side = max(width_px, height_px)
    area_px = float(abs(cv2.contourArea(q.astype(np.float32))))
    return {"projected_width_px_original": float(width_px), "projected_height_px_original": float(height_px), "projected_area_px_original": float(area_px), "quad_aspect_ratio_long_over_short": float(long_side / short_side)}
