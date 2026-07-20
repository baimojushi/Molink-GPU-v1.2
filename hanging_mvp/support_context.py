from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .geometry import fit_plane_ransac, points_from_mask
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
    normal: Optional[np.ndarray] = None,
    rules: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Estimate the real floor-wall intersection for the *current* wall plane.

    Priority order is intentionally explicit:
      1) intersect a trusted floor plane with the current wall plane;
      2) project near-wall semantic floor points onto the current wall plane;
      3) use the bottom percentile of the solid_wall_mask;
      4) fall back to the visible wall bottom percentile.

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
            "wall_floor_plane_intersection",
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
        floor_height_signed_m = float(np.dot(floor_point, g_hat))
        camera_height_above_floor_m = -floor_height_signed_m
        result = {
            "source": source,
            "t_floor_m": float(t_floor_m),
            # Legacy field kept for consumers; it has always been signed.
            "floor_height_abs_m": floor_height_signed_m,
            "floor_height_signed_m": floor_height_signed_m,
            "camera_height_above_floor_m": camera_height_above_floor_m,
            "floor_below_camera": bool(floor_height_signed_m < 0.0),
            "height_coordinate_convention": "signed_dot_with_camera_up; floor_below_camera_is_negative",
            "floor_point_on_wall_m": [float(x) for x in floor_point],
            "points_used": int(count),
            "debug": debug,
        }
        result.update(extra)
        debug["selected_source"] = source
        debug["selected_t_floor_m"] = float(t_floor_m)
        debug["selected_floor_height_abs_m"] = floor_height_signed_m
        debug["selected_floor_height_signed_m"] = floor_height_signed_m
        debug["selected_camera_height_above_floor_m"] = camera_height_above_floor_m
        return result

    if floor_mask is not None:
        raw_floor_points = int(np.count_nonzero(floor_mask))
        floor_sampling_mask = floor_mask
        floor_normal_debug: Dict[str, Any] = {
            "enabled": False,
            "raw_floor_mask_points": raw_floor_points,
            "source": "semantic_floor_mask_unfiltered",
        }
        if (
            normal is not None
            and isinstance(normal, np.ndarray)
            and normal.shape[:2] == floor_mask.shape[:2]
            and normal.ndim == 3
            and normal.shape[-1] == 3
        ):
            floor_normal_alignment_threshold = float(rule_value(
                rules,
                ["floor_wall_intersection", "floor_normal_alignment_dot_threshold"],
                0.72,
            ))
            normal_len = np.linalg.norm(normal, axis=-1)
            finite_normal = np.isfinite(normal).all(axis=-1) & (normal_len > 1e-8)
            normalized_normal = np.zeros_like(normal, dtype=np.float32)
            normalized_normal[finite_normal] = (
                normal[finite_normal] / normal_len[finite_normal].reshape(-1, 1)
            )
            floor_alignment = np.abs(np.sum(
                normalized_normal * g_hat.reshape(1, 1, 3), axis=-1
            ))
            floor_like_mask = (
                (floor_mask > 0)
                & finite_normal
                & (floor_alignment >= floor_normal_alignment_threshold)
            ).astype(np.uint8) * 255
            floor_like_points = int(np.count_nonzero(floor_like_mask))
            floor_normal_debug = {
                "enabled": True,
                "raw_floor_mask_points": raw_floor_points,
                "floor_like_points": floor_like_points,
                "floor_normal_alignment_dot_threshold": floor_normal_alignment_threshold,
                "source": "semantic_floor_mask_intersect_gravity_aligned_normals",
                "used": bool(floor_like_points >= min_floor_points),
            }
            if floor_like_points >= min_floor_points:
                floor_sampling_mask = floor_like_mask

        debug["floor_normal_prefilter"] = floor_normal_debug
        floor_pts = points_from_mask(xyz, floor_sampling_mask, max_points=max_floor_points)
        floor_pts = floor_pts[np.isfinite(floor_pts).all(axis=1)] if floor_pts.size else floor_pts
        fit_cfg = dict(rules.get("room_corner_floor_fit", {}) or {})
        fit_attempt = {"source": "wall_floor_plane_intersection", "raw_floor_points": int(len(floor_pts))}
        if bool(fit_cfg.get("enabled", True)) and len(floor_pts) >= int(fit_cfg.get("min_floor_points", 500)):
            try:
                threshold = float(fit_cfg.get("floor_ransac_threshold_m", 0.04))
                floor_plane, inlier_ratio = fit_plane_ransac(floor_pts, iterations=int(fit_cfg.get("floor_ransac_iterations", 700)), threshold=threshold)
                floor_n = np.array(floor_plane.n, np.float32); floor_d = float(floor_plane.d)
                if float(np.dot(floor_n, g_hat)) < 0.0: floor_n, floor_d = -floor_n, -floor_d
                gravity_angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(floor_n, g_hat)), 0.0, 1.0))))
                wall_floor_angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(floor_n, n)), 0.0, 1.0))))
                denom = float(np.dot(floor_n, v)); s_ref = float(0.5 * (s_min + s_max))
                rejects = []
                if inlier_ratio < float(fit_cfg.get("min_floor_ransac_inlier_ratio", 0.60)): rejects.append("floor_plane_inlier_ratio_below_threshold")
                if gravity_angle > float(fit_cfg.get("max_corner_gravity_angle_deg", 15.0)): rejects.append("floor_plane_not_gravity_aligned")
                if wall_floor_angle < float(fit_cfg.get("min_wall_floor_normal_angle_deg", 75.0)): rejects.append("wall_floor_planes_not_near_perpendicular")
                if abs(denom) < 0.75: rejects.append("wall_vertical_axis_nearly_parallel_to_floor")
                fit_attempt.update({"floor_ransac_inlier_ratio": float(inlier_ratio), "floor_normal_to_gravity_angle_deg": gravity_angle, "wall_floor_normal_angle_deg": wall_floor_angle, "reject_reasons": rejects})
                if not rejects:
                    t_floor = -float(np.dot(floor_n, origin + u * s_ref) + floor_d) / denom
                    dt_ds = -float(np.dot(floor_n, u)) / denom
                    debug["attempts"].append(dict(fit_attempt, status="accepted"))
                    return _finish("wall_floor_plane_intersection", t_floor, int(len(floor_pts)), {
                        "metric_trusted": True,
                        "floor_plane": {"n": [float(x) for x in floor_n], "d": floor_d, "ransac_inlier_ratio": float(inlier_ratio)},
                        "intersection_line_wall_coordinates": {"s_reference_m": s_ref, "t_at_reference_m": float(t_floor), "dt_ds": float(dt_ds)},
                        "floor_normal_to_gravity_angle_deg": gravity_angle, "wall_floor_normal_angle_deg": wall_floor_angle,
                    })
            except Exception as exc:
                fit_attempt.update({"status": "error", "error": str(exc)})
        debug["attempts"].append(fit_attempt)
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
            near_wall = keep & (np.abs(signed_dist) <= max_floor_wall_projection_distance_m)
            t_near_wall = t[near_wall]
            dist_near_wall = np.abs(signed_dist[near_wall])
            attempt.update({
                "kept_after_current_wall_s_filter": int(t_kept.size),
                "kept_after_wall_distance_filter": int(t_near_wall.size),
                "projected_s_range_m": [float(np.percentile(s[keep], 5)), float(np.percentile(s[keep], 95))] if np.any(keep) else None,
                "signed_distance_to_wall_p50_m": dist_p50,
                "near_wall_signed_distance_p50_m": float(np.percentile(dist_near_wall, 50)) if dist_near_wall.size else None,
                "max_floor_wall_projection_distance_m": float(max_floor_wall_projection_distance_m),
                "t_percentile_used": float(floor_t_percentile),
                "distance_gate_mode": "select_points_near_current_wall_not_global_distance_median",
            })
            if t_near_wall.size >= min_floor_points:
                t_floor = float(np.percentile(t_near_wall, floor_t_percentile))
                debug["attempts"].append(attempt)
                return _finish(
                    "floor_mask_projected_to_current_wall_plane",
                    t_floor,
                    int(t_near_wall.size),
                    {
                        "metric_trusted": True,
                        "floor_projected_t_percentile": float(floor_t_percentile),
                        "floor_wall_projection_distance_p50_m": float(np.percentile(dist_near_wall, 50)),
                    },
                )
            if t_kept.size >= min_floor_points:
                attempt["rejected_reason"] = "not_enough_floor_points_near_current_wall_plane"
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


def _object_sampling_mask(obj: Dict[str, Any], shape_hw: tuple[int, int]) -> tuple[Optional[np.ndarray], str]:
    """Return a SAM mask, or a conservative inner 2D-box fallback.

    Worker/seed semantic payloads may retain ``bbox_xyxy_processed`` while the
    runtime mask is unavailable.  Sampling only the inner box reduces wall and
    floor contamination; depth-to-current-wall checks remain mandatory later.
    """
    mask = obj.get("mask")
    if mask is not None:
        arr = np.asarray(mask)
        if arr.ndim == 2 and arr.shape[:2] == tuple(shape_hw) and np.count_nonzero(arr) > 0:
            return arr, "semantic_mask"

    bbox = obj.get("bbox_xyxy_processed")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None, "missing_mask_and_processed_bbox"
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    except (TypeError, ValueError):
        return None, "invalid_processed_bbox"
    h, w = int(shape_hw[0]), int(shape_hw[1])
    if not all(np.isfinite([x1, y1, x2, y2])) or x2 <= x1 or y2 <= y1:
        return None, "invalid_processed_bbox"
    inset_x = max(1.0, 0.12 * (x2 - x1))
    inset_y = max(1.0, 0.12 * (y2 - y1))
    ix1 = max(0, min(w, int(np.floor(x1 + inset_x))))
    iy1 = max(0, min(h, int(np.floor(y1 + inset_y))))
    ix2 = max(0, min(w, int(np.ceil(x2 - inset_x))))
    iy2 = max(0, min(h, int(np.ceil(y2 - inset_y))))
    if ix2 <= ix1 or iy2 <= iy1:
        return None, "processed_bbox_inner_region_empty"
    fallback = np.zeros((h, w), dtype=np.uint8)
    fallback[iy1:iy2, ix1:ix2] = 255
    return fallback, "processed_bbox_inner_fallback"


def _object_points_near_axis(
    obj: Dict[str, Any],
    xyz: np.ndarray,
    origin: np.ndarray,
    u: np.ndarray,
    axis_s_m: float,
    margin_m: float,
    max_points: int = 30000,
) -> tuple[np.ndarray, str]:
    sample_mask, point_source = _object_sampling_mask(obj, xyz.shape[:2])
    pts = points_from_mask(xyz, sample_mask, max_points=max_points)
    if pts.size == 0:
        return pts, point_source
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.size == 0:
        return pts, point_source
    rel = pts - origin.reshape(1, 3)
    s_vals = rel @ u.reshape(3)
    return pts[np.abs(s_vals - float(axis_s_m)) <= float(margin_m)], point_source


def _estimate_same_wall_art_height_anchor(
    *,
    objects: Dict[str, Dict[str, Any]],
    xyz: np.ndarray,
    plane,
    basis,
    s_range_m,
    rules: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Use existing artwork on the same fitted wall as a direct height anchor.

    This is deliberately not a synthetic floor estimate.  In a crowded room
    the floor can be entirely occluded while one or more already-hung works are
    visible.  Their wall-local centres are valid metric positions for proposing
    another work on that same plane.  Plane distance, horizontal wall range and
    minimum point support all have to pass; labels such as signs, hooks and
    shelves are never accepted as anchors.
    """
    rules = rules or {}
    cfg = dict(rules.get("existing_art_height_anchor", {}) or {})
    if not bool(cfg.get("enabled", True)) or plane is None:
        return None
    labels = {
        str(v).strip().lower()
        for v in cfg.get(
            "labels",
            ["existing_art", "framed_art", "painting", "artwork", "poster"],
        )
        if str(v).strip()
    }
    min_points = int(cfg.get("min_points", 30))
    max_plane_distance_m = float(cfg.get("max_plane_distance_m", 0.22))
    min_near_plane_ratio = float(cfg.get("min_near_plane_ratio", 0.30))
    s_margin_m = float(cfg.get("s_margin_m", 0.20))
    max_anchor_spread_m = float(cfg.get("max_anchor_spread_m", 0.45))

    origin = np.asarray(basis.origin, dtype=np.float32).reshape(3)
    u = np.asarray(basis.u, dtype=np.float32).reshape(3)
    v = np.asarray(basis.v, dtype=np.float32).reshape(3)
    g_hat = np.asarray(basis.g_hat_cam, dtype=np.float32).reshape(3)
    n = np.asarray(plane.n, dtype=np.float32).reshape(3)
    s_min, s_max = sorted((float(s_range_m[0]), float(s_range_m[1])))
    anchors: List[Dict[str, Any]] = []

    for obj in objects.values():
        label = str(obj.get("label") or "").strip().lower()
        if label not in labels:
            continue
        mask, point_source = _object_sampling_mask(obj, xyz.shape[:2])
        if mask is None:
            continue
        pts = points_from_mask(xyz, mask, max_points=15000)
        pts = pts[np.isfinite(pts).all(axis=1)] if pts.size else pts
        if len(pts) < min_points:
            continue
        dist = np.abs(pts @ n + float(plane.d))
        near = dist <= max_plane_distance_m
        near_count = int(np.count_nonzero(near))
        near_ratio = float(near_count / max(1, len(pts)))
        if near_count < min_points or near_ratio < min_near_plane_ratio:
            continue
        wall_pts = pts[near]
        rel = wall_pts - origin.reshape(1, 3)
        s_vals = rel @ u
        t_vals = rel @ v
        s_center = float(np.median(s_vals))
        if s_center < s_min - s_margin_m or s_center > s_max + s_margin_m:
            continue
        anchors.append({
            "object_id": obj.get("object_id"),
            "label": label,
            "point_source": point_source,
            "point_count": int(len(pts)),
            "near_plane_point_count": near_count,
            "near_plane_ratio": near_ratio,
            "plane_distance_p50_m": float(np.percentile(dist[near], 50)),
            "center_s_m": s_center,
            "center_t_m": float(np.median(t_vals)),
            "center_height_abs_m": float(np.median(wall_pts @ g_hat)),
        })

    if not anchors:
        return None
    center_heights = np.asarray([a["center_height_abs_m"] for a in anchors], dtype=np.float32)
    center_ts = np.asarray([a["center_t_m"] for a in anchors], dtype=np.float32)
    spread = float(np.max(center_heights) - np.min(center_heights)) if len(center_heights) > 1 else 0.0
    if spread > max_anchor_spread_m:
        # Keep the densest height cluster instead of averaging works hung on
        # intentionally different rows.  A single isolated row is still valid.
        best_cluster: List[int] = []
        for idx, value in enumerate(center_heights):
            members = [j for j, other in enumerate(center_heights) if abs(float(other - value)) <= max_anchor_spread_m / 2.0]
            if len(members) > len(best_cluster):
                best_cluster = members
        anchors = [anchors[i] for i in best_cluster]
        center_heights = np.asarray([a["center_height_abs_m"] for a in anchors], dtype=np.float32)
        center_ts = np.asarray([a["center_t_m"] for a in anchors], dtype=np.float32)

    return {
        "source": "same_wall_existing_art_center",
        "metric_trusted": True,
        "center_height_abs_m": float(np.median(center_heights)),
        "center_t_m": float(np.median(center_ts)),
        "anchor_count": len(anchors),
        "anchors": anchors,
        "policy": "direct_same_wall_height_anchor_never_relabel_as_floor",
    }


def _infer_wall_adjacent_geometry_support(
    *,
    xyz: np.ndarray,
    origin: np.ndarray,
    u: np.ndarray,
    g_hat: np.ndarray,
    plane,
    axis_s_m: float,
    margin_m: float,
    floor_height_abs_m: float,
    rules: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Infer an unlabeled counter/cabinet top close to the candidate wall.

    This is a fallback for scenes where VLM/SAM misses the wall-adjacent
    furniture entirely.  It intentionally excludes the wall plane itself and
    distant foreground furniture, then looks for a dense metric-height band in
    the plausible furniture range.
    """
    if plane is None or not bool(rule_value(rules, ["support_surface_constraints", "geometry_fallback_enabled"], True)):
        return None
    pts = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.size == 0:
        return None
    n = np.asarray(plane.n, dtype=np.float32).reshape(3)
    rel = pts - origin.reshape(1, 3)
    s_vals = rel @ u.reshape(3)
    h_abs = pts @ g_hat.reshape(3)
    h_above_floor = h_abs - float(floor_height_abs_m)
    wall_dist = np.abs(pts @ n + float(plane.d))
    min_wall_dist = float(rule_value(rules, ["support_surface_constraints", "geometry_min_front_distance_m"], 0.12))
    max_wall_dist = float(rule_value(rules, ["support_surface_constraints", "geometry_max_front_distance_m"], 0.90))
    min_height = float(rule_value(rules, ["support_surface_constraints", "geometry_min_support_height_above_floor_m"], 0.55))
    max_height = float(rule_value(rules, ["support_surface_constraints", "geometry_max_support_height_above_floor_m"], 1.45))
    bin_size = max(0.02, float(rule_value(rules, ["support_surface_constraints", "geometry_height_bin_m"], 0.04)))
    min_bin_points = int(rule_value(rules, ["support_surface_constraints", "geometry_min_height_bin_points"], 200))
    keep = (
        (np.abs(s_vals - float(axis_s_m)) <= float(margin_m))
        & (wall_dist >= min_wall_dist)
        & (wall_dist <= max_wall_dist)
        & (h_above_floor >= min_height)
        & (h_above_floor <= max_height)
    )
    if int(np.count_nonzero(keep)) < min_bin_points:
        return None
    kept_h = h_above_floor[keep]
    kept_s = s_vals[keep]
    edges = np.arange(min_height, max_height + bin_size * 1.5, bin_size, dtype=np.float32)
    counts, edges = np.histogram(kept_h, bins=edges)
    if counts.size == 0 or int(np.max(counts)) < min_bin_points:
        return None
    dense_threshold = max(min_bin_points, int(np.ceil(float(np.max(counts)) * 0.50)))
    qualifying = np.flatnonzero(counts >= dense_threshold)
    if qualifying.size == 0:
        return None
    # A vertical cabinet face can populate many bands similarly. The highest
    # dense band is the relevant support top; np.argmax would choose the first
    # (lowest) band and reproduce the too-low placement bug.
    best_idx = int(qualifying[-1])
    low = float(edges[best_idx])
    high = float(edges[best_idx + 1])
    band = (kept_h >= low - bin_size * 0.5) & (kept_h <= high + bin_size * 0.5)
    band_count = int(np.count_nonzero(band))
    if band_count < min_bin_points:
        return None
    s_span = float(np.percentile(kept_s[band], 90) - np.percentile(kept_s[band], 10))
    min_s_span = float(rule_value(rules, ["support_surface_constraints", "geometry_min_support_span_m"], 0.25))
    if s_span < min_s_span:
        return None
    support_height_above_floor = float(np.percentile(kept_h[band], 85))
    support_top_abs = float(floor_height_abs_m) + support_height_above_floor
    return {
        "object_id": "geometry_support_at_axis",
        "label": "geometry_support",
        "valid_points_near_axis": int(np.count_nonzero(keep)),
        "valid_points_near_current_wall": band_count,
        "axis_margin_m": float(margin_m),
        "support_top_abs_m_p85": support_top_abs,
        "support_bottom_abs_m_p05": None,
        "support_height_above_floor_m": support_height_above_floor,
        "point_source": "unlabeled_wall_adjacent_geometry_height_band",
        "wall_association_passed": True,
        "geometry_wall_distance_range_m": [min_wall_dist, max_wall_dist],
        "geometry_height_range_above_floor_m": [min_height, max_height],
        "geometry_selected_height_band_m": [low, high],
        "geometry_height_band_points": band_count,
        "geometry_dense_bin_threshold_points": int(dense_threshold),
        "geometry_support_s_span_m": s_span,
    }


def _resolve_furniture_height(
    *,
    selected_label: str,
    support_top_abs_m: Optional[float],
    floor_height_abs_m: float,
    artwork_height_m: float,
    rules: Dict[str, Any],
    is_semantic_proxy: bool = False,
) -> Dict[str, Any]:
    """Resolve furniture-aware height with label-specific bottom clearance."""
    cfg_key = selected_label if selected_label in {"table", "cabinet", "counter", "bar"} else "cabinet"
    cfg = rules.get("height_rules", {}).get(cfg_key, rules.get("height_rules", {}).get("cabinet", {}))
    preferred_center_height_m = float(cfg.get("center_height_m", 1.35))
    floor_cfg = rules.get("height_rules", {}).get("floor", {})
    floor_default_center_m = float(floor_cfg.get("center_height_above_floor_m", 1.52))
    # When the label came from geometry_support remap, the height is known from
    # point-cloud geometry — don't let the floor_default override it.
    never_below_floor_default = (
        bool(rule_value(
            rules,
            ["support_surface_constraints", "furniture_never_below_floor_default"],
            True,
        ))
        and not is_semantic_proxy
    )
    base_center_height_m = (
        max(preferred_center_height_m, floor_default_center_m)
        if never_below_floor_default
        else preferred_center_height_m
    )
    floor_abs = float(floor_height_abs_m)
    center_abs = floor_abs + base_center_height_m
    global_gap = float(rule_value(
        rules,
        ["support_surface_constraints", "furniture_min_bottom_gap_m"],
        0.15,
    ))
    min_bottom_gap_m = float(cfg.get("min_bottom_gap_to_support_m", global_gap))
    bottom_gap = None
    rule_id = f"{selected_label}_floor_default_min"
    clearance_center_abs = None
    if support_top_abs_m is not None:
        clearance_center_abs = (
            float(support_top_abs_m)
            + min_bottom_gap_m
            + float(artwork_height_m) / 2.0
        )
        if clearance_center_abs > center_abs:
            center_abs = clearance_center_abs
            rule_id = f"{selected_label}_bottom_clearance_{int(round(min_bottom_gap_m * 100))}cm"
        bottom_gap = float(center_abs - float(artwork_height_m) / 2.0 - float(support_top_abs_m))
    return {
        "center_height_abs_m": float(center_abs),
        "center_height_above_floor_m": float(center_abs - floor_abs),
        "bottom_gap_to_support_m": bottom_gap,
        "min_bottom_gap_to_support_m": float(min_bottom_gap_m),
        "clearance_center_height_abs_m": float(clearance_center_abs) if clearance_center_abs is not None else None,
        "floor_default_center_height_above_floor_m": float(floor_default_center_m),
        "preferred_furniture_center_height_above_floor_m": float(preferred_center_height_m),
        "furniture_never_below_floor_default": bool(never_below_floor_default),
        "rule_id": rule_id,
    }


def _estimate_axis_support_context(
    objects: Dict[str, Dict[str, Any]],
    xyz: np.ndarray,
    basis,
    axis_s_m: float,
    artwork_w_m: float,
    artwork_h_m: float,
    rules: Dict[str, Any],
    floor_height_abs_m: float,
    *,
    plane=None,
    floor_source_trusted: bool = False,
) -> Dict[str, Any]:
    origin = np.array(basis.origin, np.float32)
    u = np.array(basis.u, np.float32)
    g_hat = np.array(basis.g_hat_cam, np.float32)
    axis_extra = float(rule_value(rules, ["support_surface_constraints", "axis_neighborhood_extra_margin_m"], 0.20))
    margin_m = max(0.15, float(artwork_w_m) * 0.50 + axis_extra)
    min_valid_points = int(rule_value(rules, ["support_surface_constraints", "min_valid_points"], 200))
    min_floor_valid_points = int(rule_value(rules, ["support_surface_constraints", "min_floor_valid_points"], 50))
    max_wall_distance_m = float(rule_value(rules, ["support_surface_constraints", "max_distance_to_wall_m"], 0.10))
    bbox_max_wall_distance_m = float(rule_value(rules, ["support_surface_constraints", "bbox_fallback_max_distance_to_wall_m"], 0.90))
    min_touch_ratio = float(rule_value(rules, ["support_surface_constraints", "min_touch_ratio"], 0.08))
    distance_percentile = 20.0
    labels_by_priority = ["sofa", "counter", "counter_fixture", "cabinet", "sideboard", "table", "geometry_support", "floor"]
    probes: List[Dict[str, Any]] = []
    best_by_label: Dict[str, Dict[str, Any]] = {}
    for obj in objects.values():
        label = str(obj.get("label", ""))
        if label not in set(labels_by_priority):
            continue
        pts, point_source = _object_points_near_axis(obj, xyz, origin, u, axis_s_m, margin_m)
        raw_count = int(len(pts))
        if raw_count <= 0:
            probes.append({"object_id": obj.get("object_id"), "label": label, "valid_points_near_axis": 0, "point_source": point_source})
            continue
        association_max_distance_m = bbox_max_wall_distance_m if point_source == "processed_bbox_inner_fallback" else max_wall_distance_m
        association_debug: Dict[str, Any] = {
            "max_distance_to_current_wall_m": float(association_max_distance_m),
            "min_touch_ratio": float(min_touch_ratio),
            "distance_percentile": float(distance_percentile),
            "point_source": point_source,
        }
        if plane is not None:
            plane_n = np.asarray(plane.n, dtype=np.float32).reshape(3)
            wall_dist = np.abs(pts @ plane_n + float(plane.d))
            wall_dist = wall_dist[np.isfinite(wall_dist)]
            dist_p20 = float(np.percentile(wall_dist, distance_percentile)) if wall_dist.size else None
            touch_ratio = float(np.mean(wall_dist <= association_max_distance_m)) if wall_dist.size else 0.0
            associated = bool(
                dist_p20 is not None
                and dist_p20 <= association_max_distance_m
                and touch_ratio >= min_touch_ratio
            )
            association_debug.update({
                "wall_distance_p20_m": dist_p20,
                "wall_touch_ratio": touch_ratio,
                "wall_association_passed": associated,
            })
            if not associated:
                probes.append({
                    "object_id": obj.get("object_id"),
                    "label": label,
                    "valid_points_near_axis": raw_count,
                    "valid_points_near_current_wall": 0,
                    "rejected_reason": "support_not_associated_with_current_wall",
                    **association_debug,
                })
                continue
            pts = pts[np.abs(pts @ plane_n + float(plane.d)) <= association_max_distance_m]
        count = int(len(pts))
        if count <= 0:
            probes.append({
                "object_id": obj.get("object_id"),
                "label": label,
                "valid_points_near_axis": raw_count,
                "valid_points_near_current_wall": 0,
                **association_debug,
            })
            continue
        h_abs = pts @ g_hat.reshape(3)
        top_abs = _safe_percentile(h_abs, 85, None)
        bottom_abs = _safe_percentile(h_abs, 5, None)
        probe = {
            "object_id": obj.get("object_id"),
            "label": label,
            "valid_points_near_axis": raw_count,
            "valid_points_near_current_wall": count,
            "axis_margin_m": float(margin_m),
            "support_top_abs_m_p85": top_abs,
            "support_bottom_abs_m_p05": bottom_abs,
            **association_debug,
        }
        probes.append(probe)
        required_points = min_floor_valid_points if label == "floor" else min_valid_points
        if count >= required_points:
            prev = best_by_label.get(label)
            if prev is None or count > int(prev.get("valid_points_near_current_wall", 0)):
                best_by_label[label] = probe
    semantic_furniture_found = any(label in best_by_label for label in labels_by_priority if label not in {"geometry_support", "floor"})
    if not semantic_furniture_found:
        geometry_probe = _infer_wall_adjacent_geometry_support(
            xyz=xyz,
            origin=origin,
            u=u,
            g_hat=g_hat,
            plane=plane,
            axis_s_m=axis_s_m,
            margin_m=margin_m,
            floor_height_abs_m=floor_height_abs_m,
            rules=rules,
        )
        if geometry_probe is not None:
            probes.append(geometry_probe)
            best_by_label["geometry_support"] = geometry_probe
    selected_label = None; selected_probe = None
    for label in labels_by_priority:
        if label in best_by_label:
            selected_label = label; selected_probe = best_by_label[label]; break
    if selected_label is None:
        selected_label = "floor" if floor_source_trusted else "unknown"
        selected_probe = None

    # Remap geometry_support to nearest semantic furniture label.
    # geometry_support is a point-cloud fallback — VLM may have detected a
    # nearby table/cabinet/etc whose points failed wall association.  Use its
    # label for user-facing output regardless of height match.
    if selected_label == "geometry_support":
        geo_top = selected_probe.get("support_top_abs_m_p85")
        nearest_sem_label: Optional[str] = None
        nearest_sem_dist = float("inf")
        for p in probes:
            p_label = p.get("label", "")
            if p_label in {"sofa", "table", "cabinet", "counter", "counter_fixture", "sideboard", "bar"}:
                p_top = p.get("support_top_abs_m_p85")
                if p_top is not None and geo_top is not None:
                    dist = abs(float(p_top) - float(geo_top))
                else:
                    dist = 0.0  # no height data — treat as nearest by presence
                if dist < nearest_sem_dist:
                    nearest_sem_dist = dist
                    nearest_sem_label = p_label
        if nearest_sem_label is not None:
            selected_label = nearest_sem_label
            # Keep the geometry probe for height computation but tag it
            selected_probe = {**selected_probe, "semantic_proxy_for": nearest_sem_label}

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
    elif selected_label in {"table", "cabinet", "counter", "counter_fixture", "sideboard", "bar", "geometry_support"}:
        support_top_abs = selected_probe.get("support_top_abs_m_p85") if selected_probe else None
        is_proxy = isinstance(selected_probe, dict) and "semantic_proxy_for" in selected_probe
        furniture_height = _resolve_furniture_height(
            selected_label=selected_label,
            support_top_abs_m=support_top_abs,
            floor_height_abs_m=floor_abs,
            artwork_height_m=artwork_h_m,
            rules=rules,
            is_semantic_proxy=is_proxy,
        )
        center_abs = float(furniture_height["center_height_abs_m"])
        center_height_above_floor_m = float(furniture_height["center_height_above_floor_m"])
        bottom_gap = furniture_height["bottom_gap_to_support_m"]
        rule_id = str(furniture_height["rule_id"])
        mode = "metric_rule"
    elif selected_label == "floor":
        cfg = rules.get("height_rules", {}).get("floor", {})
        center_height_m = float(cfg.get("center_height_above_floor_m", 1.52))
        center_abs = floor_abs + center_height_m
        center_height_above_floor_m = center_height_m
        rule_id = "floor_center_152"; bottom_gap = None; mode = "metric_rule"
    else:
        # Unknown support remains a provisional visual seed. FLUX must not decide
        # a new installation height when the floor source is not trusted; web
        # image metric scale can still be unreliable in that path.
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
        "source": (
            ("geometry_wall_adjacent_support_probe" if selected_label == "geometry_support" else "semantic_support_probe") if selected_probe
            else ("trusted_metric_floor_fallback" if selected_label == "floor" and floor_source_trusted else "geometry_floor_fallback_for_flux_seed")
        ),
        "support_top_abs_m": selected_probe.get("support_top_abs_m_p85") if selected_probe else None,
        "support_bottom_abs_m": selected_probe.get("support_bottom_abs_m_p05") if selected_probe else None,
        "floor_height_abs_m": float(floor_abs),
        "center_height_abs_m": float(center_abs),
        "center_height_above_floor_m": float(center_height_above_floor_m),
        "bottom_gap_to_support_m": bottom_gap,
        "furniture_height_policy": furniture_height if selected_label in {"table", "cabinet", "counter", "counter_fixture", "sideboard", "geometry_support"} else None,
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
        "probe_debug": {
            "axis_s_m": float(axis_s_m),
            "axis_margin_m": float(margin_m),
            "min_valid_points": int(min_valid_points),
            "max_distance_to_current_wall_m": float(max_wall_distance_m),
            "bbox_fallback_max_distance_to_wall_m": float(bbox_max_wall_distance_m),
            "min_touch_ratio": float(min_touch_ratio),
            "support_requires_current_wall_association": plane is not None,
            "probes": probes[:24],
        },
        **({"semantic_proxy_for": selected_probe["semantic_proxy_for"]} if isinstance(selected_probe, dict) and "semantic_proxy_for" in selected_probe else {}),
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
