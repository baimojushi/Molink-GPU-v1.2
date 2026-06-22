from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .geometry import points_from_mask
from .rules_access import rule_value

def _candidate_structural_clearance(
    structural_masks: List[Dict[str, Any]],
    xyz: np.ndarray,
    plane,
    basis,
    art_s_min: float,
    art_s_max: float,
    art_t_min: float,
    art_t_max: float,
    artwork_w_m: float,
    rules: Dict[str, Any],
) -> Dict[str, Any]:
    """Fine-grained structural safety check in wall-local coordinates.

    Product rule: a front-wall subregion may pass even when the global wall
    quality score is weak, but every artwork vertical edge must stay at least
    artwork_width * clearance_fraction away from doors/windows/openings/other
    non-hangable structural regions on the same wall band.

    This is intentionally a hard gate, while low wall/plane/normal scores are
    downgraded into risk and sent to hard-paste + VLM review.
    """
    fraction = float(rule_value(
        rules,
        ["structural_clearance_policy", "min_edge_clearance_fraction_of_artwork_width"],
        1.0 / 3.0,
    ))
    required_clearance_m = float(artwork_w_m * fraction)
    vertical_overlap_margin_m = float(rule_value(
        rules,
        ["structural_clearance_policy", "vertical_overlap_margin_m"],
        0.05,
    ))
    max_plane_dist_m = float(rule_value(
        rules,
        ["structural_clearance_policy", "max_structural_wall_plane_distance_m"],
        0.75,
    ))
    min_points = int(rule_value(
        rules,
        ["structural_clearance_policy", "min_structural_points_for_clearance"],
        20,
    ))

    origin = np.asarray(basis.origin, dtype=np.float32).reshape(1, 3)
    u = np.asarray(basis.u, dtype=np.float32).reshape(3)
    v = np.asarray(basis.v, dtype=np.float32).reshape(3)
    n = np.asarray(plane.n, dtype=np.float32).reshape(3)
    d = float(plane.d)

    considered: List[Dict[str, Any]] = []
    blocking: List[Dict[str, Any]] = []
    min_clearance: Optional[float] = None

    for obj in structural_masks:
        mask = obj.get("mask")
        if mask is None or int(np.count_nonzero(mask)) == 0:
            continue
        pts = points_from_mask(xyz, mask, max_points=50000)
        pts = pts[np.isfinite(pts).all(axis=1)] if pts.size else pts
        if len(pts) < min_points:
            considered.append({
                "object_id": obj.get("object_id"),
                "label": obj.get("label"),
                "skipped": True,
                "reason": "not_enough_metric_points",
                "points": int(len(pts)),
            })
            continue

        signed_dist = pts @ n + d
        rel = pts - origin
        s_vals = rel @ u
        t_vals = rel @ v
        keep = (
            np.isfinite(s_vals)
            & np.isfinite(t_vals)
            & np.isfinite(signed_dist)
            & (np.abs(signed_dist) <= max_plane_dist_m)
            & (t_vals >= float(art_t_min) - vertical_overlap_margin_m)
            & (t_vals <= float(art_t_max) + vertical_overlap_margin_m)
        )
        if int(np.count_nonzero(keep)) < min_points:
            considered.append({
                "object_id": obj.get("object_id"),
                "label": obj.get("label"),
                "skipped": True,
                "reason": "no_same_wall_vertical_overlap",
                "points_after_filter": int(np.count_nonzero(keep)),
                "max_structural_wall_plane_distance_m": float(max_plane_dist_m),
            })
            continue

        ss = s_vals[keep]
        tt = t_vals[keep]
        dist = np.abs(signed_dist[keep])
        s_min = float(np.percentile(ss, 5))
        s_max = float(np.percentile(ss, 95))
        t_min = float(np.percentile(tt, 5))
        t_max = float(np.percentile(tt, 95))

        if s_max < art_s_min:
            clearance = float(art_s_min - s_max)
            side = "left_of_artwork"
        elif s_min > art_s_max:
            clearance = float(s_min - art_s_max)
            side = "right_of_artwork"
        else:
            overlap_m = float(min(float(art_s_max), s_max) - max(float(art_s_min), s_min))
            clearance = -max(0.0, overlap_m)
            side = "overlaps_artwork_band"

        min_clearance = clearance if min_clearance is None else min(float(min_clearance), clearance)
        item = {
            "object_id": obj.get("object_id"),
            "label": obj.get("label"),
            "points_after_filter": int(np.count_nonzero(keep)),
            "s_range_m": [s_min, s_max],
            "t_range_m": [t_min, t_max],
            "art_s_range_m": [float(art_s_min), float(art_s_max)],
            "art_t_range_m": [float(art_t_min), float(art_t_max)],
            "edge_clearance_m": float(clearance),
            "relative_position": side,
            "plane_distance_p50_m": float(np.percentile(dist, 50)),
        }
        considered.append(item)
        if clearance < required_clearance_m:
            blocking.append(item)

    return {
        "enabled": True,
        "policy": "artwork_vertical_edges_clear_structural_by_width_fraction",
        "min_edge_clearance_fraction_of_artwork_width": float(fraction),
        "required_clearance_m": float(required_clearance_m),
        "min_clearance_m": float(min_clearance) if min_clearance is not None else None,
        "passed": len(blocking) == 0,
        "considered_objects": considered,
        "blocking_objects": blocking,
        "notes": "Doors/windows/openings are hard-blocking only when their wall-local horizontal clearance to the artwork edge is below artwork_width * fraction. Other low geometry scores become risk, not hard rejection.",
    }
