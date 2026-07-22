from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .schemas import Intrinsics, WallPlane, WallBasis

EPS = 1e-8
CAMERA_UP = np.array([0.0, -1.0, 0.0], np.float32)

def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < EPS:
        return v
    return v / n


def orient_vector_to_camera_up(v: np.ndarray) -> np.ndarray:
    """Resolve arbitrary normal sign to OpenCV camera-up (negative y)."""
    value = np.asarray(v, dtype=np.float32).reshape(3)
    if not np.isfinite(value).all() or float(np.linalg.norm(value)) < EPS:
        return CAMERA_UP.copy()
    value = normalize(value)
    if float(np.dot(value, CAMERA_UP)) < 0.0:
        value = -value
    return value.astype(np.float32)


def select_gravity_consensus(
    candidates: List[Dict[str, Any]],
    *,
    max_inlier_angle_deg: float = 12.0,
    conflict_angle_deg: float = 18.0,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Select and fuse mutually consistent gravity references.

    Metric3D normals are expressed in camera coordinates, so a pitched camera
    legitimately produces a gravity vector that is not ``[0,-1,0]``.  We must
    not "repair" that camera pose.  Instead, compare independent floor,
    ceiling/table and global-normal estimates, choose the weighted angular
    medoid, and fuse only references agreeing with it.  A conflicting semantic
    mask can no longer tilt all wall bases by winning on pixel count alone.
    """
    normalized: List[Dict[str, Any]] = []
    for item in candidates or []:
        try:
            confidence = max(0.0, float(item.get("confidence", 0.0)))
            vector = orient_vector_to_camera_up(np.asarray(item.get("g_hat"), np.float32).reshape(3))
        except Exception:
            continue
        if confidence <= 0.0 or not np.isfinite(vector).all():
            continue
        normalized.append({**item, "confidence": confidence, "g_hat": vector})
    if not normalized:
        return CAMERA_UP.copy(), 0.0, {"reason": "no_usable_gravity_candidates", "candidate_count": 0}

    vectors = np.stack([item["g_hat"] for item in normalized], axis=0)
    weights = np.asarray([item["confidence"] for item in normalized], np.float64)
    dots = np.clip(np.abs(vectors @ vectors.T), 0.0, 1.0)
    angles = np.degrees(np.arccos(dots))
    weighted_cost = (angles * weights.reshape(1, -1)).sum(axis=1) / max(float(weights.sum()), EPS)
    medoid_index = int(np.argmin(weighted_cost))
    medoid = vectors[medoid_index]
    medoid_angles = angles[medoid_index]
    inlier_mask = medoid_angles <= float(max_inlier_angle_deg)
    if not np.any(inlier_mask):
        inlier_mask[medoid_index] = True
    fused = np.sum(vectors[inlier_mask] * weights[inlier_mask].reshape(-1, 1), axis=0)
    fused = orient_vector_to_camera_up(fused)
    fused_angles = np.degrees(np.arccos(np.clip(np.abs(vectors @ fused), 0.0, 1.0)))
    inlier_weight = float(weights[inlier_mask].sum())
    total_weight = max(float(weights.sum()), EPS)
    coherence = max(0.0, 1.0 - float(np.average(fused_angles[inlier_mask], weights=weights[inlier_mask])) / max(float(max_inlier_angle_deg), EPS))
    confidence = min(1.0, float(np.average(weights[inlier_mask])) * (inlier_weight / total_weight) * (0.65 + 0.35 * coherence))
    strong = weights >= 0.35
    strong_pairwise = angles[np.ix_(strong, strong)] if int(np.count_nonzero(strong)) >= 2 else np.zeros((0, 0), np.float64)
    strong_max_angle = float(strong_pairwise.max()) if strong_pairwise.size else 0.0
    conflict = bool(strong_max_angle > float(conflict_angle_deg))
    strong_inlier_count = int(np.count_nonzero(strong & inlier_mask))
    unresolved_conflict = bool(conflict and strong_inlier_count < 2)
    debug = {
        "policy": "weighted_angular_medoid_then_inlier_fusion_in_camera_coordinates",
        "camera_pose_is_preserved": True,
        "candidate_count": len(normalized),
        "selected_medoid": str(normalized[medoid_index].get("reference_label") or normalized[medoid_index].get("source") or medoid_index),
        "max_inlier_angle_deg": float(max_inlier_angle_deg),
        "conflict_angle_deg": float(conflict_angle_deg),
        "strong_reference_max_pairwise_angle_deg": strong_max_angle,
        "gravity_reference_conflict": conflict,
        "strong_consensus_inlier_count": strong_inlier_count,
        "gravity_consensus_unresolved": unresolved_conflict,
        "fused_g_hat_cam": [float(x) for x in fused],
        "candidates": [
            {
                "reference_label": str(item.get("reference_label") or item.get("source") or idx),
                "confidence": float(item["confidence"]),
                "g_hat_cam": [float(x) for x in item["g_hat"]],
                "angle_to_fused_deg": float(fused_angles[idx]),
                "included_in_consensus": bool(inlier_mask[idx]),
            }
            for idx, item in enumerate(normalized)
        ],
    }
    return fused.astype(np.float32), float(confidence), debug

def backproject(depth: np.ndarray, intr: Intrinsics) -> np.ndarray:
    h, w = depth.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    z = depth.astype(np.float32)
    x = (xx.astype(np.float32) - intr.cx) / intr.fx * z
    y = (yy.astype(np.float32) - intr.cy) / intr.fy * z
    return np.stack([x, y, z], axis=-1)

def points_from_mask(xyz: np.ndarray, mask: np.ndarray, max_points: int = 50000) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.empty((0, 3), np.float32)
    idx = np.arange(len(xs))
    if len(idx) > max_points:
        idx = np.random.choice(idx, max_points, replace=False)
    return xyz[ys[idx], xs[idx]].astype(np.float32)

def fit_plane_ransac(points: np.ndarray, iterations: int = 500, threshold: float = 0.06) -> Tuple[WallPlane, float]:
    if len(points) < 50:
        raise ValueError("not enough points for plane fitting")
    best_n = None
    best_d = 0.0
    best_inliers = 0
    npts = len(points)
    for _ in range(iterations):
        ids = np.random.choice(npts, 3, replace=False)
        p0, p1, p2 = points[ids]
        n = np.cross(p1 - p0, p2 - p0)
        n = normalize(n)
        if np.linalg.norm(n) < EPS:
            continue
        d = -float(np.dot(n, p0))
        dist = np.abs(points @ n + d)
        inliers = int(np.count_nonzero(dist < threshold))
        if inliers > best_inliers:
            best_inliers = inliers
            best_n = n
            best_d = d
    if best_n is None:
        raise ValueError("RANSAC failed")
    dist = np.abs(points @ best_n + best_d)
    inlier_mask = dist < threshold
    inlier_ratio = float(np.mean(inlier_mask))
    # refine by PCA on inliers
    P = points[inlier_mask]
    c = P.mean(axis=0)
    _, _, vh = np.linalg.svd(P - c, full_matrices=False)
    n = normalize(vh[-1])
    d = -float(np.dot(n, c))
    return WallPlane(tuple(map(float, n)), float(d)), inlier_ratio


def fit_vertical_plane_ransac(
    points: np.ndarray,
    g_hat_cam: np.ndarray,
    iterations: int = 700,
    threshold: float = 0.08,
) -> Tuple[WallPlane, float]:
    """Fit a plane whose normal is constrained perpendicular to gravity.

    Coarse semantic wall regions can contain a larger ceiling, floor or cove
    plane.  Unconstrained RANSAC then correctly finds the largest plane but it
    is the wrong *kind* of plane.  This fallback samples a horizontal wall
    tangent and constructs only vertical-plane hypotheses.  The final PCA
    normal is projected back onto the gravity-orthogonal subspace so refinement
    cannot drift to a horizontal surface again.
    """
    pts = np.asarray(points, dtype=np.float32)
    pts = pts[np.isfinite(pts).all(axis=1)] if pts.ndim == 2 else pts
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 50:
        raise ValueError("not enough points for vertical plane fitting")
    g = normalize(np.asarray(g_hat_cam, dtype=np.float32).reshape(3))
    if float(np.linalg.norm(g)) < EPS:
        raise ValueError("invalid gravity reference")

    best_n = None
    best_d = 0.0
    best_inliers = 0
    npts = len(pts)
    for _ in range(int(iterations)):
        i0, i1 = np.random.choice(npts, 2, replace=False)
        p0, p1 = pts[i0], pts[i1]
        tangent = p1 - p0
        tangent = tangent - float(np.dot(tangent, g)) * g
        if float(np.linalg.norm(tangent)) < EPS:
            continue
        n = normalize(np.cross(g, tangent))
        if float(np.linalg.norm(n)) < EPS:
            continue
        d = -float(np.dot(n, p0))
        inliers = int(np.count_nonzero(np.abs(pts @ n + d) < float(threshold)))
        if inliers > best_inliers:
            best_inliers = inliers
            best_n = n
            best_d = d
    if best_n is None or best_inliers < 3:
        raise ValueError("vertical RANSAC failed")

    first_inliers = np.abs(pts @ best_n + best_d) < float(threshold)
    P = pts[first_inliers]
    center = P.mean(axis=0)
    _, _, vh = np.linalg.svd(P - center, full_matrices=False)
    pca_n = np.asarray(vh[-1], dtype=np.float32)
    n = pca_n - float(np.dot(pca_n, g)) * g
    if float(np.linalg.norm(n)) < EPS:
        n = best_n
    n = normalize(n)
    d = -float(np.median(P @ n))
    final_inliers = np.abs(pts @ n + d) < float(threshold)
    inlier_ratio = float(np.mean(final_inliers))
    return WallPlane(tuple(map(float, n)), float(d)), inlier_ratio


def estimate_segmented_depth_plane_yaw(
    points_3d: np.ndarray,
    wall_mask: np.ndarray,
    g_hat_cam: np.ndarray,
    *,
    grid_size: int = 4,
    min_points_per_cell: int = 50,
    min_cell_inlier_ratio: float = 0.4,
    max_cell_tilt_deg: float = 15.0,
    cluster_angle_deg: float = 6.0,
    min_cluster_cells: int = 4,
    min_cluster_coverage: float = 0.30,
) -> Tuple[Optional[float], bool, Dict[str, Any]]:
    """Estimate wall heading from local depth plane clustering.

    Divides the wall into a grid of cells, fits a plane in each cell via
    RANSAC, and clusters the resulting yaw angles.  The majority-cluster
    yaw is returned as a more robust alternative to whole-wall RANSAC
    when the wall contains occluding furniture or depth noise.

    Returns (yaw_rad_or_None, reliable, debug_dict).
    """
    pts = np.asarray(points_3d, dtype=np.float32)
    mask = np.asarray(wall_mask, dtype=np.uint8)
    g = normalize(np.asarray(g_hat_cam, dtype=np.float32).reshape(3))

    debug: Dict[str, Any] = {
        "enabled": True,
        "grid_size": int(grid_size),
        "total_points": int(len(pts)),
    }

    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < min_points_per_cell * 4:
        debug["reliable"] = False
        debug["reason"] = "insufficient_points_for_segmented_depth"
        return None, False, debug

    # Compute wall bounding box from non-zero mask pixels
    mask_ys, mask_xs = np.nonzero(mask > 0)
    if len(mask_ys) < grid_size * grid_size:
        debug["reliable"] = False
        debug["reason"] = "insufficient_mask_pixels"
        return None, False, debug

    y_min, y_max = int(mask_ys.min()), int(mask_ys.max()) + 1
    x_min, x_max = int(mask_xs.min()), int(mask_xs.max()) + 1
    cell_h = max(1, (y_max - y_min) // grid_size)
    cell_w = max(1, (x_max - x_min) // grid_size)

    # Assign each 3D point to a grid cell via its projected pixel coordinate
    # We need pixel coordinates — derive from mask position heuristic:
    # points_3d is in camera coords; use x,z as horizontal, y as vertical.
    # For grid assignment we use the point's index relative to mask ordering.
    # A simpler approach: use the 3D point's y (vertical) and x (horizontal)
    # to map into grid cells.
    finite_mask = np.isfinite(pts).all(axis=1)
    valid_pts = pts[finite_mask]
    if len(valid_pts) < min_points_per_cell:
        debug["reliable"] = False
        debug["reason"] = "insufficient_finite_points"
        return None, False, debug

    # Map 3D points to grid cells using camera-projected positions
    # Use x (horizontal) and -y (vertical, since camera y is down) for grid
    cell_y_coords = (-valid_pts[:, 1])  # vertical in world
    cell_x_coords = valid_pts[:, 0]     # horizontal in world

    # Normalize to [0, 1] range
    y_range = float(cell_y_coords.max() - cell_y_coords.min())
    x_range = float(cell_x_coords.max() - cell_x_coords.min())
    if y_range < EPS or x_range < EPS:
        debug["reliable"] = False
        debug["reason"] = "degenerate_point_distribution"
        return None, False, debug

    norm_y = (cell_y_coords - cell_y_coords.min()) / y_range
    norm_x = (cell_x_coords - cell_x_coords.min()) / x_range

    grid_y = np.clip((norm_y * grid_size).astype(int), 0, grid_size - 1)
    grid_x = np.clip((norm_x * grid_size).astype(int), 0, grid_size - 1)
    cell_ids = grid_y * grid_size + grid_x

    cell_yaws: List[float] = []
    cell_details: List[Dict[str, Any]] = []

    for cell_id in range(grid_size * grid_size):
        cell_mask = cell_ids == cell_id
        cell_pts = valid_pts[cell_mask]
        if len(cell_pts) < min_points_per_cell:
            cell_details.append({
                "cell": int(cell_id),
                "points": int(len(cell_pts)),
                "skipped": True,
                "reason": "insufficient_points",
            })
            continue
        try:
            plane, inlier_ratio = fit_plane_ransac(cell_pts, iterations=300, threshold=0.06)
        except (ValueError, Exception):
            cell_details.append({
                "cell": int(cell_id),
                "points": int(len(cell_pts)),
                "skipped": True,
                "reason": "ransac_failed",
            })
            continue

        if inlier_ratio < min_cell_inlier_ratio:
            cell_details.append({
                "cell": int(cell_id),
                "points": int(len(cell_pts)),
                "inlier_ratio": round(inlier_ratio, 4),
                "skipped": True,
                "reason": "low_inlier_ratio",
            })
            continue

        n = np.asarray(plane.normal, dtype=np.float32)
        tilt_deg = float(np.degrees(np.arcsin(np.clip(abs(float(np.dot(n, g))), 0.0, 1.0))))
        if tilt_deg > max_cell_tilt_deg:
            cell_details.append({
                "cell": int(cell_id),
                "points": int(len(cell_pts)),
                "tilt_deg": round(tilt_deg, 2),
                "skipped": True,
                "reason": "excessive_tilt",
            })
            continue

        # Extract horizontal yaw: project normal onto gravity-orthogonal plane
        horizontal_n = n - float(np.dot(n, g)) * g
        if float(np.linalg.norm(horizontal_n)) < EPS:
            continue
        horizontal_n = normalize(horizontal_n)
        yaw = float(np.arctan2(float(horizontal_n[2]), float(horizontal_n[0])))
        cell_yaws.append(yaw)
        cell_details.append({
            "cell": int(cell_id),
            "points": int(len(cell_pts)),
            "inlier_ratio": round(inlier_ratio, 4),
            "tilt_deg": round(tilt_deg, 2),
            "yaw_deg": round(float(np.degrees(yaw)), 2),
        })

    debug["cells"] = cell_details
    debug["yaw_count"] = len(cell_yaws)

    if len(cell_yaws) < min_cluster_cells:
        debug["reliable"] = False
        debug["reason"] = "insufficient_cells_with_valid_yaws"
        return None, False, debug

    # Cluster yaws using angular medoid
    yaw_arr = np.array(cell_yaws, dtype=np.float64)
    cluster_threshold = float(np.radians(cluster_angle_deg))

    # Simple greedy clustering
    clusters: List[List[int]] = []
    assigned = [False] * len(yaw_arr)
    for i in range(len(yaw_arr)):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        for j in range(i + 1, len(yaw_arr)):
            if assigned[j]:
                continue
            diff = abs(float(yaw_arr[i]) - float(yaw_arr[j]))
            diff = min(diff, 2.0 * math.pi - diff)
            if diff <= cluster_threshold:
                cluster.append(j)
                assigned[j] = True
        clusters.append(cluster)

    # Find majority cluster
    clusters.sort(key=len, reverse=True)
    best_cluster = clusters[0]
    cluster_size = len(best_cluster)
    cluster_coverage = cluster_size / len(cell_yaws)

    cluster_yaws = yaw_arr[best_cluster]
    consensus_yaw = float(np.median(cluster_yaws))

    # Compute cluster spread
    diffs = np.abs(cluster_yaws - consensus_yaw)
    diffs = np.minimum(diffs, 2.0 * math.pi - diffs)
    cluster_p90 = float(np.degrees(np.percentile(diffs, 90))) if len(diffs) > 0 else 180.0

    reliable = bool(
        cluster_size >= min_cluster_cells
        and cluster_coverage >= min_cluster_coverage
        and cluster_p90 <= cluster_angle_deg
    )

    debug["reliable"] = reliable
    debug["consensus_yaw_deg"] = round(float(np.degrees(consensus_yaw)), 2)
    debug["cluster_size"] = cluster_size
    debug["cluster_coverage"] = round(cluster_coverage, 4)
    debug["cluster_p90_deg"] = round(cluster_p90, 2)
    debug["cluster_count"] = len(clusters)
    debug["all_cluster_sizes"] = [len(c) for c in clusters]

    if reliable:
        debug["source"] = "segmented_depth_plane_yaw_consensus"
    else:
        debug["reason"] = "cluster_quality_gate_failed"

    return consensus_yaw, reliable, debug


def estimate_wall_normal_consensus(
    normal_xyz: np.ndarray,
    evidence_mask: np.ndarray,
    reference_normal: np.ndarray,
    g_hat_cam: np.ndarray,
    *,
    min_samples: int = 1000,
    max_p90_spread_deg: float = 4.0,
    max_vertical_tilt_deg: float = 5.0,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Validate wall verticality and material stability with direct normals.

    The direct-normal head is useful independent evidence for deciding whether
    the sampled region behaves like one stable vertical material surface.  Its
    horizontal heading is not authoritative for artwork perspective.  Signs
    are aligned to the depth plane only so angular spread can be measured
    consistently; callers must not replace depth-plane yaw with this result.
    """
    if (
        normal_xyz is None
        or evidence_mask is None
        or normal_xyz.ndim != 3
        or normal_xyz.shape[-1] != 3
        or evidence_mask.shape[:2] != normal_xyz.shape[:2]
    ):
        return None, {"reliable": False, "reason": "normal_evidence_shape_mismatch"}

    samples = np.asarray(normal_xyz[evidence_mask > 0], dtype=np.float32).reshape(-1, 3)
    finite = np.isfinite(samples).all(axis=1)
    lengths = np.linalg.norm(samples, axis=1)
    samples = samples[finite & (lengths > EPS)]
    if len(samples) < int(min_samples):
        return None, {
            "reliable": False,
            "reason": "insufficient_wall_normal_samples",
            "sample_count": int(len(samples)),
            "min_samples": int(min_samples),
        }
    samples = samples / np.maximum(np.linalg.norm(samples, axis=1, keepdims=True), EPS)
    reference = normalize(np.asarray(reference_normal, dtype=np.float32).reshape(3))
    flip = (samples @ reference) < 0.0
    samples = samples.copy()
    samples[flip] *= -1.0
    consensus = normalize(np.median(samples, axis=0).astype(np.float32))
    if float(np.dot(consensus, reference)) < 0.0:
        consensus = -consensus

    angular_error = np.degrees(np.arccos(np.clip(samples @ consensus, -1.0, 1.0)))
    p50 = float(np.percentile(angular_error, 50))
    p90 = float(np.percentile(angular_error, 90))
    p95 = float(np.percentile(angular_error, 95))
    g_hat = normalize(np.asarray(g_hat_cam, dtype=np.float32).reshape(3))
    vertical_tilt = float(np.degrees(np.arcsin(np.clip(abs(float(np.dot(consensus, g_hat))), 0.0, 1.0))))
    reliable = bool(
        len(samples) >= int(min_samples)
        and p90 <= float(max_p90_spread_deg)
        and vertical_tilt <= float(max_vertical_tilt_deg)
    )
    debug = {
        "reliable": reliable,
        "sample_count": int(len(samples)),
        "flipped_sample_count": int(np.count_nonzero(flip)),
        "consensus_normal": [float(x) for x in consensus],
        "angular_spread_p50_deg": p50,
        "angular_spread_p90_deg": p90,
        "angular_spread_p95_deg": p95,
        "vertical_tilt_deg": vertical_tilt,
        "min_samples": int(min_samples),
        "max_p90_spread_deg": float(max_p90_spread_deg),
        "max_vertical_tilt_deg": float(max_vertical_tilt_deg),
    }
    if not reliable:
        debug["reason"] = "wall_normal_consensus_quality_gate_failed"
    return consensus.astype(np.float32), debug


def _undirected_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=np.float64).reshape(-1)
    bv = np.asarray(b, dtype=np.float64).reshape(-1)
    an = float(np.linalg.norm(av))
    bn = float(np.linalg.norm(bv))
    if an < EPS or bn < EPS:
        return 180.0
    dot = abs(float(np.dot(av / an, bv / bn)))
    return float(np.degrees(np.arccos(np.clip(dot, 0.0, 1.0))))


def _camera_matrix(intr: Intrinsics) -> np.ndarray:
    return np.array(
        [[float(intr.fx), 0.0, float(intr.cx)], [0.0, float(intr.fy), float(intr.cy)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _local_image_direction(vanishing_h: np.ndarray, point_xy: np.ndarray) -> Optional[np.ndarray]:
    vp = np.asarray(vanishing_h, dtype=np.float64).reshape(3)
    point = np.asarray(point_xy, dtype=np.float64).reshape(2)
    if not np.isfinite(vp).all():
        return None
    if abs(float(vp[2])) > 1e-8:
        direction = vp[:2] / float(vp[2]) - point
    else:
        direction = vp[:2]
    length = float(np.linalg.norm(direction))
    if length < EPS:
        return None
    return direction / length


def estimate_wall_horizontal_structure_direction(
    image_rgb: Optional[np.ndarray],
    wall_mask: Optional[np.ndarray],
    intr: Intrinsics,
    expected_u_cam: np.ndarray,
    g_hat_cam: np.ndarray,
    *,
    prefilter_angle_deg: float = 24.0,
    min_unique_lines: int = 4,
    ransac_inlier_angle_deg: float = 2.0,
    max_p90_residual_deg: float = 2.5,
    max_horizontal_tilt_deg: float = 12.0,
    correction_min_inliers: int = 6,
    correction_max_p90_deg: float = 1.5,
    min_wall_interior_fraction: float = 0.82,
    min_inlier_wall_length_fraction: float = 0.85,
    min_inlier_wall_line_fraction: float = 0.75,
    min_post_projection_inlier_retention: float = 0.75,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Estimate a calibrated wall-horizontal vanishing direction.

    Long straight segments are collected strictly from the wall interior,
    where tile seams and panel joints provide repeated horizontal evidence.
    Every image line ``l`` is
    converted to its calibrated interpretation-plane normal ``K.T @ l``.  The
    common null-space direction is the 3-D vanishing direction, including the
    valid point-at-infinity case.  This avoids unstable finite vanishing-point
    division for nearly frontal walls.
    """
    if image_rgb is None or wall_mask is None:
        return None, {"reliable": False, "reason": "horizontal_structure_inputs_missing"}
    image = np.asarray(image_rgb)
    mask = np.asarray(wall_mask)
    if image.ndim not in (2, 3) or mask.ndim != 2 or image.shape[:2] != mask.shape[:2]:
        return None, {
            "reliable": False,
            "reason": "horizontal_structure_shape_mismatch",
            "image_shape": list(image.shape[:2]) if image.ndim >= 2 else None,
            "mask_shape": list(mask.shape[:2]) if mask.ndim >= 2 else None,
        }
    ys, xs = np.where(mask > 0)
    if len(xs) < 100:
        return None, {"reliable": False, "reason": "horizontal_structure_wall_mask_too_small", "mask_pixels": int(len(xs))}

    h, w = mask.shape
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    box_w = max(1, x1 - x0 + 1)
    box_h = max(1, y1 - y0 + 1)
    x_margin = max(24, int(round(0.45 * box_w)))
    top_expand = max(24, int(round(0.55 * box_h)))
    bottom_expand = max(8, int(round(0.08 * box_h)))
    context_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.rectangle(
        context_mask,
        (max(0, x0 - x_margin), max(0, y0 - top_expand)),
        (min(w - 1, x1 + x_margin), min(h - 1, y1 + bottom_expand)),
        255,
        -1,
    )
    wall_evidence_mask = (mask > 0).astype(np.uint8) * 255

    if image.ndim == 2:
        gray = np.clip(image, 0, 255).astype(np.uint8)
    else:
        rgb = np.clip(image[..., :3], 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    context_values = gray[context_mask > 0]
    median_value = float(np.median(context_values)) if len(context_values) else 96.0
    canny_low = int(np.clip(0.55 * median_value, 24, 110))
    canny_high = int(np.clip(1.45 * median_value, canny_low + 30, 235))
    edges = cv2.Canny(gray, canny_low, canny_high, L2gradient=True)
    # ROOT CAUSE / DO NOT REGRESS (2026-07-21): the former implementation
    # detected lines in a context rectangle extending 55% of the wall height
    # above the SAM wall.  In the failing room, 29/32 RANSAC inliers were
    # ceiling light-slot edges and 93.78% of their length was outside the wall.
    # Those lines produced a numerically coherent but visually false vanishing
    # direction.  Context pixels may still set Canny thresholds, but only the
    # authoritative wall mask may contribute Hough evidence.
    edges[wall_evidence_mask == 0] = 0
    min_line_length = max(24, int(round(0.16 * box_w)))
    hough_threshold = max(18, int(round(0.09 * box_w)))
    max_line_gap = max(8, int(round(0.06 * box_w)))
    detected = cv2.HoughLinesP(
        edges,
        1.0,
        np.pi / 1440.0,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if detected is None:
        return None, {
            "reliable": False,
            "reason": "no_horizontal_structure_lines_detected",
            "wall_bbox_xyxy_processed": [x0, y0, x1, y1],
            "canny_thresholds": [canny_low, canny_high],
        }

    K = _camera_matrix(intr)
    expected_u = normalize(np.asarray(expected_u_cam, dtype=np.float32).reshape(3)).astype(np.float64)
    g_hat = normalize(np.asarray(g_hat_cam, dtype=np.float32).reshape(3)).astype(np.float64)
    expected_vp = K @ expected_u
    gravity_vp = K @ g_hat

    candidates: List[Dict[str, Any]] = []
    for values in detected[:, 0, :]:
        x_start, y_start, x_end, y_end = [float(value) for value in values]
        delta = np.array([x_end - x_start, y_end - y_start], dtype=np.float64)
        length = float(np.linalg.norm(delta))
        if length < float(min_line_length):
            continue
        midpoint = np.array([(x_start + x_end) * 0.5, (y_start + y_end) * 0.5], dtype=np.float64)
        mx = int(np.clip(round(midpoint[0]), 0, w - 1))
        my = int(np.clip(round(midpoint[1]), 0, h - 1))
        if context_mask[my, mx] == 0:
            continue
        sample_count = max(5, int(round(length / 5.0)))
        t = np.linspace(0.0, 1.0, sample_count)
        sample_x = np.clip(np.rint(x_start + t * (x_end - x_start)).astype(np.int32), 0, w - 1)
        sample_y = np.clip(np.rint(y_start + t * (y_end - y_start)).astype(np.int32), 0, h - 1)
        context_coverage = float(np.mean(context_mask[sample_y, sample_x] > 0))
        if context_coverage < 0.88:
            continue
        expected_local = _local_image_direction(expected_vp, midpoint)
        gravity_local = _local_image_direction(gravity_vp, midpoint)
        if expected_local is None or gravity_local is None:
            continue
        expected_angle = _undirected_angle_deg(delta, expected_local)
        gravity_angle = _undirected_angle_deg(delta, gravity_local)
        if expected_angle > float(prefilter_angle_deg) or gravity_angle < 30.0:
            continue
        p1 = np.array([x_start, y_start, 1.0], dtype=np.float64)
        p2 = np.array([x_end, y_end, 1.0], dtype=np.float64)
        line = np.cross(p1, p2)
        line_norm = float(np.linalg.norm(line[:2]))
        if line_norm < EPS:
            continue
        line /= line_norm
        if line[2] < 0.0:
            line = -line
        interpretation_normal = K.T @ line
        interpretation_length = float(np.linalg.norm(interpretation_normal))
        if interpretation_length < EPS:
            continue
        interpretation_normal /= interpretation_length
        # Furniture/texture filter: lines in bottom 30% of wall are likely
        # furniture edges; very short lines are likely marble texture.
        wall_height_pixels = max(1, y1 - y0 + 1)
        midpoint_y_ratio = (midpoint[1] - y0) / wall_height_pixels
        is_bottom_region = midpoint_y_ratio > 0.70
        is_short_texture = length < 0.25 * box_w
        if is_bottom_region:
            continue  # skip furniture edges in lower wall
        if is_short_texture:
            continue  # skip marble texture false positives
        interior_fraction = float(np.mean(mask[sample_y, sample_x] > 0))
        if interior_fraction < float(min_wall_interior_fraction):
            continue
        candidates.append({
            "xyxy": [x_start, y_start, x_end, y_end],
            "midpoint": midpoint,
            "length_px": length,
            "line": line,
            "interpretation_normal": interpretation_normal,
            "expected_local_angle_deg": expected_angle,
            "gravity_local_angle_deg": gravity_angle,
            "wall_interior_fraction": interior_fraction,
            "wall_interior_length_px": float(length * interior_fraction),
            "is_bottom_region": is_bottom_region,
        })

    # Suppress duplicate Hough fragments from the same physical edge.  Keep
    # distinct parallel seams because their offset diversity constrains the
    # vanishing direction.
    unique: List[Dict[str, Any]] = []
    for item in sorted(candidates, key=lambda value: value["length_px"], reverse=True):
        duplicate = False
        for kept in unique:
            line_a = item["line"]
            line_b = kept["line"]
            normal_dot = float(np.dot(line_a[:2], line_b[:2]))
            aligned_c = float(line_a[2] if normal_dot >= 0.0 else -line_a[2])
            angle = _undirected_angle_deg(line_a[:2], line_b[:2])
            if angle <= 0.65 and abs(aligned_c - float(line_b[2])) <= 2.5:
                duplicate = True
                break
        if not duplicate:
            unique.append(item)
        if len(unique) >= 200:
            break

    if len(unique) < int(min_unique_lines):
        return None, {
            "reliable": False,
            "reason": "insufficient_unique_horizontal_structure_lines",
            "raw_line_count": int(len(detected)),
            "filtered_line_count": int(len(candidates)),
            "unique_line_count": int(len(unique)),
            "min_unique_lines": int(min_unique_lines),
            "wall_bbox_xyxy_processed": [x0, y0, x1, y1],
            "canny_thresholds": [canny_low, canny_high],
        }

    M = np.stack([item["interpretation_normal"] for item in unique], axis=0)
    lengths = np.asarray([item["length_px"] for item in unique], dtype=np.float64)
    weights = np.sqrt(np.maximum(lengths, 1.0))
    weights *= np.asarray([item["wall_interior_fraction"] for item in unique], dtype=np.float64)
    midpoints = np.stack([item["midpoint"] for item in unique], axis=0)
    rng = np.random.default_rng(20260721)
    pair_count = len(unique) * (len(unique) - 1) // 2
    if pair_count <= 3000:
        pairs = [(i, j) for i in range(len(unique)) for j in range(i + 1, len(unique))]
    else:
        pairs = [tuple(int(x) for x in rng.choice(len(unique), 2, replace=False)) for _ in range(3000)]

    best: Optional[Tuple[Tuple[float, int, float], np.ndarray, np.ndarray, np.ndarray]] = None
    residual_threshold = float(ransac_inlier_angle_deg)
    center_xy = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float64)
    center_direction = _local_image_direction(expected_vp, center_xy)
    if center_direction is None:
        center_direction = np.array([1.0, 0.0], dtype=np.float64)
    cross_axis = np.array([-center_direction[1], center_direction[0]], dtype=np.float64)
    cross_coordinates = midpoints @ cross_axis
    for i, j in pairs:
        direction = np.cross(M[i], M[j])
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm < 1e-8:
            continue
        direction /= direction_norm
        if float(np.dot(direction, expected_u)) < 0.0:
            direction = -direction
        residuals = np.degrees(np.arcsin(np.clip(np.abs(M @ direction), 0.0, 1.0)))
        inliers = residuals <= residual_threshold
        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count < int(min_unique_lines):
            continue
        offset_span = float(np.ptp(cross_coordinates[inliers])) if inlier_count >= 2 else 0.0
        score = float(np.sum(weights[inliers])) * (1.0 + 0.15 * min(1.0, offset_span / max(20.0, 0.20 * box_h)))
        key = (score, inlier_count, offset_span)
        if best is None or key > best[0]:
            best = (key, direction.copy(), residuals, inliers.copy())

    if best is None:
        return None, {
            "reliable": False,
            "reason": "horizontal_structure_vanishing_ransac_failed",
            "raw_line_count": int(len(detected)),
            "filtered_line_count": int(len(candidates)),
            "unique_line_count": int(len(unique)),
        }

    inliers = best[3]
    weighted_matrix = M[inliers] * np.sqrt(weights[inliers]).reshape(-1, 1)
    _, _, vh = np.linalg.svd(weighted_matrix, full_matrices=False)
    raw_direction = normalize(np.asarray(vh[-1], dtype=np.float32)).astype(np.float64)
    if float(np.dot(raw_direction, expected_u)) < 0.0:
        raw_direction = -raw_direction
    raw_residuals = np.degrees(np.arcsin(np.clip(np.abs(M @ raw_direction), 0.0, 1.0)))
    raw_inliers = raw_residuals <= residual_threshold
    raw_inlier_count = int(np.count_nonzero(raw_inliers))
    raw_inlier_residuals = raw_residuals[raw_inliers]
    raw_p50 = float(np.percentile(raw_inlier_residuals, 50)) if len(raw_inlier_residuals) else 180.0
    raw_p90 = float(np.percentile(raw_inlier_residuals, 90)) if len(raw_inlier_residuals) else 180.0
    horizontal_tilt = float(np.degrees(np.arcsin(np.clip(abs(float(np.dot(raw_direction, g_hat))), 0.0, 1.0))))
    horizontal_direction = raw_direction - float(np.dot(raw_direction, g_hat)) * g_hat
    if float(np.linalg.norm(horizontal_direction)) < EPS:
        return None, {
            "reliable": False,
            "reason": "horizontal_structure_direction_parallel_to_gravity",
            "horizontal_tilt_deg": horizontal_tilt,
        }
    horizontal_direction = normalize(horizontal_direction.astype(np.float32)).astype(np.float64)
    if float(np.dot(horizontal_direction, expected_u)) < 0.0:
        horizontal_direction = -horizontal_direction
    final_residuals = np.degrees(np.arcsin(np.clip(np.abs(M @ horizontal_direction), 0.0, 1.0)))
    # ROOT CAUSE / DO NOT REGRESS (2026-07-21): gravity projection changes the
    # candidate direction.  v14 computed this post-projection residual but then
    # decided `reliable` from the pre-projection inliers/p90.  The failed task
    # therefore passed with raw p90=1.68 deg even though the direction actually
    # used for projection had p90=6.77 deg.  Every quality statistic below must
    # be recomputed for the final, gravity-orthogonal direction.
    final_inliers = final_residuals <= residual_threshold
    projected_raw_inlier_p90 = float(np.percentile(final_residuals[raw_inliers], 90)) if raw_inlier_count else 180.0
    post_projection_inlier_retention = (
        float(np.count_nonzero(final_inliers & raw_inliers)) / max(1, raw_inlier_count)
    )
    inlier_count = int(np.count_nonzero(final_inliers))
    final_inlier_residuals = final_residuals[final_inliers]
    p50 = float(np.percentile(final_inlier_residuals, 50)) if len(final_inlier_residuals) else 180.0
    p90 = float(np.percentile(final_inlier_residuals, 90)) if len(final_inlier_residuals) else 180.0
    projected_p90 = p90
    offset_span = float(np.ptp(cross_coordinates[final_inliers])) if inlier_count >= 2 else 0.0
    total_inlier_length = float(np.sum(lengths[final_inliers]))
    inlier_wall_length = float(np.sum([
        item["wall_interior_length_px"] for index, item in enumerate(unique) if bool(final_inliers[index])
    ]))
    inlier_wall_length_fraction = inlier_wall_length / max(total_inlier_length, EPS)
    inlier_wall_line_fraction = float(np.mean([
        item["wall_interior_fraction"] >= float(min_wall_interior_fraction)
        for index, item in enumerate(unique) if bool(final_inliers[index])
    ])) if inlier_count else 0.0
    direction_angle = _undirected_angle_deg(horizontal_direction, expected_u)
    observed_vp = K @ horizontal_direction
    # VP containment guard: if the vanishing point projects inside the wall
    # body, the detected lines are likely marble texture or furniture edges,
    # not real structural seams.  This is a strong quality penalty.
    vp_inside_wall_body = False
    if abs(float(observed_vp[2])) > 1e-8:
        vp_px = observed_vp[:2] / float(observed_vp[2])
        if (x0 <= vp_px[0] <= x1 and y0 <= vp_px[1] <= y1):
            vp_inside_wall_body = True
    predicted_local = _local_image_direction(expected_vp, center_xy)
    observed_local = _local_image_direction(observed_vp, center_xy)
    local_image_angle = _undirected_angle_deg(predicted_local, observed_local) if predicted_local is not None and observed_local is not None else None
    min_offset_span = max(20.0, 0.08 * box_h)
    # VP containment penalty: when VP falls inside the wall body, require
    # stronger inlier evidence (≥6 lines) to trust the structure direction.
    vp_penalty = bool(vp_inside_wall_body and inlier_count < 6)
    reliable = bool(
        inlier_count >= int(min_unique_lines)
        and not vp_penalty
        and p90 <= float(max_p90_residual_deg)
        and projected_raw_inlier_p90 <= float(max_p90_residual_deg)
        and post_projection_inlier_retention >= float(min_post_projection_inlier_retention)
        and horizontal_tilt <= float(max_horizontal_tilt_deg)
        and offset_span >= min_offset_span
        and total_inlier_length >= 0.80 * box_w
        and inlier_wall_length_fraction >= float(min_inlier_wall_length_fraction)
        and inlier_wall_line_fraction >= float(min_inlier_wall_line_fraction)
    )
    correction_confident = bool(
        reliable
        and inlier_count >= int(correction_min_inliers)
        and p90 <= float(correction_max_p90_deg)
        and offset_span >= max(35.0, 0.12 * box_h)
    )
    line_debug = []
    for index, item in enumerate(unique[:64]):
        line_debug.append({
            "xyxy_processed": [float(v) for v in item["xyxy"]],
            "length_px": float(item["length_px"]),
            "wall_interior_fraction": float(item["wall_interior_fraction"]),
            "expected_local_angle_deg": float(item["expected_local_angle_deg"]),
            "residual_deg": float(raw_residuals[index]),
            "post_gravity_projection_residual_deg": float(final_residuals[index]),
            "raw_direction_inlier": bool(raw_inliers[index]),
            "inlier": bool(final_inliers[index]),
        })
    debug = {
        "reliable": reliable,
        "correction_confident": correction_confident,
        "method": "calibrated_image_line_nullspace_ransac_with_gravity_projection",
        "evidence_policy": "authoritative_wall_mask_interior_only_no_ceiling_or_adjacent_trim",
        "vanishing_point_inside_wall_body": vp_inside_wall_body,
        "vp_penalty_applied": vp_penalty,
        "raw_line_count": int(len(detected)),
        "filtered_line_count": int(len(candidates)),
        "unique_line_count": int(len(unique)),
        "inlier_count": inlier_count,
        "inlier_total_length_px": total_inlier_length,
        "inlier_wall_interior_length_px": inlier_wall_length,
        "inlier_wall_interior_length_fraction": inlier_wall_length_fraction,
        "inlier_wall_line_fraction": inlier_wall_line_fraction,
        "min_wall_interior_fraction": float(min_wall_interior_fraction),
        "min_inlier_wall_length_fraction": float(min_inlier_wall_length_fraction),
        "min_inlier_wall_line_fraction": float(min_inlier_wall_line_fraction),
        "line_offset_span_px": offset_span,
        "min_line_offset_span_px": min_offset_span,
        "raw_direction_residual_p50_deg": raw_p50,
        "raw_direction_residual_p90_deg": raw_p90,
        "residual_p50_deg": p50,
        "residual_p90_deg": p90,
        "post_gravity_projection_residual_p90_deg": projected_p90,
        "post_gravity_projection_raw_inlier_residual_p90_deg": projected_raw_inlier_p90,
        "post_gravity_projection_inlier_retention": post_projection_inlier_retention,
        "min_post_projection_inlier_retention": float(min_post_projection_inlier_retention),
        "max_p90_residual_deg": float(max_p90_residual_deg),
        "raw_direction_horizontal_tilt_deg": horizontal_tilt,
        "max_horizontal_tilt_deg": float(max_horizontal_tilt_deg),
        "direction_angle_to_depth_basis_u_deg": direction_angle,
        "local_image_angle_to_K_basis_u_deg": local_image_angle,
        "observed_horizontal_direction_cam": [float(x) for x in horizontal_direction],
        "observed_vanishing_homogeneous_px": [float(x) for x in observed_vp],
        "predicted_vanishing_homogeneous_px": [float(x) for x in expected_vp],
        "wall_bbox_xyxy_processed": [x0, y0, x1, y1],
        "context_bbox_xyxy_processed": [max(0, x0 - x_margin), max(0, y0 - top_expand), min(w - 1, x1 + x_margin), min(h - 1, y1 + bottom_expand)],
        "canny_thresholds": [canny_low, canny_high],
        "hough_threshold": int(hough_threshold),
        "min_line_length_px": int(min_line_length),
        "ransac_inlier_angle_deg": residual_threshold,
        "line_segments": line_debug,
    }
    if not reliable:
        debug["reason"] = "horizontal_structure_quality_gate_failed"
    return horizontal_direction.astype(np.float32), debug


def resolve_wall_projection_plane(
    depth_plane: WallPlane,
    anchor_points: np.ndarray,
    normal_xyz: np.ndarray,
    normal_evidence_mask: np.ndarray,
    g_hat_cam: np.ndarray,
    *,
    max_direct_depth_tilt_deg: float = 2.5,
    normal_recovery_angle_deg: float = 12.0,
    hard_conflict_angle_deg: float = 35.0,
    normal_min_samples: int = 1000,
    normal_max_p90_spread_deg: float = 4.0,
    normal_max_vertical_tilt_deg: float = 5.0,
    image_rgb: Optional[np.ndarray] = None,
    wall_mask: Optional[np.ndarray] = None,
    intr: Optional[Intrinsics] = None,
    direct_normal_fail_closed_on_quality_failure: bool = True,
    horizontal_structure_enabled: bool = True,
    horizontal_structure_angle_threshold_deg: float = 6.0,
    horizontal_structure_max_correction_deg: float = 16.0,
    horizontal_structure_prefilter_angle_deg: float = 24.0,
    horizontal_structure_min_unique_lines: int = 4,
    horizontal_structure_ransac_inlier_angle_deg: float = 2.0,
    horizontal_structure_max_p90_residual_deg: float = 2.5,
    horizontal_structure_max_horizontal_tilt_deg: float = 12.0,
    horizontal_structure_correction_min_inliers: int = 6,
    horizontal_structure_correction_max_p90_deg: float = 1.5,
    horizontal_structure_min_wall_interior_fraction: float = 0.82,
    horizontal_structure_min_inlier_wall_length_fraction: float = 0.85,
    horizontal_structure_min_inlier_wall_line_fraction: float = 0.75,
    horizontal_structure_min_post_projection_inlier_retention: float = 0.75,
    normal_conflict_structure_support_margin_deg: float = 3.0,
    controlled_structural_quorum_enabled: bool = False,
    controlled_structural_quorum_angle_threshold_deg: float = 6.0,
    segmented_depth_plane_yaw: Optional[float] = None,
    segmented_depth_plane_confident: bool = False,
) -> Tuple[WallPlane, Dict[str, Any]]:
    """Resolve the plane orientation used for artwork perspective.

    Depth remains authoritative for distance.  Its normal is projected onto the
    gravity-orthogonal subspace, removing impossible vertical tilt.  Direct
    normals independently validate verticality/material stability.  When their
    horizontal headings disagree beyond ``normal_recovery_angle_deg``, neither
    is allowed to win alone: wall-interior structure lines must clearly support
    one hypothesis or the wall fails closed.  A resolved heading always comes
    from the calibrated structure direction, never by copying a direct normal.
    """
    depth_n = normalize(np.asarray(depth_plane.n, dtype=np.float32).reshape(3))
    g_hat = normalize(np.asarray(g_hat_cam, dtype=np.float32).reshape(3))
    points = np.asarray(anchor_points, dtype=np.float32).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        return depth_plane, {"hard_reject": True, "reason": "no_projection_plane_anchor_points"}
    anchor = np.median(points, axis=0).astype(np.float32)
    if float(np.dot(depth_n, -anchor)) < 0.0:
        depth_n = -depth_n

    depth_vertical_tilt = float(np.degrees(np.arcsin(np.clip(abs(float(np.dot(depth_n, g_hat))), 0.0, 1.0))))
    depth_vertical_n = depth_n - float(np.dot(depth_n, g_hat)) * g_hat
    if float(np.linalg.norm(depth_vertical_n)) < EPS:
        return depth_plane, {
            "hard_reject": True,
            "reason": "depth_plane_parallel_to_gravity",
            "depth_plane_vertical_tilt_deg": depth_vertical_tilt,
        }
    depth_vertical_n = normalize(depth_vertical_n)
    if float(np.dot(depth_vertical_n, depth_n)) < 0.0:
        depth_vertical_n = -depth_vertical_n

    normal_consensus, normal_debug = estimate_wall_normal_consensus(
        normal_xyz,
        normal_evidence_mask,
        depth_n,
        g_hat,
        min_samples=int(normal_min_samples),
        max_p90_spread_deg=float(normal_max_p90_spread_deg),
        max_vertical_tilt_deg=float(normal_max_vertical_tilt_deg),
    )
    disagreement = None
    yaw_disagreement = None
    direct_vertical_n = None
    direct_u = None
    chosen_n = depth_vertical_n
    source = "depth_plane_gravity_verticalized"
    if normal_consensus is not None:
        disagreement = float(np.degrees(np.arccos(np.clip(abs(float(np.dot(depth_n, normal_consensus))), 0.0, 1.0))))
        if (
            bool(direct_normal_fail_closed_on_quality_failure)
            and not bool(normal_debug.get("reliable"))
            and int(normal_debug.get("sample_count", 0)) >= int(normal_min_samples)
        ):
            return depth_plane, {
                "hard_reject": True,
                "reason": "direct_normal_wall_validation_failed",
                "depth_plane_vertical_tilt_deg": depth_vertical_tilt,
                "depth_normal_disagreement_deg": disagreement,
                "direct_normal_used_for_yaw": False,
                "normal_consensus": normal_debug,
            }

        if bool(normal_debug.get("reliable")):
            direct_vertical_n = np.asarray(normal_consensus, dtype=np.float32).reshape(3)
            direct_vertical_n = direct_vertical_n - float(np.dot(direct_vertical_n, g_hat)) * g_hat
            if float(np.linalg.norm(direct_vertical_n)) >= EPS:
                direct_vertical_n = normalize(direct_vertical_n)
                if float(np.dot(direct_vertical_n, depth_vertical_n)) < 0.0:
                    direct_vertical_n = -direct_vertical_n
                yaw_disagreement = _undirected_angle_deg(depth_vertical_n, direct_vertical_n)
                direct_u = normalize(np.cross(g_hat, direct_vertical_n))

    expected_u = normalize(np.cross(g_hat, chosen_n))
    structure_direction = None
    structure_debug: Dict[str, Any] = {
        "reliable": False,
        "reason": "horizontal_structure_validation_disabled",
    }
    if bool(horizontal_structure_enabled) and image_rgb is not None and wall_mask is not None and intr is not None:
        structure_direction, structure_debug = estimate_wall_horizontal_structure_direction(
            image_rgb,
            wall_mask,
            intr,
            expected_u,
            g_hat,
            prefilter_angle_deg=float(horizontal_structure_prefilter_angle_deg),
            min_unique_lines=int(horizontal_structure_min_unique_lines),
            ransac_inlier_angle_deg=float(horizontal_structure_ransac_inlier_angle_deg),
            max_p90_residual_deg=float(horizontal_structure_max_p90_residual_deg),
            max_horizontal_tilt_deg=float(horizontal_structure_max_horizontal_tilt_deg),
            correction_min_inliers=int(horizontal_structure_correction_min_inliers),
            correction_max_p90_deg=float(horizontal_structure_correction_max_p90_deg),
            min_wall_interior_fraction=float(horizontal_structure_min_wall_interior_fraction),
            min_inlier_wall_length_fraction=float(horizontal_structure_min_inlier_wall_length_fraction),
            min_inlier_wall_line_fraction=float(horizontal_structure_min_inlier_wall_line_fraction),
            min_post_projection_inlier_retention=float(horizontal_structure_min_post_projection_inlier_retention),
        )
    structure_reliable = bool(structure_direction is not None and structure_debug.get("reliable"))
    normal_yaw_conflict = bool(
        yaw_disagreement is not None
        and yaw_disagreement > float(normal_recovery_angle_deg)
    )

    # ROOT CAUSE / DO NOT REGRESS (2026-07-21): v10-v14 treated
    # normal_recovery_angle_deg as a diagnostic-only value and preserved depth
    # yaw even when a coherent direct-normal field disagreed by 20.76 degrees.
    # A VLM later approved the visibly warped paste.  For a real sensor conflict
    # (>12 deg by default), wall-internal seams must form a three-way quorum and
    # clearly support exactly one heading.  Visual review is not an arbiter for
    # unresolved metric geometry.
    if normal_yaw_conflict:
        if not structure_reliable or direct_u is None:
            # Controlled structural quorum fallback: when the standard
            # three-evidence quorum cannot form, attempt recovery using
            # segmented depth planes, reliable direct normals, and
            # enhanced structure direction.
            if bool(controlled_structural_quorum_enabled):
                direct_normal_yaw_val = None
                if direct_u is not None and bool(normal_debug.get("reliable")):
                    direct_normal_yaw_val = float(np.arctan2(
                        float(direct_u[2]), float(direct_u[0])
                    ))
                structure_yaw_val = None
                if structure_direction is not None and structure_reliable:
                    structure_yaw_val = float(np.arctan2(
                        float(structure_direction[2]), float(structure_direction[0])
                    ))
                quorum_recovered, quorum_source, quorum_debug = evaluate_controlled_structural_quorum(
                    depth_yaw=segmented_depth_plane_yaw,
                    depth_yaw_confident=bool(segmented_depth_plane_confident),
                    direct_normal_yaw=direct_normal_yaw_val,
                    direct_normal_confident=bool(normal_debug.get("reliable")),
                    structure_yaw=structure_yaw_val,
                    structure_confident=structure_reliable,
                    angle_threshold_deg=float(controlled_structural_quorum_angle_threshold_deg),
                )
                if quorum_recovered:
                    consensus_yaw = float(np.radians(quorum_debug.get("consensus_yaw_deg", 0.0)))
                    recovered_u = np.array([
                        math.cos(consensus_yaw), 0.0, math.sin(consensus_yaw)
                    ], dtype=np.float64)
                    recovered_n = normalize(np.cross(recovered_u, g_hat))
                    if float(np.dot(recovered_n, depth_n)) < 0.0:
                        recovered_n = -recovered_n
                    corrected_d = -float(np.dot(recovered_n, anchor))
                    return WallPlane(tuple(map(float, recovered_n)), corrected_d), {
                        "hard_reject": False,
                        "policy": "controlled_structural_quorum_recovery",
                        "projection_orientation_source": quorum_source,
                        "depth_support_plane_preserved": True,
                        "depth_horizontal_heading_preserved": False,
                        "direct_normal_used_for_yaw": False,
                        "depth_plane_normal": [float(x) for x in depth_n],
                        "depth_plane_vertical_tilt_deg": depth_vertical_tilt,
                        "max_direct_depth_tilt_deg": float(max_direct_depth_tilt_deg),
                        "depth_plane_required_verticalization": bool(depth_vertical_tilt > float(max_direct_depth_tilt_deg)),
                        "depth_normal_disagreement_deg": disagreement,
                        "depth_direct_normal_yaw_disagreement_deg": yaw_disagreement,
                        "normal_yaw_conflict_requires_structure_quorum": True,
                        "normal_recovery_angle_deg": float(normal_recovery_angle_deg),
                        "hard_conflict_angle_deg": float(hard_conflict_angle_deg),
                        "resolved_projection_normal": [float(x) for x in recovered_n],
                        "resolved_projection_normal_dot_gravity_abs": abs(float(np.dot(recovered_n, g_hat))),
                        "metric_anchor_m": [float(x) for x in anchor],
                        "normal_consensus": normal_debug,
                        "horizontal_structure": structure_debug,
                        "controlled_structural_quorum": quorum_debug,
                    }
            return depth_plane, {
                "hard_reject": True,
                "reason": "depth_direct_normal_yaw_conflict_without_wall_internal_structure_quorum",
                "depth_plane_vertical_tilt_deg": depth_vertical_tilt,
                "depth_normal_disagreement_deg": disagreement,
                "depth_direct_normal_yaw_disagreement_deg": yaw_disagreement,
                "normal_recovery_angle_deg": float(normal_recovery_angle_deg),
                "direct_normal_used_for_yaw": False,
                "normal_consensus": normal_debug,
                "horizontal_structure": structure_debug,
            }
        structure_to_depth = _undirected_angle_deg(structure_direction, expected_u)
        structure_to_direct = _undirected_angle_deg(structure_direction, direct_u)
        support_threshold = float(horizontal_structure_angle_threshold_deg)
        support_margin = float(normal_conflict_structure_support_margin_deg)
        supports_depth = bool(
            structure_to_depth <= support_threshold
            and structure_to_depth + support_margin <= structure_to_direct
        )
        supports_direct = bool(
            structure_to_direct <= support_threshold
            and structure_to_direct + support_margin <= structure_to_depth
        )
        structure_debug.update({
            "normal_conflict_arbitration": True,
            "angle_to_depth_heading_deg": structure_to_depth,
            "angle_to_direct_normal_heading_deg": structure_to_direct,
            "support_threshold_deg": support_threshold,
            "support_margin_deg": support_margin,
            "supports_depth_heading": supports_depth,
            "supports_direct_normal_heading": supports_direct,
        })
        if supports_depth == supports_direct:
            structure_debug["action"] = "fail_closed_no_unique_three_evidence_quorum"
            return depth_plane, {
                "hard_reject": True,
                "reason": "depth_direct_normal_yaw_conflict_not_uniquely_resolved_by_wall_structure",
                "depth_plane_vertical_tilt_deg": depth_vertical_tilt,
                "depth_normal_disagreement_deg": disagreement,
                "depth_direct_normal_yaw_disagreement_deg": yaw_disagreement,
                "direct_normal_used_for_yaw": False,
                "normal_consensus": normal_debug,
                "horizontal_structure": structure_debug,
            }
        if supports_depth:
            chosen_n = depth_vertical_n
            source = "depth_yaw_validated_by_wall_internal_structure_quorum"
            structure_debug["action"] = "depth_heading_wins_three_evidence_quorum"
        else:
            correction_angle = _undirected_angle_deg(structure_direction, expected_u)
            if correction_angle > float(hard_conflict_angle_deg):
                structure_debug["action"] = "fail_closed_structure_correction_exceeds_hard_limit"
                return depth_plane, {
                    "hard_reject": True,
                    "reason": "wall_structure_heading_correction_exceeds_hard_conflict_limit",
                    "depth_plane_vertical_tilt_deg": depth_vertical_tilt,
                    "depth_normal_disagreement_deg": disagreement,
                    "depth_direct_normal_yaw_disagreement_deg": yaw_disagreement,
                    "direct_normal_used_for_yaw": False,
                    "normal_consensus": normal_debug,
                    "horizontal_structure": structure_debug,
                }
            corrected_n = normalize(np.cross(structure_direction, g_hat))
            if float(np.dot(corrected_n, depth_n)) < 0.0:
                corrected_n = -corrected_n
            chosen_n = corrected_n
            source = "wall_internal_structure_quorum_supports_direct_normal_heading"
            structure_debug["action"] = "structure_heading_wins_three_evidence_quorum"
    elif structure_reliable:
        structure_angle = _undirected_angle_deg(structure_direction, expected_u)
        structure_debug["angle_threshold_deg"] = float(horizontal_structure_angle_threshold_deg)
        structure_debug["max_correction_deg"] = float(horizontal_structure_max_correction_deg)
        if structure_angle > float(horizontal_structure_angle_threshold_deg):
            if bool(structure_debug.get("correction_confident")) and structure_angle <= float(horizontal_structure_max_correction_deg):
                corrected_n = normalize(np.cross(structure_direction, g_hat))
                if float(np.dot(corrected_n, depth_n)) < 0.0:
                    corrected_n = -corrected_n
                chosen_n = corrected_n
                source = "depth_yaw_corrected_by_horizontal_structure_vanishing_point"
                structure_debug["action"] = "corrected_horizontal_heading"
            else:
                structure_debug["action"] = "fail_closed"
                return depth_plane, {
                    "hard_reject": True,
                    "reason": "horizontal_structure_and_depth_yaw_conflict_unresolved",
                    "depth_plane_vertical_tilt_deg": depth_vertical_tilt,
                    "depth_normal_disagreement_deg": disagreement,
                    "direct_normal_used_for_yaw": False,
                    "normal_consensus": normal_debug,
                    "horizontal_structure": structure_debug,
                }
        else:
            structure_debug["action"] = "validated_depth_horizontal_heading"

    chosen_d = -float(np.dot(chosen_n, anchor))
    debug = {
        "hard_reject": False,
        "policy": "gravity_verticalized_three_evidence_wall_internal_structure_quorum",
        "projection_orientation_source": source,
        "depth_support_plane_preserved": True,
        "depth_horizontal_heading_preserved": source in {
            "depth_plane_gravity_verticalized",
            "depth_yaw_validated_by_wall_internal_structure_quorum",
        },
        "direct_normal_used_for_yaw": False,
        "depth_plane_normal": [float(x) for x in depth_n],
        "depth_plane_vertical_tilt_deg": depth_vertical_tilt,
        "max_direct_depth_tilt_deg": float(max_direct_depth_tilt_deg),
        "depth_plane_required_verticalization": bool(depth_vertical_tilt > float(max_direct_depth_tilt_deg)),
        "depth_normal_disagreement_deg": disagreement,
        "depth_direct_normal_yaw_disagreement_deg": yaw_disagreement,
        "normal_yaw_conflict_requires_structure_quorum": normal_yaw_conflict,
        "normal_recovery_angle_deg": float(normal_recovery_angle_deg),
        "hard_conflict_angle_deg": float(hard_conflict_angle_deg),
        "resolved_projection_normal": [float(x) for x in chosen_n],
        "resolved_projection_normal_dot_gravity_abs": abs(float(np.dot(chosen_n, g_hat))),
        "metric_anchor_m": [float(x) for x in anchor],
        "normal_consensus": normal_debug,
        "horizontal_structure": structure_debug,
    }
    return WallPlane(tuple(map(float, chosen_n)), chosen_d), debug


def evaluate_controlled_structural_quorum(
    *,
    depth_yaw: Optional[float],
    depth_yaw_confident: bool,
    direct_normal_yaw: Optional[float],
    direct_normal_confident: bool,
    structure_yaw: Optional[float],
    structure_confident: bool,
    angle_threshold_deg: float = 6.0,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Evaluate 2-of-3 controlled structural quorum for wall heading recovery.

    When the normal_yaw_conflict branch in resolve_wall_projection_plane()
    cannot form a three-evidence quorum, this function attempts recovery using
    segmented depth planes, reliable direct normals, and enhanced structure
    direction.

    Recovery conditions:
    - ≥2 sources agree within angle_threshold_deg
    - Third source (if available) is low-confidence, not high-confidence opposing
    - Direct normal alone cannot unlock recovery

    Returns (recovered, source_label, debug_dict).
    """
    debug: Dict[str, Any] = {
        "enabled": True,
        "angle_threshold_deg": float(angle_threshold_deg),
        "depth_yaw": round(float(depth_yaw), 4) if depth_yaw is not None else None,
        "depth_yaw_confident": depth_yaw_confident,
        "direct_normal_yaw": round(float(direct_normal_yaw), 4) if direct_normal_yaw is not None else None,
        "direct_normal_confident": direct_normal_confident,
        "structure_yaw": round(float(structure_yaw), 4) if structure_yaw is not None else None,
        "structure_confident": structure_confident,
    }

    sources: List[Tuple[str, float, bool]] = []
    if depth_yaw is not None:
        sources.append(("depth", float(depth_yaw), depth_yaw_confident))
    if direct_normal_yaw is not None:
        sources.append(("direct_normal", float(direct_normal_yaw), direct_normal_confident))
    if structure_yaw is not None:
        sources.append(("structure", float(structure_yaw), structure_confident))

    if len(sources) < 2:
        debug["recovered"] = False
        debug["reason"] = "insufficient_sources_for_quorum"
        return False, "insufficient_sources", debug

    # Check pairwise agreement
    threshold = float(np.radians(angle_threshold_deg))
    agreements: List[Tuple[str, str, float]] = []
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            name_i, yaw_i, conf_i = sources[i]
            name_j, yaw_j, conf_j = sources[j]
            diff = abs(yaw_i - yaw_j)
            diff = min(diff, 2.0 * math.pi - diff)
            if diff <= threshold:
                agreements.append((name_i, name_j, float(np.degrees(diff))))

    debug["pairwise_agreements"] = [
        {"sources": [a[0], a[1]], "angle_deg": round(a[2], 2)}
        for a in agreements
    ]

    if not agreements:
        debug["recovered"] = False
        debug["reason"] = "no_pair_within_threshold"
        return False, "no_agreement", debug

    # Find the best agreeing pair
    best_agreement = min(agreements, key=lambda x: x[2])
    agreeing_sources = {best_agreement[0], best_agreement[1]}
    agreeing_yaws = [
        yaw for name, yaw, _ in sources if name in agreeing_sources
    ]
    consensus_yaw = float(np.median(agreeing_yaws))

    # Check third source: must not be high-confidence opposing
    third_sources = [
        (name, yaw, conf) for name, yaw, conf in sources
        if name not in agreeing_sources
    ]
    for name, yaw, conf in third_sources:
        diff = abs(yaw - consensus_yaw)
        diff = min(diff, 2.0 * math.pi - diff)
        if conf and diff > threshold:
            # High-confidence opposing source → cannot recover
            debug["recovered"] = False
            debug["reason"] = f"third_source_{name}_confidently_opposes"
            debug["opposing_source"] = name
            debug["opposing_angle_deg"] = round(float(np.degrees(diff)), 2)
            return False, "opposed_by_confident_third", debug

    # Direct normal alone cannot unlock
    if len(agreeing_sources) == 1 and "direct_normal" in agreeing_sources:
        debug["recovered"] = False
        debug["reason"] = "direct_normal_alone_cannot_unlock"
        return False, "direct_normal_only", debug

    source_label = f"controlled_quorum_{'_'.join(sorted(agreeing_sources))}"
    debug["recovered"] = True
    debug["consensus_yaw_deg"] = round(float(np.degrees(consensus_yaw)), 2)
    debug["agreeing_sources"] = sorted(agreeing_sources)
    debug["source"] = source_label
    return True, source_label, debug


def estimate_g_hat_from_floor_normals(
    normal_xyz: np.ndarray,
    floor_mask: Optional[np.ndarray],
    *,
    normal_y_max: float = -0.75,
    abs_z_max: float = 0.45,
    min_pixels: int = 1000,
    angular_variance_soft_limit_rad: float = 0.08,
    full_confidence_pixels: int = 5000,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    if floor_mask is None or floor_mask.shape[:2] != normal_xyz.shape[:2]:
        return CAMERA_UP.copy(), 0.0, {"reason": "no_floor_mask"}
    n = normal_xyz[floor_mask > 0].reshape(-1, 3)
    if len(n) == 0:
        return CAMERA_UP.copy(), 0.0, {"reason": "empty_floor_mask"}
    n = n.astype(np.float32)
    finite = np.isfinite(n).all(axis=1)
    lengths = np.linalg.norm(n, axis=1)
    n = n[finite & (lengths > EPS)]
    if len(n) == 0:
        return CAMERA_UP.copy(), 0.0, {"reason": "no_finite_floor_normals"}
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), EPS)
    abs_y_min = abs(float(normal_y_max))
    sel = n[(np.abs(n[:, 1]) > abs_y_min) & (np.abs(n[:, 2]) < float(abs_z_max))]
    if len(sel) < int(min_pixels):
        return CAMERA_UP.copy(), 0.0, {
            "reason": "not_enough_floor_normal_pixels",
            "count": int(len(sel)),
            "min_pixels": int(min_pixels),
            "normal_y_max": float(normal_y_max),
            "abs_y_min": float(abs_y_min),
            "abs_z_max": float(abs_z_max),
        }
    flip = (sel @ CAMERA_UP) < 0.0
    sel = sel.copy()
    sel[flip] *= -1.0
    mean = orient_vector_to_camera_up(np.mean(sel, axis=0).astype(np.float32))
    # Angular variance around mean.
    dots = np.clip(sel @ mean, -1, 1)
    ang = np.arccos(dots)
    var = float(np.var(ang))
    conf = 1.0
    soft_limit = float(angular_variance_soft_limit_rad)
    if var > soft_limit:
        conf *= max(0.0, soft_limit / var)
    conf *= min(1.0, len(sel) / float(full_confidence_pixels))
    return mean.astype(np.float32), float(conf), {
        "count": int(len(sel)),
        "flipped_normal_count": int(np.count_nonzero(flip)),
        "direction_dot_camera_up": float(np.dot(mean, CAMERA_UP)),
        "angular_variance_rad": var,
        "normal_y_max": float(normal_y_max),
        "abs_y_min": float(abs_y_min),
        "abs_z_max": float(abs_z_max),
        "min_pixels": int(min_pixels),
        "angular_variance_soft_limit_rad": soft_limit,
        "full_confidence_pixels": int(full_confidence_pixels),
    }


def estimate_g_hat_from_horizontal_reference_normals(
    normal_xyz: np.ndarray,
    reference_mask: Optional[np.ndarray],
    *,
    reference_label: str,
    abs_y_min: float = 0.75,
    abs_z_max: float = 0.45,
    min_pixels: int = 1000,
    angular_variance_soft_limit_rad: float = 0.08,
    full_confidence_pixels: int = 5000,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Estimate camera-up from a horizontal semantic reference surface.

    Roof/ceiling and table normals may be emitted with either sign by a normal
    backend.  Select near-horizontal normals by magnitude, then orient every
    sample toward camera-up (negative camera-y in the current Metric3D/OpenCV
    convention) before averaging.
    """
    label = str(reference_label or "horizontal_reference").strip().lower()
    default_up = CAMERA_UP
    if reference_mask is None or reference_mask.shape[:2] != normal_xyz.shape[:2]:
        return default_up, 0.0, {"reason": f"no_{label}_mask", "reference_label": label}

    n = normal_xyz[reference_mask > 0].reshape(-1, 3).astype(np.float32)
    if len(n) == 0:
        return default_up, 0.0, {"reason": f"empty_{label}_mask", "reference_label": label}

    finite = np.isfinite(n).all(axis=1)
    norm = np.linalg.norm(n, axis=1)
    n = n[finite & (norm > EPS)]
    if len(n) == 0:
        return default_up, 0.0, {"reason": f"no_finite_{label}_normals", "reference_label": label}
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), EPS)

    sel = n[(np.abs(n[:, 1]) > float(abs_y_min)) & (np.abs(n[:, 2]) < float(abs_z_max))]
    if len(sel) < int(min_pixels):
        return default_up, 0.0, {
            "reason": f"not_enough_{label}_normal_pixels",
            "reference_label": label,
            "count": int(len(sel)),
            "min_pixels": int(min_pixels),
            "abs_y_min": float(abs_y_min),
            "abs_z_max": float(abs_z_max),
        }

    # Resolve the arbitrary normal sign before averaging.  This lets ceiling
    # normals and table-top normals act as the same camera-up reference.
    flip = (sel @ default_up) < 0.0
    sel = sel.copy()
    sel[flip] *= -1.0
    mean = orient_vector_to_camera_up(np.mean(sel, axis=0).astype(np.float32))
    dots = np.clip(sel @ mean, -1.0, 1.0)
    ang = np.arccos(dots)
    var = float(np.var(ang))
    soft_limit = float(angular_variance_soft_limit_rad)
    conf = 1.0
    if var > soft_limit:
        conf *= max(0.0, soft_limit / var)
    conf *= min(1.0, len(sel) / float(full_confidence_pixels))
    return mean.astype(np.float32), float(conf), {
        "reference_label": label,
        "count": int(len(sel)),
        "flipped_normal_count": int(np.count_nonzero(flip)),
        "angular_variance_rad": var,
        "abs_y_min": float(abs_y_min),
        "abs_z_max": float(abs_z_max),
        "min_pixels": int(min_pixels),
        "angular_variance_soft_limit_rad": soft_limit,
        "full_confidence_pixels": int(full_confidence_pixels),
    }


def estimate_g_hat_from_global_floor_like_normals(
    normal_xyz: np.ndarray,
    *,
    normal_y_max: float = -0.75,
    abs_z_max: float = 0.45,
    min_pixels: int = 1000,
    angular_variance_soft_limit_rad: float = 0.08,
    full_confidence_pixels: int = 5000,
    lower_image_start_ratio: float = 0.55,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Global Metric3D floor-like fallback used when semantic references fail."""
    default_up = CAMERA_UP
    if normal_xyz is None or normal_xyz.ndim != 3 or normal_xyz.shape[-1] != 3:
        return default_up, 0.0, {"reason": "normal_missing_or_invalid"}

    h, w, _ = normal_xyz.shape
    finite = np.isfinite(normal_xyz).all(axis=-1)
    lengths = np.linalg.norm(normal_xyz, axis=-1)
    normalized = np.zeros_like(normal_xyz, dtype=np.float32)
    valid = finite & (lengths > EPS)
    normalized[valid] = normal_xyz[valid] / lengths[valid].reshape(-1, 1)
    abs_y_min = abs(float(normal_y_max))
    floor_like = valid & (np.abs(normalized[..., 1]) > abs_y_min) & (np.abs(normalized[..., 2]) < float(abs_z_max))
    lower = np.zeros((h, w), dtype=bool)
    lower[int(round(h * float(lower_image_start_ratio))):, :] = True

    selected_region = "lower_image"
    cand = floor_like & lower
    if int(np.count_nonzero(cand)) < int(min_pixels):
        selected_region = "full_image"
        cand = floor_like
    count = int(np.count_nonzero(cand))
    if count < int(min_pixels):
        return default_up, 0.0, {
            "reason": "insufficient_global_floor_like_normals",
            "count": count,
            "min_pixels": int(min_pixels),
            "normal_y_max": float(normal_y_max),
            "abs_y_min": float(abs_y_min),
            "abs_z_max": float(abs_z_max),
            "lower_image_start_ratio": float(lower_image_start_ratio),
        }

    sel = normalized[cand].reshape(-1, 3).astype(np.float32)
    flip = (sel @ default_up) < 0.0
    sel = sel.copy()
    sel[flip] *= -1.0
    mean = orient_vector_to_camera_up(np.mean(sel, axis=0).astype(np.float32))
    dots = np.clip(sel @ mean, -1.0, 1.0)
    ang = np.arccos(dots)
    var = float(np.var(ang))
    soft_limit = float(angular_variance_soft_limit_rad)
    conf = 1.0
    if var > soft_limit:
        conf *= max(0.0, soft_limit / var)
    conf *= min(1.0, len(sel) / float(full_confidence_pixels))
    return mean.astype(np.float32), float(conf), {
        "count": int(len(sel)),
        "flipped_normal_count": int(np.count_nonzero(flip)),
        "direction_dot_camera_up": float(np.dot(mean, default_up)),
        "selected_region": selected_region,
        "angular_variance_rad": var,
        "normal_y_max": float(normal_y_max),
        "abs_y_min": float(abs_y_min),
        "abs_z_max": float(abs_z_max),
        "min_pixels": int(min_pixels),
        "angular_variance_soft_limit_rad": soft_limit,
        "full_confidence_pixels": int(full_confidence_pixels),
        "lower_image_start_ratio": float(lower_image_start_ratio),
    }

def make_wall_basis(plane: WallPlane, points: np.ndarray, g_hat_cam: np.ndarray) -> WallBasis:
    n = np.array(plane.n, np.float32)
    d = float(plane.d)
    # Project the robust point centroid onto the selected plane.  Using the raw
    # centroid can place the artwork basis several centimetres off-plane after
    # a constrained orientation recovery.
    centroid = points.mean(axis=0).astype(np.float32)
    n = normalize(n)
    origin = centroid - float(np.dot(n, centroid) + d) * n
    # Force normal toward camera origin: plane point c; if n points away from camera, flip.
    to_cam = -origin
    if np.dot(n, to_cam) < 0:
        n = -n
    # wall horizontal u = cross(g, n)
    u = normalize(np.cross(g_hat_cam, n))
    if np.linalg.norm(u) < EPS:
        # fallback horizontal axis
        u = normalize(np.array([1, 0, 0], np.float32))
    v = normalize(np.cross(n, u))
    if np.dot(v, g_hat_cam) < 0:
        v = -v
        u = -u
    return WallBasis(
        origin=tuple(map(float, origin)),
        u=tuple(map(float, u)),
        v=tuple(map(float, v)),
        n=tuple(map(float, n)),
        g_hat_cam=tuple(map(float, g_hat_cam)),
    )

def project_points(points: np.ndarray, intr: Intrinsics) -> np.ndarray:
    z = np.maximum(points[:, 2], 1e-6)
    u = intr.fx * points[:, 0] / z + intr.cx
    v = intr.fy * points[:, 1] / z + intr.cy
    return np.stack([u, v], axis=1)

def project_quad(origin, u, v, intr: Intrinsics, axis_s_m: float, center_t_m: float, width_m: float, height_m: float) -> np.ndarray:
    origin = np.array(origin, np.float32)
    u = np.array(u, np.float32)
    v = np.array(v, np.float32)
    C = origin + u * axis_s_m + v * center_t_m
    pts = np.stack([
        C - u * width_m / 2 + v * height_m / 2,
        C + u * width_m / 2 + v * height_m / 2,
        C + u * width_m / 2 - v * height_m / 2,
        C - u * width_m / 2 - v * height_m / 2,
    ], axis=0)
    return project_points(pts, intr)

def polygon_mask(size_hw: Tuple[int, int], quad: np.ndarray) -> np.ndarray:
    h, w = size_hw
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.round(quad).astype(np.int32)], 255)
    return mask

def ray_plane_depth_for_mask(mask: np.ndarray, intr: Intrinsics, plane: WallPlane) -> np.ndarray:
    h, w = mask.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    n = np.array(plane.n, np.float32)
    d = float(plane.d)
    # ray = [(u-cx)/fx, (v-cy)/fy, 1]
    rx = (xx.astype(np.float32) - intr.cx) / intr.fx
    ry = (yy.astype(np.float32) - intr.cy) / intr.fy
    rz = np.ones_like(rx)
    denom = n[0] * rx + n[1] * ry + n[2] * rz
    t = -d / np.where(np.abs(denom) < 1e-6, np.nan, denom)
    depth = t * rz
    return depth.astype(np.float32)

def scale_points_to_original(points: np.ndarray, scale_xy: Tuple[float, float]) -> list[list[float]]:
    sx, sy = scale_xy
    return [[float(x * sx), float(y * sy)] for x, y in points]
