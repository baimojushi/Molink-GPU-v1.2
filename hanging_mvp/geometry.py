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
    # Force normal toward camera origin: plane point c; if n points away from camera, flip.
    origin = points.mean(axis=0).astype(np.float32)
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
