from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .schemas import Intrinsics, WallPlane, WallBasis

EPS = 1e-8

def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < EPS:
        return v
    return v / n

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
        return np.array([0, -1, 0], np.float32), 0.0, {"reason": "no_floor_mask"}
    n = normal_xyz[floor_mask > 0].reshape(-1, 3)
    if len(n) == 0:
        return np.array([0, -1, 0], np.float32), 0.0, {"reason": "empty_floor_mask"}
    # Metric3D/OpenCV convention in current dataset: floor-like normals often
    # have negative camera-y and limited camera-z. Keep thresholds configurable
    # because different depth backends may expose a different normal convention.
    sel = n[(n[:, 1] < float(normal_y_max)) & (np.abs(n[:, 2]) < float(abs_z_max))]
    if len(sel) < int(min_pixels):
        return np.array([0, -1, 0], np.float32), 0.0, {
            "reason": "not_enough_floor_normal_pixels",
            "count": int(len(sel)),
            "min_pixels": int(min_pixels),
            "normal_y_max": float(normal_y_max),
            "abs_z_max": float(abs_z_max),
        }
    mean = normalize(np.mean(sel, axis=0).astype(np.float32))
    # Angular variance around mean.
    dots = np.clip(sel @ mean, -1, 1)
    ang = np.arccos(dots)
    var = float(np.var(ang))
    conf = 1.0
    soft_limit = float(angular_variance_soft_limit_rad)
    if var > soft_limit:
        conf *= max(0.0, soft_limit / var)
    conf *= min(1.0, len(sel) / float(full_confidence_pixels))
    # Use +up direction. If floor normals point upward in camera y negative, keep it as g_hat "up".
    g_hat = mean
    return g_hat.astype(np.float32), float(conf), {
        "count": int(len(sel)),
        "angular_variance_rad": var,
        "normal_y_max": float(normal_y_max),
        "abs_z_max": float(abs_z_max),
        "min_pixels": int(min_pixels),
        "angular_variance_soft_limit_rad": soft_limit,
        "full_confidence_pixels": int(full_confidence_pixels),
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
