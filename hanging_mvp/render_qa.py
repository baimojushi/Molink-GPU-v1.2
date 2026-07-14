from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def _load_mask(path: Path, shape_hw: tuple[int, int]) -> np.ndarray:
    raw = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    if raw.shape[:2] != shape_hw:
        raw = cv2.resize(raw, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return raw > 0


def _kernel(radius_px: int) -> np.ndarray:
    r = max(0, int(radius_px))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def _rgb_luma(arr: np.ndarray) -> np.ndarray:
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return np.tensordot(arr.astype(np.float32), weights, axes=([-1], [0]))


def _chroma_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Compare color residual after removing neutral luma.  Neutral shadow should
    # have small chroma delta even when it is visibly darker.
    la = _rgb_luma(a)[..., None]
    lb = _rgb_luma(b)[..., None]
    ca = a.astype(np.float32) - la
    cb = b.astype(np.float32) - lb
    return np.mean(np.abs(cb - ca), axis=2)


def masked_global_ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Small dependency-free masked SSIM-style score over RGB pixels."""
    idx = mask > 0
    if int(np.count_nonzero(idx)) < 16:
        return 0.0
    x = a[idx].astype(np.float64)
    y = b[idx].astype(np.float64)
    scores = []
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    for ch in range(3):
        xc = x[:, ch]
        yc = y[:, ch]
        mux = float(np.mean(xc))
        muy = float(np.mean(yc))
        varx = float(np.var(xc))
        vary = float(np.var(yc))
        cov = float(np.mean((xc - mux) * (yc - muy)))
        numerator = (2.0 * mux * muy + c1) * (2.0 * cov + c2)
        denominator = (mux * mux + muy * muy + c1) * (varx + vary + c2)
        scores.append(float(numerator / denominator) if denominator > 0 else 0.0)
    return float(max(-1.0, min(1.0, np.mean(scores))))


def _masked_highfreq_ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray, *, blur_px: int = 5) -> float:
    """Structure score after removing low-frequency relight.

    This lets QA accept explainable low-frequency exposure changes while still
    rejecting actual artwork rewrites, crop drift, or high-frequency repainting.
    """
    idx = mask > 0
    if int(np.count_nonzero(idx)) < 16:
        return 0.0
    la = _rgb_luma(a).astype(np.float32)
    lb = _rgb_luma(b).astype(np.float32)
    radius = max(1, int(blur_px))
    ha = la - cv2.GaussianBlur(la, (0, 0), sigmaX=float(radius), sigmaY=float(radius))
    hb = lb - cv2.GaussianBlur(lb, (0, 0), sigmaX=float(radius), sigmaY=float(radius))
    x = ha[idx].astype(np.float64)
    y = hb[idx].astype(np.float64)
    mux = float(np.mean(x))
    muy = float(np.mean(y))
    varx = float(np.var(x))
    vary = float(np.var(y))
    cov = float(np.mean((x - mux) * (y - muy)))
    c1 = 1e-4
    c2 = (0.03 * 255.0) ** 2
    denominator = (mux * mux + muy * muy + c1) * (varx + vary + c2)
    score = ((2.0 * mux * muy + c1) * (2.0 * cov + c2) / denominator) if denominator > 0 else 0.0
    return float(max(-1.0, min(1.0, score)))


def _safe_median(values: np.ndarray, default: float = 0.0) -> float:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return float(default)
    return float(np.median(arr))


def _adaptive_relight_metrics(
    *,
    target: np.ndarray,
    render: np.ndarray,
    art: np.ndarray,
    core: np.ndarray,
    ambient_inner_px: int,
    ambient_outer_px: int,
    self_glow_margin: float,
    max_core_chroma_delta: float,
    min_identity_hf_ssim: float,
) -> dict[str, Any]:
    target_luma = _rgb_luma(target)
    render_luma = _rgb_luma(render)
    luma_diff = render_luma - target_luma
    art_u8 = (art > 0).astype(np.uint8)
    inner = cv2.dilate(art_u8, _kernel(max(0, int(ambient_inner_px)))) > 0
    outer = cv2.dilate(art_u8, _kernel(max(1, int(ambient_outer_px)))) > 0
    ambient_ring = outer & (~inner)
    if int(np.count_nonzero(ambient_ring)) < 32:
        ambient_ring = outer & (~(art_u8 > 0))
    if int(np.count_nonzero(ambient_ring)) < 32:
        ambient_ring = np.zeros_like(core, dtype=bool)

    core_luma_delta = _safe_median(luma_diff[core]) if np.any(core) else 0.0
    ambient_luma_delta = _safe_median(luma_diff[ambient_ring]) if np.any(ambient_ring) else 0.0

    core_target_luma = target_luma[core] if np.any(core) else np.asarray([], dtype=np.float32)
    if core_target_luma.size:
        p05 = float(np.percentile(core_target_luma, 5))
        p50 = float(np.percentile(core_target_luma, 50))
        p95 = float(np.percentile(core_target_luma, 95))
    else:
        p05 = p50 = p95 = 0.0
    dynamic_range = max(0.0, p95 - p05)
    core_rgb = target[core] if np.any(core) else np.zeros((0, 3), dtype=np.float32)
    if core_rgb.size:
        sat_proxy = np.max(core_rgb, axis=1) - np.min(core_rgb, axis=1)
        luma_core_flat = core_target_luma.reshape(-1)
        paper_like = (sat_proxy <= 18.0) & (luma_core_flat >= max(45.0, p50 - 10.0))
        paper_like_ratio = float(np.mean(paper_like)) if paper_like.size else 0.0
        paper_luma_median = _safe_median(luma_core_flat[paper_like], p50) if np.any(paper_like) else p50
    else:
        paper_like_ratio = 0.0
        paper_luma_median = 0.0

    under = 0.0
    if p95 < 165.0:
        under += 0.35 * (1.0 - max(0.0, p95 - 110.0) / 55.0)
    if paper_like_ratio > 0.12 and paper_luma_median < 180.0:
        under += 0.40 * min(1.0, (180.0 - paper_luma_median) / 70.0)
    if dynamic_range < 75.0 and p95 < 190.0:
        under += 0.25 * (1.0 - dynamic_range / 75.0)
    underexposed_score = _clamp01(under)

    scene_light_score = _clamp01(max(ambient_luma_delta, 0.0) / 14.0)
    positive_core = max(core_luma_delta, 0.0)
    positive_env = max(ambient_luma_delta, 0.0)
    if positive_core <= 1.0:
        delta_consistency = 1.0
    else:
        delta_consistency = _clamp01(1.0 - abs(positive_core - positive_env) / 32.0)

    edge_ring = (cv2.dilate(art_u8, _kernel(max(1, int(ambient_inner_px)))) > 0) & (~(art_u8 > 0))
    halo_threshold = max(8.0, positive_core + 6.0)
    halo_ratio = float(np.mean(luma_diff[edge_ring] > halo_threshold)) if np.any(edge_ring) else 0.0
    render_p95 = float(np.percentile(render_luma[core], 95)) if np.any(core) else 0.0
    bloom_risk = bool(render_p95 > 245.0 and underexposed_score < 0.55 and positive_core > 8.0)
    core_chroma_delta = float(np.mean(_chroma_delta(target, render)[core])) if np.any(core) else 0.0
    highfreq_ssim = _masked_highfreq_ssim(target, render, core)

    render_coherence_score = _clamp01(
        0.35 * scene_light_score
        + 0.25 * delta_consistency
        + 0.30 * max(0.0, highfreq_ssim)
        + 0.10 * (1.0 - min(1.0, halo_ratio * 4.0))
        - (0.30 if bloom_risk else 0.0)
    )
    allowed_brightening = float(
        6.0
        + 0.8 * positive_env
        + 16.0 * underexposed_score
        + 8.0 * render_coherence_score
    )
    unexplained_brightening = max(0.0, positive_core - allowed_brightening)
    self_glow_risk = bool(unexplained_brightening > float(self_glow_margin))
    # A core relight is explainable when the brightness budget covers it and
    # the chroma / high-frequency structure still looks like the hard-paste.
    darken_explained = core_luma_delta <= 0.0 and abs(core_luma_delta) <= 40.0
    brighten_explained = positive_core <= allowed_brightening + float(self_glow_margin)
    core_luma_change_explained = bool(
        (darken_explained or brighten_explained)
        and not self_glow_risk
        and core_chroma_delta <= float(max_core_chroma_delta)
        and highfreq_ssim >= float(min_identity_hf_ssim)
    )
    return {
        "enabled": True,
        "method": "adaptive_explainable_relight_budget",
        "core_luma_delta": float(core_luma_delta),
        "ambient_luma_delta": float(ambient_luma_delta),
        "underexposed_score": float(underexposed_score),
        "scene_light_score": float(scene_light_score),
        "render_coherence_score": float(render_coherence_score),
        "allowed_brightening": float(allowed_brightening),
        "unexplained_brightening": float(unexplained_brightening),
        "self_glow_risk": bool(self_glow_risk),
        "core_luma_change_explained": bool(core_luma_change_explained),
        "core_chroma_delta": float(core_chroma_delta),
        "core_highfreq_ssim": float(highfreq_ssim),
        "halo_ratio": float(halo_ratio),
        "bloom_risk": bool(bloom_risk),
        "target_core_luma": {"p05": p05, "p50": p50, "p95": p95, "dynamic_range": dynamic_range},
        "paper_like": {"area_ratio": paper_like_ratio, "median_luma": float(paper_luma_median)},
        "thresholds": {
            "self_glow_margin": float(self_glow_margin),
            "max_core_chroma_delta": float(max_core_chroma_delta),
            "min_identity_hf_ssim": float(min_identity_hf_ssim),
            "ambient_inner_px": int(ambient_inner_px),
            "ambient_outer_px": int(ambient_outer_px),
        },
    }


def evaluate_mask_anchored_render_qa(
    *,
    hard_paste_path: Path,
    render_path: Path,
    art_region_mask_path: Path,
    core_erode_px: int = 2,
    allowed_frame_shadow_px: int = 6,
    forbidden_outer_guard_px: int = 20,
    changed_pixel_delta: float = 18.0,
    min_core_masked_ssim: float = 0.96,
    max_core_color_delta: float = 10.0,
    max_forbidden_outer_guard_changed_ratio: float = 0.12,
    ignore_shadow_in_forbidden_guard: bool = True,
    shadow_min_darkening_delta: float = 4.0,
    shadow_max_brighten_delta: float = 3.0,
    shadow_max_chroma_delta: float = 6.0,
    adaptive_relight_qa_enabled: bool = True,
    adaptive_ambient_inner_px: int = 6,
    adaptive_ambient_outer_px: int = 32,
    adaptive_self_glow_margin: float = 8.0,
    adaptive_max_core_chroma_delta: float = 16.0,
    adaptive_min_identity_hf_ssim: float = 0.55,
) -> dict[str, Any]:
    """QA render output using deterministic masks plus explainable relight.

    Geometry remains anchored to the hard-paste art mask.  Low-frequency luma
    changes are no longer judged merely by similarity to hard-paste; they pass
    when underexposure, nearby environment light, and render coherence can
    explain the change while artwork chroma/structure stays stable.
    """
    target = _load_rgb(Path(hard_paste_path))
    render = _load_rgb(Path(render_path))
    if target.shape != render.shape:
        return {
            "status": "failed",
            "passed": False,
            "reason": "render_size_mismatch",
            "target_shape": list(target.shape),
            "render_shape": list(render.shape),
        }
    h, w = target.shape[:2]
    art = _load_mask(Path(art_region_mask_path), (h, w)).astype(np.uint8)
    if int(np.count_nonzero(art)) <= 0:
        return {"status": "skipped_empty_art_region_mask", "passed": True, "reason": "empty_art_region_mask"}

    core = cv2.erode(art, _kernel(core_erode_px)) > 0 if core_erode_px > 0 else art > 0
    if int(np.count_nonzero(core)) < 16:
        core = art > 0
    allowed = cv2.dilate(art, _kernel(allowed_frame_shadow_px)) > 0
    guard_outer = cv2.dilate(art, _kernel(forbidden_outer_guard_px)) > 0
    forbidden_guard = guard_outer & (~allowed)

    delta = np.mean(np.abs(render - target), axis=2)
    core_ssim = masked_global_ssim(target, render, core)
    core_delta = float(np.mean(delta[core])) if np.any(core) else 0.0
    adaptive = _adaptive_relight_metrics(
        target=target,
        render=render,
        art=art,
        core=core,
        ambient_inner_px=int(adaptive_ambient_inner_px),
        ambient_outer_px=int(adaptive_ambient_outer_px),
        self_glow_margin=float(adaptive_self_glow_margin),
        max_core_chroma_delta=float(adaptive_max_core_chroma_delta),
        min_identity_hf_ssim=float(adaptive_min_identity_hf_ssim),
    ) if bool(adaptive_relight_qa_enabled) else {"enabled": False, "reason": "disabled"}
    adaptive_explained = bool(adaptive.get("core_luma_change_explained"))

    changed_forbidden = (delta > float(changed_pixel_delta)) & forbidden_guard
    raw_guard_ratio = float(np.mean(delta[forbidden_guard] > float(changed_pixel_delta))) if np.any(forbidden_guard) else 0.0
    ignored_shadow_count = 0
    shadow_like_ratio = 0.0
    if bool(ignore_shadow_in_forbidden_guard) and np.any(forbidden_guard):
        target_luma = _rgb_luma(target)
        render_luma = _rgb_luma(render)
        luma_diff = render_luma - target_luma
        max_channel_brighten = np.max(render - target, axis=2)
        chroma_diff = _chroma_delta(target, render)
        shadow_like = (
            forbidden_guard
            & changed_forbidden
            & (luma_diff <= -float(shadow_min_darkening_delta))
            & (max_channel_brighten <= float(shadow_max_brighten_delta))
            & (chroma_diff <= float(shadow_max_chroma_delta))
        )
        ignored_shadow_count = int(np.count_nonzero(shadow_like))
        shadow_like_ratio = float(np.mean(shadow_like[forbidden_guard])) if np.any(forbidden_guard) else 0.0
        changed_forbidden = changed_forbidden & (~shadow_like)

    guard_ratio = float(np.mean(changed_forbidden[forbidden_guard])) if np.any(forbidden_guard) else 0.0

    failures: list[str] = []
    if core_ssim < float(min_core_masked_ssim) and not adaptive_explained:
        failures.append("core_masked_ssim_below_threshold")
    if core_delta > float(max_core_color_delta) and not adaptive_explained:
        failures.append("core_color_delta_above_threshold")
    if bool(adaptive.get("self_glow_risk")):
        failures.append("adaptive_self_glow_unexplained_brightening")
    if guard_ratio > float(max_forbidden_outer_guard_changed_ratio):
        failures.append("forbidden_outer_guard_changed_ratio_above_threshold")

    return {
        "status": "passed" if not failures else "failed",
        "passed": not failures,
        "method": "mask_anchored_adaptive_relight_and_forbidden_outer_guard",
        "geometry_anchor": "art_region_mask_from_hardpaste",
        "bbox_redetection_used_for_decision": False,
        "core_masked_ssim": core_ssim,
        "core_color_delta": core_delta,
        "forbidden_outer_guard_changed_ratio": guard_ratio,
        "raw_forbidden_outer_guard_changed_ratio": raw_guard_ratio,
        "ignored_shadow_like_guard_ratio": shadow_like_ratio,
        "ignored_shadow_like_guard_pixel_count": ignored_shadow_count,
        "changed_pixel_delta": float(changed_pixel_delta),
        "adaptive_relight": adaptive,
        "thresholds": {
            "min_core_masked_ssim": float(min_core_masked_ssim),
            "max_core_color_delta": float(max_core_color_delta),
            "max_forbidden_outer_guard_changed_ratio": float(max_forbidden_outer_guard_changed_ratio),
        },
        "shadow_aware_guard": {
            "enabled": bool(ignore_shadow_in_forbidden_guard),
            "shadow_min_darkening_delta": float(shadow_min_darkening_delta),
            "shadow_max_brighten_delta": float(shadow_max_brighten_delta),
            "shadow_max_chroma_delta": float(shadow_max_chroma_delta),
            "policy": "neutral_darken_only_projection_pixels_do_not_fail_outer_guard",
        },
        "bands_px": {
            "core_erode_px": int(core_erode_px),
            "allowed_frame_shadow_px": int(allowed_frame_shadow_px),
            "forbidden_outer_guard_px": int(forbidden_outer_guard_px),
        },
        "failures": failures,
    }
