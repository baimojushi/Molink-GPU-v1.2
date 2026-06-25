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


def _masked_global_ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
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
) -> dict[str, Any]:
    """QA FLUX output using the deterministic art mask as the only geometry anchor.

    No SAM/VLM/bbox re-detection is used for the pass/fail decision.
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
    core_ssim = _masked_global_ssim(target, render, core)
    core_delta = float(np.mean(delta[core])) if np.any(core) else 0.0

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
    if core_ssim < float(min_core_masked_ssim):
        failures.append("core_masked_ssim_below_threshold")
    if core_delta > float(max_core_color_delta):
        failures.append("core_color_delta_above_threshold")
    if guard_ratio > float(max_forbidden_outer_guard_changed_ratio):
        failures.append("forbidden_outer_guard_changed_ratio_above_threshold")

    return {
        "status": "passed" if not failures else "failed",
        "passed": not failures,
        "method": "mask_anchored_core_ssim_and_forbidden_outer_guard",
        "geometry_anchor": "art_region_mask_from_hardpaste",
        "bbox_redetection_used_for_decision": False,
        "core_masked_ssim": core_ssim,
        "core_color_delta": core_delta,
        "forbidden_outer_guard_changed_ratio": guard_ratio,
        "raw_forbidden_outer_guard_changed_ratio": raw_guard_ratio,
        "ignored_shadow_like_guard_ratio": shadow_like_ratio,
        "ignored_shadow_like_guard_pixel_count": ignored_shadow_count,
        "changed_pixel_delta": float(changed_pixel_delta),
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
