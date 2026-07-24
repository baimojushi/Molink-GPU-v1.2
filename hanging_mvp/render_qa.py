from __future__ import annotations

import os
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


def enforce_protected_artwork_composite(
    *,
    authoritative_image_path: Path,
    provider_render_path: Path,
    protect_mask_path: Path,
    output_path: Path,
    feather_inside_px: float = 0.0,
) -> dict[str, Any]:
    """Keep the provider's full-frame generation while protecting the artwork.

    This is deliberately the inverse of the legacy identity-locked shadow
    compositor.  The provider render is the authoritative full-frame result;
    only the supplied artwork protection region is restored from the Stage A
    image.  No crop is made, no outside-edit pixels are rolled back, and no
    provider contribution is reduced to a synthetic shadow.

    ``feather_inside_px`` creates an *inside-only* transition.  Pixels outside
    the protection mask always remain provider pixels, so there is no pasted
    crop rectangle or halo.  With a value of zero the protected region is
    restored exactly, which is appropriate for Stage B styling.  A small value
    (normally 1-2 px) is useful for Stage A render recovery because it keeps the
    artwork interior fixed while letting the generated edge lighting meet the
    generated wall naturally.
    """
    with Image.open(authoritative_image_path) as source_image:
        source = np.asarray(source_image.convert("RGB"), dtype=np.uint8)
    with Image.open(provider_render_path) as render_image:
        render = np.asarray(render_image.convert("RGB"), dtype=np.uint8)

    height, width = source.shape[:2]
    if render.shape[:2] != (height, width):
        render = cv2.resize(render, (width, height), interpolation=cv2.INTER_LANCZOS4)
    protect = _load_mask(Path(protect_mask_path), (height, width))
    if not np.any(protect):
        raise ValueError("protected-artwork composite requires a non-empty protection mask")

    feather = max(0.0, float(feather_inside_px))
    if feather <= 0.0:
        alpha = protect.astype(np.float32)
    else:
        # OpenCV's distance transform is positive only inside the mask.  The
        # first inside pixel receives a partial source weight and the weight
        # reaches 1.0 deeper inside, avoiding a hard generated/original seam.
        distance = cv2.distanceTransform(protect.astype(np.uint8), cv2.DIST_L2, 5)
        alpha = np.clip(distance / max(feather, 1e-6), 0.0, 1.0).astype(np.float32)
        alpha[~protect] = 0.0

    alpha_rgb = alpha[..., None]
    composed = np.clip(
        render.astype(np.float32) * (1.0 - alpha_rgb)
        + source.astype(np.float32) * alpha_rgb,
        0.0,
        255.0,
    ).astype(np.uint8)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_name(f".{output.name}.tmp")
    try:
        with tmp_output.open("wb") as handle:
            Image.fromarray(composed, mode="RGB").save(handle, format="PNG")
            handle.flush()
            os.fsync(handle.fileno())
        tmp_output.replace(output)
    except Exception:
        tmp_output.unlink(missing_ok=True)
        raise

    exact_lock = protect & (alpha >= 0.999)
    transition = protect & (alpha > 0.001) & (alpha < 0.999)
    outside = ~protect
    provider_delta = np.mean(
        np.abs(render.astype(np.int16) - source.astype(np.int16)), axis=2
    )
    return {
        "enabled": True,
        "mode": "full_frame_provider_with_protected_artwork_v1",
        "output_path": str(output),
        "authoritative_image_path": str(authoritative_image_path),
        "provider_render_path": str(provider_render_path),
        "protect_mask_path": str(protect_mask_path),
        "full_frame_provider_preserved": True,
        "local_crop_used": False,
        "outside_protect_restored": False,
        "provider_pixels_outside_protect_preserved": True,
        "protected_area_px": int(np.count_nonzero(protect)),
        "exact_lock_area_px": int(np.count_nonzero(exact_lock)),
        "transition_area_px": int(np.count_nonzero(transition)),
        "feather_inside_px": float(feather),
        "provider_changed_ratio_outside_protect": (
            float(np.mean(provider_delta[outside] > 1.0)) if np.any(outside) else 0.0
        ),
        "provider_mean_delta_outside_protect": (
            float(np.mean(provider_delta[outside])) if np.any(outside) else 0.0
        ),
    }


def enforce_identity_locked_shadow_composite(
    *,
    hard_paste_path: Path,
    provider_render_path: Path,
    artwork_front_mask_path: Path,
    mount_mask_path: Path,
    edit_mask_path: Path,
    output_path: Path,
    protect_dilate_px: int = 4,
    shadow_min_darkening_delta: float = 4.0,
    shadow_max_brighten_delta: float = 3.0,
    shadow_max_chroma_delta: float = 6.0,
    shadow_max_darkening_delta: float = 24.0,
    shadow_blur_px: float = 1.25,
    shadow_min_component_area_px: int = 4,
    provider_identity_trusted: bool = True,
    fallback_shadow_max_darkening_delta: float = 14.0,
) -> dict[str, Any]:
    """Lock artwork identity/geometry and retain only explainable contact shadow.

    Some image-edit endpoints cannot accept a real mask. For textured walls the
    provider therefore sees the full room and may repaint or relocate the tiny
    artwork even when the prompt contains exact coordinates. This compositor
    makes the local contract deterministic:

    * the cleaned hard-paste is the base image;
    * artwork-front and precomposed mount pixels stay byte-identical;
    * pixels outside the edit mask stay byte-identical;
    * provider shadow is admitted only when raw artwork identity passed QA;
    * otherwise a bounded soft contact shadow is projected from the known mount
      mask, so displaced provider content cannot survive as a rectangular ghost.

    QA must still run on the returned image; this function does not bypass any
    delivery threshold.
    """
    # Keep full-resolution rooms in uint8 and promote only the working delta.
    # A 3648x2736 frame is common here; loading both inputs as float32 and then
    # allocating another RGB delta needlessly adds hundreds of MB of peak RAM.
    with Image.open(hard_paste_path) as hard_image:
        hard = np.asarray(hard_image.convert("RGB"), dtype=np.uint8)
    with Image.open(provider_render_path) as render_image:
        render = np.asarray(render_image.convert("RGB"), dtype=np.uint8)
    height, width = hard.shape[:2]
    if render.shape[:2] != (height, width):
        render = cv2.resize(render, (width, height), interpolation=cv2.INTER_LANCZOS4)

    front = _load_mask(Path(artwork_front_mask_path), (height, width))
    mount = _load_mask(Path(mount_mask_path), (height, width))
    edit = _load_mask(Path(edit_mask_path), (height, width))
    if not np.any(front):
        raise ValueError("identity lock requires a non-empty artwork front mask")
    if not np.any(mount):
        raise ValueError("identity lock requires a non-empty mount mask")
    if not np.any(edit):
        raise ValueError("identity lock requires a non-empty edit mask")
    if np.any(front & ~mount):
        raise ValueError("artwork front mask must be contained by the mount mask")
    if np.any(mount & ~edit):
        raise ValueError("mount mask must be contained by the edit mask")

    protect_radius = max(0, int(protect_dilate_px))
    protected = mount.copy()
    if protect_radius > 0:
        protected = cv2.dilate(mount.astype(np.uint8), _kernel(protect_radius), iterations=1) > 0

    delta = render.astype(np.int16) - hard.astype(np.int16)
    luma_delta = (
        delta[:, :, 0].astype(np.float32) * 0.2126
        + delta[:, :, 1].astype(np.float32) * 0.7152
        + delta[:, :, 2].astype(np.float32) * 0.0722
    )
    chroma_delta = (
        np.abs(delta[:, :, 0].astype(np.float32) - luma_delta)
        + np.abs(delta[:, :, 1].astype(np.float32) - luma_delta)
        + np.abs(delta[:, :, 2].astype(np.float32) - luma_delta)
    ) / 3.0
    max_channel_delta = np.max(delta, axis=2)
    if provider_identity_trusted:
        shadow_like = (
            edit
            & ~protected
            & (luma_delta <= -abs(float(shadow_min_darkening_delta)))
            & (max_channel_delta <= float(shadow_max_brighten_delta))
            & (chroma_delta <= float(shadow_max_chroma_delta))
        )
    else:
        shadow_like = np.zeros_like(edit, dtype=bool)
    del delta, chroma_delta, max_channel_delta

    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        shadow_like.astype(np.uint8), connectivity=8
    )
    kept_shadow = np.zeros_like(shadow_like, dtype=bool)
    min_area = max(1, int(shadow_min_component_area_px))
    retained_components = 0
    for component_id in range(1, int(component_count)):
        if int(stats[component_id, cv2.CC_STAT_AREA]) < min_area:
            continue
        kept_shadow[labels == component_id] = True
        retained_components += 1

    max_darkening = abs(float(shadow_max_darkening_delta))
    shadow_field = np.where(kept_shadow, np.clip(luma_delta, -max_darkening, 0.0), 0.0).astype(np.float32)
    blur_sigma = max(0.0, float(shadow_blur_px))
    if blur_sigma > 0.0 and np.any(kept_shadow):
        shadow_field = cv2.GaussianBlur(
            shadow_field,
            (0, 0),
            sigmaX=blur_sigma,
            sigmaY=blur_sigma,
            borderType=cv2.BORDER_REPLICATE,
        )
    if np.any(kept_shadow):
        allowed_shadow_region = edit & ~protected
        shadow_field = np.where(allowed_shadow_region, np.minimum(shadow_field, 0.0), 0.0)
        shadow_source = "provider_neutral_darken_only"
        fallback_shadow_trace: dict[str, Any] = {"applied": False, "reason": "trusted_provider_shadow_available"}
    else:
        # Build a small conventional down-right contact shadow from the exact
        # mount geometry. This is intentionally local and low-amplitude. It is
        # preferable to accepting neutral-looking fragments of a relocated
        # provider object, which can form an L-shaped or rectangular ghost.
        ys, xs = np.where(mount)
        bbox_width = int(xs.max() - xs.min() + 1)
        bbox_height = int(ys.max() - ys.min() + 1)
        min_dimension = max(1, min(bbox_width, bbox_height))
        offset_px = max(1, min(8, int(round(min_dimension * 0.018))))
        fallback_blur_sigma = max(1.25, min(8.0, float(min_dimension) * 0.025))
        shifted_mount = np.zeros_like(mount, dtype=np.float32)
        shifted_mount[offset_px:, offset_px:] = mount[:-offset_px, :-offset_px].astype(np.float32)
        projected_alpha = cv2.GaussianBlur(
            shifted_mount,
            (0, 0),
            sigmaX=fallback_blur_sigma,
            sigmaY=fallback_blur_sigma,
            borderType=cv2.BORDER_REPLICATE,
        )
        projected_alpha = np.clip(projected_alpha, 0.0, 1.0)
        yy, xx = np.indices(mount.shape)
        directional_half_space = (yy >= int(ys.min())) & (xx >= int(xs.min()))
        projected_region = edit & ~protected & directional_half_space
        fallback_darkening = abs(float(fallback_shadow_max_darkening_delta))
        shadow_field = np.where(projected_region, -fallback_darkening * projected_alpha, 0.0).astype(np.float32)
        shadow_source = "mask_projected_contact_shadow"
        fallback_shadow_trace = {
            "applied": True,
            "reason": "provider_identity_untrusted" if not provider_identity_trusted else "no_connected_provider_shadow",
            "offset_x_px": int(offset_px),
            "offset_y_px": int(offset_px),
            "blur_sigma_px": float(fallback_blur_sigma),
            "max_darkening_delta": float(fallback_darkening),
        }

    # The hard-paste remains authoritative for color, texture, perspective and
    # every pixel outside the local neutral-shadow scope.
    composed = hard.copy()
    for channel in range(3):
        composed[:, :, channel] = np.clip(
            hard[:, :, channel].astype(np.float32) + shadow_field,
            0.0,
            255.0,
        ).astype(np.uint8)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_name(f".{output.name}.tmp")
    try:
        with tmp_output.open("wb") as handle:
            Image.fromarray(composed, mode="RGB").save(handle, format="PNG")
            handle.flush()
            os.fsync(handle.fileno())
        tmp_output.replace(output)
    except Exception:
        tmp_output.unlink(missing_ok=True)
        raise

    applied = shadow_field < -0.25
    return {
        "enabled": True,
        "mode": "identity_locked_shadow_composite_v2",
        "output_path": str(output),
        "provider_render_path": str(provider_render_path),
        "hard_paste_path": str(hard_paste_path),
        "artwork_front_mask_path": str(artwork_front_mask_path),
        "mount_mask_path": str(mount_mask_path),
        "edit_mask_path": str(edit_mask_path),
        "scope_enforced_locally": True,
        "artwork_geometry_locked_to_candidate": True,
        "hardpaste_core_pixel_restore": True,
        "hardpaste_mount_pixel_restore": True,
        "outside_edit_mask_restored": True,
        "provider_identity_trusted": bool(provider_identity_trusted),
        "shadow_source": shadow_source,
        "provider_contribution": "neutral_darken_only_contact_shadow" if shadow_source.startswith("provider_") else "none",
        "provider_contribution_present": bool(shadow_source.startswith("provider_") and np.any(applied)),
        "fallback_shadow": fallback_shadow_trace,
        "front_area_px": int(np.count_nonzero(front)),
        "mount_area_px": int(np.count_nonzero(mount)),
        "edit_area_px": int(np.count_nonzero(edit)),
        "protected_area_px": int(np.count_nonzero(protected)),
        "raw_shadow_like_area_px": int(np.count_nonzero(shadow_like)),
        "retained_shadow_component_count": int(retained_components),
        "retained_shadow_seed_area_px": int(np.count_nonzero(kept_shadow)),
        "applied_shadow_area_px": int(np.count_nonzero(applied)),
        "applied_shadow_min_luma_delta": float(np.min(shadow_field)) if np.any(applied) else 0.0,
        "thresholds": {
            "protect_dilate_px": int(protect_radius),
            "shadow_min_darkening_delta": float(abs(float(shadow_min_darkening_delta))),
            "shadow_max_brighten_delta": float(shadow_max_brighten_delta),
            "shadow_max_chroma_delta": float(shadow_max_chroma_delta),
            "shadow_max_darkening_delta": float(max_darkening),
            "shadow_blur_px": float(blur_sigma),
            "shadow_min_component_area_px": int(min_area),
            "fallback_shadow_max_darkening_delta": float(abs(float(fallback_shadow_max_darkening_delta))),
        },
    }


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
