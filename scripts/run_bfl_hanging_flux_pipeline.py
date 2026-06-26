#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hanging_mvp.hardpaste import load_rgba, save_rgb, warp_artwork_mount_to_geometry, warp_artwork_to_quad
from hanging_mvp.render_qa import evaluate_mask_anchored_render_qa
from image_editing.pipeline import cleanup_objects, render_hanging
from image_editing.provider_registry import build_image_edit_provider
from image_editing.providers import ImageEditResult


MOCK_PROVIDER_WARNING = (
    "🔴 WARNING: provider=mock is a local no-op dry run. "
    "Final images are copied/intermediate inputs and must not be interpreted as BFL success or failure."
)


def _is_mock_provider(name: str | None) -> bool:
    return str(name or "").strip().lower() in {"mock", "dry_run", "noop"}


def _normalize_render_provider_name(name: str | None) -> str:
    key = str(name or os.getenv("HANGING_RENDER_PROVIDER") or "apiyi_gpt_image2_vip").strip().lower()
    aliases = {
        "apiyi": "apiyi_gpt_image2_vip",
        "apiyi_gpt_image2_vip": "apiyi_gpt_image2_vip",
        "gpt_image2_vip": "apiyi_gpt_image2_vip",
        "gpt-image-2-vip": "apiyi_gpt_image2_vip",
        "bfl": "bfl",
        "bfl_official": "bfl",
        "bfl_flux2": "bfl",
        "mock": "mock",
        "dry_run": "mock",
        "noop": "mock",
    }
    return aliases.get(key, "apiyi_gpt_image2_vip")


def _red(text: str) -> str:
    return f"\033[31m{text}\033[0m"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in str(value)).strip("_") or "item"


def _extract_zip(zip_path: Path, work_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(work_dir)
    return work_dir


def _resolve_render_seed(cli_seed: int | None) -> int | None:
    if cli_seed is not None:
        return int(cli_seed)
    env_value = os.getenv("MOLINK_RENDER_FIXED_SEED")
    if env_value is None or not env_value.strip():
        return None
    return int(env_value.strip())


def _resolve_render_prompt_upsampling(cli_value: bool | None) -> bool:
    if cli_value is not None:
        return bool(cli_value)
    return str(os.getenv("MOLINK_RENDER_PROMPT_UPSAMPLING", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_render_mask_scope(cli_value: str | None) -> str:
    raw = str(cli_value or os.getenv("MOLINK_RENDER_MASK_SCOPE", "edit_region")).strip().lower()
    if raw in {"front_only", "front"}:
        return "front_only"
    if raw in {"edit_region", "edit", "legacy"}:
        return "edit_region"
    return "art_only"


def _select_render_mask_path(
    *,
    qa_art_mask: Path,
    art_region_mask: Path,
    flux2_edit_mask: Path,
    scope: str,
) -> tuple[Path, str]:
    scope = _resolve_render_mask_scope(scope)
    if scope == "front_only":
        return qa_art_mask, "qa_art_mask_front_only"
    if scope == "edit_region":
        return flux2_edit_mask, "flux2_edit_mask_edit_region"
    return art_region_mask, "art_region_mask_art_only"


def _resolve_light_context_reference_mode(provider_name: str) -> str:
    default = "erased_room" if _normalize_render_provider_name(provider_name) == "apiyi_gpt_image2_vip" else "none"
    raw = str(os.getenv("MOLINK_RENDER_LIGHT_CONTEXT_REFERENCE", default)).strip().lower()
    if raw in {"0", "false", "off", "none", "no"}:
        return "none"
    if raw in {"erase", "erased", "erased_room", "clean_room", "cleaned_room"}:
        return "erased_room"
    return "none"


def _ambient_light_hint(
    *,
    image_path: Path,
    edit_mask_path: Path | None,
    object_mask_path: Path | None,
    band_px: int = 64,
    exclude_px: int = 8,
) -> tuple[str, dict[str, Any]]:
    """Build a tiny numeric light hint for GPT Image from deterministic pixels.

    This is intentionally advisory text only.  The hard-paste target remains the
    authoritative canvas; the hint gives GPT Image information it loses when we
    send only a local crop.
    """
    try:
        rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
        h, w = rgb.shape[:2]
        mask_source = object_mask_path if object_mask_path is not None and object_mask_path.exists() else edit_mask_path
        if mask_source is None or not mask_source.exists():
            return "", {"enabled": False, "reason": "missing_mask"}
        raw = np.asarray(Image.open(mask_source).convert("L"), dtype=np.uint8)
        if raw.shape[:2] != (h, w):
            raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
        alpha = raw > 0
        bbox = _alpha_bbox(alpha.astype(np.float32), threshold=0.2)
        if bbox is None:
            return "", {"enabled": False, "reason": "empty_mask"}
        band = max(8, int(band_px))
        exclude = max(0, int(exclude_px))
        dil_band = _dilate_alpha(alpha.astype(np.float32), band) > 0.05
        dil_exclude = _dilate_alpha(alpha.astype(np.float32), exclude) > 0.05
        sample = dil_band & (~dil_exclude)
        if edit_mask_path is not None and edit_mask_path.exists():
            edit_raw = np.asarray(Image.open(edit_mask_path).convert("L"), dtype=np.uint8)
            if edit_raw.shape[:2] != (h, w):
                edit_raw = cv2.resize(edit_raw, (w, h), interpolation=cv2.INTER_NEAREST)
            # Prefer pixels around the target, but do not require only editable pixels;
            # nearby non-object wall often carries the cleanest ambient evidence.
            sample = sample & (edit_raw <= 250) | (sample & (edit_raw > 0))
        count = int(np.count_nonzero(sample))
        if count < 32:
            return "", {"enabled": False, "reason": "insufficient_samples", "sample_count": count}
        weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        luma = np.tensordot(rgb, weights, axes=([-1], [0]))
        med_rgb = np.median(rgb[sample], axis=0)
        med_luma = float(np.median(luma[sample]))
        x0, y0, x1, y1 = bbox
        yy, xx = np.indices((h, w))
        sides: dict[str, float] = {}
        side_masks = {
            "left": sample & (xx < x0),
            "right": sample & (xx >= x1),
            "top": sample & (yy < y0),
            "bottom": sample & (yy >= y1),
        }
        for name, m in side_masks.items():
            if int(np.count_nonzero(m)) >= 16:
                sides[name] = float(np.median(luma[m]))
        brightest = max(sides, key=sides.get) if sides else None
        darkest = min(sides, key=sides.get) if sides else None
        hint = (
            "环境光采样提示：挂画附近干净墙面中位 RGB≈"
            f"({int(round(float(med_rgb[0])))},{int(round(float(med_rgb[1])))},{int(round(float(med_rgb[2])))})，"
            f"中位亮度≈{med_luma:.1f}/255。"
        )
        if brightest and darkest and brightest != darkest:
            hint += f"周围墙面相对亮度：{brightest} 侧更亮，{darkest} 侧更暗；投影和画芯明度应与这个环境光关系一致。"
        hint += "只把这些数值作为光影/明度参考，不要因此移动、缩放、重绘或替换挂画。"
        return hint, {
            "enabled": True,
            "sample_count": count,
            "median_wall_rgb": [float(v) for v in med_rgb],
            "median_wall_luma": med_luma,
            "side_luma": sides,
            "brightest_side": brightest,
            "darkest_side": darkest,
            "band_px": int(band),
            "exclude_px": int(exclude),
        }
    except Exception as exc:
        return "", {"enabled": False, "reason": "exception", "error": str(exc)}


def _fit_affine_luma_residual(luma: np.ndarray, sample: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(sample)
    if xs.size < 32:
        return 0.0, float(np.std(luma[sample])) if xs.size else 0.0
    h, w = luma.shape[:2]
    # Fit and remove a smooth affine lighting plane so a plain wall with a broad
    # light/shadow gradient does not trigger the full-space render path.
    A = np.stack([
        xs.astype(np.float32) / max(1.0, float(w)),
        ys.astype(np.float32) / max(1.0, float(h)),
        np.ones_like(xs, dtype=np.float32),
    ], axis=1)
    b = luma[sample].astype(np.float32)
    try:
        coeff, *_ = np.linalg.lstsq(A, b, rcond=None)
        pred = A @ coeff
        return float(np.std(b - pred)), float(np.std(b))
    except Exception:
        return float(np.std(b)), float(np.std(b))


def _render_full_space_texture_gate(
    *,
    image_path: Path,
    edit_mask_path: Path | None,
    object_mask_path: Path | None,
    outer_px: int = 192,
    inner_guard_px: int = 12,
    min_sample_px: int = 512,
    residual_std_threshold: float = 3.0,
    hf_mad_threshold: float = 1.25,
    edge_p95_threshold: float = 5.5,
    chroma_std_threshold: float = 2.6,
) -> tuple[bool, dict[str, Any]]:
    """Decide whether GPT Image needs the full room instead of a local crop.

    Local crop rendering is safe for flat, nearly solid-color walls.  It becomes
    unstable on walls with visible material texture, high-frequency grain, subtle
    low-frequency mottling, or hard light/shadow boundaries because the cropped
    request loses full-space texture continuity.  The detector samples wall pixels
    around the mounted object, removes a simple smooth lighting plane, and then
    tests residual texture/edge energy.  Smooth gradual light transitions alone
    should therefore not trigger the full-space path.
    """
    if not _env_truthy("MOLINK_RENDER_TEXTURE_GATE_ENABLED", "1"):
        return False, {"enabled": False, "reason": "disabled_by_env"}
    try:
        rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
        h, w = rgb.shape[:2]
        if edit_mask_path is None or not edit_mask_path.exists():
            return False, {"enabled": True, "full_space_recommended": False, "reason": "missing_edit_mask"}
        edit = np.asarray(Image.open(edit_mask_path).convert("L"), dtype=np.uint8)
        if edit.shape[:2] != (h, w):
            edit = cv2.resize(edit, (w, h), interpolation=cv2.INTER_NEAREST)
        if object_mask_path is not None and object_mask_path.exists():
            obj = np.asarray(Image.open(object_mask_path).convert("L"), dtype=np.uint8)
            if obj.shape[:2] != (h, w):
                obj = cv2.resize(obj, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            obj = edit.copy()

        edit_bool = edit > 0
        obj_bool = obj > 0
        anchor = obj_bool if int(np.count_nonzero(obj_bool)) else edit_bool
        if not int(np.count_nonzero(anchor)):
            return False, {"enabled": True, "full_space_recommended": False, "reason": "empty_masks"}

        outer = _dilate_alpha(anchor.astype(np.float32), max(1, int(outer_px))) > 0.05
        inner = _dilate_alpha(anchor.astype(np.float32), max(0, int(inner_guard_px))) > 0.05
        sample = (outer & (~inner)) | (edit_bool & (~inner))
        sample_count = int(np.count_nonzero(sample))
        if sample_count < int(min_sample_px):
            bbox = _alpha_bbox(anchor.astype(np.float32), threshold=0.2)
            if bbox is not None:
                x0, y0, x1, y1 = bbox
                pad = max(24, int(outer_px))
                rx0, ry0 = max(0, x0 - pad), max(0, y0 - pad)
                rx1, ry1 = min(w, x1 + pad), min(h, y1 + pad)
                rect = np.zeros((h, w), bool)
                rect[ry0:ry1, rx0:rx1] = True
                sample = rect & (~inner)
                sample_count = int(np.count_nonzero(sample))
        if sample_count < int(min_sample_px):
            return False, {
                "enabled": True,
                "full_space_recommended": False,
                "reason": "insufficient_wall_samples",
                "sample_count": sample_count,
                "min_sample_px": int(min_sample_px),
            }

        weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        luma = np.tensordot(rgb, weights, axes=([-1], [0])).astype(np.float32)
        residual_std, raw_luma_std = _fit_affine_luma_residual(luma, sample)
        blur_small = cv2.GaussianBlur(luma, (0, 0), sigmaX=3.0, sigmaY=3.0)
        high = luma - blur_small
        hf_values = high[sample]
        hf_mad = float(np.median(np.abs(hf_values - np.median(hf_values))) * 1.4826)
        blur_edge = cv2.GaussianBlur(luma, (0, 0), sigmaX=2.0, sigmaY=2.0)
        gx = cv2.Sobel(blur_edge, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur_edge, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        edge_p95 = float(np.percentile(grad[sample], 95))
        # Chroma residual catches colored plaster / material mottling even when
        # luminance is almost constant.
        rg = rgb[:, :, 0] - rgb[:, :, 1]
        yb = 0.5 * (rgb[:, :, 0] + rgb[:, :, 1]) - rgb[:, :, 2]
        chroma_std = max(_fit_affine_luma_residual(rg.astype(np.float32), sample)[0], _fit_affine_luma_residual(yb.astype(np.float32), sample)[0])
        score_parts = {
            "affine_removed_luma_residual_std": float(residual_std) / max(1e-6, float(residual_std_threshold)),
            "high_frequency_luma_mad": float(hf_mad) / max(1e-6, float(hf_mad_threshold)),
            "blurred_edge_gradient_p95": float(edge_p95) / max(1e-6, float(edge_p95_threshold)),
            "affine_removed_chroma_residual_std": float(chroma_std) / max(1e-6, float(chroma_std_threshold)),
        }
        texture_score = float(max(score_parts.values()))
        recommended = bool(texture_score >= 1.0)
        dominant_reason = max(score_parts, key=score_parts.get)
        return recommended, {
            "enabled": True,
            "full_space_recommended": recommended,
            "reason": "candidate_wall_not_pure_color" if recommended else "candidate_wall_looks_pure_or_smooth_gradient_only",
            "dominant_reason": dominant_reason,
            "texture_score": texture_score,
            "score_parts": score_parts,
            "sample_count": sample_count,
            "outer_px": int(outer_px),
            "inner_guard_px": int(inner_guard_px),
            "thresholds": {
                "residual_std_threshold": float(residual_std_threshold),
                "hf_mad_threshold": float(hf_mad_threshold),
                "edge_p95_threshold": float(edge_p95_threshold),
                "chroma_std_threshold": float(chroma_std_threshold),
            },
            "metrics": {
                "raw_luma_std": float(raw_luma_std),
                "affine_removed_luma_residual_std": float(residual_std),
                "high_frequency_luma_mad": float(hf_mad),
                "blurred_edge_gradient_p95": float(edge_p95),
                "affine_removed_chroma_residual_std": float(chroma_std),
            },
            "smooth_lighting_transition_exclusion": "affine_luma_plane_removed_before_texture_decision",
        }
    except Exception as exc:
        return False, {"enabled": True, "full_space_recommended": False, "reason": "exception", "error": str(exc)}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


def _low_frequency_luma_delta_image(original: Image.Image, generated: Image.Image, *, blur_px: int, max_delta: float, strength: float) -> np.ndarray:
    radius = max(1, int(blur_px))
    orig_blur = np.asarray(original.convert("RGB").filter(ImageFilter.GaussianBlur(radius=float(radius))), dtype=np.float32)
    gen_blur = np.asarray(generated.convert("RGB").filter(ImageFilter.GaussianBlur(radius=float(radius))), dtype=np.float32)
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    delta = np.tensordot(gen_blur, weights, axes=([-1], [0])) - np.tensordot(orig_blur, weights, axes=([-1], [0]))
    clamp = float(max(0.0, max_delta))
    if clamp > 0:
        delta = np.clip(delta, -clamp, clamp)
    return delta[..., None] * float(strength)


def _darken_only_luma_transfer_full(original_arr: np.ndarray, generated_arr: np.ndarray, *, max_darkening: float) -> np.ndarray:
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    orig_luma = np.tensordot(original_arr, weights, axes=([-1], [0]))
    gen_luma = np.tensordot(generated_arr, weights, axes=([-1], [0]))
    dark_delta = np.minimum(gen_luma - orig_luma, 0.0)
    clamp = float(max(0.0, max_darkening))
    if clamp > 0:
        dark_delta = np.clip(dark_delta, -clamp, 0.0)
    return np.clip(original_arr + dark_delta[..., None], 0, 255)


def _alpha_bbox(alpha: np.ndarray, *, threshold: float = 0.2) -> tuple[int, int, int, int] | None:
    arr = np.asarray(alpha)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    ys, xs = np.where(arr > float(threshold))
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _dilate_alpha(alpha: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return np.asarray(alpha, dtype=np.float32)
    arr = np.asarray(alpha, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    k = max(1, int(radius_px) * 2 + 1)
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate((np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8), kernel, iterations=1).astype(np.float32) / 255.0


def _estimate_anchor_similarity_full(
    *,
    hard_rgb: Image.Image,
    render_rgb: Image.Image,
    anchor_alpha: np.ndarray,
    max_shift_px: int,
    min_score: float,
    scale_min: float,
    scale_max: float,
    scale_steps: int,
) -> dict[str, Any]:
    max_shift = max(0, int(max_shift_px))
    bbox = _alpha_bbox(anchor_alpha, threshold=0.2)
    if bbox is None or max_shift <= 0:
        return {
            "applied": False,
            "reason": "missing_anchor_or_search_disabled",
            "dx": 0,
            "dy": 0,
            "scale": 1.0,
            "score": None,
            "matrix_hard_to_render": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        }
    w, h = hard_rgb.size
    x0, y0, x1, y1 = bbox
    pad = 2
    x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    bw, bh = x1 - x0, y1 - y0
    if bw < 6 or bh < 6:
        return {
            "applied": False,
            "reason": "anchor_too_small",
            "dx": 0,
            "dy": 0,
            "scale": 1.0,
            "score": None,
            "anchor_bbox": [x0, y0, x1, y1],
            "matrix_hard_to_render": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        }
    hard_gray = np.asarray(hard_rgb.convert("L"), dtype=np.float32)
    render_gray = np.asarray(render_rgb.convert("L"), dtype=np.float32)
    template0 = hard_gray[y0:y1, x0:x1]
    if float(np.std(template0)) < 3.0:
        return {
            "applied": False,
            "reason": "anchor_low_texture",
            "dx": 0,
            "dy": 0,
            "scale": 1.0,
            "score": None,
            "anchor_bbox": [x0, y0, x1, y1],
            "matrix_hard_to_render": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        }

    s_min = max(0.75, float(scale_min))
    s_max = min(1.35, float(scale_max))
    if s_max < s_min:
        s_min, s_max = s_max, s_min
    steps = max(1, int(scale_steps))
    scales = [1.0] if steps == 1 else [float(v) for v in np.linspace(s_min, s_max, steps)]
    if 1.0 not in scales:
        scales.append(1.0)
    scales = sorted(set(round(float(v), 4) for v in scales))

    sx0 = max(0, x0 - max_shift)
    sy0 = max(0, y0 - max_shift)
    sx1 = min(w, x1 + max_shift)
    sy1 = min(h, y1 + max_shift)
    search = render_gray[sy0:sy1, sx0:sx1]
    best: dict[str, Any] | None = None
    for scale in scales:
        tw = max(3, int(round(float(bw) * float(scale))))
        th = max(3, int(round(float(bh) * float(scale))))
        if tw > search.shape[1] or th > search.shape[0]:
            continue
        template = template0 if (tw == bw and th == bh) else cv2.resize(template0, (tw, th), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
        if float(np.std(template)) < 3.0:
            continue
        result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(result)
        rec = {
            "score": float(max_val),
            "scale": float(scale),
            "gen_x0": int(sx0 + max_loc[0]),
            "gen_y0": int(sy0 + max_loc[1]),
            "template_w": int(tw),
            "template_h": int(th),
        }
        if best is None or float(rec["score"]) > float(best["score"]):
            best = rec
    if best is None:
        return {
            "applied": False,
            "reason": "search_region_too_small",
            "dx": 0,
            "dy": 0,
            "scale": 1.0,
            "score": None,
            "anchor_bbox": [x0, y0, x1, y1],
            "matrix_hard_to_render": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        }

    scale = float(best["scale"])
    gen_x0 = int(best["gen_x0"])
    gen_y0 = int(best["gen_y0"])
    tx = float(gen_x0) - scale * float(x0)
    ty = float(gen_y0) - scale * float(y0)
    raw_dx = gen_x0 - x0
    raw_dy = gen_y0 - y0
    applied = bool(float(best["score"]) >= float(min_score) and (abs(raw_dx) > 0 or abs(raw_dy) > 0 or abs(scale - 1.0) >= 0.003))
    return {
        "applied": applied,
        "reason": None if applied else "score_below_threshold_or_identity_transform",
        "dx": int(raw_dx) if applied else 0,
        "dy": int(raw_dy) if applied else 0,
        "raw_dx": int(raw_dx),
        "raw_dy": int(raw_dy),
        "scale": float(scale if applied else 1.0),
        "raw_scale": float(scale),
        "score": float(best["score"]),
        "min_score": float(min_score),
        "max_shift_px": int(max_shift),
        "scale_search_min": float(s_min),
        "scale_search_max": float(s_max),
        "scale_search_steps": int(steps),
        "anchor_bbox": [int(x0), int(y0), int(x1), int(y1)],
        "matched_bbox": [int(gen_x0), int(gen_y0), int(gen_x0 + int(best["template_w"])), int(gen_y0 + int(best["template_h"]))],
        "matrix_hard_to_render": [[float(scale), 0.0, float(tx)], [0.0, float(scale), float(ty)]] if applied else [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "resampling": "lanczos4_for_locked_rgb_and_linear_antialias_for_masks",
    }




def _translate_full_rgb(render_rgb: Image.Image, *, dx: int, dy: int, fill_rgb: Image.Image) -> Image.Image:
    if dx == 0 and dy == 0:
        return render_rgb.convert("RGB")
    arr = np.asarray(render_rgb.convert("RGB"), dtype=np.uint8)
    fill = np.asarray(fill_rgb.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    matrix = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
    shifted = cv2.warpAffine(arr, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    valid = cv2.warpAffine(np.ones((h, w), dtype=np.uint8) * 255, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    shifted[valid == 0] = fill[valid == 0]
    return Image.fromarray(shifted, mode="RGB")


def _warp_full_rgb_similarity(image: Image.Image, *, matrix: list[list[float]], size_wh: tuple[int, int]) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    m = np.array(matrix, dtype=np.float32)
    out_w, out_h = int(size_wh[0]), int(size_wh[1])
    warped = cv2.warpAffine(arr, m, (out_w, out_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return Image.fromarray(warped, mode="RGB")


def _warp_full_alpha_similarity(alpha: np.ndarray, *, matrix: list[list[float]], size_wh: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(alpha, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    m = np.array(matrix, dtype=np.float32)
    out_w, out_h = int(size_wh[0]), int(size_wh[1])
    warped = cv2.warpAffine(np.clip(arr, 0.0, 1.0), m, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return np.clip(warped.astype(np.float32), 0.0, 1.0)


def _correct_render_wall_rgb_full(
    *,
    hard_rgb: Image.Image,
    render_rgb: Image.Image,
    edit_alpha: np.ndarray,
    object_alpha: np.ndarray,
    exclusion_px: int,
    sample_max_delta: float,
    max_rgb_delta: float,
    strength: float,
    apply_to_wall_only: bool = False,
) -> tuple[Image.Image, dict[str, Any]]:
    hard = np.asarray(hard_rgb.convert("RGB"), dtype=np.float32)
    render = np.asarray(render_rgb.convert("RGB"), dtype=np.float32)
    obj_dilated = _dilate_alpha(object_alpha, int(exclusion_px))
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    hard_luma = np.tensordot(hard, weights, axes=([-1], [0]))
    render_luma = np.tensordot(render, weights, axes=([-1], [0]))
    luma_delta = render_luma - hard_luma
    rgb_delta_abs = np.max(np.abs(render - hard), axis=2)
    sample_mask = (
        (np.asarray(edit_alpha, dtype=np.float32) > 0.15)
        & (obj_dilated < 0.05)
        & (np.abs(luma_delta) <= float(sample_max_delta))
        & (rgb_delta_abs <= float(sample_max_delta) * 2.0)
    )
    sample_count = int(np.count_nonzero(sample_mask))
    if sample_count < 32:
        return render_rgb.convert("RGB"), {
            "applied": False,
            "reason": "insufficient_clean_wall_samples",
            "sample_count": sample_count,
            "exclusion_px": int(exclusion_px),
            "sample_max_delta": float(sample_max_delta),
        }
    hard_med = np.median(hard[sample_mask], axis=0)
    render_med = np.median(render[sample_mask], axis=0)
    delta = (hard_med - render_med) * float(strength)
    clamp = float(max(0.0, max_rgb_delta))
    if clamp > 0:
        delta = np.clip(delta, -clamp, clamp)
    corrected_full = np.clip(render + delta.reshape(1, 1, 3), 0, 255)
    if apply_to_wall_only:
        # Only normalize wall pixels.  Keep GPT Image's rendered artwork, frame,
        # mat/core and object pixels exactly as the provider returned them; this
        # avoids a second hard-paste/locked-texture pass that can create bottom
        # frame stacking when render geometry shifts or scales.
        wall_apply_alpha = np.clip(np.asarray(edit_alpha, dtype=np.float32) * (1.0 - np.clip(object_alpha, 0.0, 1.0)), 0.0, 1.0)[..., None]
        corrected = render * (1.0 - wall_apply_alpha) + corrected_full * wall_apply_alpha
        apply_scope = "wall_pixels_only_inside_edit_mask_object_unchanged"
    else:
        corrected = corrected_full
        apply_scope = "entire_render_image"
    return Image.fromarray(corrected.astype(np.uint8), mode="RGB"), {
        "applied": True,
        "sample_count": sample_count,
        "exclusion_px": int(exclusion_px),
        "sample_max_delta": float(sample_max_delta),
        "max_rgb_delta": float(max_rgb_delta),
        "strength": float(strength),
        "hard_wall_median_rgb": [float(v) for v in hard_med],
        "render_wall_median_rgb": [float(v) for v in render_med],
        "applied_delta_rgb": [float(v) for v in delta],
        "apply_scope": apply_scope,
    }


def _apply_final_photometric_lock(
    *,
    hard_paste: Path,
    render: Path,
    object_mask: Path,
    edit_mask: Path | None,
    out_path: Path,
    blur_px: int,
    max_delta: float,
    strength: float,
    feather_px: int,
    shadow_feather_px: int,
    relocation_search_px: int,
    relocation_min_score: float,
    relocation_scale_min: float,
    relocation_scale_max: float,
    relocation_scale_steps: int,
    wall_sample_exclusion_px: int,
    wall_sample_max_delta: float,
    wall_rgb_max_delta: float,
    wall_rgb_correction_strength: float,
    object_edge_guard_px: int,
    postprocess_mode: str = "photometric_lock",
) -> dict[str, Any]:
    with Image.open(hard_paste) as hard_img, Image.open(render) as render_img, Image.open(object_mask) as object_mask_img:
        hard_rgb = hard_img.convert("RGB")
        render_rgb = render_img.convert("RGB")
        if render_rgb.size != hard_rgb.size:
            render_rgb = render_rgb.resize(hard_rgb.size, Image.Resampling.LANCZOS)

        object_mask_l = object_mask_img.convert("L")
        if object_mask_l.size != hard_rgb.size:
            object_mask_l = object_mask_l.resize(hard_rgb.size, Image.Resampling.NEAREST)
        object_mask_for_light = object_mask_l
        if feather_px > 0:
            object_mask_for_light = object_mask_for_light.filter(ImageFilter.GaussianBlur(radius=float(feather_px)))
        object_alpha = np.asarray(object_mask_for_light, dtype=np.float32) / 255.0

        if edit_mask is not None and edit_mask.exists():
            with Image.open(edit_mask) as edit_mask_img:
                edit_mask_l = edit_mask_img.convert("L")
                if edit_mask_l.size != hard_rgb.size:
                    edit_mask_l = edit_mask_l.resize(hard_rgb.size, Image.Resampling.NEAREST)
            edit_mask_available = True
        else:
            edit_mask_l = object_mask_l
            edit_mask_available = False
        if shadow_feather_px > 0:
            edit_mask_for_shadow = edit_mask_l.filter(ImageFilter.GaussianBlur(radius=float(shadow_feather_px)))
        else:
            edit_mask_for_shadow = edit_mask_l
        edit_alpha_2d = np.asarray(edit_mask_for_shadow, dtype=np.float32) / 255.0

        normalized_postprocess_mode = str(postprocess_mode or "photometric_lock").strip().lower().replace("-", "_")
        no_wall_rgb_modes = {"photometric_lock_no_wall_rgb", "fullspace_object_lock", "full_space_object_lock"}
        if normalized_postprocess_mode in {"wall_rgb_only", "wall_rgb_correction_only", "render_wall_rgb_only"}:
            corrected_render_rgb, wall_rgb_trace = _correct_render_wall_rgb_full(
                hard_rgb=hard_rgb,
                render_rgb=render_rgb,
                edit_alpha=edit_alpha_2d,
                object_alpha=object_alpha,
                exclusion_px=wall_sample_exclusion_px,
                sample_max_delta=wall_sample_max_delta,
                max_rgb_delta=wall_rgb_max_delta,
                strength=wall_rgb_correction_strength,
                apply_to_wall_only=True,
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            corrected_render_rgb.save(out_path)
            return {
                "enabled": True,
                "mode": "post_provider_wall_rgb_only_no_locked_texture",
                "postprocess_mode": normalized_postprocess_mode,
                "input_render_path": str(render),
                "hard_paste_path": str(hard_paste),
                "object_mask_path": str(object_mask),
                "edit_mask_path": str(edit_mask) if edit_mask is not None else None,
                "edit_mask_available": bool(edit_mask_available),
                "output_path": str(out_path),
                "locked_texture_pasted": False,
                "hardpaste_frame_core_repasted": False,
                "anchor_relocation_applied": False,
                "render_pixels_preserved": True,
                "wall_rgb_correction": wall_rgb_trace,
                "feather_px": int(feather_px),
                "shadow_feather_px": int(shadow_feather_px),
            }

        relocation_trace = _estimate_anchor_similarity_full(
            hard_rgb=hard_rgb,
            render_rgb=render_rgb,
            anchor_alpha=object_alpha,
            max_shift_px=relocation_search_px,
            min_score=relocation_min_score,
            scale_min=relocation_scale_min,
            scale_max=relocation_scale_max,
            scale_steps=relocation_scale_steps,
        )
        transform_matrix = relocation_trace.get("matrix_hard_to_render") or [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        locked_rgb = _warp_full_rgb_similarity(hard_rgb, matrix=transform_matrix, size_wh=hard_rgb.size)
        locked_object_alpha = _warp_full_alpha_similarity(object_alpha, matrix=transform_matrix, size_wh=hard_rgb.size)

        wall_exclusion_alpha = np.maximum(object_alpha, locked_object_alpha)
        if normalized_postprocess_mode in no_wall_rgb_modes:
            wall_rgb_trace = {
                "applied": False,
                "reason": "disabled_for_full_space_provider_render",
                "apply_scope": "none",
            }
        else:
            render_rgb, wall_rgb_trace = _correct_render_wall_rgb_full(
                hard_rgb=hard_rgb,
                render_rgb=render_rgb,
                edit_alpha=edit_alpha_2d,
                object_alpha=wall_exclusion_alpha,
                exclusion_px=wall_sample_exclusion_px,
                sample_max_delta=wall_sample_max_delta,
                max_rgb_delta=wall_rgb_max_delta,
                strength=wall_rgb_correction_strength,
            )

        # Preserve provider-rendered contact shadow / projection outside the
        # locked artwork object, while pasting a high-quality resampled locked
        # texture at the provider-estimated translation+scale.
        object_guard_alpha = _dilate_alpha(locked_object_alpha, int(object_edge_guard_px))
        object_alpha_3d = locked_object_alpha[..., None]
        edit_alpha_3d = edit_alpha_2d[..., None]
        guard_alpha_3d = np.clip(edit_alpha_3d * np.clip(object_guard_alpha - locked_object_alpha, 0.0, 1.0)[..., None], 0.0, 1.0)
        shadow_alpha = np.clip(edit_alpha_3d * (1.0 - object_guard_alpha[..., None]), 0.0, 1.0)

        hard_arr = np.asarray(hard_rgb, dtype=np.float32)
        locked_arr = np.asarray(locked_rgb, dtype=np.float32)
        render_arr = np.asarray(render_rgb, dtype=np.float32)
        delta = _low_frequency_luma_delta_image(
            locked_rgb,
            render_rgb,
            blur_px=blur_px,
            max_delta=max_delta,
            strength=strength,
        )
        # Apply render-derived lightness to the locked texture, including the
        # slightly grayer frame appearance, without changing artwork identity.
        adjusted_object = np.clip(locked_arr + delta, 0, 255)
        guard_shadow = _darken_only_luma_transfer_full(
            hard_arr,
            render_arr,
            max_darkening=max_delta,
        )

        combined_alpha = np.clip(object_alpha_3d + guard_alpha_3d + shadow_alpha, 0.0, 1.0)
        out = (
            hard_arr * (1.0 - combined_alpha)
            + adjusted_object * object_alpha_3d
            + guard_shadow * guard_alpha_3d
            + render_arr * shadow_alpha
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB").save(out_path)
    return {
        "enabled": True,
        "mode": (
            "post_provider_shadow_preserving_photometric_object_lock_relocated_no_wall_rgb"
            if normalized_postprocess_mode in no_wall_rgb_modes
            else "post_provider_shadow_preserving_photometric_object_lock_relocated_wall_corrected"
        ),
        "postprocess_mode": str(postprocess_mode or "photometric_lock").strip().lower().replace("-", "_"),
        "input_render_path": str(render),
        "hard_paste_path": str(hard_paste),
        "object_mask_path": str(object_mask),
        "edit_mask_path": str(edit_mask) if edit_mask is not None else None,
        "edit_mask_available": bool(edit_mask_available),
        "output_path": str(out_path),
        "outside_object_pixels_preserved_from_hardpaste": False,
        "rendered_shadow_preserved_outside_object": True,
        "generated_pixels_pasted": (
            "outside_similarity_locked_object_within_edit_mask_no_wall_rgb_correction"
            if normalized_postprocess_mode in no_wall_rgb_modes
            else "outside_similarity_locked_object_within_edit_mask_after_wall_rgb_correction"
        ),
        "locked_texture_transform": "hardpaste_texture_warped_to_provider_translation_and_scale",
        "locked_texture_resampling": "cv2.INTER_LANCZOS4_rgb_and_linear_alpha",
        "object_edge_guard_px": int(object_edge_guard_px),
        "guard_treatment": "darken_only_luma_transfer_in_narrow_object_edge_ring",
        "intrinsic_color_policy": "artwork_object_achromatic_low_frequency_luma_delta_only",
        "anchor_relocation": relocation_trace,
        "wall_rgb_correction": wall_rgb_trace,
        "blur_px": int(blur_px),
        "max_delta": float(max_delta),
        "strength": float(strength),
        "feather_px": int(feather_px),
        "shadow_feather_px": int(shadow_feather_px),
    }


def _render_edit_runtime_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "seed": _resolve_render_seed(args.seed),
        "safety_tolerance": args.safety_tolerance,
        "prompt_upsampling": _resolve_render_prompt_upsampling(getattr(args, "render_prompt_upsampling", None)),
        "render_creativity_profile": "low_variance_photometric_focus",
        "interior_artwork_regeneration_discouraged": True,
        "render_mask_scope": _resolve_render_mask_scope(getattr(args, "render_mask_scope", None)),
    }


def _load_manifest(extracted: Path) -> dict[str, Any]:
    manifest = extracted / "flux_inputs" / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"missing flux_inputs/manifest.json in hanging result: {manifest}")
    data = _read_json(manifest)
    if not isinstance(data.get("items"), list):
        raise ValueError("flux_inputs/manifest.json must contain an items list")
    return data


def _resolve_path(extracted: Path, rel_or_abs: str | None) -> Path | None:
    if not rel_or_abs:
        return None
    p = Path(rel_or_abs)
    return p if p.is_absolute() else extracted / p


def _require_existing(path: Path | None, *, label: str, asset_id: str) -> Path:
    if path is None or not path.exists():
        raise FileNotFoundError(f"manifest item {asset_id} references missing {label}: {path}")
    return path


def _copy_inputs_for_trace(item: dict[str, Any], manifest: dict[str, Any], extracted: Path, item_dir: Path) -> dict[str, str]:
    copied: dict[str, str] = {}
    keys = (
        "original_room_path",
        "artwork_path",
        "hard_paste_path",
        "remove_mask_path",
        "art_region_mask_path",
        "front_region_mask_path",
        "frame_side_mask_path",
        "mount_mask_path",
        "flux2_edit_mask_path",
        "mount_debug_path",
        "candidate_path",
        "render_prompt_path",
    )
    for key in keys:
        rel = item.get(key) or manifest.get(key)
        if not rel:
            continue
        src = _resolve_path(extracted, str(rel))
        if src is None or not src.exists():
            raise FileNotFoundError(f"manifest item {item.get('asset_id')} references missing file: {rel}")
        dst = item_dir / "inputs" / Path(str(rel)).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        copied[key] = str(dst)
    return copied


def _quad_from_item(item: dict[str, Any], extracted: Path) -> list[list[float]] | None:
    quad = item.get("target_quad_original") or item.get("polygon_px")
    if quad:
        return [[float(p[0]), float(p[1])] for p in quad[:4]]
    candidate_path = _resolve_path(extracted, item.get("candidate_path"))
    if candidate_path and candidate_path.exists():
        cand = _read_json(candidate_path)
        projection = cand.get("projection", {}) if isinstance(cand.get("projection"), dict) else {}
        quad = projection.get("projected_quad_px_original")
        if quad:
            return [[float(p[0]), float(p[1])] for p in quad[:4]]
    return None


def _is_strict_v7_item(item: dict[str, Any], manifest: dict[str, Any]) -> bool:
    erase_order = str(item.get("erase_order") or manifest.get("erase_order") or "").strip()
    if erase_order == "original_room_then_hardpaste_after_erase":
        return True
    try:
        if int(manifest.get("version", 0) or 0) >= 2:
            return True
    except Exception:
        pass
    if item.get("original_room_path") or item.get("artwork_path") or item.get("target_quad_original"):
        return True
    return False


def _hardpaste_after_erase(
    clean_room: Path,
    artwork: Path,
    quad: list[list[float]],
    out_path: Path,
    *,
    mount_policy: dict[str, Any] | None = None,
    supersample_scale: int = 4,
) -> tuple[Path, dict[str, Any]]:
    room_rgb = np.array(Image.open(clean_room).convert("RGB"))
    art_rgba = load_rgba(artwork)
    policy = mount_policy if isinstance(mount_policy, dict) else {}
    if policy.get("precompose_frame_sides"):
        mounted = warp_artwork_mount_to_geometry(
            room_rgb,
            art_rgba,
            front_quad_original=policy.get("front_quad_original") or quad,
            side_faces_original=policy.get("visible_side_faces_original") or [],
            edge_sample_px=int(policy.get("edge_sample_px", 8)),
            edge_average_depth_px=int(policy.get("edge_average_depth_px", max(1, min(2, int(policy.get("edge_sample_px", 8)))))),
            side_shading_strength=float(policy.get("side_shading_strength", 0.18)),
            supersample_scale=int(supersample_scale),
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_rgb(str(out_path), mounted["image"])
        return out_path, {**(mounted.get("debug") or {}), "policy": policy}
    pasted, _art_region = warp_artwork_to_quad(room_rgb, art_rgba, quad, supersample_scale=int(supersample_scale))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_rgb(str(out_path), pasted)
    return out_path, {"mode": "flat_hardpaste", "policy": policy}


def _run_mask_anchored_qa(args: argparse.Namespace, *, hard_paste: Path, render: Path, art_mask: Path) -> dict[str, Any]:
    return evaluate_mask_anchored_render_qa(
        hard_paste_path=hard_paste,
        render_path=render,
        art_region_mask_path=art_mask,
        core_erode_px=int(args.qa_core_erode_px),
        allowed_frame_shadow_px=int(args.qa_allowed_frame_shadow_px),
        forbidden_outer_guard_px=int(args.qa_forbidden_outer_guard_px),
        changed_pixel_delta=float(args.qa_changed_pixel_delta),
        min_core_masked_ssim=float(args.qa_min_core_masked_ssim),
        max_core_color_delta=float(args.qa_max_core_color_delta),
        max_forbidden_outer_guard_changed_ratio=float(args.qa_max_forbidden_outer_guard_changed_ratio),
        ignore_shadow_in_forbidden_guard=bool(args.qa_ignore_shadow_in_forbidden_guard),
        shadow_min_darkening_delta=float(args.qa_shadow_min_darkening_delta),
        shadow_max_brighten_delta=float(args.qa_shadow_max_brighten_delta),
        shadow_max_chroma_delta=float(args.qa_shadow_max_chroma_delta),
    )


def _is_nonempty_image_file(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        p = Path(path)
        return p.exists() and p.is_file() and p.stat().st_size > 0
    except Exception:
        return False


def _env_truthy(name: str, default: str = "1") -> bool:
    value = os.getenv(name, default)
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def _qa_hard_block_reasons(qa: dict[str, Any] | None, args: argparse.Namespace) -> list[str]:
    if not isinstance(qa, dict) or qa.get("passed"):
        return []
    reasons: list[str] = []
    try:
        if float(qa.get("core_masked_ssim", 1.0)) < float(args.qa_hard_min_core_masked_ssim):
            reasons.append("hard_core_masked_ssim_too_low")
    except Exception:
        pass
    try:
        if float(qa.get("core_color_delta", 0.0)) > float(args.qa_hard_max_core_color_delta):
            reasons.append("hard_core_color_delta_too_high")
    except Exception:
        pass
    try:
        if float(qa.get("forbidden_outer_guard_changed_ratio", 0.0)) > float(args.qa_hard_max_forbidden_outer_guard_changed_ratio):
            reasons.append("hard_forbidden_outer_guard_changed_ratio_too_high")
    except Exception:
        pass
    return reasons


def _qa_quality_score(qa: dict[str, Any] | None) -> float:
    if not isinstance(qa, dict):
        return -1.0
    if qa.get("passed"):
        return 100.0
    score = 0.0
    try:
        score += float(qa.get("core_masked_ssim", 0.0)) * 40.0
    except Exception:
        pass
    try:
        score -= min(float(qa.get("core_color_delta", 0.0)), 120.0) / 3.0
    except Exception:
        pass
    try:
        score -= min(float(qa.get("forbidden_outer_guard_changed_ratio", 0.0)), 1.0) * 30.0
    except Exception:
        pass
    return score


def _render_failure_diagnosis(
    *,
    qa: dict[str, Any] | None,
    render_trace: dict[str, Any] | None,
    using_apiyi_gpt_image2_vip: bool,
    flux2_input_mode: str,
) -> dict[str, Any]:
    qa = qa if isinstance(qa, dict) else {}
    trace = render_trace if isinstance(render_trace, dict) else {}
    failures = list(qa.get("failures") or [])
    provider = str(trace.get("provider") or "unknown")
    diagnosis: dict[str, Any] = {
        "provider": provider,
        "using_apiyi_gpt_image2_vip": bool(using_apiyi_gpt_image2_vip),
        "flux2_input_mode": flux2_input_mode,
        "reference_count": trace.get("reference_count"),
        "mask_sent_to_flux2": bool(trace.get("mask_sent_to_flux2")),
        "mask_enforced_locally": bool(trace.get("mask_enforced_locally") or trace.get("mask_local_composite")),
        "failures": failures,
    }
    try:
        diagnosis["core_masked_ssim"] = float(qa.get("core_masked_ssim"))
    except Exception:
        pass
    try:
        diagnosis["core_color_delta"] = float(qa.get("core_color_delta"))
    except Exception:
        pass
    try:
        diagnosis["forbidden_outer_guard_changed_ratio"] = float(qa.get("forbidden_outer_guard_changed_ratio"))
    except Exception:
        pass

    if (
        provider == "bfl"
        and not using_apiyi_gpt_image2_vip
        and bool(trace.get("mask_sent_to_flux2"))
        and ("core_masked_ssim_below_threshold" in failures or "core_color_delta_above_threshold" in failures)
    ):
        diagnosis["root_cause"] = (
            "native FLUX/BFL final edit was allowed to repaint the masked artwork core; "
            "the prompt/mask constrained outer geometry, but did not preserve the tiny painting identity inside the edit mask"
        )
        diagnosis["recommended_provider"] = "apiyi_gpt_image2_vip"
        diagnosis["recommended_mode"] = "photometric_composite"
    elif using_apiyi_gpt_image2_vip and bool(trace.get("mask_photometric_composite")):
        diagnosis["root_cause"] = "post_photometric_composite_qa_failure"
    else:
        diagnosis["root_cause"] = "undetermined_render_qa_failure"
    return diagnosis


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    requested_provider_name = str(getattr(args, "provider", None) or os.getenv("HANGING_RENDER_PROVIDER") or "apiyi_gpt_image2_vip").strip().lower()
    provider_name = _normalize_render_provider_name(requested_provider_name)
    args.provider = provider_name
    provider = build_image_edit_provider(provider_name)
    using_mock_provider = _is_mock_provider(provider_name)
    using_apiyi_gpt_image2_vip = provider_name == "apiyi_gpt_image2_vip"
    if using_mock_provider:
        print(_red(MOCK_PROVIDER_WARNING), file=sys.stderr)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_dir = output_dir / "final_hd"
    final_dir.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="hanging_flux_") as td:
        extracted = _extract_zip(Path(args.hanging_zip).resolve(), Path(td))
        manifest = _load_manifest(extracted)
        items = manifest.get("items", [])
        if args.limit is not None:
            items = items[: args.limit]
        aggregate: dict[str, Any] = {
            "version": 2,
            "source_zip": str(Path(args.hanging_zip).resolve()),
            "provider": provider_name,
            "requested_provider": requested_provider_name,
            "provider_warning": MOCK_PROVIDER_WARNING if using_mock_provider else None,
            "render_provider": provider_name,
            "cleanup_provider": "bfl_erase_v1" if using_apiyi_gpt_image2_vip else provider_name,
            "started_at_epoch": time.time(),
            "manifest_scope": manifest.get("scope"),
            "manifest_version": manifest.get("version"),
            "input_item_count": len(items),
            "erase_order": "original_room_then_hardpaste_after_erase",
            "flux2_mask_required": not using_apiyi_gpt_image2_vip,
            "flux2_mask_sent": not using_apiyi_gpt_image2_vip,
            "final_render_mask_required": not using_apiyi_gpt_image2_vip,
            "final_render_contract": (
                "gpt-image-2-vip /v1/images/edits with local bounded composite; "
                "pure-color walls use local crop/composite/QA to enforce edit scope; "
                "textured candidate walls send the full room and enforce the remote edit scope with a compressed-pixel coordinate matrix prompt"
            ) if using_apiyi_gpt_image2_vip else "masked FLUX.2 edit",
            "outputs": [],
            "errors": [],
            "warnings": ([{
                "provider": provider_name,
                "warning_label": "🔴 MOCK PROVIDER / NO-OP RESULT",
                "message": MOCK_PROVIDER_WARNING,
                "production_usable": False,
            }] if using_mock_provider else []),
            "dual_ab_compare": {
                "enabled": bool(args.dual_ab_compare),
                "requested_limit": int(args.dual_ab_compare_limit),
                "executed_count": 0,
                "note": "Compare safe dual_erased_room_plus_target against a single-input baseline before expanding dual coverage.",
            },
            "render_size_qa": {
                "method": "mask_anchored_core_ssim_and_forbidden_outer_guard",
                "bbox_redetection_used_for_decision": False,
                "decision_mode": str(getattr(args, "finalize_policy", "qa_advisory")),
                "guarantee_final_hd": bool(getattr(args, "guarantee_final_hd", True)),
            },
        }
        dual_ab_executed = 0
        for idx, item in enumerate(items):
            asset_id = _safe_name(item.get("asset_id") or f"item_{idx:03d}")
            item_dir = output_dir / asset_id
            item_dir.mkdir(parents=True, exist_ok=True)
            try:
                copied = _copy_inputs_for_trace(item, manifest, extracted, item_dir)
                hard_paste_legacy = _require_existing(_resolve_path(extracted, item.get("hard_paste_path")), label="hard_paste_path", asset_id=asset_id)
                remove_mask = _require_existing(_resolve_path(extracted, item.get("remove_mask_path")), label="remove_mask_path", asset_id=asset_id)
                art_region_mask = _require_existing(_resolve_path(extracted, item.get("art_region_mask_path")), label="art_region_mask_path", asset_id=asset_id)
                front_region_candidate = _resolve_path(extracted, item.get("front_region_mask_path"))
                if front_region_candidate is not None and front_region_candidate.exists():
                    qa_art_mask = front_region_candidate
                    qa_art_mask_source = "front_region_mask_path"
                else:
                    qa_art_mask = art_region_mask
                    qa_art_mask_source = "art_region_mask_path_legacy_fallback"

                flux2_edit_mask_candidate = _resolve_path(extracted, item.get("flux2_edit_mask_path"))
                if flux2_edit_mask_candidate is not None and flux2_edit_mask_candidate.exists():
                    flux2_edit_mask = flux2_edit_mask_candidate
                    flux2_edit_mask_source = "flux2_edit_mask_path"
                else:
                    # Backwards-compatible fallback for older manifests.  New
                    # service exports must include flux2_edit_mask_path.
                    flux2_edit_mask = qa_art_mask
                    flux2_edit_mask_source = f"missing_flux2_edit_mask_path_fallback_to_{qa_art_mask_source}"
                    aggregate["warnings"].append({
                        "asset_id": asset_id,
                        "mode": "missing_flux2_edit_mask_path",
                        "message": "Manifest did not include flux2_edit_mask_path; falling back to QA art mask so FLUX.2 is still masked.",
                    })
                render_mask_scope = _resolve_render_mask_scope(getattr(args, "render_mask_scope", None))
                render_mask, render_mask_source = _select_render_mask_path(
                    qa_art_mask=qa_art_mask,
                    art_region_mask=art_region_mask,
                    flux2_edit_mask=flux2_edit_mask,
                    scope=render_mask_scope,
                )
                mount_mask_candidate = _resolve_path(extracted, item.get("mount_mask_path"))
                if mount_mask_candidate is not None and mount_mask_candidate.exists():
                    render_object_mask = mount_mask_candidate
                    render_object_mask_source = "mount_mask_path"
                else:
                    render_object_mask = art_region_mask
                    render_object_mask_source = "art_region_mask_path"
                prompt = item.get("render_prompt")
                if not prompt and item.get("render_prompt_path"):
                    prompt = _require_existing(_resolve_path(extracted, item["render_prompt_path"]), label="render_prompt_path", asset_id=asset_id).read_text(encoding="utf-8")
                if not prompt:
                    raise ValueError(f"manifest item {asset_id} has no render prompt")

                original_room = Path(args.original_room).resolve() if args.original_room else _resolve_path(extracted, item.get("original_room_path") or manifest.get("original_room_path"))
                artwork_path = _resolve_path(extracted, item.get("artwork_path") or manifest.get("artwork_path"))
                if artwork_path is None and args.artwork_reference:
                    artwork_path = Path(args.artwork_reference).resolve()
                quad = _quad_from_item(item, extracted)

                legacy_fallback_reasons: list[str] = []
                if original_room is None:
                    legacy_fallback_reasons.append("missing_original_room_path")
                elif not original_room.exists():
                    legacy_fallback_reasons.append(f"original_room_not_found:{original_room}")
                if artwork_path is None:
                    legacy_fallback_reasons.append("missing_artwork_path")
                elif not artwork_path.exists():
                    legacy_fallback_reasons.append(f"artwork_not_found:{artwork_path}")
                if not quad:
                    legacy_fallback_reasons.append("missing_target_quad_original")

                strict_v7_item = _is_strict_v7_item(item, manifest)
                if strict_v7_item and legacy_fallback_reasons:
                    raise FileNotFoundError(
                        f"strict v7 manifest item {asset_id} cannot run erase-before-hardpaste; "
                        f"missing/invalid inputs={legacy_fallback_reasons}. "
                        "Refusing legacy hardpaste->erase fallback because it makes erase see the newly pasted artwork."
                    )

                sequence_mode = "v7_original_erase_hardpaste_flux2"
                mount_policy = item.get("artwork_mount_policy") if isinstance(item.get("artwork_mount_policy"), dict) else {}
                if not legacy_fallback_reasons:
                    erase_input = original_room
                    erase_result = cleanup_objects(
                        provider,
                        erase_input,
                        remove_mask,
                        output_dir=item_dir,
                        output_name="01_erased.png",
                        metadata={
                            "asset_id": asset_id,
                            "candidate_id": item.get("candidate_id"),
                            "wall_id": item.get("wall_id"),
                            "dilate_pixels": args.erase_dilate_pixels,
                            "erase_input_semantics": "original_room",
                        },
                    )
                    # Defensive guard: refuse to hardpaste if erase output size differs from original room.
                    with Image.open(erase_result.image) as erased_img, Image.open(original_room) as orig_img:
                        if erased_img.size != orig_img.size:
                            raise RuntimeError(
                                f"cleanup output size {erased_img.size} differs from original_room size {orig_img.size}; "
                                "refusing to hardpaste original-space quad onto resized erase canvas"
                            )
                    hard_paste_after_erase, mount_trace = _hardpaste_after_erase(
                        erase_result.image,
                        artwork_path,
                        quad,
                        item_dir / "02_hard_paste_after_erase.png",
                        mount_policy=mount_policy,
                        supersample_scale=int(args.hardpaste_supersample_scale),
                    )
                else:
                    # Backwards compatibility for older manifests. This path is
                    # intentionally marked legacy because erase sees the pasted art.
                    sequence_mode = "legacy_hardpaste_erase_flux2"
                    warning = (
                        f"WARNING legacy_hardpaste_erase_flux2 fallback for asset_id={asset_id}; "
                        f"reasons={legacy_fallback_reasons}; v7 manifests should provide "
                        "original_room_path, artwork_path and target_quad_original. "
                        "Output quality may degrade because erase sees already-pasted artwork."
                    )
                    print(warning, file=sys.stderr)
                    aggregate["warnings"].append({
                        "asset_id": asset_id,
                        "sequence_mode": sequence_mode,
                        "reasons": list(legacy_fallback_reasons),
                        "message": warning,
                    })
                    erase_input = hard_paste_legacy
                    erase_result = cleanup_objects(
                        provider,
                        erase_input,
                        remove_mask,
                        output_dir=item_dir,
                        output_name="01_erased.png",
                        metadata={
                            "asset_id": asset_id,
                            "candidate_id": item.get("candidate_id"),
                            "wall_id": item.get("wall_id"),
                            "dilate_pixels": args.erase_dilate_pixels,
                            "erase_input_semantics": "legacy_hard_paste",
                        },
                    )
                    hard_paste_after_erase = erase_result.image
                    # Defensive guard: refuse to hardpaste if erase output size differs from erase input.
                    with Image.open(erase_result.image) as erased_img, Image.open(erase_input) as inp_img:
                        if erased_img.size != inp_img.size:
                            raise RuntimeError(
                                f"cleanup output size {erased_img.size} differs from erase_input size {inp_img.size}; "
                                "refusing to hardpaste onto resized erase canvas"
                            )
                    mount_trace = {"mode": "legacy_hardpaste_erase_flux2", "policy": mount_policy}

                requested_flux2_input_mode = str(item.get("flux2_input_mode") or "single_after_erase_hardpaste")
                flux2_input_mode = requested_flux2_input_mode
                render_primary = hard_paste_after_erase
                render_refs: list[Path] = []
                if requested_flux2_input_mode in {"dual_erased_room_plus_target", "dual_original_plus_target"} and sequence_mode.startswith("v7"):
                    flux2_input_mode = "dual_erased_room_plus_target"
                    render_refs = [erase_result.image]
                    render_input_semantics = {
                        "input_image": "cleaned_hard_paste_target_authoritative_canvas",
                        "input_image_2": "erased_room_texture_lighting_reference",
                        "order_policy": "input_image_is_authoritative_target; input_image_2_is_clean_reference_only; raw_original_reference_prohibited",
                    }
                    if requested_flux2_input_mode == "dual_original_plus_target":
                        warning = f"WARNING downgraded legacy raw-original dual mode to clean-reference dual for asset_id={asset_id}"
                        print(warning, file=sys.stderr)
                        aggregate["warnings"].append({
                            "asset_id": asset_id,
                            "mode": "raw_original_dual_downgraded",
                            "message": warning,
                        })
                else:
                    flux2_input_mode = "single_after_erase_hardpaste"
                    render_input_semantics = {
                        "input_image": "cleaned_hard_paste_target_authoritative",
                        "raw_original_reference_prohibited": True,
                    }

                light_context_reference_mode = _resolve_light_context_reference_mode(provider_name)
                if (
                    using_apiyi_gpt_image2_vip
                    and light_context_reference_mode == "erased_room"
                    and sequence_mode.startswith("v7")
                    and erase_result.image not in render_refs
                ):
                    render_refs.append(erase_result.image)
                    render_input_semantics["input_image_2"] = "erased_room_full_space_environment_light_reference"
                    render_input_semantics["light_context_reference_policy"] = (
                        "use erased full-space image only for environment light direction, wall brightness, and material context; "
                        "do not remove or override the authoritative hard-paste artwork in input_image"
                    )

                render_light_hint, render_light_hint_trace = _ambient_light_hint(
                    image_path=hard_paste_after_erase,
                    edit_mask_path=render_mask,
                    object_mask_path=render_object_mask,
                    band_px=max(8, int(os.getenv("MOLINK_RENDER_LIGHT_HINT_BAND_PX", "64"))),
                    exclude_px=max(0, int(os.getenv("MOLINK_RENDER_LIGHT_HINT_EXCLUDE_PX", "8"))),
                ) if _env_truthy("MOLINK_RENDER_LIGHT_HINT_PROMPT", "1") else ("", {"enabled": False, "reason": "disabled_by_env"})
                prompt_parts = [prompt]
                if render_light_hint:
                    prompt_parts.append(render_light_hint)
                mount_policy_render = item.get("artwork_mount_policy") if isinstance(item.get("artwork_mount_policy"), dict) else {}
                visible_side_faces = mount_policy_render.get("visible_side_faces_original") if isinstance(mount_policy_render.get("visible_side_faces_original"), list) else []
                visible_side_edges = [
                    str(face.get("source_edge"))
                    for face in visible_side_faces
                    if isinstance(face, dict) and face.get("source_edge") in {"left", "right", "top", "bottom"}
                ]
                hidden_side_edges = [edge for edge in ("left", "right", "top", "bottom") if edge not in visible_side_edges]
                if mount_policy_render.get("precompose_frame_sides"):
                    vis = ", ".join(visible_side_edges) if visible_side_edges else "the already visible side faces"
                    hid = ", ".join(hidden_side_edges) if hidden_side_edges else "the remaining perimeter"
                    prompt_parts.append(
                        "FRAME-SIDE LOCK: the hard-paste target already contains the correct visible thickness faces. "
                        f"Only these side faces may be visible: {vis}. "
                        f"Do not invent frame thickness, a matte surround, or a perimeter border on hidden edges such as {hid}. "
                        "Do not draw a thin border around all four sides of the front artwork."
                    )
                prompt_for_render = "\n\n".join(part for part in prompt_parts if part)

                full_space_texture_render, wall_texture_gate_trace = _render_full_space_texture_gate(
                    image_path=hard_paste_after_erase,
                    edit_mask_path=render_mask,
                    object_mask_path=render_object_mask,
                    outer_px=max(16, int(os.getenv("MOLINK_RENDER_TEXTURE_GATE_OUTER_PX", "192"))),
                    inner_guard_px=max(0, int(os.getenv("MOLINK_RENDER_TEXTURE_GATE_INNER_GUARD_PX", "12"))),
                    min_sample_px=max(32, int(os.getenv("MOLINK_RENDER_TEXTURE_GATE_MIN_SAMPLE_PX", "512"))),
                    residual_std_threshold=max(0.1, _env_float("MOLINK_RENDER_TEXTURE_GATE_RESIDUAL_STD", 3.0)),
                    hf_mad_threshold=max(0.1, _env_float("MOLINK_RENDER_TEXTURE_GATE_HF_MAD", 1.25)),
                    edge_p95_threshold=max(0.1, _env_float("MOLINK_RENDER_TEXTURE_GATE_EDGE_P95", 5.5)),
                    chroma_std_threshold=max(0.1, _env_float("MOLINK_RENDER_TEXTURE_GATE_CHROMA_STD", 2.6)),
                )
                full_space_coordinate_prompt_requested = bool(using_apiyi_gpt_image2_vip and full_space_texture_render)
                if full_space_coordinate_prompt_requested:
                    render_input_semantics["final_render_scope"] = "full_space_image_with_remote_pixel_coordinate_matrix_prompt"
                    render_input_semantics["mask_crop_disabled_reason"] = "candidate_wall_texture_not_pure_color"
                    render_input_semantics["coordinate_space"] = "exact_compressed_api_upload_pixels"
                else:
                    render_input_semantics["final_render_scope"] = "local_mask_crop_composite_or_native_mask"

                render_front_polygon_px = item.get("front_polygon_px") or item.get("polygon_px") or quad
                render_front_bbox_px = item.get("front_bbox_px") or item.get("bbox_px")
                render_mount_bbox_px = item.get("mount_bbox_px")

                render_result = render_hanging(
                    provider,
                    render_primary,
                    prompt_for_render,
                    art_mask=render_mask,
                    references=render_refs,
                    output_dir=item_dir,
                    output_name="03_flux2_render.png" if sequence_mode.startswith("v7") else "02_flux2_render.png",
                    metadata={
                        "asset_id": asset_id,
                        "candidate_id": item.get("candidate_id"),
                        "wall_id": item.get("wall_id"),
                        **_render_edit_runtime_metadata(args),
                        "sequence_mode": sequence_mode,
                        "flux2_input_mode": flux2_input_mode,
                        "render_input_semantics": render_input_semantics,
                        "render_reference_count": len(render_refs),
                        "light_context_reference_mode": light_context_reference_mode,
                        "render_light_hint": render_light_hint_trace,
                        "wall_texture_gate": wall_texture_gate_trace,
                        "full_space_coordinate_prompt_requested": full_space_coordinate_prompt_requested,
                        "full_space_render_reason": "candidate_wall_texture_not_pure_color" if full_space_coordinate_prompt_requested else None,
                        "remote_pixel_coordinate_matrix_supported": bool(using_apiyi_gpt_image2_vip),
                        "remote_pixel_coordinate_matrix_sent": bool(full_space_coordinate_prompt_requested),
                        "front_polygon_px": render_front_polygon_px,
                        "polygon_px": render_front_polygon_px,
                        "front_bbox_px": render_front_bbox_px,
                        "bbox_px": render_front_bbox_px,
                        "mount_bbox_px": render_mount_bbox_px,
                        "target_quad_original": quad,
                        "artwork_mount_policy": item.get("artwork_mount_policy") if isinstance(item.get("artwork_mount_policy"), dict) else None,
                        "art_region_mask_usage": "front_region_mask_for_core_qa; selected_render_mask_sent_to_provider",
                        "render_mask_path": str(render_mask),
                        "render_mask_source": render_mask_source,
                        "qa_art_mask_path": str(qa_art_mask),
                        "qa_art_mask_source": qa_art_mask_source,
                        "art_region_mask_path": str(art_region_mask),
                        "mount_mask_path": str(render_object_mask),
                        "mount_mask_source": render_object_mask_source,
                        "render_object_mask_path": str(render_object_mask),
                        "render_object_mask_source": render_object_mask_source,
                        "flux2_edit_mask_path": str(flux2_edit_mask),
                        "flux2_edit_mask_source": flux2_edit_mask_source,
                        "core_photometric_adjustment_allowed": True,
                        "hardpaste_core_pixel_restore_allowed": False,
                    },
                )
                post_render_photometric_lock_trace: dict[str, Any] | None = None
                if _env_truthy("MOLINK_FINAL_PHOTOMETRIC_LOCK", "1"):
                    final_postprocess_mode_env = (
                        os.getenv("MOLINK_FINAL_POSTPROCESS_MODE")
                        or os.getenv("MOLINK_FINAL_PHOTOMETRIC_LOCK_MODE")
                        or "photometric_lock"
                    ).strip().lower().replace("-", "_")
                    if full_space_coordinate_prompt_requested:
                        final_postprocess_mode = (
                            os.getenv("MOLINK_FULLSPACE_FINAL_POSTPROCESS_MODE")
                            or "photometric_lock_no_wall_rgb"
                        ).strip().lower().replace("-", "_")
                    else:
                        final_postprocess_mode = final_postprocess_mode_env
                    if final_postprocess_mode in {"wall_rgb_only", "wall_rgb_correction_only", "render_wall_rgb_only"}:
                        locked_path = item_dir / ("03_flux2_render_wall_rgb_corrected.png" if sequence_mode.startswith("v7") else "02_flux2_render_wall_rgb_corrected.png")
                    else:
                        locked_path = item_dir / ("03_flux2_render_photometric_locked.png" if sequence_mode.startswith("v7") else "02_flux2_render_photometric_locked.png")
                    post_render_photometric_lock_trace = _apply_final_photometric_lock(
                        hard_paste=hard_paste_after_erase,
                        render=render_result.image,
                        object_mask=render_object_mask,
                        edit_mask=render_mask,
                        out_path=locked_path,
                        blur_px=max(1, int(os.getenv("MOLINK_FINAL_PHOTOMETRIC_LOCK_BLUR_PX", "18"))),
                        max_delta=max(0.0, _env_float("MOLINK_FINAL_PHOTOMETRIC_LOCK_MAX_DELTA", 32.0)),
                        strength=max(0.0, _env_float("MOLINK_FINAL_PHOTOMETRIC_LOCK_STRENGTH", 1.0)),
                        feather_px=max(0, int(os.getenv("MOLINK_FINAL_PHOTOMETRIC_LOCK_FEATHER_PX", "1"))),
                        shadow_feather_px=max(0, int(os.getenv("MOLINK_FINAL_SHADOW_FEATHER_PX", "2"))),
                        relocation_search_px=max(0, int(os.getenv("MOLINK_FINAL_RELOCATION_SEARCH_PX", "24"))),
                        relocation_min_score=max(0.0, _env_float("MOLINK_FINAL_RELOCATION_MIN_SCORE", 0.38)),
                        relocation_scale_min=_env_float("MOLINK_FINAL_RELOCATION_SCALE_MIN", 0.92),
                        relocation_scale_max=_env_float("MOLINK_FINAL_RELOCATION_SCALE_MAX", 1.08),
                        relocation_scale_steps=max(1, int(os.getenv("MOLINK_FINAL_RELOCATION_SCALE_STEPS", "17"))),
                        wall_sample_exclusion_px=max(0, int(os.getenv("MOLINK_FINAL_WALL_SAMPLE_EXCLUSION_PX", "12"))),
                        wall_sample_max_delta=max(0.0, _env_float("MOLINK_FINAL_WALL_SAMPLE_MAX_DELTA", 18.0)),
                        wall_rgb_max_delta=max(0.0, _env_float("MOLINK_FINAL_WALL_RGB_MAX_DELTA", 24.0)),
                        wall_rgb_correction_strength=max(0.0, _env_float("MOLINK_FINAL_WALL_RGB_CORRECTION_STRENGTH", 1.0)),
                        object_edge_guard_px=max(0, int(os.getenv("MOLINK_FINAL_OBJECT_EDGE_GUARD_PX", "2"))),
                        postprocess_mode=final_postprocess_mode,
                    )
                    render_result = ImageEditResult(
                        image=locked_path,
                        trace={
                            **dict(render_result.trace),
                            "post_render_photometric_lock": post_render_photometric_lock_trace,
                            "output_path_before_photometric_lock": dict(render_result.trace).get("output_path"),
                            "output_path": str(locked_path),
                        },
                    )

                primary_render_size_qa = _run_mask_anchored_qa(
                    args,
                    hard_paste=hard_paste_after_erase,
                    render=render_result.image,
                    art_mask=qa_art_mask,
                )

                dual_ab_compare_trace: dict[str, Any] | None = None
                if (
                    args.dual_ab_compare
                    and flux2_input_mode == "dual_erased_room_plus_target"
                    and sequence_mode.startswith("v7")
                    and dual_ab_executed < int(args.dual_ab_compare_limit)
                ):
                    try:
                        baseline_prompt = (
                            f"{prompt} A/B comparison baseline: use this single input image as the authoritative target; "
                            "do not use any original-room reference image for this baseline render. "
                            "Keep the visible frame/border and painting core identity-locked; only apply low-frequency achromatic light/shadow and brightness adaptation. Do not change intrinsic colors and do not edit any non-artwork region."
                        )
                        baseline_result = render_hanging(
                            provider,
                            hard_paste_after_erase,
                            baseline_prompt,
                            art_mask=render_mask,
                            references=[],
                            output_dir=item_dir,
                            output_name="03b_flux2_render_single_baseline.png",
                            metadata={
                                "asset_id": asset_id,
                                "candidate_id": item.get("candidate_id"),
                                "wall_id": item.get("wall_id"),
                                **_render_edit_runtime_metadata(args),
                                "sequence_mode": sequence_mode,
                                "flux2_input_mode": "single_after_erase_hardpaste_ab_baseline",
                                "render_input_semantics": {
                                    "input_image": "cleaned_hard_paste_target_authoritative",
                                    "ab_compare_role": "single_input_baseline_for_dual_mode_validation",
                                },
                                "art_region_mask_usage": "front_region_mask_for_core_qa; selected_render_mask_sent_to_provider",
                        "render_mask_path": str(render_mask),
                        "render_mask_source": render_mask_source,
                                "qa_art_mask_path": str(qa_art_mask),
                                "qa_art_mask_source": qa_art_mask_source,
                                "art_region_mask_path": str(art_region_mask),
                                "flux2_edit_mask_path": str(flux2_edit_mask),
                                "flux2_edit_mask_source": flux2_edit_mask_source,
                                "core_photometric_adjustment_allowed": True,
                                "hardpaste_core_pixel_restore_allowed": False,
                                "ab_compare": True,
                            },
                        )
                        baseline_final_path = output_dir / "final_hd_ab_compare" / f"{asset_id}__single_baseline.png"
                        baseline_final_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(baseline_result.image, baseline_final_path)
                        dual_ab_executed += 1
                        aggregate["dual_ab_compare"]["executed_count"] = dual_ab_executed
                        dual_ab_compare_trace = {
                            "enabled": True,
                            "purpose": "validate whether safe dual_erased_room_plus_target improves or degrades the same candidate against a single-input baseline",
                            "single_baseline_path": str(baseline_result.image),
                            "single_baseline_final_path": str(baseline_final_path),
                            "single_baseline_trace": baseline_result.trace,
                            "dual_primary_path": str(render_result.image),
                        }
                    except Exception as ab_exc:
                        warning = f"WARNING dual_ab_compare baseline failed for asset_id={asset_id}: {ab_exc}"
                        print(warning, file=sys.stderr)
                        aggregate["warnings"].append({
                            "asset_id": asset_id,
                            "mode": "dual_ab_compare",
                            "message": warning,
                        })
                        dual_ab_compare_trace = {
                            "enabled": True,
                            "error": str(ab_exc),
                            "dual_primary_path": str(render_result.image),
                        }

                selected_render_result = render_result
                selected_render_size_qa = primary_render_size_qa
                render_fallback_trace: dict[str, Any] | None = None
                if not bool(primary_render_size_qa.get("passed")) and flux2_input_mode == "dual_erased_room_plus_target":
                    fallback_prompt = (
                        f"{prompt} SAFE FALLBACK: use this single cleaned hard-paste input as the only authoritative canvas. "
                        "Do not use any reference image. Preserve the exact painting-core pixel footprint, preserve the visible frame/border appearance, and allow only low-frequency achromatic light/shadow and brightness adaptation. Do not change intrinsic colors and do not edit any non-artwork region."
                    )
                    fallback_result = render_hanging(
                        provider,
                        hard_paste_after_erase,
                        fallback_prompt,
                        art_mask=render_mask,
                        references=[],
                        output_dir=item_dir,
                        output_name="03c_flux2_render_single_fallback.png",
                        metadata={
                            "asset_id": asset_id,
                            "candidate_id": item.get("candidate_id"),
                            "wall_id": item.get("wall_id"),
                            **_render_edit_runtime_metadata(args),
                            "sequence_mode": sequence_mode,
                            "flux2_input_mode": "single_after_erase_hardpaste_size_qa_fallback",
                            "art_region_mask_usage": "front_region_mask_for_core_qa; selected_render_mask_sent_to_provider",
                        "render_mask_path": str(render_mask),
                        "render_mask_source": render_mask_source,
                            "qa_art_mask_path": str(qa_art_mask),
                            "qa_art_mask_source": qa_art_mask_source,
                            "art_region_mask_path": str(art_region_mask),
                            "flux2_edit_mask_path": str(flux2_edit_mask),
                            "flux2_edit_mask_source": flux2_edit_mask_source,
                            "core_photometric_adjustment_allowed": True,
                            "hardpaste_core_pixel_restore_allowed": False,
                            "size_qa_fallback": True,
                        },
                    )
                    fallback_size_qa = _run_mask_anchored_qa(
                        args,
                        hard_paste=hard_paste_after_erase,
                        render=fallback_result.image,
                        art_mask=qa_art_mask,
                    )
                    render_fallback_trace = {
                        "trigger": "primary_render_size_qa_failed",
                        "mode": "single_after_erase_hardpaste",
                        "render_path": str(fallback_result.image),
                        "render_trace": fallback_result.trace,
                        "render_size_qa": fallback_size_qa,
                    }
                    if bool(fallback_size_qa.get("passed")):
                        selected_render_result = fallback_result
                        selected_render_size_qa = fallback_size_qa
                    elif _qa_quality_score(fallback_size_qa) > _qa_quality_score(selected_render_size_qa):
                        selected_render_result = fallback_result
                        selected_render_size_qa = fallback_size_qa
                        render_fallback_trace["selected_despite_qa_failure"] = True

                hardpaste_preview_path: Path | None = None
                final_path: Path | None = None
                qa_passed = bool(selected_render_size_qa.get("passed"))
                qa_hard_block_reasons = _qa_hard_block_reasons(selected_render_size_qa, args)
                qa_hard_blocked = bool(qa_hard_block_reasons)
                finalize_policy = str(getattr(args, "finalize_policy", "qa_advisory") or "qa_advisory").strip().lower()
                guarantee_final_hd = bool(getattr(args, "guarantee_final_hd", True))
                selected_flux_file_ok = _is_nonempty_image_file(selected_render_result.image)
                selected_trace = dict(selected_render_result.trace or {})
                selected_mask_sent = bool(selected_trace.get("mask_sent_to_flux2"))
                selected_apiyi_gpt_image2_vip = bool(selected_trace.get("apiyi_images_edit_request")) and str(selected_trace.get("provider")) == "apiyi_gpt_image2_vip"
                selected_remote_native_mask_sent = bool(selected_mask_sent and not selected_apiyi_gpt_image2_vip and not using_mock_provider)
                selected_mask_enforced_locally = bool(selected_trace.get("mask_enforced_locally") or selected_trace.get("mask_local_composite"))
                selected_remote_coordinate_matrix = bool(selected_trace.get("remote_coordinate_matrix_sent") or selected_trace.get("full_space_coordinate_prompt"))
                selected_edit_scope_ok = selected_mask_sent or (selected_apiyi_gpt_image2_vip and (selected_mask_enforced_locally or selected_remote_coordinate_matrix))
                selected_edit_scope = (
                    "mask_sent"
                    if selected_mask_sent
                    else (
                        "gpt_image2_vip_mask_photometric_composite"
                        if bool(selected_trace.get("mask_photometric_composite"))
                        else (
                            "gpt_image2_vip_full_space_coordinate_prompt"
                            if selected_remote_coordinate_matrix
                            else ("gpt_image2_vip_mask_local_composite" if selected_mask_enforced_locally else "unmasked_unknown")
                        )
                    )
                )
                render_failure_diagnosis = _render_failure_diagnosis(
                    qa=selected_render_size_qa,
                    render_trace=selected_trace,
                    using_apiyi_gpt_image2_vip=selected_apiyi_gpt_image2_vip,
                    flux2_input_mode=flux2_input_mode,
                )
                allow_qa_advisory_final = finalize_policy in {"qa_advisory", "lax", "always_final"}
                should_finalize_flux = (
                    selected_edit_scope_ok
                    and not qa_hard_blocked
                    and (qa_passed or (guarantee_final_hd and allow_qa_advisory_final and selected_flux_file_ok))
                )

                finalization_trace: dict[str, Any] = {
                    "policy": finalize_policy,
                    "raw_flux_passed": qa_passed,
                    "qa_hard_blocked": qa_hard_blocked,
                    "qa_hard_block_reasons": qa_hard_block_reasons,
                    "qa_hard_thresholds": {
                        "min_core_masked_ssim": float(args.qa_hard_min_core_masked_ssim),
                        "max_core_color_delta": float(args.qa_hard_max_core_color_delta),
                        "max_forbidden_outer_guard_changed_ratio": float(args.qa_hard_max_forbidden_outer_guard_changed_ratio),
                    },
                    "guarantee_final_hd": guarantee_final_hd,
                    "selected_flux_file_ok": selected_flux_file_ok,
                    "selected_flux2_mask_sent": selected_remote_native_mask_sent,
                    "selected_apiyi_gpt_image2_vip": selected_apiyi_gpt_image2_vip,
                    "selected_mask_enforced_locally": selected_mask_enforced_locally,
                    "selected_remote_coordinate_matrix": selected_remote_coordinate_matrix,
                    "selected_edit_scope": selected_edit_scope,
                    "selected_edit_scope_ok": selected_edit_scope_ok,
                    "never_silent_hardpaste_success": True,
                    "hardpaste_delivery_allowed": False,
                    "render_failure_diagnosis": render_failure_diagnosis,
                }
                if should_finalize_flux:
                    if qa_passed:
                        item_exit_code = "OK_SELECTED_SINGLE_FALLBACK" if render_fallback_trace and selected_render_result is not render_result else "OK_SELECTED"
                    else:
                        item_exit_code = "OK_SELECTED_QA_ADVISORY_ACCEPTED"
                        aggregate["warnings"].append({
                            "asset_id": asset_id,
                            "mode": "flux_qa_advisory_accepted",
                            "message": (
                                f"Final render QA did not pass for {asset_id}, but a non-empty final render was produced under the configured edit scope; "
                                "QA is advisory and the render was promoted to final_hd. Severe artwork-identity QA remains a hard gate."
                            ),
                            "qa": selected_render_size_qa,
                        })
                    final_path = final_dir / f"{asset_id}.png"
                    shutil.copyfile(selected_render_result.image, final_path)
                    production_usable = not using_mock_provider
                else:
                    partial_dir = output_dir / "partial_review"
                    partial_dir.mkdir(parents=True, exist_ok=True)
                    preview_suffix = "frame_side_hardpaste_preview" if bool(mount_policy.get("precompose_frame_sides")) else "hardpaste_preview"
                    hardpaste_preview_path = partial_dir / f"{asset_id}__{preview_suffix}.png"
                    shutil.copyfile(hard_paste_after_erase, hardpaste_preview_path)
                    item_exit_code = "PARTIAL_HARDPASTE_REVIEW"
                    production_usable = False
                    aggregate["warnings"].append({
                        "asset_id": asset_id,
                        "mode": "final_render_unavailable_or_strict_gate",
                        "message": f"No deliverable final render was available for {asset_id}; hardpaste preview saved for manual review",
                        "qa": selected_render_size_qa,
                        "qa_hard_block_reasons": qa_hard_block_reasons,
                        "selected_flux2_mask_sent": selected_remote_native_mask_sent,
                        "render_failure_diagnosis": render_failure_diagnosis,
                    })
                finalization_trace.update({
                    "final_path": str(final_path) if final_path is not None else None,
                    "production_usable": bool(production_usable),
                })

                item_trace = {
                    "asset_id": asset_id,
                    "candidate_id": item.get("candidate_id"),
                    "wall_id": item.get("wall_id"),
                    "score": item.get("score"),
                    "risk_level": item.get("risk_level"),
                    "sequence_mode": sequence_mode,
                    "legacy_fallback_reasons": list(legacy_fallback_reasons),
                    "erase_input": str(erase_input),
                    "hard_paste_after_erase": str(hard_paste_after_erase),
                    "mount_trace": mount_trace,
                    "requested_flux2_input_mode": requested_flux2_input_mode,
                    "flux2_input_mode": flux2_input_mode,
                    "render_input_semantics": render_input_semantics,
                    "render_reference_count": len(render_refs),
                    "light_context_reference_mode": locals().get("light_context_reference_mode"),
                    "render_light_hint": locals().get("render_light_hint_trace"),
                    "wall_texture_gate": locals().get("wall_texture_gate_trace"),
                    "full_space_coordinate_prompt_requested": locals().get("full_space_coordinate_prompt_requested"),
                    "remote_pixel_coordinate_matrix_supported": bool(using_apiyi_gpt_image2_vip),
                    "remote_pixel_coordinate_matrix_sent": bool(selected_remote_coordinate_matrix),
                    "flux2_mask_sent": selected_remote_native_mask_sent,
                    "final_render_edit_scope": selected_edit_scope,
                    "selected_apiyi_gpt_image2_vip": selected_apiyi_gpt_image2_vip,
                    "flux2_edit_mask_path": str(flux2_edit_mask),
                    "flux2_edit_mask_source": flux2_edit_mask_source,
                    "art_region_mask_usage": "front_region_mask_for_core_qa; selected_render_mask_sent_to_provider",
                    "render_mask_path": str(render_mask),
                    "render_mask_source": render_mask_source,
                    "qa_art_mask_path": str(qa_art_mask),
                    "qa_art_mask_source": qa_art_mask_source,
                    "art_region_mask_path": str(art_region_mask),
                    "mount_mask_path": str(render_object_mask),
                    "mount_mask_source": render_object_mask_source,
                    "render_object_mask_path": str(render_object_mask),
                    "render_object_mask_source": render_object_mask_source,
                    "render_size_qa_status": selected_render_size_qa.get("status"),
                    "primary_render_size_qa": primary_render_size_qa,
                    "selected_render_size_qa": selected_render_size_qa,
                    "post_flux_finalization": finalization_trace,
                    "post_render_photometric_lock": post_render_photometric_lock_trace,
                    "render_failure_diagnosis": render_failure_diagnosis,
                    "render_fallback": render_fallback_trace,
                    "exit_code": item_exit_code,
                    "production_usable": bool(production_usable),
                    "input_files": copied,
                    "erase_trace": erase_result.trace,
                    "render_trace": render_result.trace,
                    "selected_render_trace": selected_render_result.trace,
                    "selected_render_path": str(selected_render_result.image),
                    "mock_provider_warning": MOCK_PROVIDER_WARNING if using_mock_provider else None,
                    "final_path": str(final_path) if final_path is not None else None,
                    "hardpaste_preview_path": str(hardpaste_preview_path) if hardpaste_preview_path is not None else None,
                    "exit_review": item.get("exit_review"),
                    "manifest_selection": item.get("manifest_selection"),
                    "dual_ab_compare": dual_ab_compare_trace,
                }
                (item_dir / "edit_trace.json").write_text(json.dumps(item_trace, ensure_ascii=False, indent=2), encoding="utf-8")
                aggregate["outputs"].append(item_trace)
            except Exception as exc:
                error = {"asset_id": asset_id, "error": str(exc)}
                aggregate["errors"].append(error)
                if not args.continue_on_error:
                    raise
        if using_mock_provider:
            print(_red(MOCK_PROVIDER_WARNING), file=sys.stderr)
        aggregate["finished_at_epoch"] = time.time()
        aggregate["elapsed_s"] = round(aggregate["finished_at_epoch"] - aggregate["started_at_epoch"], 3)
        aggregate["exit_code_counts"] = {}
        for output in aggregate["outputs"]:
            code = str(output.get("exit_code") or "UNKNOWN")
            aggregate["exit_code_counts"][code] = int(aggregate["exit_code_counts"].get(code, 0)) + 1
        aggregate["production_usable_output_count"] = sum(1 for output in aggregate["outputs"] if output.get("production_usable"))
        (output_dir / "edit_trace.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
        return aggregate


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cleanup + final image render for approved hanging assets.")
    parser.add_argument("--hanging-zip", required=True, help="Zip returned by /analyze_hanging_mvp")
    parser.add_argument("--original-room", default=None, help="Optional original room image override for legacy manifests")
    parser.add_argument("--artwork-reference", default=None, help="Original artwork image override for legacy manifests")
    parser.add_argument("--output-dir", required=True, help="Directory for final_hd images and traces")
    parser.add_argument("--provider", default="apiyi_gpt_image2_vip", choices=["apiyi_gpt_image2_vip", "apiyi", "gpt_image2_vip", "gpt-image-2-vip", "bfl", "mock"], help="Use apiyi_gpt_image2_vip for GPT Image 2 VIP final render, bfl for original FLUX.2 render, mock for dry run")
    parser.add_argument("--erase-dilate-pixels", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--safety-tolerance", type=int, default=None)
    parser.add_argument("--render-prompt-upsampling", dest="render_prompt_upsampling", action="store_true", default=None, help="Enable provider-side prompt upsampling for final render; disabled by default to reduce creative drift.")
    parser.add_argument("--no-render-prompt-upsampling", dest="render_prompt_upsampling", action="store_false", help="Disable provider-side prompt upsampling for final render.")
    parser.add_argument("--render-mask-scope", default=os.getenv("MOLINK_RENDER_MASK_SCOPE", "edit_region"), choices=["art_only", "front_only", "edit_region"], help="Final render mask scope: art_only forbids edits outside the mounted artwork object; front_only restricts to the visible front face; edit_region keeps the legacy wider halo.")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N approved items after manifest selection")
    parser.add_argument("--dual-ab-compare", action="store_true", help="For safe dual_erased_room_plus_target items, also render a single-input baseline for validation.")
    parser.add_argument("--dual-ab-compare-limit", type=int, default=1, help="Maximum safe dual-mode items to render with the extra single-input A/B baseline.")
    parser.add_argument("--qa-core-erode-px", type=int, default=int(os.getenv("MOLINK_FLUX_QA_CORE_ERODE_PX", "1")))
    parser.add_argument("--qa-allowed-frame-shadow-px", type=int, default=int(os.getenv("MOLINK_FLUX_QA_ALLOWED_FRAME_SHADOW_PX", "16")))
    parser.add_argument("--qa-forbidden-outer-guard-px", type=int, default=int(os.getenv("MOLINK_FLUX_QA_FORBIDDEN_OUTER_GUARD_PX", "8")))
    parser.add_argument("--qa-changed-pixel-delta", type=float, default=float(os.getenv("MOLINK_FLUX_QA_CHANGED_PIXEL_DELTA", "30")))
    parser.add_argument("--qa-min-core-masked-ssim", type=float, default=float(os.getenv("MOLINK_FLUX_QA_MIN_CORE_MASKED_SSIM", "0.78")))
    parser.add_argument("--qa-max-core-color-delta", type=float, default=float(os.getenv("MOLINK_FLUX_QA_MAX_CORE_COLOR_DELTA", "45")))
    parser.add_argument("--qa-max-forbidden-outer-guard-changed-ratio", type=float, default=float(os.getenv("MOLINK_FLUX_QA_MAX_FORBIDDEN_OUTER_GUARD_CHANGED_RATIO", "0.65")))
    parser.add_argument("--qa-ignore-shadow-in-forbidden-guard", dest="qa_ignore_shadow_in_forbidden_guard", action="store_true", default=_env_truthy("MOLINK_FLUX_QA_IGNORE_GENERATED_SHADOW", "1"), help="Do not fail QA for neutral darken-only projection/shadow pixels in the outer guard band.")
    parser.add_argument("--qa-no-ignore-shadow-in-forbidden-guard", dest="qa_ignore_shadow_in_forbidden_guard", action="store_false")
    parser.add_argument("--qa-shadow-min-darkening-delta", type=float, default=float(os.getenv("MOLINK_FLUX_QA_SHADOW_MIN_DARKENING_DELTA", "4")))
    parser.add_argument("--qa-shadow-max-brighten-delta", type=float, default=float(os.getenv("MOLINK_FLUX_QA_SHADOW_MAX_BRIGHTEN_DELTA", "3")))
    parser.add_argument("--qa-shadow-max-chroma-delta", type=float, default=float(os.getenv("MOLINK_FLUX_QA_SHADOW_MAX_CHROMA_DELTA", "6")))
    parser.add_argument("--qa-hard-min-core-masked-ssim", type=float, default=float(os.getenv("MOLINK_FLUX_QA_HARD_MIN_CORE_MASKED_SSIM", "0.45")), help="Hard delivery gate: below this artwork-core SSIM, a Flux render cannot enter final_hd even under qa_advisory.")
    parser.add_argument("--qa-hard-max-core-color-delta", type=float, default=float(os.getenv("MOLINK_FLUX_QA_HARD_MAX_CORE_COLOR_DELTA", "90")), help="Hard delivery gate: above this artwork-core color delta, a Flux render cannot enter final_hd even under qa_advisory.")
    parser.add_argument("--qa-hard-max-forbidden-outer-guard-changed-ratio", type=float, default=float(os.getenv("MOLINK_FLUX_QA_HARD_MAX_FORBIDDEN_OUTER_GUARD_CHANGED_RATIO", "0.85")), help="Hard delivery gate for severe outer-region change around the target artwork.")
    parser.add_argument("--finalize-policy", default=os.getenv("MOLINK_FLUX_FINALIZE_POLICY", "qa_advisory"), choices=["qa_gate_only", "qa_advisory", "lax", "always_final"], help="qa_advisory keeps QA metrics but allows non-empty Flux renders into final_hd.")
    parser.add_argument("--guarantee-final-hd", dest="guarantee_final_hd", action="store_true", default=_env_truthy("MOLINK_FLUX_GUARANTEE_FINAL_HD", "1"), help="Promote a non-empty Flux render to final_hd even when advisory QA fails.")
    parser.add_argument("--no-guarantee-final-hd", dest="guarantee_final_hd", action="store_false")
    parser.add_argument("--hardpaste-supersample-scale", type=int, default=4, help="Supersampling scale for hard-paste after erase; improves small artwork anti-aliasing without rejecting small projections.")
    parser.add_argument("--size-locked-transition-feather-px", type=int, default=2, help="Feather width for size-locked composite edge band.")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    trace = run_pipeline(parse_args())
    print(json.dumps({"output_count": len(trace["outputs"]), "error_count": len(trace["errors"]), "trace": "edit_trace.json"}, ensure_ascii=False, indent=2))
