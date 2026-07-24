"""worker/render_api.py —— 把离线脚本 run_bfl_hanging_flux_pipeline.run_pipeline
封装为 worker 内可调用函数，复用既有 QA 与 final_hd/partial_review 语义。

通过 bfl.parse_args([]) 取脚本自身的 argparse 默认值，
而非手工镜像——脚本新增参数时无需同步修改此处。
"""
from __future__ import annotations

import os
import json
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Mapping

import cv2
import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import run_bfl_hanging_flux_pipeline as bfl  # noqa: E402
from hanging_mvp.flux_export import build_stage_b_prompt  # noqa: E402
from hanging_mvp.render_qa import enforce_protected_artwork_composite, masked_global_ssim  # noqa: E402
from image_editing.pipeline import render_hanging  # noqa: E402
from image_editing.provider_registry import build_image_edit_provider  # noqa: E402
from semantic_mvp.vlm_client import parse_styling_analysis_with_vlm  # noqa: E402


def normalize_render_provider(value: str | None) -> str:
    key = str(value or os.getenv("HANGING_RENDER_PROVIDER") or "apiyi_gpt_image2_vip").strip().lower()
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
    normalized = aliases.get(key)
    if normalized:
        return normalized
    print(f"[worker/render_api] unknown render provider={key!r}; fallback to apiyi_gpt_image2_vip", flush=True)
    return "apiyi_gpt_image2_vip"


def _call_bfl_parse_args(argv):
    """兼容 bfl.parse_args(argv) 和 bfl.parse_args() 两种签名。"""
    import inspect
    import sys

    argv = list(argv or [])

    try:
        sig = inspect.signature(bfl.parse_args)
        can_take_argv = len(sig.parameters) >= 1
    except Exception:
        can_take_argv = False

    if can_take_argv:
        return bfl.parse_args(argv)

    old_argv = sys.argv[:]
    try:
        sys.argv = [getattr(bfl, "__file__", "run_bfl_hanging_flux_pipeline.py")] + argv
        return bfl.parse_args()
    finally:
        sys.argv = old_argv


def _default_args():
    """获取脚本 argparse 默认值；用占位 required 参数避免读取真实 sys.argv。"""
    import argparse

    try:
        return _call_bfl_parse_args([
            "--hanging-zip", "__dummy_hanging__.zip",
            "--output-dir", "__dummy_output__",
        ])
    except (SystemExit, Exception) as exc:
        print(f"[worker/render_api] parse_args fallback: {type(exc).__name__}: {exc}", flush=True)

    # 极端兜底：保持与 run_bfl_hanging_flux_pipeline.py 当前默认参数一致
    return argparse.Namespace(
        hanging_zip="",
        original_room=None,
        artwork_reference=None,
        output_dir="",
        provider="apiyi_gpt_image2_vip",
        erase_dilate_pixels=10,
        seed=None,
        safety_tolerance=None,
        limit=None,
        wall_ids=None,
        candidate_ids=None,
        dual_ab_compare=False,
        dual_ab_compare_limit=1,
        qa_core_erode_px=1,
        qa_allowed_frame_shadow_px=16,
        qa_forbidden_outer_guard_px=8,
        qa_changed_pixel_delta=30.0,
        qa_min_core_masked_ssim=0.78,
        qa_max_core_color_delta=45.0,
        qa_max_forbidden_outer_guard_changed_ratio=0.65,
        hardpaste_supersample_scale=4,
        size_locked_transition_feather_px=2,
        continue_on_error=False,
        finalize_policy="qa_advisory",
        guarantee_final_hd=True,
        render_prompt_upsampling=None,
        render_mask_scope=os.getenv("MOLINK_RENDER_MASK_SCOPE", "edit_region"),
    )
def run_pipeline(result_zip: str, output_dir: str, provider: str = "apiyi_gpt_image2_vip", limit: int | None = None, target_wall_ids: list[str] | None = None, target_candidate_ids: list[str] | None = None) -> Dict[str, Any]:
    args = _default_args()
    # 调用者覆盖：必要字段 + worker 层有意与脚本默认不同的字段
    args.hanging_zip = str(result_zip)
    args.output_dir = str(output_dir)
    args.provider = normalize_render_provider(provider)
    # worker 层容错：单张失败继续处理其余张，而非终止整个批次。
    # QA 仅作为质量诊断，不再把有效 Flux 输出挡在 final_hd 外。
    args.continue_on_error = True
    args.finalize_policy = "qa_advisory"
    args.guarantee_final_hd = True
    args.qa_core_erode_px = int(os.getenv("MOLINK_FLUX_QA_CORE_ERODE_PX", "1"))
    args.qa_allowed_frame_shadow_px = int(os.getenv("MOLINK_FLUX_QA_ALLOWED_FRAME_SHADOW_PX", "16"))
    args.qa_forbidden_outer_guard_px = int(os.getenv("MOLINK_FLUX_QA_FORBIDDEN_OUTER_GUARD_PX", "8"))
    args.qa_changed_pixel_delta = float(os.getenv("MOLINK_FLUX_QA_CHANGED_PIXEL_DELTA", "30"))
    args.qa_min_core_masked_ssim = float(os.getenv("MOLINK_FLUX_QA_MIN_CORE_MASKED_SSIM", "0.78"))
    args.qa_max_core_color_delta = float(os.getenv("MOLINK_FLUX_QA_MAX_CORE_COLOR_DELTA", "45"))
    args.qa_max_forbidden_outer_guard_changed_ratio = float(os.getenv("MOLINK_FLUX_QA_MAX_FORBIDDEN_OUTER_GUARD_CHANGED_RATIO", "0.65"))
    if limit is not None:
        args.limit = limit
    # 用户等待页选择非首选墙后的补渲染过滤条件。
    # 只影响 render manifest 的 item 选择，不改变 Metric3D/Semantic/Hanging 时序。
    if target_wall_ids:
        args.wall_ids = [str(x) for x in target_wall_ids if str(x).strip()]
    if target_candidate_ids:
        args.candidate_ids = [str(x) for x in target_candidate_ids if str(x).strip()]
    return bfl.run_pipeline(args)


def normalize_wallpaper_recolor(value: Any) -> dict[str, Any] | None:
    """Normalize the phase-2 wallpaper input without making a recommendation."""
    if not isinstance(value, Mapping) or not bool(value.get("enabled")):
        return None
    raw_tone = value.get("tone_rgb")
    if not isinstance(raw_tone, (list, tuple)) or len(raw_tone) < 3:
        raw_tone = [58, 62, 71]
    try:
        tone = [max(0, min(255, int(round(float(v))))) for v in list(raw_tone)[:3]]
    except (TypeError, ValueError):
        tone = [58, 62, 71]
    return {"enabled": True, "tone_rgb": tone}


def _load_required_stage_b_zone(path: Path | None, size: tuple[int, int], label: str) -> np.ndarray:
    width, height = size
    if path is None:
        raise FileNotFoundError(f"Stage B requires {label}, but the manifest has no path")
    if not path.is_file():
        raise FileNotFoundError(f"Stage B {label} does not exist: {path}")
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L").resize((width, height), Image.Resampling.NEAREST), dtype=np.uint8)
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    if not np.any(mask):
        raise ValueError(f"Stage B {label} is empty: {path}")
    return mask


def _load_optional_stage_b_zone(path: Path | None, size: tuple[int, int]) -> np.ndarray | None:
    if path is None or not path.is_file():
        return None
    width, height = size
    try:
        with Image.open(path) as image:
            mask = np.asarray(image.convert("L").resize((width, height), Image.Resampling.NEAREST), dtype=np.uint8)
    except Exception:
        return None
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    return mask if np.any(mask) else None


def _stage_b_anchor_from_manifest(
    *,
    extracted: Path,
    item: Mapping[str, Any],
    size: tuple[int, int],
    item_dir: Path,
) -> tuple[Path, np.ndarray, str]:
    """Build the only local Stage B protection: the artwork front anchor.

    Prefer the exact front-region mask.  Legacy archives may only contain the
    old dilated Stage-B protect mask, so polygon/bbox reconstruction is tried
    before falling back to that broad mask.
    """
    width, height = size
    candidates = [
        ("front_region_mask", item.get("front_region_mask_path")),
        ("art_region_mask", item.get("art_region_mask_path")),
    ]
    for source, rel in candidates:
        path = _resolve_extracted_path(extracted, rel)
        mask = _load_optional_stage_b_zone(path, size)
        if mask is not None:
            break
    else:
        source = "front_polygon"
        mask = np.zeros((height, width), dtype=np.uint8)
        polygon = item.get("front_polygon_px") or item.get("polygon_px") or item.get("target_quad_original")
        try:
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            if points.shape[0] >= 3 and np.all(np.isfinite(points)):
                cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 255)
        except Exception:
            pass
        if not np.any(mask):
            source = "front_bbox"
            bbox = item.get("front_bbox_px") or item.get("bbox_px")
            try:
                x0, y0, x1, y1 = [int(round(float(v))) for v in list(bbox)[:4]]
                x0, x1 = sorted((max(0, x0), min(width, x1)))
                y0, y1 = sorted((max(0, y0), min(height, y1)))
                if x1 > x0 and y1 > y0:
                    mask[y0:y1, x0:x1] = 255
            except Exception:
                pass
        if not np.any(mask):
            source = "legacy_stage_b_protect_zone"
            legacy_path = _resolve_extracted_path(extracted, item.get("stage_b_protect_zone_mask_path"))
            legacy = _load_optional_stage_b_zone(legacy_path, size)
            if legacy is None:
                raise FileNotFoundError("Stage B requires an artwork front anchor mask or polygon")
            mask = legacy

    erode_px = max(0, int(os.getenv("MOLINK_STAGE_B_ARTWORK_ANCHOR_ERODE_PX", "0")))
    if erode_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        eroded = cv2.erode(mask, kernel)
        if np.any(eroded):
            mask = eroded
            source = f"{source}_eroded_{erode_px}px"

    item_dir.mkdir(parents=True, exist_ok=True)
    anchor_path = item_dir / "stage_b_artwork_anchor_mask.png"
    Image.fromarray(mask, mode="L").save(anchor_path)
    return anchor_path, mask, source


def _masked_luma_gradient_metrics(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """Compare wall illumination/texture gradients while allowing a new base color."""
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    luma_a = np.tensordot(np.asarray(a, dtype=np.float32), weights, axes=([-1], [0]))
    luma_b = np.tensordot(np.asarray(b, dtype=np.float32), weights, axes=([-1], [0]))
    selected = np.asarray(mask, dtype=bool)
    valid_x = selected[:, 1:] & selected[:, :-1]
    valid_y = selected[1:, :] & selected[:-1, :]
    gradients_a = np.concatenate([
        (luma_a[:, 1:] - luma_a[:, :-1])[valid_x],
        (luma_a[1:, :] - luma_a[:-1, :])[valid_y],
    ])
    gradients_b = np.concatenate([
        (luma_b[:, 1:] - luma_b[:, :-1])[valid_x],
        (luma_b[1:, :] - luma_b[:-1, :])[valid_y],
    ])
    if gradients_a.size < 32:
        return {
            "sample_count": int(gradients_a.size),
            "gradient_correlation": 1.0,
            "base_gradient_rms": 0.0,
            "styled_gradient_rms": 0.0,
            "gradient_energy_ratio": 1.0,
            "flat_base_wall": True,
        }
    base_rms = float(np.sqrt(np.mean(np.square(gradients_a))))
    styled_rms = float(np.sqrt(np.mean(np.square(gradients_b))))
    denominator = float(np.linalg.norm(gradients_a) * np.linalg.norm(gradients_b))
    correlation = float(np.dot(gradients_a, gradients_b) / denominator) if denominator > 1e-8 else (1.0 if styled_rms < 0.75 else 0.0)
    return {
        "sample_count": int(gradients_a.size),
        "gradient_correlation": float(max(-1.0, min(1.0, correlation))),
        "base_gradient_rms": base_rms,
        "styled_gradient_rms": styled_rms,
        "gradient_energy_ratio": float(styled_rms / max(base_rms, 0.5)),
        "flat_base_wall": bool(base_rms < 0.05),
    }


def _masked_stage_b_composite_and_qa(
    stage_a_path: Path,
    generated_path: Path,
    protect_mask_path: Path,
    output_path: Path,
    *,
    decor_zone: np.ndarray,
    wall_zone: np.ndarray,
    soft_furnishing_requested: bool,
    wallpaper_enabled: bool,
    wallpaper_target_rgb: list[int] | tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    """Protect only the artwork after an unrestricted full-frame styling call.

    ``decor_zone`` and ``wall_zone`` are diagnostic/goal-evaluation regions;
    they are never used to crop, mask, or composite the provider output.  This
    lets the model place accessories in physically plausible volumes beyond the
    original support segmentation while the mounted artwork remains immutable.
    """
    with Image.open(stage_a_path) as base_image, Image.open(generated_path) as generated_image:
        base_size = tuple(base_image.size)
        generated_size = tuple(generated_image.size)
    if generated_size != base_size:
        return {
            "passed": False,
            "reason": "provider_output_size_mismatch",
            "stage_a_size": list(base_size),
            "generated_size": list(generated_size),
        }

    protection_trace = enforce_protected_artwork_composite(
        authoritative_image_path=stage_a_path,
        provider_render_path=generated_path,
        protect_mask_path=protect_mask_path,
        output_path=output_path,
        feather_inside_px=0.0,
    )
    with Image.open(stage_a_path) as base_image, Image.open(output_path) as output_image, Image.open(protect_mask_path) as protect_image:
        base = base_image.convert("RGB")
        composited = output_image.convert("RGB")
        protect = protect_image.convert("L").resize(base.size, Image.Resampling.NEAREST)

    base_arr = np.asarray(base, dtype=np.int16)
    out_arr = np.asarray(composited, dtype=np.int16)
    protect_arr = np.asarray(protect, dtype=np.uint8) > 0
    qa_anchor_guard_px = max(0, int(os.getenv("MOLINK_STAGE_B_QA_ARTWORK_GUARD_PX", "8")))
    if qa_anchor_guard_px > 0:
        guard_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * qa_anchor_guard_px + 1, 2 * qa_anchor_guard_px + 1),
        )
        artwork_qa_guard = cv2.dilate(protect_arr.astype(np.uint8), guard_kernel) > 0
    else:
        artwork_qa_guard = protect_arr.copy()
    changed = np.any(base_arr != out_arr, axis=2)
    delta = np.mean(np.abs(out_arr - base_arr), axis=2)
    protect_changed = int(np.count_nonzero(changed & protect_arr))
    failures: list[str] = []
    content_quality: dict[str, Any] = {}

    # Whole-frame guard: creativity is welcome, a complete unrelated rewrite is
    # not.  This gate is intentionally broad and does not impose a spatial mask.
    scene_eval = ~protect_arr
    scene_area = int(np.count_nonzero(scene_eval))
    scene_changed_delta = float(os.getenv("MOLINK_STAGE_B_QA_SCENE_CHANGED_PIXEL_DELTA", "18"))
    scene_changed_ratio = float(np.mean(delta[scene_eval] > scene_changed_delta)) if scene_area else 0.0
    scene_ssim = masked_global_ssim(base_arr, out_arr, scene_eval) if scene_area >= 16 else 0.0
    min_scene_ssim = float(os.getenv("MOLINK_STAGE_B_QA_MIN_SCENE_SSIM", "0.22"))
    min_scene_changed_ratio_for_ssim = float(
        os.getenv("MOLINK_STAGE_B_QA_MIN_SCENE_CHANGED_RATIO_FOR_SSIM", "0.20")
    )
    max_scene_changed_ratio = float(os.getenv("MOLINK_STAGE_B_QA_MAX_SCENE_CHANGED_RATIO", "0.88"))
    if scene_area < 16:
        failures.append("scene_qa_region_too_small")
    else:
        # Global SSIM is unstable on nearly flat rooms when a small accessory
        # changes a low-variance patch.  Use it only after a material fraction
        # of the full scene changed; tiny localized edits are governed by the
        # explicit changed-ratio and goal-effect gates below.
        if scene_changed_ratio >= min_scene_changed_ratio_for_ssim and scene_ssim < min_scene_ssim:
            failures.append("full_scene_identity_below_threshold")
        if scene_changed_ratio > max_scene_changed_ratio and scene_ssim < 0.55:
            failures.append("full_scene_changed_ratio_above_threshold")
    content_quality["scene"] = {
        "area_px": scene_area,
        "masked_ssim": scene_ssim,
        "changed_ratio": scene_changed_ratio,
        "thresholds": {
            "min_masked_ssim": min_scene_ssim,
            "max_changed_ratio": max_scene_changed_ratio,
            "min_changed_ratio_for_ssim": min_scene_changed_ratio_for_ssim,
            "changed_pixel_delta": scene_changed_delta,
        },
    }

    effective_change_detected = False
    if soft_furnishing_requested:
        decor_eval = np.asarray(decor_zone) > 0
        if wallpaper_enabled:
            decor_eval &= ~(np.asarray(wall_zone) > 0)
        # Do not score frame/side relighting or contact-shadow pixels as soft
        # furnishing changes.  This is a QA-only guard; it is not a generation
        # mask and it does not roll provider pixels back.
        decor_eval &= ~artwork_qa_guard
        decor_area = int(np.count_nonzero(decor_eval))
        min_ssim = float(os.getenv("MOLINK_STAGE_B_QA_MIN_DECOR_SSIM", "0.40"))
        max_changed_ratio = float(os.getenv("MOLINK_STAGE_B_QA_MAX_DECOR_CHANGED_RATIO", "0.70"))
        min_changed_ratio = float(os.getenv("MOLINK_STAGE_B_QA_MIN_DECOR_CHANGED_RATIO", "0.003"))
        min_changed_ratio_for_ssim = float(os.getenv("MOLINK_STAGE_B_QA_MIN_CHANGED_RATIO_FOR_SSIM", "0.08"))
        changed_delta = float(os.getenv("MOLINK_STAGE_B_QA_CHANGED_PIXEL_DELTA", "24"))
        if decor_area >= 16:
            decor_ssim = masked_global_ssim(base_arr, out_arr, decor_eval)
            decor_changed_ratio = float(np.mean(delta[decor_eval] > changed_delta))
            if decor_changed_ratio < min_changed_ratio:
                failures.append("decor_effect_below_minimum")
            else:
                effective_change_detected = True
            if decor_changed_ratio >= min_changed_ratio_for_ssim and decor_ssim < min_ssim:
                failures.append("decor_masked_ssim_below_threshold")
            if decor_changed_ratio > max_changed_ratio:
                failures.append("decor_changed_ratio_above_threshold")
            content_quality["decor"] = {
                "area_px": decor_area,
                "masked_ssim": decor_ssim,
                "changed_ratio": decor_changed_ratio,
                "zone_usage": "diagnostic_and_goal_evaluation_only_not_generation_mask",
                "thresholds": {
                    "min_masked_ssim": min_ssim,
                    "min_changed_ratio": min_changed_ratio,
                    "max_changed_ratio": max_changed_ratio,
                    "min_changed_ratio_for_ssim": min_changed_ratio_for_ssim,
                    "changed_pixel_delta": changed_delta,
                },
            }
        else:
            failures.append("decor_qa_region_too_small")
            content_quality["decor"] = {"area_px": decor_area, "status": "too_small"}

    if wallpaper_enabled:
        wall_eval = (np.asarray(wall_zone) > 0) & ~artwork_qa_guard
        wall_area = int(np.count_nonzero(wall_eval))
        min_gradient_correlation = float(os.getenv("MOLINK_STAGE_B_QA_MIN_WALL_GRADIENT_CORRELATION", "0.45"))
        max_gradient_energy_ratio = float(os.getenv("MOLINK_STAGE_B_QA_MAX_WALL_GRADIENT_ENERGY_RATIO", "4.0"))
        min_wall_changed_ratio = float(os.getenv("MOLINK_STAGE_B_QA_MIN_WALL_CHANGED_RATIO", "0.02"))
        min_target_improvement = float(os.getenv("MOLINK_STAGE_B_QA_MIN_WALL_TARGET_IMPROVEMENT", "0.08"))
        wall_changed_delta = float(os.getenv("MOLINK_STAGE_B_QA_WALL_CHANGED_PIXEL_DELTA", "10"))
        if wall_area >= 32:
            wall_metrics = _masked_luma_gradient_metrics(base_arr, out_arr, wall_eval)
            wall_changed_ratio = float(np.mean(delta[wall_eval] > wall_changed_delta))
            if wall_changed_ratio < min_wall_changed_ratio:
                failures.append("wall_recolor_effect_below_minimum")
            else:
                effective_change_detected = True
            if (
                not wall_metrics["flat_base_wall"]
                and wall_metrics["gradient_correlation"] < min_gradient_correlation
            ):
                failures.append("wall_luma_gradient_correlation_below_threshold")
            if wall_metrics["gradient_energy_ratio"] > max_gradient_energy_ratio:
                failures.append("wall_gradient_energy_ratio_above_threshold")

            target_metrics: dict[str, Any] = {"enabled": False}
            if isinstance(wallpaper_target_rgb, (list, tuple)) and len(wallpaper_target_rgb) >= 3:
                target = np.asarray(list(wallpaper_target_rgb)[:3], dtype=np.float32)
                base_median = np.median(base_arr[wall_eval].astype(np.float32), axis=0)
                styled_median = np.median(out_arr[wall_eval].astype(np.float32), axis=0)
                before_distance = float(np.linalg.norm(base_median - target))
                after_distance = float(np.linalg.norm(styled_median - target))
                improvement = float((before_distance - after_distance) / max(before_distance, 1.0))
                target_metrics = {
                    "enabled": True,
                    "target_rgb": [int(v) for v in target.tolist()],
                    "base_median_rgb": [float(v) for v in base_median.tolist()],
                    "styled_median_rgb": [float(v) for v in styled_median.tolist()],
                    "distance_before": before_distance,
                    "distance_after": after_distance,
                    "relative_improvement": improvement,
                    "min_relative_improvement": min_target_improvement,
                }
                if before_distance > 8.0 and improvement < min_target_improvement:
                    failures.append("wall_target_color_not_approached")
            content_quality["wall"] = {
                "area_px": wall_area,
                **wall_metrics,
                "changed_ratio": wall_changed_ratio,
                "target_color": target_metrics,
                "zone_usage": "diagnostic_and_goal_evaluation_only_not_generation_mask",
                "thresholds": {
                    "min_gradient_correlation": min_gradient_correlation,
                    "max_gradient_energy_ratio": max_gradient_energy_ratio,
                    "min_changed_ratio": min_wall_changed_ratio,
                    "changed_pixel_delta": wall_changed_delta,
                },
            }
        else:
            failures.append("wall_qa_region_too_small")
            content_quality["wall"] = {"area_px": wall_area, "status": "too_small"}

    if protect_changed:
        failures.append("protected_pixels_changed_after_local_restore")
    if (soft_furnishing_requested or wallpaper_enabled) and not effective_change_detected:
        failures.append("styling_no_effect")
    return {
        "passed": not failures,
        "reason": "ok" if not failures else "content_or_effect_quality_gate_failed",
        "generation_scope": "full_frame_unmasked",
        "generation_mask_used": False,
        "zones_used_as_generation_masks": False,
        "artwork_protection": protection_trace,
        "protect_changed_pixel_count": protect_changed,
        "artwork_qa_guard_px": qa_anchor_guard_px,
        "content_quality": content_quality,
        "effective_change_detected": bool(effective_change_detected),
        "failures": failures,
        "output_size": list(composited.size),
    }


def _resolve_extracted_path(root: Path, value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def run_stage_b_pipeline(
    result_zip: str,
    output_dir: str,
    *,
    provider: str = "apiyi_gpt_image2_vip",
    soft_furnishing_requested: bool = False,
    wallpaper_recolor: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run at most one Stage B edit per Stage A image.

    Stage A images are copied to ``pre_styling_hd`` before any edit.  A failed
    Stage B item leaves its original ``final_hd`` file untouched, so delivery is
    never blocked by this optional channel.
    """
    started = time.time()
    output_root = Path(output_dir).resolve()
    final_dir = output_root / "final_hd"
    pre_dir = output_root / "pre_styling_hd"
    normalized_wallpaper = normalize_wallpaper_recolor(wallpaper_recolor)
    requested = bool(soft_furnishing_requested or normalized_wallpaper)
    aggregate: Dict[str, Any] = {
        "version": 3,
        "requested": requested,
        "soft_furnishing_requested": bool(soft_furnishing_requested),
        "wallpaper_recolor": normalized_wallpaper,
        "zone_source": "qa_only_hanging_semantic_sam_or_bbox_export",
        "generation_scope": "full_frame_unmasked_with_artwork_protection",
        "provider": normalize_render_provider(provider),
        "call_count": 0,
        "outputs": [],
        "errors": [],
    }
    if not final_dir.is_dir():
        aggregate["elapsed_s"] = round(time.time() - started, 3)
        return aggregate
    try:
        pre_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        aggregate["errors"].append({"asset_id": None, "error": f"Stage B archive setup failed: {exc}"})
        aggregate["default_styling_status"] = "failed_fallback_to_plain" if requested else "not_requested"
        aggregate["elapsed_s"] = round(time.time() - started, 3)
        return aggregate

    stage_a_images = {path.stem: path for path in sorted(final_dir.glob("*.png"))}
    for asset_id, image_path in stage_a_images.items():
        record = {
            "asset_id": asset_id,
            "styling_status": "not_requested" if not requested else "pending",
            "pre_styling_path": str(pre_dir / image_path.name),
            "final_path": str(image_path),
        }
        try:
            shutil.copyfile(image_path, pre_dir / image_path.name)
        except Exception as exc:
            record["styling_status"] = "failed_fallback_to_plain" if requested else "not_requested"
            record["pre_styling_path"] = None
            record["error"] = f"could not preserve Stage A image: {exc}"
            aggregate["errors"].append({"asset_id": asset_id, "error": record["error"]})
        aggregate["outputs"].append(record)
    if not requested or not stage_a_images:
        aggregate["elapsed_s"] = round(time.time() - started, 3)
        return aggregate

    try:
        provider_adapter = build_image_edit_provider(normalize_render_provider(provider))
        extracted_context = tempfile.TemporaryDirectory(prefix="molink_stage_b_")
        extracted = Path(extracted_context.name)
        with zipfile.ZipFile(Path(result_zip).resolve(), "r") as archive:
            archive.extractall(extracted)
        manifest_path = extracted / "flux_inputs" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        manifest_index = {
            str(item.get("asset_id")): item
            for item in (manifest.get("items") or [])
            if isinstance(item, dict) and item.get("asset_id")
        }
        artwork_path = _resolve_extracted_path(extracted, manifest.get("artwork_path") or "flux_inputs/artwork.png")
        if artwork_path is None or not artwork_path.is_file():
            raise FileNotFoundError("Stage B styling analysis requires flux_inputs/artwork.png")
        styling_analysis = parse_styling_analysis_with_vlm(
            artwork_path.read_bytes(),
            os.getenv("STYLING_VLM_MODE", os.getenv("VLM_MODE", "auto")),
            soft_furnishing_requested=bool(soft_furnishing_requested),
            wallpaper_recolor=normalized_wallpaper,
        )
        aggregate["styling_analysis"] = styling_analysis
        analysis_dir = output_root / "stage_b"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        analysis_path = analysis_dir / "styling_analysis.json"
        analysis_path.write_text(json.dumps(styling_analysis, ensure_ascii=False, indent=2), encoding="utf-8")
        aggregate["styling_analysis_path"] = str(analysis_path)
    except Exception as exc:
        for record in aggregate["outputs"]:
            if record.get("styling_status") == "pending":
                record["styling_status"] = "failed_fallback_to_plain"
                record["error"] = f"Stage B setup failed: {exc}"
                aggregate["errors"].append({"asset_id": record.get("asset_id"), "error": record["error"]})
        aggregate["succeeded_count"] = 0
        aggregate["failed_count"] = sum(1 for item in aggregate["outputs"] if item.get("styling_status") == "failed_fallback_to_plain")
        aggregate["elapsed_s"] = round(time.time() - started, 3)
        return aggregate

    try:
        for record in aggregate["outputs"]:
            if record.get("styling_status") != "pending" or not record.get("pre_styling_path"):
                continue
            asset_id = str(record["asset_id"])
            stage_a_path = Path(record["pre_styling_path"])
            final_path = Path(record["final_path"])
            item = manifest_index.get(asset_id, {})
            try:
                with Image.open(stage_a_path) as source:
                    size = tuple(int(v) for v in source.size)
                item_dir = output_root / asset_id / "stage_b"
                item_dir.mkdir(parents=True, exist_ok=True)

                protect_path, protect_zone, protect_source = _stage_b_anchor_from_manifest(
                    extracted=extracted,
                    item=item,
                    size=size,
                    item_dir=item_dir,
                )

                decor_zone = np.zeros((size[1], size[0]), dtype=np.uint8)
                wall_zone = np.zeros((size[1], size[0]), dtype=np.uint8)
                decor_zone_source = "not_requested"
                wall_zone_source = "not_requested"
                if soft_furnishing_requested:
                    decor_path = _resolve_extracted_path(extracted, item.get("stage_b_decor_zone_mask_path"))
                    loaded_decor = _load_optional_stage_b_zone(decor_path, size)
                    if loaded_decor is not None:
                        decor_zone = loaded_decor
                        decor_zone_source = "semantic_diagnostic_zone"
                    else:
                        # Generation is full-frame and must not fail because an
                        # older semantic archive lacks an advisory QA mask.
                        decor_zone[:] = 255
                        decor_zone[protect_zone > 0] = 0
                        decor_zone_source = "full_scene_fallback_diagnostic_zone"
                if normalized_wallpaper:
                    wall_path = _resolve_extracted_path(extracted, item.get("stage_b_wall_zone_mask_path"))
                    loaded_wall = _load_optional_stage_b_zone(wall_path, size)
                    if loaded_wall is not None:
                        wall_zone = loaded_wall
                        wall_zone_source = "semantic_wall_diagnostic_zone"
                    else:
                        wall_zone[:] = 255
                        wall_zone[protect_zone > 0] = 0
                        wall_zone_source = "full_scene_fallback_wall_diagnostic_zone"
                artwork_context = dict(item.get("artwork_analysis") or {}) if isinstance(item.get("artwork_analysis"), dict) else {}
                artwork_context["physical_width_m"] = item.get("artwork_physical_width_m")
                artwork_context["physical_height_m"] = item.get("artwork_physical_height_m")
                prompt = build_stage_b_prompt(
                    bool(soft_furnishing_requested),
                    normalized_wallpaper,
                    styling_analysis,
                    artwork_context,
                )
                (item_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
                aggregate["call_count"] += 1
                result = render_hanging(
                    provider_adapter,
                    stage_a_path,
                    prompt,
                    art_mask=None,
                    output_dir=item_dir,
                    output_name="styled.png",
                    metadata={
                        "task": "stage_b_spatial_styling",
                        "asset_id": asset_id,
                        "candidate_id": item.get("candidate_id"),
                        "wall_id": item.get("wall_id"),
                        "stage_b_single_call": True,
                        "full_space_edit_mode": "styling_artwork_protected",
                        "full_space_coordinate_prompt_requested": True,
                        "full_space_render_reason": "styling_creativity_no_generation_mask",
                        "protected_artwork_mask_path": str(protect_path),
                        "protected_artwork_mask_source": protect_source,
                        "front_polygon_px": item.get("front_polygon_px") or item.get("polygon_px") or item.get("target_quad_original"),
                        "front_bbox_px": item.get("front_bbox_px") or item.get("bbox_px"),
                        "mount_bbox_px": item.get("mount_bbox_px"),
                        "stage_b_generation_mask_used": False,
                        "stage_b_zones_usage": "qa_and_goal_evaluation_only",
                        "stage_b_zone_source": item.get("styling_zone_source") or aggregate["zone_source"],
                        "stage_b_decor_zone_source": decor_zone_source,
                        "stage_b_wall_zone_source": wall_zone_source,
                    },
                )
                composited_path = item_dir / "styled_masked_composite.png"
                qa = _masked_stage_b_composite_and_qa(
                    stage_a_path,
                    Path(result.image),
                    protect_path,
                    composited_path,
                    decor_zone=decor_zone,
                    wall_zone=wall_zone,
                    soft_furnishing_requested=bool(soft_furnishing_requested),
                    wallpaper_enabled=bool(normalized_wallpaper),
                    wallpaper_target_rgb=(normalized_wallpaper or {}).get("tone_rgb"),
                )
                if not qa.get("passed"):
                    record.update({
                        "styling_status": "qa_rejected_fallback_to_plain",
                        "qa": qa,
                        "prompt_path": str(item_dir / "prompt.txt"),
                        "styling_zone_source": item.get("styling_zone_source") or aggregate["zone_source"],
                        "trace": dict(result.trace),
                    })
                    continue
                pending_path = final_path.with_name(f".{final_path.stem}.stage_b_pending{final_path.suffix}")
                shutil.copyfile(composited_path, pending_path)
                os.replace(pending_path, final_path)
                record.update({
                    "styling_status": "succeeded",
                    "styling_zone_source": item.get("styling_zone_source") or aggregate["zone_source"],
                    "artwork_anchor_source": protect_source,
                    "decor_zone_source": decor_zone_source,
                    "wall_zone_source": wall_zone_source,
                    "prompt_path": str(item_dir / "prompt.txt"),
                    "qa": qa,
                    "trace": dict(result.trace),
                })
            except Exception as exc:  # optional styling must never block Stage A delivery
                record["styling_status"] = "failed_fallback_to_plain"
                record["error"] = str(exc)
                aggregate["errors"].append({"asset_id": asset_id, "error": str(exc)})
    finally:
        try:
            extracted_context.cleanup()
        except Exception as exc:
            aggregate["errors"].append({"asset_id": None, "error": f"Stage B temporary cleanup failed: {exc}"})

    aggregate["succeeded_count"] = sum(1 for item in aggregate["outputs"] if item.get("styling_status") == "succeeded")
    aggregate["failed_count"] = sum(1 for item in aggregate["outputs"] if item.get("styling_status") == "failed_fallback_to_plain")
    aggregate["qa_rejected_count"] = sum(1 for item in aggregate["outputs"] if item.get("styling_status") == "qa_rejected_fallback_to_plain")
    aggregate["elapsed_s"] = round(time.time() - started, 3)
    return aggregate
