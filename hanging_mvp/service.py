from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageOps

from .metric3d_io import load_metric3d_zip
from .masks import load_semantic_masks, masks_by_label, masks_by_role, semantic_label_matches, canonical_semantic_label
from .geometry import (
    backproject, estimate_g_hat_from_floor_normals, estimate_g_hat_from_global_floor_like_normals,
    estimate_g_hat_from_horizontal_reference_normals, fit_plane_ransac, make_wall_basis,
    orient_vector_to_camera_up,
    points_from_mask, project_points, project_quad, polygon_mask, ray_plane_depth_for_mask, scale_points_to_original
)
from .hardpaste import load_rgba, warp_artwork_mount_to_geometry, warp_artwork_to_quad
from .debug_assets import png_bytes as debug_png_bytes, mask_preview, build_wall_debug_overlay, draw_rejected_wall_overlay
from .client_result import build_client_result
from semantic_mvp.debug_assets import draw_semantic_wall_debug
from semantic_mvp.vlm_client import parse_artwork_frame_with_vlm
from .rules_access import parse_json, rule_value
from .numeric_utils import _safe_percentile, _mask_bbox, _clamp01
from .structural_clearance import _candidate_structural_clearance
from .support_context import (
    _estimate_floor_wall_intersection, _object_points_near_axis,
    _estimate_axis_support_context, _quad_metrics_original,
)
from .response_assets import _png_bytes, _draw_overlay, _zip_response
from .exit_review import review_hardpaste_exit
from .flux_export import (
    build_flux2_edit_mask, build_remove_mask_for_hardpaste, build_render_prompt, resize_mask_to_original,
    safe_asset_id, select_flux_manifest_items,
)
from .candidate_safety import (
    choose_flux2_input_policy, compute_edge_corner_support_debug,
    evaluate_wall_support_gate, semantic_axis_policy,
)
from .height_fit_contract import (
    evaluate_semantic_clear_rect_vertical_fit,
    floor_source_is_metric_trusted,
    height_contract_gate_reasons,
    resolve_candidate_center_t,
    room_corner_floor_fit_contract,
)
from .styling_metrics import analyze_artwork_luminance, measure_candidate_wall_luminance, should_suggest_dark_wallpaper

try:
    from molink_causal import imaging as causal_imaging
    from molink_causal import light_model as causal_light_model
except Exception:  # pragma: no cover - causal layer is optional at runtime
    causal_imaging = None
    causal_light_model = None

API_KEY = os.getenv("API_KEY", "")
ALLOW_UNAUTH_LOCAL = os.getenv("ALLOW_UNAUTH_LOCAL", "").strip().lower() in {"1", "true", "yes", "on"}
PORT = int(os.getenv("HANGING_MVP_PORT", "6010"))
HANGING_PIPELINE_BUILD_ID = "2026-07-17-render-identity-review-state-v6"


def _truthy_env(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default)
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def _causal_enabled() -> bool:
    return _truthy_env("MOLINK_CAUSAL_ENABLED", "0") and causal_imaging is not None and causal_light_model is not None


def _causal_decision_enabled() -> bool:
    return _causal_enabled() and _truthy_env("HANGING_CAUSAL_DECISION", "0")


def _clamp01_local(value: float) -> float:
    try:
        return float(max(0.0, min(1.0, float(value))))
    except Exception:
        return 0.0



DEFAULT_REMOVABLE_EXISTING_CONTENT_LABELS = {
    "existing_art", "painting", "poster", "framed_art", "artwork",
    "wall_clutter", "wall_label", "hook", "nail", "green_board", "notice_board",
    "small_sign", "signage", "sticker", "label",
}

DEFAULT_BLOCKING_EXISTING_CONTENT_LABELS = {
    "small_shelf", "shelf", "wall_mounted_shelf", "built_in_shelf",
    "cabinet", "storage", "counter_fixture", "counter", "appliance",
    "bookcase", "bookshelf", "rack", "tv_stand", "sideboard",
}


def _rule_label_set(rules: Mapping[str, Any], path: list[str], default: set[str]) -> set[str]:
    raw = rule_value(dict(rules or {}), path, None)
    if isinstance(raw, (list, tuple, set)):
        values = {canonical_semantic_label(x) for x in raw if str(x or "").strip()}
    else:
        values = set()
    return values or {canonical_semantic_label(x) for x in default}


def _classify_existing_content_object(
    obj: Mapping[str, Any],
    *,
    removable_labels: set[str],
    blocking_labels: set[str],
) -> str:
    """Classify existing_content for hanging safety.

    Only flat wall visual elements (old art, posters, labels, hooks, nails) may
    be cleaned and counted as wall support. Physical fixtures such as shelves
    and cabinets occupy volume and must block placement instead of becoming
    synthetic support.
    """
    label = canonical_semantic_label(obj.get("label"))
    if label in blocking_labels or semantic_label_matches(label, blocking_labels):
        return "blocking"
    if label in removable_labels or semantic_label_matches(label, removable_labels):
        return "removable"
    # Keep old behavior for unknown existing_content labels unless rules expand
    # the blocking list. This avoids over-rejecting harmless wall clutter while
    # still hard-blocking known physical fixtures.
    return "removable"


def _union_masks_for_shape(items: List[Mapping[str, Any]], shape_hw: tuple[int, int]) -> np.ndarray | None:
    if not items:
        return None
    out = np.zeros(shape_hw, np.uint8)
    for item in items:
        m = item.get("mask") if isinstance(item, Mapping) else None
        if m is None:
            continue
        try:
            arr = np.asarray(m)
            if arr.shape[:2] == shape_hw:
                out[arr > 0] = 255
        except Exception:
            continue
    return out if np.any(out) else None


_STAGE_B_DECOR_SUPPORT_LABELS = {"floor", "sofa", "table", "cabinet", "sideboard", "chair"}


def _resize_binary_mask(mask: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    width, height = [int(v) for v in size_wh]
    arr = np.asarray(mask, dtype=np.uint8)
    if arr.shape[:2] != (height, width):
        arr = cv2.resize(arr, (width, height), interpolation=cv2.INTER_NEAREST)
    return np.where(arr > 0, 255, 0).astype(np.uint8)


def _build_stage_b_decor_zone(
    objects: Mapping[str, Mapping[str, Any]],
    *,
    processed_shape_hw: tuple[int, int],
    original_size_wh: tuple[int, int],
    wall_items: List[Mapping[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    support_items: list[Mapping[str, Any]] = []
    for item in (objects or {}).values():
        label = str(item.get("label") or "").strip().lower()
        role = str(item.get("role") or "").strip().lower()
        mask = item.get("mask")
        if role != "support" or label not in _STAGE_B_DECOR_SUPPORT_LABELS:
            continue
        if isinstance(mask, np.ndarray) and mask.shape[:2] == processed_shape_hw and np.any(mask):
            support_items.append(item)

    if support_items:
        processed = np.zeros(processed_shape_hw, np.uint8)
        sources: list[str] = []
        for item in support_items:
            processed[item["mask"] > 0] = 255
            semantic_source = str(item.get("semantic_source") or item.get("mask_source") or "bbox")
            sources.append("sam" if "sam" in semantic_source else "bbox")
        zone = _resize_binary_mask(processed, original_size_wh)
        source_set = set(sources)
        source = "sam_support_union" if source_set == {"sam"} else ("sam_bbox_support_union" if "sam" in source_set else "bbox_support_union")
        return zone, {
            "styling_zone_source": source,
            "support_object_ids": [str(item.get("object_id") or "") for item in support_items],
            "support_labels": [str(item.get("label") or "") for item in support_items],
            "support_mask_area_px_original": int(np.count_nonzero(zone)),
        }

    width, height = [int(v) for v in original_size_wh]
    fallback = np.zeros((height, width), np.uint8)
    fallback[int(round(height * 0.52)):, :] = 255
    wall_union = _union_masks_for_shape(wall_items, processed_shape_hw)
    if wall_union is not None:
        fallback[_resize_binary_mask(wall_union, original_size_wh) > 0] = 0
    return fallback, {
        "styling_zone_source": "fallback_lower_region",
        "support_object_ids": [],
        "support_labels": [],
        "support_mask_area_px_original": int(np.count_nonzero(fallback)),
    }


def _scope_stage_b_decor_zone_to_candidate(
    decor_zone: np.ndarray,
    wall_zone: np.ndarray,
    *,
    front_bbox_px: list[int] | None,
    base_debug: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep support edits near the candidate artwork instead of opening the room."""
    decor = np.where(np.asarray(decor_zone) > 0, 255, 0).astype(np.uint8)
    wall = np.asarray(wall_zone) > 0
    height, width = decor.shape[:2]
    if front_bbox_px and len(front_bbox_px) >= 4:
        art_x0, art_y0, art_x1, _art_y1 = [int(v) for v in front_bbox_px[:4]]
        art_width = max(1, art_x1 - art_x0)
        center_x = 0.5 * (art_x0 + art_x1)
        corridor_width = min(float(width) * 0.78, max(float(width) * 0.40, float(art_width) * 3.0))
        x0 = max(0, int(round(center_x - 0.5 * corridor_width)))
        x1 = min(width, int(round(center_x + 0.5 * corridor_width)))
        y0 = max(0, int(round(float(art_y0) - float(height) * 0.12)))
    else:
        x0 = int(round(width * 0.20))
        x1 = int(round(width * 0.80))
        y0 = int(round(height * 0.30))

    scope = np.zeros((height, width), dtype=bool)
    scope[y0:, x0:x1] = True
    scoped = np.where((decor > 0) & scope & (~wall), 255, 0).astype(np.uint8)
    source = str(base_debug.get("styling_zone_source") or "bbox_support_union")
    fallback_used = False
    if not np.any(scoped):
        fallback_used = True
        fallback = np.zeros((height, width), np.uint8)
        fallback[int(round(height * 0.52)):, x0:x1] = 255
        fallback[wall] = 0
        scoped = fallback
        source = "fallback_lower_region"
    return scoped, {
        **dict(base_debug),
        "styling_zone_source": source,
        "candidate_spatial_scope": "artwork_centered_near_wall_corridor",
        "candidate_scope_bbox_px": [int(x0), int(y0), int(x1), int(height)],
        "support_mask_area_px_before_scope": int(np.count_nonzero(decor)),
        "support_mask_area_px_original": int(np.count_nonzero(scoped)),
        "candidate_scope_fallback_used": bool(fallback_used),
        "wall_overlap_removed_px": int(np.count_nonzero((decor > 0) & scope & wall)),
    }


def _build_stage_b_protect_zone(
    mount_mask: np.ndarray,
    *,
    front_bbox_px: list[int] | None,
    candidate: Mapping[str, Any],
) -> tuple[np.ndarray, int]:
    height, width = mount_mask.shape[:2]
    margin_px = max(8, int(round(min(width, height) * 0.012)))
    hard = candidate.get("hard_constraints", {}) if isinstance(candidate, Mapping) else {}
    try:
        art_w_m = float(hard.get("artwork_width_m") or 0.0)
        art_h_m = float(hard.get("artwork_height_m") or 0.0)
        if front_bbox_px and art_w_m > 0 and art_h_m > 0:
            bbox_w = max(1.0, float(front_bbox_px[2]) - float(front_bbox_px[0]))
            bbox_h = max(1.0, float(front_bbox_px[3]) - float(front_bbox_px[1]))
            px_per_m = 0.5 * (bbox_w / art_w_m + bbox_h / art_h_m)
            margin_px = int(round(max(8.0, min(96.0, px_per_m * 0.10))))
    except Exception:
        pass
    kernel_size = margin_px * 2 + 1
    protected = cv2.dilate((mount_mask > 0).astype(np.uint8) * 255, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
    return protected, int(margin_px)


def _analyze_artwork_image_with_vlm(artwork_raw: bytes, art_rgba: np.ndarray, seed_analysis: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Inspect the artwork image itself for frame/surface/thickness policy.

    The semantic room parser only sees the room photo.  Product artwork image
    traits must be inferred from ``artwork_image`` here; otherwise every artwork
    falls back to ``uncertain`` and the renderer always uses the default side
    thickness.  Fail closed to the prior seed/default analysis so hanging never
    depends on VLM availability.
    """
    base = dict(seed_analysis or {})
    if not _truthy_env("ARTWORK_FRAME_VLM_ENABLED", "1"):
        base.setdefault("enabled", False)
        base.setdefault("status", "disabled")
        base.setdefault("reason", "ARTWORK_FRAME_VLM_ENABLED is disabled")
        return base
    try:
        h, w = art_rgba.shape[:2]
        mode = os.getenv("ARTWORK_FRAME_VLM_MODE", os.getenv("VLM_MODE", "auto"))
        result = parse_artwork_frame_with_vlm(artwork_raw, mode, int(w), int(h))
        merged = {**base, **(result or {})}
        merged.setdefault("enabled", True)
        merged.setdefault("source", "artwork_image_vlm")
        return merged
    except Exception as exc:
        base.update({
            "enabled": True,
            "source": base.get("source") or "artwork_image_vlm",
            "status": "error",
            "reason": f"artwork_frame_vlm_failed: {exc}",
        })
        return base


def _semantic_light_fields(semantics: Mapping[str, Any]) -> dict[str, Any]:
    raw = semantics.get("light") if isinstance(semantics, Mapping) else None
    if not isinstance(raw, Mapping):
        return {}
    return {k: v for k, v in raw.items() if v is not None}


def _wall_causal_semantic_light(global_light: Mapping[str, Any], wall_obj: Mapping[str, Any]) -> dict[str, Any]:
    """Return semantic light scoped to this wall.

    Top-level VLM light values are scene-level. They are useful for narration, but
    fields such as surface_active_light and landscape_competition_score must not be
    blindly applied to every candidate wall. Wall-object values override scene
    values and are the only values used for per-candidate P_view/T13 decisions.
    """
    out = dict(global_light or {})
    if "landscape_competition_score" in out:
        out["landscape_competition_scene_score"] = out.pop("landscape_competition_score")
        out.setdefault("landscape_competition_scope", "scene")
    if "surface_active_light" in out:
        out["surface_active_light_scene"] = out.pop("surface_active_light")
        out.setdefault("surface_active_light_scope", "scene")

    if isinstance(wall_obj, Mapping):
        try:
            if wall_obj.get("landscape_competition_score") is not None:
                out["landscape_competition_score"] = _clamp01_local(float(wall_obj.get("landscape_competition_score")))
                out["landscape_competition_scope"] = "wall"
        except Exception:
            pass
        sal = wall_obj.get("surface_active_light")
        if isinstance(sal, bool):
            out["surface_active_light"] = sal
            out["surface_active_light_scope"] = "wall"
    return {k: v for k, v in out.items() if v is not None}


def _estimate_artwork_visual_complexity(art_rgba: np.ndarray) -> float | None:
    try:
        arr = np.asarray(art_rgba)
        if arr.ndim != 3 or arr.shape[0] < 8 or arr.shape[1] < 8:
            return None
        rgb = arr[:, :, :3].astype(np.uint8)
        alpha = arr[:, :, 3] if arr.shape[2] >= 4 else None
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if alpha is not None and np.count_nonzero(alpha > 8) > 100:
            mask = (alpha > 8).astype(np.uint8)
        else:
            mask = np.ones(gray.shape, np.uint8)
        vals = gray[mask > 0]
        if vals.size < 100:
            return None
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        edges = cv2.Canny(gray, 40, 120)
        edge_density = float(np.count_nonzero((edges > 0) & (mask > 0)) / max(1, np.count_nonzero(mask > 0)))
        lap_energy = float(np.mean(np.abs(lap[mask > 0])) / 60.0)
        tonal_var = float(np.std(vals) / 80.0)
        score = 0.45 * min(1.0, edge_density * 8.0) + 0.35 * min(1.0, lap_energy) + 0.20 * min(1.0, tonal_var)
        return round(_clamp01_local(score), 3)
    except Exception:
        return None



def _artwork_analysis_needs_runtime_vlm(analysis: Mapping[str, Any] | None) -> bool:
    if not isinstance(analysis, Mapping) or not analysis:
        return True
    status = str(analysis.get("status") or "").strip().lower()
    source = str(analysis.get("source") or "").strip().lower()
    if status in {"ok", "uncertain"} and source not in {"", "semantic_service"}:
        return False
    return status in {"", "not_requested", "disabled"}


def _build_runtime_artwork_analysis_from_vlm(artwork_raw: bytes, vlm_mode: str = "auto") -> dict[str, Any]:
    try:
        art_img = ImageOps.exif_transpose(Image.open(io.BytesIO(artwork_raw))).convert("RGB")
        art_w, art_h = art_img.size
        analysis = parse_artwork_frame_with_vlm(artwork_raw, vlm_mode, int(art_w), int(art_h))
        if isinstance(analysis, dict):
            analysis.setdefault("source", "hanging_runtime_artwork_vlm")
            analysis.setdefault("enabled", True)
            analysis.setdefault("artwork_image_size", [int(art_w), int(art_h)])
            return analysis
        raise RuntimeError("parse_artwork_frame_with_vlm returned non-object")
    except Exception as exc:
        return {
            "enabled": True,
            "source": "hanging_runtime_artwork_vlm",
            "status": "error",
            "has_frame": None,
            "confidence": 0.0,
            "frame_style": "uncertain",
            "artwork_surface_style": "uncertain",
            "side_thickness_policy": "normal",
            "frame_width_ratio": {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0},
            "dominant_edge_colors_rgb": {"left": [0, 0, 0], "right": [0, 0, 0], "top": [0, 0, 0], "bottom": [0, 0, 0]},
            "should_precompose_frame_sides": False,
            "mood_register": None,
            "visual_complexity": None,
            "reason": f"artwork_frame_vlm_failed_at_hanging: {exc}",
        }

def _augment_artwork_analysis_with_pix(artwork_analysis: Mapping[str, Any], art_rgba: np.ndarray) -> dict[str, Any]:
    out = dict(artwork_analysis or {})
    out.update(analyze_artwork_luminance(art_rgba))
    if out.get("visual_complexity") is None and out.get("complexity_score") is None:
        complexity = _estimate_artwork_visual_complexity(art_rgba)
        if complexity is not None:
            out["visual_complexity"] = complexity
            out["visual_complexity_source"] = "pix_hf_energy"
    return out


def _light_room_mask(shape_hw: tuple[int, int], window_mask: np.ndarray | None) -> np.ndarray:
    room = np.ones(shape_hw, dtype=bool)
    if window_mask is not None:
        room[window_mask > 0] = False
    if not np.any(room):
        room[:, :] = True
    return room


def _candidate_window_clearance_cm(candidate: Mapping[str, Any]) -> float | None:
    hard = candidate.get("hard_constraints", {}) if isinstance(candidate, Mapping) else {}
    sc = hard.get("structural_clearance_policy", {}) if isinstance(hard, Mapping) else {}
    vals: list[float] = []
    if isinstance(sc, Mapping):
        for key in ("considered_objects", "blocking_objects"):
            for obj in sc.get(key, []) or []:
                if not isinstance(obj, Mapping):
                    continue
                if str(obj.get("label") or "").strip().lower() != "window":
                    continue
                try:
                    vals.append(float(obj.get("edge_clearance_m")) * 100.0)
                except Exception:
                    continue
    return round(min(vals), 1) if vals else None


def _candidate_wall_area_ratio(candidate: Mapping[str, Any]) -> float | None:
    hard = candidate.get("hard_constraints", {}) if isinstance(candidate, Mapping) else {}
    if not isinstance(hard, Mapping):
        return None
    try:
        sr = hard.get("visible_wall_s_range_m") or []
        tr = hard.get("visible_wall_t_range_m") or []
        wall_area = max(1e-6, abs(float(sr[1]) - float(sr[0])) * abs(float(tr[1]) - float(tr[0])))
        art_area = max(0.0, float(hard.get("artwork_width_m") or 0.0) * float(hard.get("artwork_height_m") or 0.0))
        if art_area <= 0.0:
            return None
        return round(float(art_area / wall_area), 4)
    except Exception:
        return None


def _candidate_visual_complexity(candidate: Mapping[str, Any]) -> float | None:
    art = candidate.get("artwork_analysis") if isinstance(candidate, Mapping) else None
    if not isinstance(art, Mapping):
        return None
    for key in ("visual_complexity", "complexity_score"):
        try:
            if art.get(key) is not None:
                return _clamp01_local(float(art.get(key)))
        except Exception:
            pass
    return None


def _causal_light_for_candidate(
    candidate: Mapping[str, Any],
    global_light: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str | None, float | None]:
    if not _causal_enabled() or not global_light:
        return None, None, None
    facts = dict(global_light.get("light_facts") or {})
    if not facts:
        return None, None, None
    semantic_light = dict(global_light.get("semantic_light") or {})
    for key in ("theta", "exterior_brightness_class", "ambient_light_mode", "shadow_presence", "temporal_light_quality"):
        if semantic_light.get(key) is not None:
            facts[key] = semantic_light[key]
    ceiling_height = semantic_light.get("ceiling_height")
    view = SimpleNamespace(
        window_edge_clearance_cm=_candidate_window_clearance_cm(candidate),
        landscape_competition_score=semantic_light.get("landscape_competition_score"),
        artwork_wall_area_ratio=_candidate_wall_area_ratio(candidate),
    )
    try:
        score = causal_light_model.evaluate(
            SimpleNamespace(**facts),
            view,
            ceiling_height=ceiling_height,
            artwork_complexity=_candidate_visual_complexity(candidate),
        )
    except Exception as exc:
        try:
            candidate.setdefault("ranking_policy", {})["causal_light_error"] = str(exc)
        except Exception:
            pass
        return None, None, None
    components = dict(score.components or {})
    components.update({
        "r_light": score.r_light,
        "bonus": score.bonus,
        "penalty": score.penalty,
        "delta": score.delta,
        "risk_band": score.risk_band,
        "penalty_band": score.penalty_band,
        "source": "molink_causal.light_model",
    })
    return components, score.risk_band, float(score.delta)


app = FastAPI(title="hanging-art-mvp-geometry-api")

def require_auth(a):
    if ALLOW_UNAUTH_LOCAL:
        return
    if not API_KEY:
        raise HTTPException(500, "API_KEY is not configured; set ALLOW_UNAUTH_LOCAL=1 for local development")
    if a != f"Bearer {API_KEY}":
        raise HTTPException(401, "Unauthorized")



def _jsonable_point(pt) -> list[float]:
    return [float(pt[0]), float(pt[1])]


def _jsonable_quad(quad) -> list[list[float]]:
    return [_jsonable_point(pt) for pt in list(quad)[:4]]


def _mask_area_for_quad(size_hw: tuple[int, int], quad: list[list[float]]) -> tuple[int, np.ndarray]:
    mask = np.zeros(size_hw, np.uint8)
    pts = np.array(quad, np.float32)
    if pts.shape[0] >= 3 and np.isfinite(pts).all():
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 255)
    return int(np.count_nonzero(mask)), mask


def _quad_edge_length(quad: list[list[float]], source_edge: str) -> float:
    """Return the screen-space length of a side face's source edge.

    Side face quads produced by _compute_artwork_mount_policy_for_candidate()
    are intentionally ordered as [back_a, back_b, front_b, front_a] for all
    four source edges. Therefore q[0] -> q[1] is always the artwork-edge
    direction used for estimated side width. source_edge is retained in the
    signature for call-site readability and future validation, but it must not
    select a different segment unless the quad construction convention changes.
    """
    _ = source_edge
    q = np.array(quad, np.float32)
    if q.shape[0] < 4:
        return 1.0
    return float(np.linalg.norm(q[1] - q[0]))


def _binary_mask_bbox_xyxy(mask: np.ndarray) -> list[int] | None:
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    ys, xs = np.where(arr > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _quad_to_int_polygon(quad: Any) -> list[list[int]] | None:
    if not isinstance(quad, (list, tuple)) or len(quad) < 4:
        return None
    pts: list[list[int]] = []
    try:
        for pt in quad[:4]:
            pts.append([int(round(float(pt[0]))), int(round(float(pt[1])))])
    except Exception:
        return None
    return pts


def _coerce_quad_points(quad: Any) -> list[list[float]] | None:
    if not isinstance(quad, (list, tuple)) or len(quad) < 4:
        return None
    pts: list[list[float]] = []
    try:
        for pt in quad[:4]:
            if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                return None
            x = float(pt[0])
            y = float(pt[1])
            if not (np.isfinite(x) and np.isfinite(y)):
                return None
            pts.append([x, y])
    except Exception:
        return None
    return pts


def _quad_bbox_xyxy(quad: Any) -> list[float] | None:
    pts = _coerce_quad_points(quad)
    if pts is None:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def _bbox_iou_xyxy(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b:
        return 0.0
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = inter_w * inter_h
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def _quad_iou_px(quad_a: Any, quad_b: Any) -> float:
    pts_a = _coerce_quad_points(quad_a)
    pts_b = _coerce_quad_points(quad_b)
    if pts_a is None or pts_b is None:
        return 0.0
    bbox_a = _quad_bbox_xyxy(pts_a)
    bbox_b = _quad_bbox_xyxy(pts_b)
    if bbox_a is None or bbox_b is None:
        return 0.0
    x0 = int(np.floor(min(bbox_a[0], bbox_b[0])))
    y0 = int(np.floor(min(bbox_a[1], bbox_b[1])))
    x1 = int(np.ceil(max(bbox_a[2], bbox_b[2])))
    y1 = int(np.ceil(max(bbox_a[3], bbox_b[3])))
    width = max(1, x1 - x0 + 3)
    height = max(1, y1 - y0 + 3)
    if width * height > 12_000_000:
        return _bbox_iou_xyxy(bbox_a, bbox_b)
    mask_a = np.zeros((height, width), np.uint8)
    mask_b = np.zeros((height, width), np.uint8)
    offset = np.array([x0 - 1, y0 - 1], np.float32)
    arr_a = np.round(np.array(pts_a, np.float32) - offset).astype(np.int32)
    arr_b = np.round(np.array(pts_b, np.float32) - offset).astype(np.int32)
    cv2.fillPoly(mask_a, [arr_a], 255)
    cv2.fillPoly(mask_b, [arr_b], 255)
    union = int(np.count_nonzero((mask_a > 0) | (mask_b > 0)))
    if union <= 0:
        return 0.0
    inter = int(np.count_nonzero((mask_a > 0) & (mask_b > 0)))
    return float(inter / union)


def _candidate_projection_footprint(candidate: Mapping[str, Any]) -> dict[str, Any] | None:
    projection = candidate.get("projection", {}) if isinstance(candidate.get("projection"), Mapping) else {}
    quad = projection.get("projected_quad_px_original") or candidate.get("target_quad_original")
    pts = _coerce_quad_points(quad)
    bbox = _quad_bbox_xyxy(pts)
    if pts is None or bbox is None:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return {
        "quad": pts,
        "bbox_xyxy": bbox,
        "center_px": [float(sum(xs) / len(xs)), float(sum(ys) / len(ys))],
        "width_px": float(max(1.0, bbox[2] - bbox[0])),
        "height_px": float(max(1.0, bbox[3] - bbox[1])),
    }


def _candidate_center_distance_px(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    ac = a.get("center_px") or [0.0, 0.0]
    bc = b.get("center_px") or [0.0, 0.0]
    return float(((float(ac[0]) - float(bc[0])) ** 2 + (float(ac[1]) - float(bc[1])) ** 2) ** 0.5)


def _suppress_projected_duplicate_candidates(
    candidates: list[dict[str, Any]],
    rules: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Drop visually duplicate same-wall candidates after projection."""
    iou_threshold = float(rule_value(rules, ["candidate_diversity", "projected_duplicate_iou_threshold"], 0.25))
    center_fraction = float(rule_value(rules, ["candidate_diversity", "projected_duplicate_center_distance_fraction_of_artwork_width"], 0.65))
    center_floor_px = float(rule_value(rules, ["candidate_diversity", "projected_duplicate_center_distance_floor_px"], 12.0))
    enabled = bool(rule_value(rules, ["candidate_diversity", "projected_duplicate_suppression_enabled"], True))
    summary: dict[str, Any] = {
        "enabled": enabled,
        "policy": "same_wall_projected_quad_iou_or_center_distance",
        "input_count": len(candidates),
        "kept_count": len(candidates),
        "dropped_count": 0,
        "projected_duplicate_iou_threshold": float(iou_threshold),
        "projected_duplicate_center_distance_fraction_of_artwork_width": float(center_fraction),
        "projected_duplicate_center_distance_floor_px": float(center_floor_px),
        "dropped": [],
    }
    if not enabled or len(candidates) <= 1:
        return candidates, [], summary

    ordered = sorted(
        candidates,
        key=lambda c: (
            float(c.get("score", 0.0) or 0.0),
            float(((c.get("ranking_policy", {}) or {}).get("policy_v6", {}) or {}).get("wall_center_composition_score", 0.0) or 0.0),
        ),
        reverse=True,
    )
    kept_by_wall: dict[str, list[dict[str, Any]]] = {}
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    for cand in ordered:
        wall_id = str(cand.get("wall_id") or "unknown_wall")
        footprint = _candidate_projection_footprint(cand)
        cand.setdefault("candidate_diversity_policy", {})["projected_duplicate_suppression"] = {
            "enabled": True,
            "selected_after_suppression": True,
            "projected_duplicate_iou_threshold": float(iou_threshold),
            "center_distance_fraction_of_artwork_width": float(center_fraction),
        }
        if footprint is None:
            kept_by_wall.setdefault(wall_id, []).append(cand)
            kept.append(cand)
            continue

        duplicate_debug: dict[str, Any] | None = None
        for prev in kept_by_wall.get(wall_id, []):
            prev_footprint = _candidate_projection_footprint(prev)
            if prev_footprint is None:
                continue
            iou = _quad_iou_px(footprint["quad"], prev_footprint["quad"])
            center_dist = _candidate_center_distance_px(footprint, prev_footprint)
            width_ref = max(1.0, min(float(footprint["width_px"]), float(prev_footprint["width_px"])))
            center_threshold = max(center_floor_px, width_ref * center_fraction)
            if iou > iou_threshold or center_dist < center_threshold:
                duplicate_debug = {
                    "kept_candidate_id": prev.get("candidate_id"),
                    "kept_axis_source": (prev.get("axis", {}) or {}).get("axis_source"),
                    "projected_quad_iou": float(iou),
                    "center_distance_px": float(center_dist),
                    "center_distance_threshold_px": float(center_threshold),
                    "drop_reason": "projected_quad_iou" if iou > iou_threshold else "projected_center_distance",
                }
                break

        if duplicate_debug is not None:
            cand.setdefault("candidate_diversity_policy", {})["projected_duplicate_suppression"].update({
                "selected_after_suppression": False,
                **duplicate_debug,
            })
            rec = {
                "candidate_id": cand.get("candidate_id"),
                "wall_id": wall_id,
                "selection_status": "dropped_by_projected_duplicate_suppression",
                "reject_reasons": ["projected_duplicate_suppressed_same_wall"],
                "soft_risk_reasons": list(cand.get("soft_risk_reasons", [])),
                "score": cand.get("score"),
                "axis": cand.get("axis"),
                "projection": cand.get("projection"),
                "candidate_diversity_policy": cand.get("candidate_diversity_policy", {}),
            }
            dropped.append(rec)
            summary["dropped"].append({
                "candidate_id": cand.get("candidate_id"),
                "wall_id": wall_id,
                **duplicate_debug,
            })
            continue

        kept_by_wall.setdefault(wall_id, []).append(cand)
        kept.append(cand)

    summary["kept_count"] = len(kept)
    summary["dropped_count"] = len(dropped)
    return kept, dropped, summary


def _largest_rect_in_binary_mask(mask_u8: np.ndarray) -> dict[str, Any] | None:
    """Largest axis-aligned all-foreground rectangle in a binary mask crop."""
    arr = (np.asarray(mask_u8) > 0).astype(np.uint8)
    if arr.ndim != 2 or arr.size == 0:
        return None
    h, w = arr.shape
    heights = np.zeros(w, dtype=np.int32)
    best_area = 0
    best: tuple[int, int, int, int] | None = None

    for y in range(h):
        heights = (heights + 1) * arr[y]
        stack: list[tuple[int, int]] = []
        for x in range(w + 1):
            cur_h = int(heights[x]) if x < w else 0
            start = x
            while stack and stack[-1][1] > cur_h:
                idx, prev_h = stack.pop()
                area = int(prev_h * (x - idx))
                if area > best_area:
                    best_area = area
                    best = (idx, y - prev_h + 1, x - 1, y)
                start = idx
            if not stack or stack[-1][1] < cur_h:
                stack.append((start, cur_h))

    if best is None or best_area <= 0:
        return None
    x1, y1, x2, y2 = best
    return {
        "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
        "area_px": int(best_area),
        "width_px": int(x2 - x1 + 1),
        "height_px": int(y2 - y1 + 1),
    }


def _semantic_solid_max_empty_rect_axes(
    *,
    semantic_clear_wall_mask: np.ndarray,
    solid_wall_mask: np.ndarray,
    xyz: np.ndarray,
    basis: Any,
    artwork_w_m: float,
    artwork_h_m: float,
    min_edge_margin_m: float,
    metric_seed_tolerance_m: float = 0.02,
    max_axes: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate fallback candidates from semantic_clear_wall_mask ∩ solid_wall_mask.

    This path is only for low safe-coverage walls: a wall can have poor full-mask
    coverage while still containing a contiguous solid rectangle large enough for
    the artwork.  The rectangle may propose a horizontal axis and a safe vertical
    interval, but it must never replace a trusted metric installation height.
    """
    safe_clear = ((semantic_clear_wall_mask > 0) & (solid_wall_mask > 0)).astype(np.uint8)
    safe_clear = cv2.morphologyEx(safe_clear, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    safe_clear = cv2.morphologyEx(safe_clear, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(safe_clear, connectivity=8)
    min_component_area_px = max(900, int(np.count_nonzero(semantic_clear_wall_mask > 0) * 0.005))
    required_w_m = float(artwork_w_m + 2.0 * min_edge_margin_m)
    required_h_m = float(artwork_h_m + 2.0 * min_edge_margin_m)
    debug: dict[str, Any] = {
        "enabled": True,
        "source": "semantic_clear_wall_mask_intersect_solid_wall_mask_max_empty_rect",
        "safe_clear_area_px": int(np.count_nonzero(safe_clear)),
        "component_count": int(max(0, num_labels - 1)),
        "min_component_area_px": int(min_component_area_px),
        "required_width_m": float(required_w_m),
        "required_height_m": float(required_h_m),
        "metric_seed_tolerance_m": float(metric_seed_tolerance_m),
        "tolerance_policy": "candidate_seed_only_exact_projected_footprint_remains_authoritative",
        "added_count": 0,
        "components": [],
    }
    axes: list[dict[str, Any]] = []
    origin = np.array(basis.origin, np.float32).reshape(1, 3)
    u = np.array(basis.u, np.float32).reshape(3)
    v = np.array(basis.v, np.float32).reshape(3)

    component_items: list[tuple[int, int]] = []
    for comp_id in range(1, num_labels):
        area_px = int(stats[comp_id, cv2.CC_STAT_AREA])
        if area_px >= min_component_area_px:
            component_items.append((area_px, comp_id))
    component_items.sort(reverse=True)

    for area_px, comp_id in component_items:
        x = int(stats[comp_id, cv2.CC_STAT_LEFT])
        y = int(stats[comp_id, cv2.CC_STAT_TOP])
        ww = int(stats[comp_id, cv2.CC_STAT_WIDTH])
        hh = int(stats[comp_id, cv2.CC_STAT_HEIGHT])
        comp_crop = (labels[y:y + hh, x:x + ww] == comp_id).astype(np.uint8)
        rect = _largest_rect_in_binary_mask(comp_crop)
        comp_debug: dict[str, Any] = {
            "component_id": int(comp_id),
            "component_area_px": int(area_px),
            "component_bbox_xywh": [int(x), int(y), int(ww), int(hh)],
            "largest_rect": rect,
        }
        debug["components"].append(comp_debug)
        if rect is None:
            comp_debug["skip_reason"] = "no_nonempty_rect"
            continue

        rx1, ry1, rx2, ry2 = rect["bbox_xyxy"]
        gx1, gy1, gx2, gy2 = x + rx1, y + ry1, x + rx2, y + ry2
        rect_mask = np.zeros_like(safe_clear, dtype=np.uint8)
        rect_mask[gy1:gy2 + 1, gx1:gx2 + 1] = 255
        rect_mask[(safe_clear == 0)] = 0
        rect_pts = points_from_mask(xyz, rect_mask, max_points=30000)
        rect_pts = rect_pts[np.isfinite(rect_pts).all(axis=1)] if rect_pts.size else rect_pts
        comp_debug["largest_rect_global_bbox_xyxy"] = [int(gx1), int(gy1), int(gx2), int(gy2)]
        comp_debug["largest_rect_point_count"] = int(len(rect_pts))
        if len(rect_pts) < 80:
            comp_debug["skip_reason"] = "largest_rect_has_too_few_metric_points"
            continue

        rel = rect_pts - origin
        rect_s = rel @ u
        rect_t = rel @ v
        rect_s = rect_s[np.isfinite(rect_s)]
        rect_t = rect_t[np.isfinite(rect_t)]
        if rect_s.size < 40 or rect_t.size < 40:
            comp_debug["skip_reason"] = "largest_rect_metric_projection_invalid"
            continue
        s_min = float(np.percentile(rect_s, 5))
        s_max = float(np.percentile(rect_s, 95))
        t_min = float(np.percentile(rect_t, 5))
        t_max = float(np.percentile(rect_t, 95))
        width_m = float(s_max - s_min)
        height_m = float(t_max - t_min)
        comp_debug.update({
            "s_range_m": [float(s_min), float(s_max)],
            "t_range_m": [float(t_min), float(t_max)],
            "width_m": float(width_m),
            "height_m": float(height_m),
        })
        # Metric3D percentile projection is noisy by a few millimetres.  This
        # rectangle only proposes a horizontal seed; the exact projected quad
        # is still checked later against wall support, edges, structures and
        # blocking fixtures.  A small tolerance here prevents a 0.957 m clear
        # region from being discarded against a 0.960 m seed requirement.
        if width_m + float(metric_seed_tolerance_m) < required_w_m:
            comp_debug["skip_reason"] = "largest_rect_too_narrow_for_artwork_with_edge_margin"
            continue
        if height_m + float(metric_seed_tolerance_m) < required_h_m:
            comp_debug["skip_reason"] = "largest_rect_too_short_for_artwork_with_edge_margin"
            continue
        comp_debug["accepted_with_metric_seed_tolerance"] = bool(
            width_m < required_w_m or height_m < required_h_m
        )

        axis_s = float(0.5 * (s_min + s_max))
        axes.append({
            "axis_s_m": axis_s,
            "component_id": int(comp_id),
            "component_area_px": int(area_px),
            "component_s_range_m": [float(s_min), float(s_max)],
            "component_t_range_m": [float(t_min), float(t_max)],
            "component_width_m": float(width_m),
            "component_height_m": float(height_m),
            "source": "semantic_clear_subregion_max_empty_rect_horizontal_axis",
            "candidate_generation_policy": "horizontal_axis_only_metric_height_remains_authoritative",
            "vertical_policy": "diagnostic_only_exact_projected_footprint_is_authoritative",
            "max_empty_rect_bbox_processed": [int(gx1), int(gy1), int(gx2), int(gy2)],
        })
        debug["added_count"] += 1
        if len(axes) >= max_axes:
            break

    debug["status"] = "available" if axes else "no_rect_large_enough"
    return axes, debug


def _cleanup_semantic_types_from_remove_mask(
    *,
    remove_mask: np.ndarray,
    existing_masks: list[Mapping[str, Any]],
    foreground_masks: list[Mapping[str, Any]],
    original_size_wh: tuple[int, int],
) -> dict[str, Any]:
    """Describe what the *exported* erase mask really covers.

    The previous metadata reused candidate-overlap booleans computed before the
    final remove_mask was built.  In v7 the erase mask may cover old artwork or
    labels elsewhere on the same wall, even when they do not overlap the new
    art front.  Downstream runners must therefore key off the exported mask and
    object intersections, not stale overlap flags.
    """
    ow, oh = original_size_wh
    rm = np.asarray(remove_mask)
    if rm.ndim == 3:
        rm = rm[:, :, 0]
    if rm.shape[:2] != (oh, ow):
        rm = cv2.resize(rm, (ow, oh), interpolation=cv2.INTER_NEAREST)
    rm_bool = rm > 0
    total_px = int(np.count_nonzero(rm_bool))

    def _objects_hit(objs: list[Mapping[str, Any]], kind: str) -> list[dict[str, Any]]:
        hits: list[dict[str, Any]] = []
        for obj in objs or []:
            mask = obj.get("mask")
            if mask is None:
                continue
            obj_mask = resize_mask_to_original(mask, original_size_wh)
            overlap_px = int(np.count_nonzero(rm_bool & (obj_mask > 0)))
            if overlap_px <= 0:
                continue
            hits.append({
                "kind": kind,
                "object_id": obj.get("object_id"),
                "label": str(obj.get("label") or ""),
                "role": str(obj.get("role") or ""),
                "overlap_px": overlap_px,
            })
        return hits

    existing_hits = _objects_hit(existing_masks, "existing_content")
    foreground_hits = _objects_hit(foreground_masks, "foreground")
    labels = sorted({x["label"] for x in existing_hits + foreground_hits if x.get("label")})
    return {
        "remove_mask_nonempty": bool(total_px > 0),
        "remove_mask_area_px": total_px,
        "person": bool(foreground_hits),
        "foreground": bool(foreground_hits),
        "non_person_existing_content": bool(existing_hits),
        "labels": labels,
        "objects": existing_hits + foreground_hits,
        "source": "exported_remove_mask_object_intersections",
    }


def _resolve_artwork_has_frame(rules: dict, artwork_analysis: dict) -> tuple[bool | None, str]:
    raw = rule_value(rules, ["artwork", "has_frame"], None)
    if isinstance(raw, bool):
        return raw, "rules_bool"
    if raw is None or str(raw).strip().lower() in {"", "auto", "none", "null"}:
        if not bool(rule_value(rules, ["artwork_mount", "auto_detect_frame"], True)):
            return None, "auto_disabled"
        min_conf = float(rule_value(rules, ["artwork_mount", "frame_detection_min_confidence"], 0.72))
        try:
            conf = float((artwork_analysis or {}).get("confidence", 0.0) or 0.0)
        except Exception:
            conf = 0.0
        if (artwork_analysis or {}).get("has_frame") is True and conf >= min_conf:
            return True, "artwork_vlm"
        if (artwork_analysis or {}).get("has_frame") is False:
            return False, "artwork_vlm"
        return None, "unknown"
    if isinstance(raw, str) and raw.strip().lower() in {"1", "true", "yes", "y", "on"}:
        return True, "rules_string"
    if isinstance(raw, str) and raw.strip().lower() in {"0", "false", "no", "n", "off"}:
        return False, "rules_string"
    return None, "invalid_rules_value"


def _compute_artwork_mount_policy_for_candidate(
    *,
    rules: dict,
    artwork_analysis: dict,
    artwork_has_frame: bool | None,
    artwork_has_frame_source: str,
    basis,
    axis_s: float,
    center_t: float,
    artwork_w_m: float,
    artwork_h_m: float,
    intr,
    scale_to_original: tuple[float, float],
    original_size_wh: tuple[int, int],
) -> dict:
    enabled = bool(rule_value(rules, ["artwork_mount", "precompose_frame_sides_when_detected"], True))
    analysis = artwork_analysis or {}
    surface_style = str(analysis.get("artwork_surface_style") or analysis.get("surface_style") or "uncertain").strip().lower()
    side_thickness_policy = str(analysis.get("side_thickness_policy") or "normal").strip().lower()
    if surface_style in {"chinese_hanging_scroll", "scroll", "hanging_scroll", "chinese_scroll", "guohua_hanging_scroll", "vertical_scroll", "立轴", "国画立轴"}:
        side_thickness_policy = "none"
    elif surface_style in {"rough_or_canvas", "rough", "coarse", "canvas", "canvas_wrap", "fabric", "oil_painting", "acrylic", "impasto"}:
        side_thickness_policy = "double"
    elif side_thickness_policy not in {"normal", "double", "none"}:
        side_thickness_policy = "normal"
    base = {
        "mode": "flat_hardpaste",
        "precompose_frame_sides": False,
        "frame_source": artwork_has_frame_source,
        "artwork_has_frame": artwork_has_frame,
        "artwork_analysis": analysis,
        "artwork_surface_style": surface_style,
        "side_thickness_policy": side_thickness_policy,
    }
    if not enabled:
        base["fallback_reason"] = "precompose_frame_sides_disabled"
        return base
    if side_thickness_policy == "none":
        base["fallback_reason"] = "artwork_vlm_classified_as_chinese_hanging_scroll_no_side_faces"
        return base

    min_depth = float(rule_value(rules, ["artwork_mount", "min_frame_depth_m"], 0.012))
    max_depth = float(rule_value(rules, ["artwork_mount", "max_frame_depth_m"], 0.055))
    default_depth = float(rule_value(rules, ["artwork_mount", "default_frame_depth_m"], 0.025))
    side_depth_multiplier = 2.0 if side_thickness_policy == "double" else 1.0
    frame_depth_m = float(max(min_depth, min(max_depth, default_depth * side_depth_multiplier)))
    max_faces = max(1, int(rule_value(rules, ["artwork_mount", "max_visible_side_faces"], 2)))
    min_area = float(rule_value(rules, ["artwork_mount", "min_side_face_area_px"], 3))
    min_width = float(rule_value(rules, ["artwork_mount", "min_side_face_width_px"], 1.5))
    edge_sample_px = int(rule_value(rules, ["artwork_mount", "edge_sample_px"], 8))
    edge_average_depth_px = int(rule_value(rules, ["artwork_mount", "edge_average_depth_px"], max(1, min(2, edge_sample_px))))
    side_shading_strength = float(rule_value(rules, ["artwork_mount", "side_shading_strength"], 0.18))

    origin = np.array(basis.origin, np.float32)
    u = np.array(basis.u, np.float32)
    v = np.array(basis.v, np.float32)
    n = np.array(basis.n, np.float32)
    C = origin + u * float(axis_s) + v * float(center_t)
    # `project_quad(...)` / candidate.projection is the visual target that
    # downstream QA and FLUX prompts lock to.  Previous code treated this quad
    # as the back face on the wall, then moved the artwork front toward the
    # camera by frame_depth_m.  That made the visible front projection slightly
    # larger/shifted than the candidate bbox, which is the source of the
    # observed vertical overrun when frame sides are enabled.
    #
    # Keep the visible painting front face anchored to the candidate quad.  The
    # deterministic side faces recede from that locked front face toward the
    # wall/behind it, so frame thickness can be precomposed without changing
    # the painting core footprint.
    front3 = np.stack([
        C - u * artwork_w_m / 2.0 + v * artwork_h_m / 2.0,
        C + u * artwork_w_m / 2.0 + v * artwork_h_m / 2.0,
        C + u * artwork_w_m / 2.0 - v * artwork_h_m / 2.0,
        C - u * artwork_w_m / 2.0 - v * artwork_h_m / 2.0,
    ], axis=0).astype(np.float32)
    back3 = (front3 - n.reshape(1, 3) * frame_depth_m).astype(np.float32)
    back_proc = project_points(back3, intr)
    front_proc = project_points(front3, intr)
    back_orig = scale_points_to_original(back_proc, scale_to_original)
    front_orig = scale_points_to_original(front_proc, scale_to_original)

    raw_faces = [
        {"source_edge": "top", "quad": [back_orig[0], back_orig[1], front_orig[1], front_orig[0]]},
        {"source_edge": "right", "quad": [back_orig[1], back_orig[2], front_orig[2], front_orig[1]]},
        {"source_edge": "bottom", "quad": [back_orig[2], back_orig[3], front_orig[3], front_orig[2]]},
        {"source_edge": "left", "quad": [back_orig[3], back_orig[0], front_orig[0], front_orig[3]]},
    ]
    ow, oh = original_size_wh
    _, front_mask = _mask_area_for_quad((oh, ow), front_orig)

    # When wall faces camera directly, back-front perspective shift may be
    # sub-pixel (< 2px). But sides are still visible when artwork center
    # is offset from image center — the side facing away from the image
    # center is always visible in perspective.
    front_center = np.mean(front_orig, axis=0)
    image_center = np.array([ow / 2.0, oh / 2.0])
    center_offset = front_center - image_center
    offset_dist = np.linalg.norm(center_offset)

    visible = []
    for face in raw_faces:
        area_total, side_mask = _mask_area_for_quad((oh, ow), face["quad"])
        side_visible = side_mask.copy()
        side_visible[front_mask > 0] = 0
        visible_area = int(np.count_nonzero(side_visible))
        edge_len = max(1.0, _quad_edge_length(face["quad"], face["source_edge"]))
        est_width = float(visible_area) / edge_len

        # Fallback: when perspective shift is sub-pixel, compute side
        # visibility from artwork-center-to-image-center offset.
        # The side whose edge faces away from image center is visible.
        if visible_area < min_area and est_width < min_width and offset_dist > 10.0:
            face_quad = np.array(face["quad"], dtype=np.float32)
            face_edge = face["source_edge"]
            # Determine which edge of the face is the "outer" edge
            # (farthest from image center)
            if face_edge == "right":
                outer_edge_pts = face_quad[[1, 2]]  # right edge
            elif face_edge == "left":
                outer_edge_pts = face_quad[[0, 3]]  # left edge
            elif face_edge == "top":
                outer_edge_pts = face_quad[[0, 1]]  # top edge
            else:  # bottom
                outer_edge_pts = face_quad[[2, 3]]  # bottom edge

            outer_edge_center = np.mean(outer_edge_pts, axis=0)
            # Check if this edge is farther from image center than front center
            dist_outer = np.linalg.norm(outer_edge_center - image_center)
            dist_front = np.linalg.norm(front_center - image_center)
            if dist_outer > dist_front + 5.0:
                # This side faces away from image center → visible
                # Estimate visible width proportional to offset distance
                est_visible_width = min(frame_depth_m * 40.0 * (offset_dist / max(100.0, dist_front)), edge_len * 0.15)
                est_visible_width = max(est_visible_width, min_width)
                visible_area = max(visible_area, int(est_visible_width * edge_len * 0.3))
                est_width = max(est_width, est_visible_width)

        rec = {
            "source_edge": face["source_edge"],
            "quad": _jsonable_quad(face["quad"]),
            "area_px": float(visible_area),
            "total_area_px": float(area_total),
            "estimated_side_width_px": float(est_width),
        }
        if visible_area >= min_area or est_width >= min_width:
            visible.append(rec)
    visible.sort(key=lambda x: (float(x.get("area_px", 0.0)), float(x.get("estimated_side_width_px", 0.0))), reverse=True)
    visible = visible[:max_faces]
    if not visible:
        base.update({
            "mode": "flat_hardpaste",
            "fallback_reason": "no_visible_side_face_after_projection",
            "frame_depth_m": frame_depth_m,
            "side_depth_multiplier": float(side_depth_multiplier),
            "side_thickness_policy": side_thickness_policy,
            "front_quad_original": _jsonable_quad(front_orig),
            "back_quad_original": _jsonable_quad(back_orig),
            "front_anchor_mode": "candidate_projection_visual_front_locked",
            "visible_side_faces_original": [],
        })
        return base
    return {
        "mode": "precomposed_frame_sides",
        "precompose_frame_sides": True,
        "frame_source": artwork_has_frame_source,
        "artwork_has_frame": artwork_has_frame,
        "frame_depth_m": float(frame_depth_m),
        "side_depth_multiplier": float(side_depth_multiplier),
        "side_thickness_policy": side_thickness_policy,
        "artwork_surface_style": surface_style,
        "front_quad_original": _jsonable_quad(front_orig),
        "back_quad_original": _jsonable_quad(back_orig),
        "front_anchor_mode": "candidate_projection_visual_front_locked",
        "visible_side_faces_original": visible,
        "edge_sample_px": int(edge_sample_px),
        "edge_average_depth_px": int(max(1, edge_average_depth_px)),
        "side_shading_strength": float(side_shading_strength),
        "max_visible_side_faces": int(max_faces),
        "edge_color_source": "original_artwork_outer_edge_average_color",
        "artwork_analysis": analysis,
    }

















@app.get("/health")
def health():
    return {
        "status": "ok",
        "api": "hanging-art-mvp-geometry-api",
        "cleanup": {
            "enabled": True,
            "stage": "candidate_wall_cleanup_then_render",
            "mask_definition": "art_region_mask ∩ current_wall_mask ∩ existing_content_mask",
            "off_wall_existing_content_protected": True,
        },
        "flux": {
            "integrated": False,
            "downstream_contract": "approved hard_paste assets are exported under flux_inputs/items for external BFL official API processing",
            "provider_count": 2,
            "providers": ["bfl_erase_v1", "bfl_flux2_edit"],
            "local_flux2_klein_default": "disabled",
        },
    }

@app.post("/analyze_hanging_mvp")
async def analyze_hanging_mvp(
    room_image: UploadFile = File(...),
    metric3d_zip: UploadFile = File(...),
    artwork_image: UploadFile = File(...),
    artwork_mask: Optional[UploadFile] = File(None),
    semantics_json: str = Form("{}"),
    rules_json: str = Form("{}"),
    authorization: Optional[str] = Header(None),
):
    require_auth(authorization)
    rules = parse_json(rules_json, "rules_json")
    semantics = parse_json(semantics_json, "semantics_json")

    metric_bytes = await metric3d_zip.read()
    bundle = load_metric3d_zip(metric_bytes)
    depth = bundle.depth_m
    normal = bundle.normal_xyz
    h, w = depth.shape
    xyz = backproject(depth, bundle.intr)

    room_raw = await room_image.read()
    room_pil = ImageOps.exif_transpose(Image.open(io.BytesIO(room_raw))).convert("RGB")
    room_rgb = np.array(room_pil)

    artwork_raw = await artwork_image.read()
    art_rgba = load_rgba(artwork_raw)

    objects = load_semantic_masks(semantics, (h, w))
    walls = masks_by_role(objects, "wall") or masks_by_label(objects, {"wall"})
    floors = masks_by_label(objects, {"floor"})
    roofs = masks_by_label(objects, {"roof", "ceiling"})
    tables = masks_by_label(objects, {"table"})
    window_masks_for_causal = masks_by_label(objects, {"window"})
    ceiling_masks_for_causal = masks_by_label(objects, {"roof", "ceiling"})
    structural_forbidden_labels = set(rules.get("forbidden_classes", ["window", "door", "arch", "opening", "tv", "mirror", "lamp", "track_lighting"]))
    existing_content_labels = set(rules.get("existing_content_classes", ["existing_art", "painting", "poster", "framed_art", "wall_clutter", "wall_label", "hook", "nail", "small_shelf", "green_board", "notice_board"]))
    seed_artwork_analysis = semantics.get("artwork_analysis") if isinstance(semantics.get("artwork_analysis"), dict) else {}
    artwork_analysis = _analyze_artwork_image_with_vlm(artwork_raw, art_rgba, seed_artwork_analysis)
    artwork_analysis = _augment_artwork_analysis_with_pix(artwork_analysis, art_rgba)
    resolved_artwork_has_frame, artwork_has_frame_source = _resolve_artwork_has_frame(rules, artwork_analysis)

    causal_semantic_light = _semantic_light_fields(semantics) if _causal_enabled() else {}
    causal_window_mask = _union_masks_for_shape(window_masks_for_causal, (h, w)) if _causal_enabled() else None
    causal_ceiling_mask = _union_masks_for_shape(ceiling_masks_for_causal, (h, w)) if _causal_enabled() else None
    causal_room_mask = _light_room_mask((h, w), causal_window_mask) if _causal_enabled() else None

    def _union_reference_masks(items):
        if not items:
            return None
        merged = np.zeros((h, w), np.uint8)
        for item in items:
            mask = item.get("mask")
            if isinstance(mask, np.ndarray) and mask.shape[:2] == (h, w):
                merged = cv2.bitwise_or(merged, mask.astype(np.uint8))
        return merged if np.count_nonzero(merged) > 0 else None

    # Preserve the existing height-anchor behavior (first semantic floor mask).
    # A union of floor boxes is used only for gravity estimation; roof/table are
    # gravity references only and never replace the floor height anchor.
    floor_mask = floors[0]["mask"] if floors else None
    floor_gravity_mask = _union_reference_masks(floors)
    normal_y_max = float(rule_value(rules, ["camera", "floor_normal_filter", "normal_y_max"], -0.75))
    abs_z_max = float(rule_value(rules, ["camera", "floor_normal_filter", "abs_z_max"], 0.45))
    min_g_pixels = int(rule_value(rules, ["camera", "floor_normal_filter", "min_pixels"], 1000))
    angular_variance_soft_limit = float(rule_value(rules, ["camera", "floor_normal_filter", "angular_variance_soft_limit_rad"], 0.08))
    full_confidence_pixels = int(rule_value(rules, ["camera", "floor_normal_filter", "full_confidence_pixels"], 5000))

    gravity_attempts = []
    g_hat = None
    g_conf = 0.0
    g_debug = {}
    gravity_source = None
    gravity_candidates = []
    reference_sets = [
        ("floor", floors, floor_gravity_mask),
        ("roof", roofs, _union_reference_masks(roofs)),
        ("table", tables, _union_reference_masks(tables)),
    ]
    for reference_label, reference_objects, reference_mask in reference_sets:
        if not reference_objects:
            gravity_attempts.append({
                "reference_label": reference_label,
                "status": "missing",
                "object_count": 0,
            })
            continue
        if reference_label == "floor":
            candidate_g, candidate_conf, candidate_debug = estimate_g_hat_from_floor_normals(
                normal,
                reference_mask,
                normal_y_max=normal_y_max,
                abs_z_max=abs_z_max,
                min_pixels=min_g_pixels,
                angular_variance_soft_limit_rad=angular_variance_soft_limit,
                full_confidence_pixels=full_confidence_pixels,
            )
        else:
            candidate_g, candidate_conf, candidate_debug = estimate_g_hat_from_horizontal_reference_normals(
                normal,
                reference_mask,
                reference_label=reference_label,
                abs_y_min=abs(normal_y_max),
                abs_z_max=abs_z_max,
                min_pixels=min_g_pixels,
                angular_variance_soft_limit_rad=angular_variance_soft_limit,
                full_confidence_pixels=full_confidence_pixels,
            )
        attempt = {
            "reference_label": reference_label,
            "status": "candidate" if float(candidate_conf) > 0.0 else "unusable",
            "object_count": len(reference_objects),
            "object_ids": [obj.get("object_id") for obj in reference_objects],
            "confidence": float(candidate_conf),
            "debug": candidate_debug,
        }
        gravity_attempts.append(attempt)
        if float(candidate_conf) > 0.0:
            gravity_candidates.append({
                "reference_label": reference_label,
                "g_hat": candidate_g,
                "confidence": float(candidate_conf),
                "debug": dict(candidate_debug),
                "attempt": attempt,
            })

    if gravity_candidates:
        # Floor remains the first tie-breaker, but a barely usable floor mask
        # must not suppress a much stronger ceiling/roof/table reference.  This
        # is particularly important when SAM's floor mask is mostly a person or
        # another vertical foreground region.
        selected_gravity = max(gravity_candidates, key=lambda item: item["confidence"])
        selected_gravity["attempt"]["status"] = "selected"
        for candidate in gravity_candidates:
            if candidate is not selected_gravity:
                candidate["attempt"]["status"] = "not_selected_lower_confidence"
        g_hat = selected_gravity["g_hat"]
        g_conf = float(selected_gravity["confidence"])
        g_debug = dict(selected_gravity["debug"])
        gravity_source = f"semantic_{selected_gravity['reference_label']}_mask"

    if g_hat is None:
        candidate_g, candidate_conf, candidate_debug = estimate_g_hat_from_global_floor_like_normals(
            normal,
            normal_y_max=normal_y_max,
            abs_z_max=abs_z_max,
            min_pixels=min_g_pixels,
            angular_variance_soft_limit_rad=angular_variance_soft_limit,
            full_confidence_pixels=full_confidence_pixels,
        )
        g_hat = candidate_g
        g_conf = float(candidate_conf)
        g_debug = dict(candidate_debug)
        gravity_source = "metric3d_global_floor_like_normals"
        gravity_attempts.append({
            "reference_label": "global_metric3d_floor_like",
            "status": "accepted" if float(candidate_conf) > 0.0 else "unusable_default_up",
            "confidence": float(candidate_conf),
            "debug": candidate_debug,
        })

    g_hat_before_canonicalization = np.asarray(g_hat, dtype=np.float32).reshape(3)
    g_hat = orient_vector_to_camera_up(g_hat_before_canonicalization)
    g_debug.update({
        "source": gravity_source,
        "g_hat_cam": [float(x) for x in g_hat],
        "camera_up_direction": [0.0, -1.0, 0.0],
        "direction_dot_camera_up": float(np.dot(g_hat, np.array([0.0, -1.0, 0.0], np.float32))),
        "direction_flip_applied_at_service_boundary": bool(float(np.dot(g_hat_before_canonicalization, g_hat)) < 0.0),
        "reference_priority": ["floor", "roof", "table", "metric3d_global_floor_like_normals"],
        "semantic_reference_selection_policy": "highest_confidence_with_floor_first_tie_break",
        "semantic_reference_counts": {
            "floor": len(floors),
            "roof": len(roofs),
            "table": len(tables),
        },
        "attempts": gravity_attempts,
    })

    auto_multiwall_debug = {
        "enabled": True,
        "source_wall_count": len(walls),
        "auto_wall_count": 0,
        "method": "semantic_first_then_geometry_fallback",
        "used": False,
        "existing_content_policy": "never_removed_globally; candidate-wall-scoped metadata/cleanup only",
        "walls": []
    }

    try:
        def _merge_x_intervals(intervals):
            if not intervals:
                return 0.0, 0, []
            intervals.sort()
            merged = []
            for x1, x2 in intervals:
                if not merged or x1 > merged[-1][1]:
                    merged.append([x1, x2])
                else:
                    merged[-1][1] = max(merged[-1][1], x2)
            cov = float(sum(x2 - x1 for x1, x2 in merged)) / max(1.0, float(w))
            return cov, max((x2 for _, x2 in merged), default=0), merged

        def _wall_mask_x_coverage(_walls):
            intervals = []
            for _wobj in _walls:
                ys, xs = np.where(_wobj["mask"] > 0)
                if xs.size == 0:
                    continue
                intervals.append([int(xs.min()), int(xs.max()) + 1])
            return _merge_x_intervals(intervals)

        def _wall_bbox_x_coverage(_walls):
            intervals = []
            for _wobj in _walls:
                box = _wobj.get("bbox_xyxy_processed") or []
                if not isinstance(box, (list, tuple)) or len(box) != 4:
                    continue
                try:
                    x1 = int(max(0, min(w - 1, round(float(box[0])))))
                    x2 = int(max(0, min(w, round(float(box[2])))))
                except Exception:
                    continue
                if x2 > x1:
                    intervals.append([x1, x2])
            return _merge_x_intervals(intervals)

        semantic_wall_coverage, semantic_wall_max_right, semantic_wall_intervals = _wall_mask_x_coverage(walls)
        semantic_wall_bbox_coverage, semantic_wall_bbox_max_right, semantic_wall_bbox_intervals = _wall_bbox_x_coverage(walls)
        auto_multiwall_debug["semantic_wall_coverage_ratio"] = semantic_wall_coverage
        auto_multiwall_debug["semantic_wall_max_right_x"] = semantic_wall_max_right
        auto_multiwall_debug["semantic_wall_x_intervals"] = semantic_wall_intervals
        auto_multiwall_debug["semantic_wall_bbox_coverage_ratio"] = semantic_wall_bbox_coverage
        auto_multiwall_debug["semantic_wall_bbox_max_right_x"] = semantic_wall_bbox_max_right
        auto_multiwall_debug["semantic_wall_bbox_x_intervals"] = semantic_wall_bbox_intervals
        semantic_multiwall_adequate = (
            len(walls) >= 2
            and semantic_wall_coverage >= 0.70
            and semantic_wall_max_right >= int(w * 0.82)
        )
        semantic_bbox_multiwall_adequate = (
            len(walls) >= 2
            and semantic_wall_bbox_coverage >= 0.70
            and semantic_wall_bbox_max_right >= int(w * 0.82)
        )

        if len(walls) >= 2 and (semantic_multiwall_adequate or semantic_bbox_multiwall_adequate):
            auto_multiwall_debug["semantic_wall_count"] = len(walls)
            if semantic_multiwall_adequate:
                auto_multiwall_debug["reason"] = "semantic_multiwall_preserved"
            else:
                auto_multiwall_debug["reason"] = "semantic_multiwall_preserved_bbox_mask_coordinate_mismatch_suspected"
                auto_multiwall_debug["semantic_wall_reason"] = "mask_coverage_low_but_bbox_coverage_ok"
                auto_multiwall_debug["used"] = False
        else:
            source_wall_mask = np.zeros((h, w), np.uint8)
            for _wobj in walls:
                source_wall_mask = cv2.bitwise_or(source_wall_mask, _wobj["mask"])
            # If the semantic wall set covers too little horizontal extent, the
            # VLM probably missed a side wall. Use the full image as geometry
            # search space, then subtract support/forbidden/foreground objects.
            if len(walls) >= 2 and not semantic_multiwall_adequate:
                auto_multiwall_debug["semantic_wall_count"] = len(walls)
                auto_multiwall_debug["semantic_wall_reason"] = "low_horizontal_coverage_or_missing_right_side"
                source_wall_mask[:, :] = 255
            if np.count_nonzero(source_wall_mask) == 0:
                source_wall_mask[:, :] = 255

            for _obj in objects.values():
                # Do not remove role=existing_content here. In auto-multiwall fallback,
                # existing artwork/clutter is not a global wall-clear mask; it is only
                # tracked later inside each candidate wall's art_region ∩ wall_mask scope.
                # This prevents hanging on one wall from clearing visual content on other walls.
                if _obj.get("role") in {"support", "forbidden", "foreground", "non_support"}:
                    source_wall_mask[_obj["mask"] > 0] = 0

            n_dot_g = np.abs(np.sum(normal * g_hat.reshape(1, 1, 3), axis=-1))
            vertical_surface = (source_wall_mask > 0) & (n_dot_g <= 0.55)

            n_proj = normal - (np.sum(normal * g_hat.reshape(1, 1, 3), axis=-1, keepdims=True) * g_hat.reshape(1, 1, 3))
            n_proj_norm = np.linalg.norm(n_proj, axis=-1)
            valid = vertical_surface & (n_proj_norm > 1e-4)

            angle = np.arctan2(n_proj[..., 0], n_proj[..., 2])
            bins = np.floor(((angle + np.pi) / (2 * np.pi)) * 16).astype(np.int32)
            bins = np.clip(bins, 0, 15)

            auto_walls = []
            min_auto_area = max(5000, int(h * w * 0.006))
            idx_auto = 0

            for b in range(16):
                bin_mask = ((bins == b) & valid).astype(np.uint8) * 255
                if np.count_nonzero(bin_mask) < min_auto_area:
                    continue
                k3 = np.ones((3, 3), np.uint8)
                k9 = np.ones((9, 9), np.uint8)
                bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, k3)
                bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, k9)

                num_labels, labels_cc, stats, cents = cv2.connectedComponentsWithStats((bin_mask > 0).astype(np.uint8), connectivity=8)
                for cid in range(1, num_labels):
                    area = int(stats[cid, cv2.CC_STAT_AREA])
                    if area < min_auto_area:
                        continue

                    x = int(stats[cid, cv2.CC_STAT_LEFT])
                    y = int(stats[cid, cv2.CC_STAT_TOP])
                    ww = int(stats[cid, cv2.CC_STAT_WIDTH])
                    hh = int(stats[cid, cv2.CC_STAT_HEIGHT])

                    if (y + hh) < int(h * 0.42):
                        continue
                    if ww < 40 or hh < 60:
                        continue

                    mask = ((labels_cc == cid).astype(np.uint8) * 255)
                    oid = f"auto_wall_{idx_auto:02d}"
                    auto_obj = {
                        "object_id": oid,
                        "label": "wall",
                        "role": "wall",
                        "confidence": 0.70,
                        "mask": mask,
                        "polygon": [[x, y], [x + ww, y], [x + ww, y + hh], [x, y + hh]],
                        "source": "geometry_normal_cluster"
                    }
                    auto_walls.append(auto_obj)
                    auto_multiwall_debug["walls"].append({
                        "object_id": oid,
                        "normal_bin": int(b),
                        "area_px": area,
                        "bbox_xywh": [x, y, ww, hh]
                    })
                    idx_auto += 1

            auto_multiwall_debug["auto_wall_count"] = len(auto_walls)
            if len(auto_walls) >= 2:
                walls = auto_walls
                auto_multiwall_debug["used"] = True
                if len(objects) and auto_multiwall_debug.get("semantic_wall_reason") == "low_horizontal_coverage_or_missing_right_side":
                    auto_multiwall_debug["reason"] = "geometry_fallback_replaced_low_coverage_semantic_walls"
                else:
                    auto_multiwall_debug["reason"] = "geometry_fallback_used"
            else:
                auto_multiwall_debug["reason"] = "geometry_fallback_not_needed_or_failed"
    except Exception as e:
        auto_multiwall_debug["used"] = False
        auto_multiwall_debug["reason"] = "geometry_fallback_exception"
        auto_multiwall_debug["error"] = str(e)

    if not walls:
        plan = {"status": "failed", "exit_code": "FAILED_NO_WALL", "fallback_reason": "semantics_json has no wall object"}
        return _zip_response({"hanging_plan.json": json.dumps(plan, ensure_ascii=False, indent=2).encode("utf-8")})

    _artwork_cfg = rules.get("artwork", {})
    if not _artwork_cfg.get("physical_width_m") or not _artwork_cfg.get("physical_height_m"):
        err = {"status": "failed", "exit_code": "FAILED_MISSING_ARTWORK_DIMENSIONS",
               "error": "artwork.physical_width_m and artwork.physical_height_m are required inputs"}
        return _zip_response({"hanging_plan.json": json.dumps(err, ensure_ascii=False, indent=2).encode("utf-8")})
    artwork_w_m = float(_artwork_cfg["physical_width_m"])
    artwork_h_m = float(_artwork_cfg["physical_height_m"])
    max_axis = int(rules.get("mvp", {}).get("max_axis_candidates_per_wall", 9))
    fine_grained_front_policy_enabled = bool(rule_value(
        rules,
        ["structural_clearance_policy", "enable_fine_grained_clear_subregions"],
        True,
    ))
    allow_low_score_as_high_risk = bool(rule_value(
        rules,
        ["structural_clearance_policy", "allow_low_quality_pass_as_high_risk"],
        True,
    ))
    high_risk_score_threshold = float(rule_value(
        rules,
        ["structural_clearance_policy", "high_risk_score_threshold"],
        0.72,
    ))
    medium_risk_score_threshold = float(rule_value(
        rules,
        ["structural_clearance_policy", "medium_risk_score_threshold"],
        0.86,
    ))
    vlm_review_for_high_risk = bool(rule_value(
        rules,
        ["structural_clearance_policy", "vlm_review_required_for_high_risk"],
        True,
    ))
    center_height_floor = float(rules.get("height_rules", {}).get("floor", {}).get("center_height_above_floor_m", 1.52))
    min_wall_inside = float(rule_value(rules, ["hard_constraints", "min_wall_inside_ratio"], 0.97))
    hard_reject_wall_inside = float(rule_value(rules, ["hard_constraints", "hard_reject_wall_inside_ratio"], 0.85))
    min_semantic_axis_safe_coverage = float(rule_value(rules, ["hard_constraints", "min_semantic_axis_safe_mask_coverage"], 0.50))
    min_edge_support_ratio = float(rule_value(rules, ["hard_constraints", "min_artwork_edge_support_ratio"], 0.80))
    min_corner_support_ratio = float(rule_value(rules, ["hard_constraints", "min_artwork_corner_support_ratio"], 0.75))
    hard_reject_edge_support_ratio = float(rule_value(rules, ["hard_constraints", "hard_reject_artwork_edge_support_ratio"], 0.50))
    hard_reject_corner_support_ratio = float(rule_value(rules, ["hard_constraints", "hard_reject_artwork_corner_support_ratio"], 0.50))
    enforce_edge_corner_support = bool(rule_value(rules, ["hard_constraints", "enforce_edge_corner_support_gate"], True))
    max_struct_forbidden = float(rule_value(rules, ["hard_constraints", "max_structural_forbidden_overlap_ratio"], 0.01))
    # Kept for backwards-compatible debug only. existing_content is no longer a
    # rejection/score penalty; cleanup is deferred and candidate-wall-scoped.
    max_existing = float(rule_value(rules, ["hard_constraints", "max_existing_content_overlap_ratio_mvp"], 0.01))
    min_edge_margin_m = float(rule_value(rules, ["hard_constraints", "min_edge_margin_m"], 0.08))
    min_projected_artwork_border_margin_px_original = float(rule_value(
        rules,
        ["hard_constraints", "min_projected_artwork_border_margin_px_original"],
        24,
    ))
    allow_cleanup_existing_as_wall_support = bool(rule_value(
        rules,
        ["existing_content_overlap_policy", "allow_cleanup_existing_content_as_wall_support"],
        True,
    ))
    max_blocking_existing_overlap = float(rule_value(
        rules,
        ["existing_content_overlap_policy", "max_blocking_existing_content_overlap_ratio"],
        0.02,
    ))
    cleanup_trigger_overlap_ratio = float(rule_value(
        rules,
        ["cleanup_policy", "cleanup_required_overlap_ratio"],
        0.0,
    ))
    max_occ = float(rule_value(rules, ["hard_constraints", "max_foreground_occlusion_ratio"], 0.08))
    # Final policy: do not reject, review, or down-rank candidates because the
    # projected artwork appears small in pixels. Small physical artworks and far
    # walls must still be eligible when wall support and geometry are safe.
    projection_size_gate_enabled = False
    max_quad_aspect_ratio_long_over_short = float(rule_value(rules, ["hard_constraints", "max_quad_aspect_ratio_long_over_short"], 8.0))
    wall_center_composition_weight = float(rule_value(rules, ["candidate_ranking", "wall_center_composition_weight"], 0.32))
    wall_corner_soft_penalty_weight = float(rule_value(rules, ["candidate_ranking", "wall_corner_soft_penalty_weight"], 0.12))
    wall_center_composition_weight = max(0.0, min(0.60, wall_center_composition_weight))
    wall_corner_soft_penalty_weight = max(0.0, min(0.40, wall_corner_soft_penalty_weight))

    # Geometry self-defense against wall openings / recesses:
    # If semantics misses an arch/opening, projected art may still land on a hole.
    # Compare actual scene depth vs projected wall-plane depth inside the art region.
    fov_deg = float(bundle.intr.fov_deg or 0.0)
    base_plane_depth_error = float(rule_value(rules, ["hard_constraints", "max_plane_depth_error_m"], 0.08))
    if fov_deg > 85:
        max_plane_depth_error_effective = 0.20
    elif fov_deg > 70:
        max_plane_depth_error_effective = max(base_plane_depth_error, 0.12)
    else:
        max_plane_depth_error_effective = base_plane_depth_error


    foreground_depth_margin_effective_m = max(0.03, min(0.12, max_plane_depth_error_effective * 0.5))
    min_plane_consistency_ratio = 0.75
    max_recess_behind_wall_ratio = 0.08

    # Adaptive depth tolerance:
    # Metric3D depth noise grows with distance and is worse under high FOV.
    # Use a per-pixel tolerance instead of a single scalar threshold.
    # This avoids false rejection of real wall regions at 4-8m while still
    # rejecting true openings whose depth is usually far behind the wall plane.
    default_depth_relative_tol = 0.035 if fov_deg > 85 else (0.025 if fov_deg > 70 else 0.015)
    default_depth_tolerance_max_m = 0.35 if fov_deg > 85 else (0.25 if fov_deg > 70 else 0.15)

    depth_tolerance_relative = float(rule_value(
        rules,
        ["hard_constraints", "plane_depth_relative_tol"],
        default_depth_relative_tol
    ))
    depth_tolerance_max_m = float(rule_value(
        rules,
        ["hard_constraints", "plane_depth_tolerance_max_m"],
        default_depth_tolerance_max_m
    ))

    def make_depth_tol_map(expected_depth):
        return np.clip(
            np.maximum(max_plane_depth_error_effective, expected_depth * depth_tolerance_relative),
            max_plane_depth_error_effective,
            depth_tolerance_max_m
        )


    # Normal consistency inside the projected artwork region:
    # This catches side returns / angled reveal surfaces whose depth may be close
    # to the main wall plane but whose normals do not match the target wall.
    min_normal_consistency_ratio = 0.75
    normal_consistency_dot_threshold = 0.65
    max_normal_mismatch_ratio = 0.20

    candidates = []
    rejected = []
    debug_overlays = []
    candidate_idx = 0
    wall_debug_registry = {}
    wall_object_registry = {}

    # Pre-collect masks
    structural_masks = [
        o for o in objects.values()
        if o.get("role") == "forbidden" or semantic_label_matches(o.get("label"), structural_forbidden_labels)
    ]
    existing_masks = [
        o for o in objects.values()
        if o.get("role") == "existing_content" or semantic_label_matches(o.get("label"), existing_content_labels)
    ]
    removable_existing_labels = _rule_label_set(
        rules,
        ["existing_content_overlap_policy", "removable_labels"],
        DEFAULT_REMOVABLE_EXISTING_CONTENT_LABELS,
    )
    blocking_existing_labels = _rule_label_set(
        rules,
        ["existing_content_overlap_policy", "blocking_labels"],
        DEFAULT_BLOCKING_EXISTING_CONTENT_LABELS,
    )
    removable_existing_masks = []
    blocking_existing_masks = []
    for _existing_obj in existing_masks:
        _existing_kind = _classify_existing_content_object(
            _existing_obj,
            removable_labels=removable_existing_labels,
            blocking_labels=blocking_existing_labels,
        )
        _existing_obj["existing_content_safety_class"] = _existing_kind
        if _existing_kind == "blocking":
            blocking_existing_masks.append(_existing_obj)
        else:
            removable_existing_masks.append(_existing_obj)
    # HANGING_POLICY_PATCH_V6_BEGIN_FOREGROUND_MASKS
    foreground_masks = [
        o for o in objects.values()
        if o.get("role") == "foreground"
        or str(o.get("label", "")).lower() in {"foreground", "foreground_object", "person", "human", "people"}
    ]
    # HANGING_POLICY_PATCH_V6_END_FOREGROUND_MASKS
    semantic_object_inventory = {
        "status": semantics.get("semantic_object_inventory_status") or semantics.get("object_inventory", {}).get("status") or "unknown",
        "source": "semantic_mvp.object_inventory",
        "per_wall_high_res_crop_used": bool(semantics.get("object_inventory", {}).get("per_wall_high_res_crop_used", False)),
        "existing_content_count": int(len(existing_masks)),
        "removable_existing_content_count": int(len(removable_existing_masks)),
        "blocking_existing_content_count": int(len(blocking_existing_masks)),
        "blocking_existing_content_labels": sorted(blocking_existing_labels),
        "foreground_count": int(len(foreground_masks)),
        "counts_by_role": (semantics.get("object_inventory", {}) or {}).get("counts_by_role", {}),
        "counts_by_label": (semantics.get("object_inventory", {}) or {}).get("counts_by_label", {}),
        "warning": "empty semantic cleanup inventory; cleanup masks may be empty because semantic objects were not detected" if not existing_masks and not foreground_masks else None,
    }

    for wall_obj in walls:
        wall_id = wall_obj["object_id"]
        wall_object_registry[wall_id] = wall_obj
        wall_debug_registry.setdefault(wall_id, {"wall_id": wall_id, "wall_source": wall_obj.get("source"), "wall_mask": wall_obj.get("mask").copy(), "summary": {}})
        raw_wall_mask_before_structural = wall_obj["mask"].copy()
        wall_mask = raw_wall_mask_before_structural.copy()
        # subtract structural forbidden from wall mask
        for fm in structural_masks:
            wall_mask[fm["mask"] > 0] = 0
        # Adaptive vertical-wall filter:
        # Coarse MVP semantic wall boxes may include floor/ceiling pixels.
        # Filter candidate wall pixels using abs(dot(normal_xyz, g_hat_cam)) <= thr,
        # where smaller values are more wall-like (normal approximately horizontal,
        # perpendicular to gravity). Use adaptive thresholds so large pitch/yaw shots
        # do not fail just because the strict threshold leaves too few points.
        wall_normal_dot_g = np.abs(np.sum(normal * g_hat.reshape(1, 1, 3), axis=-1))
        raw_wall_points_count = int(np.count_nonzero(wall_mask))

        selected_threshold = None
        selected_vertical_points_count = 0
        selected_wall_mask = wall_mask.copy()
        threshold_trials = [0.45, 0.60, 0.75]
        threshold_debug = []

        for thr in threshold_trials:
            candidate_mask = wall_mask.copy()
            candidate_mask[wall_normal_dot_g > thr] = 0
            candidate_count = int(np.count_nonzero(candidate_mask))
            threshold_debug.append({
                "threshold": float(thr),
                "vertical_wall_points": candidate_count
            })
            if candidate_count >= 1000:
                selected_threshold = float(thr)
                selected_vertical_points_count = candidate_count
                selected_wall_mask = candidate_mask
                break

        if selected_threshold is not None:
            wall_mask = selected_wall_mask
            wall_filter_debug = {
                "enabled": True,
                "mode": "adaptive_vertical_wall_filter",
                "selected_threshold": selected_threshold,
                "threshold_trials": threshold_debug,
                "raw_wall_points": raw_wall_points_count,
                "vertical_wall_points": selected_vertical_points_count
            }
        else:
            wall_filter_debug = {
                "enabled": False,
                "mode": "fallback_raw_wall_mask",
                "reason": "not_enough_vertical_wall_points_for_all_thresholds",
                "threshold_trials": threshold_debug,
                "raw_wall_points": raw_wall_points_count,
                "vertical_wall_points": 0
            }

        if wall_id in wall_debug_registry:
            wall_debug_registry[wall_id]["filtered_wall_mask"] = wall_mask.copy()
            wall_debug_registry[wall_id]["summary"]["wall_filter_debug"] = wall_filter_debug

        causal_wall_light: dict[str, Any] = {}
        if _causal_enabled():
            try:
                wall_mask_bool = wall_mask > 0
                stats = causal_imaging.light_statistics(
                    room_rgb,
                    room_mask=causal_room_mask,
                    window_mask=causal_window_mask,
                    wall_mask=wall_mask_bool,
                    ceiling_mask=causal_ceiling_mask,
                )
                noise = causal_imaging.surface_visual_noise(room_rgb, wall_mask_bool)
                causal_wall_light = {
                    "light_facts": {
                        "rho": stats.rho,
                        "wall_lambda": stats.wall_lambda,
                        "lambda_ceil": stats.lambda_ceil,
                        "surface_visual_noise_score": noise.score,
                    },
                    "surface_visual_noise_band": noise.band,
                    "surface_visual_noise_debug": {
                        "edge_density": noise.edge_density,
                        "hf_energy": noise.hf_energy,
                        "local_std": noise.local_std,
                    },
                    "semantic_light": _wall_causal_semantic_light(causal_semantic_light, wall_obj),
                }
                if wall_id in wall_debug_registry:
                    wall_debug_registry[wall_id]["summary"]["causal_light"] = causal_wall_light
            except Exception as _causal_e:
                causal_wall_light = {"error": str(_causal_e)}

        pts = points_from_mask(xyz, wall_mask)
        if len(pts) < 1000:
            rejected.append({
                "wall_id": wall_id,
                "reason": "wall_too_small_after_vertical_filter",
                "points": int(len(pts)),
                "wall_filter_debug": wall_filter_debug
            })
            continue
        try:
            plane, inlier_ratio = fit_plane_ransac(pts, threshold=0.08)

            # Post-RANSAC vertical-wall validation:
            # For a true vertical wall, wall normal should be roughly perpendicular to gravity.
            plane_n = np.array(plane.n, np.float32)
            fitted_plane_abs_dot_g = float(abs(np.dot(plane_n, g_hat)))
            wall_filter_debug["fitted_plane_abs_dot_g"] = fitted_plane_abs_dot_g
            wall_filter_debug["wall_plane_verticality_pass_threshold"] = 0.55

            if fitted_plane_abs_dot_g > 0.55:
                rejected.append({
                    "wall_id": wall_id,
                    "reason": "fitted_plane_is_not_vertical_wall",
                    "fitted_plane_abs_dot_g": fitted_plane_abs_dot_g,
                    "wall_filter_debug": wall_filter_debug
                })
                continue

            basis = make_wall_basis(plane, pts, g_hat)
        except Exception as e:
            rejected.append({"wall_id": wall_id, "reason": f"wall_plane_fit_failed: {e}", "wall_filter_debug": wall_filter_debug})
            continue

        origin = np.array(basis.origin, np.float32)
        u = np.array(basis.u, np.float32)
        v = np.array(basis.v, np.float32)

        # Keep global wall-local coordinates for height fallback and t_floor.
        rel = pts - origin
        s_vals = rel @ u
        t_vals = rel @ v
        if len(s_vals) == 0:
            continue

        # Build solid wall components before axis generation.
        # A coarse semantic wall may include openings. We keep only pixels that:
        #   1) lie close to the fitted wall plane in depth
        #   2) have normals consistent with the fitted target wall normal
        wall_depth_full = ray_plane_depth_for_mask(wall_mask, bundle.intr, plane)
        wall_valid = (wall_mask > 0) & np.isfinite(wall_depth_full)
        wall_depth_err = np.abs(depth - wall_depth_full)
        wall_depth_tol = make_depth_tol_map(wall_depth_full)

        wall_n_np_for_solid = np.array(basis.n, np.float32)
        wall_normal_alignment_for_solid = np.abs(np.sum(normal * wall_n_np_for_solid.reshape(1, 1, 3), axis=-1))

        solid_wall_bool = (
            wall_valid
            & (wall_depth_err <= wall_depth_tol)
            & (wall_normal_alignment_for_solid >= normal_consistency_dot_threshold)
        )

        solid_wall_mask_u8 = (solid_wall_bool.astype(np.uint8) * 255)

        k3 = np.ones((3, 3), np.uint8)
        k7 = np.ones((7, 7), np.uint8)
        solid_wall_mask_u8 = cv2.morphologyEx(solid_wall_mask_u8, cv2.MORPH_OPEN, k3)
        solid_wall_mask_u8 = cv2.morphologyEx(solid_wall_mask_u8, cv2.MORPH_CLOSE, k7)

        if wall_id in wall_debug_registry:
            wall_debug_registry[wall_id].update({
                "plane_consistency_mask": ((wall_valid & (wall_depth_err <= wall_depth_tol)).astype(np.uint8) * 255),
                "normal_consistency_mask": ((wall_valid & (wall_normal_alignment_for_solid >= normal_consistency_dot_threshold)).astype(np.uint8) * 255),
                "solid_wall_mask": solid_wall_mask_u8.copy(),
                "opening_recess_mask": (((wall_mask > 0) & ~(solid_wall_mask_u8 > 0)).astype(np.uint8) * 255),
            })

        num_labels, comp_labels, comp_stats, comp_centroids = cv2.connectedComponentsWithStats(
            (solid_wall_mask_u8 > 0).astype(np.uint8),
            connectivity=8
        )

        raw_wall_area_px = int(np.count_nonzero(wall_mask))
        solid_wall_area_px = int(np.count_nonzero(solid_wall_mask_u8))
        min_component_area_px = max(2500, int(raw_wall_area_px * 0.015))

        axes = []
        solid_component_debug = {
            "enabled": True,
            "mode": "solid_wall_connected_components_v2",
            "raw_wall_area_px": raw_wall_area_px,
            "solid_wall_area_px": solid_wall_area_px,
            "depth_tolerance_relative": float(depth_tolerance_relative),
            "depth_tolerance_max_m": float(depth_tolerance_max_m),
            "wall_depth_tolerance_p50_m": float(np.nanpercentile(wall_depth_tol[wall_valid], 50)) if np.any(wall_valid) else None,
            "normal_consistency_dot_threshold": float(normal_consistency_dot_threshold),
            "min_component_area_px": int(min_component_area_px),
            "min_edge_margin_m": float(min_edge_margin_m),
            "required_component_width_m": float(artwork_w_m + 2.0 * min_edge_margin_m),
            "component_count": int(max(0, num_labels - 1)),
            "components": []
        }

        for comp_id in range(1, num_labels):
            area_px = int(comp_stats[comp_id, cv2.CC_STAT_AREA])
            if area_px < min_component_area_px:
                continue

            comp_mask = ((comp_labels == comp_id).astype(np.uint8) * 255)
            comp_pts = points_from_mask(xyz, comp_mask, max_points=30000)
            if len(comp_pts) < 300:
                continue

            comp_rel = comp_pts - origin
            comp_s = comp_rel @ u
            comp_t = comp_rel @ v
            comp_s_min = float(np.percentile(comp_s, 5))
            comp_s_max = float(np.percentile(comp_s, 95))
            comp_t_min = float(np.percentile(comp_t, 5))
            comp_t_max = float(np.percentile(comp_t, 95))
            comp_width_m = comp_s_max - comp_s_min
            comp_height_m = comp_t_max - comp_t_min

            comp_debug = {
                "component_id": int(comp_id),
                "area_px": area_px,
                "s_range_m": [comp_s_min, comp_s_max],
                "t_range_m": [comp_t_min, comp_t_max],
                "width_m": float(comp_width_m),
                "height_m": float(comp_height_m)
            }
            solid_component_debug["components"].append(comp_debug)

            required_component_width_m = float(artwork_w_m + 2.0 * min_edge_margin_m)
            comp_debug["required_component_width_m"] = required_component_width_m
            if comp_width_m < required_component_width_m:
                comp_debug["skip_reason"] = "component_too_narrow_for_artwork_with_edge_margin"
                continue

            center_s = 0.5 * (comp_s_min + comp_s_max)
            axes.append({
                "axis_s_m": float(center_s),
                "component_id": int(comp_id),
                "component_area_px": area_px,
                "component_s_range_m": [float(comp_s_min), float(comp_s_max)],
                "component_width_m": float(comp_width_m),
                "source": "solid_component_center"
            })

            if comp_width_m >= artwork_w_m * 1.8:
                axes.append({
                    "axis_s_m": float(comp_s_min + comp_width_m / 3.0),
                    "component_id": int(comp_id),
                    "component_area_px": area_px,
                    "component_s_range_m": [float(comp_s_min), float(comp_s_max)],
                    "component_width_m": float(comp_width_m),
                    "source": "solid_component_one_third"
                })
                axes.append({
                    "axis_s_m": float(comp_s_min + 2.0 * comp_width_m / 3.0),
                    "component_id": int(comp_id),
                    "component_area_px": area_px,
                    "component_s_range_m": [float(comp_s_min), float(comp_s_max)],
                    "component_width_m": float(comp_width_m),
                    "source": "solid_component_two_third"
                })

        semantic_occupied_mask = np.zeros_like(wall_mask, dtype=np.uint8)
        semantic_occupied_objects: list[Dict[str, Any]] = []
        for _sem_obj in list(existing_masks) + list(foreground_masks):
            _sem_mask = _sem_obj.get("mask")
            if _sem_mask is None:
                continue
            _overlap = int(np.count_nonzero((_sem_mask > 0) & (wall_mask > 0)))
            if _overlap <= 0:
                continue
            semantic_occupied_mask[_sem_mask > 0] = 255
            semantic_occupied_objects.append({
                "object_id": _sem_obj.get("object_id"),
                "label": _sem_obj.get("label"),
                "role": _sem_obj.get("role"),
                "wall_overlap_px": _overlap,
            })
        semantic_clear_wall_mask = wall_mask.copy()
        semantic_clear_wall_mask[semantic_occupied_mask > 0] = 0
        safe_axis_wall_mask = solid_wall_mask_u8.copy()
        safe_axis_wall_mask[semantic_occupied_mask > 0] = 0
        semantic_axis_gate = semantic_axis_policy(
            semantic_wall_mask=semantic_clear_wall_mask,
            safe_axis_wall_mask=safe_axis_wall_mask,
            min_safe_coverage=min_semantic_axis_safe_coverage,
        )
        semantic_axis_debug = {
            "enabled": bool(fine_grained_front_policy_enabled),
            "source": "semantic_clear_wall_mask_after_structural_and_object_subtraction",
            "added_count": 0,
            "semantic_object_exclusion": {
                "enabled": True,
                "excluded_object_count": len(semantic_occupied_objects),
                "excluded_objects": semantic_occupied_objects,
                "policy": "semantic_clear_subregion subtracts existing_content and foreground masks; no per-wall high-res crop is used",
            },
            **semantic_axis_gate,
        }
        if wall_id in wall_debug_registry:
            wall_debug_registry[wall_id]["semantic_clear_wall_mask"] = semantic_clear_wall_mask.copy()
            wall_debug_registry[wall_id]["safe_axis_wall_mask"] = safe_axis_wall_mask.copy()
            wall_debug_registry[wall_id]["summary"]["semantic_axis_status"] = semantic_axis_gate.get("status")
            wall_debug_registry[wall_id]["summary"]["semantic_axis_safe_mask_coverage"] = semantic_axis_gate.get("semantic_axis_safe_mask_coverage")
            wall_debug_registry[wall_id]["summary"]["semantic_object_exclusion_count"] = len(semantic_occupied_objects)
        if fine_grained_front_policy_enabled and semantic_axis_gate.get("allowed"):
            sem_pts = points_from_mask(xyz, semantic_clear_wall_mask, max_points=30000)
            sem_pts = sem_pts[np.isfinite(sem_pts).all(axis=1)] if sem_pts.size else sem_pts
            if len(sem_pts) >= 50:
                sem_rel = sem_pts - origin.reshape(1, 3)
                sem_s_values = sem_rel @ u.reshape(3)
                sem_s_min = float(np.percentile(sem_s_values, 5))
                sem_s_max = float(np.percentile(sem_s_values, 95))
            else:
                sem_s_min = sem_s_max = 0.0
                semantic_axis_debug["skip_reason"] = "semantic_clear_wall_mask_has_too_few_points_after_object_subtraction"
            sem_width_m = float(sem_s_max - sem_s_min)
            semantic_axis_debug.update({
                "s_range_m": [sem_s_min, sem_s_max],
                "width_m": sem_width_m,
                "required_width_m": float(artwork_w_m + 2.0 * min_edge_margin_m),
            })
            if sem_width_m >= artwork_w_m + 2.0 * min_edge_margin_m:
                semantic_positions = [
                    (0.50, "semantic_clear_subregion_center"),
                    (1.0 / 3.0, "semantic_clear_subregion_one_third"),
                    (2.0 / 3.0, "semantic_clear_subregion_two_third"),
                    (0.25, "semantic_clear_subregion_one_quarter"),
                    (0.75, "semantic_clear_subregion_three_quarter"),
                ]
                existing_axis_s = [float(a.get("axis_s_m", 0.0)) for a in axes]
                min_axis_separation_m = max(0.04, artwork_w_m * 0.10)
                for frac, source_name in semantic_positions:
                    axis_s_sem = float(sem_s_min + sem_width_m * frac)
                    if any(abs(axis_s_sem - ex) < min_axis_separation_m for ex in existing_axis_s):
                        continue
                    axes.append({
                        "axis_s_m": axis_s_sem,
                        "component_id": None,
                        "component_area_px": raw_wall_area_px,
                        "component_s_range_m": [float(sem_s_min), float(sem_s_max)],
                        "component_width_m": sem_width_m,
                        "source": source_name,
                        "candidate_generation_policy": "fine_grained_clear_subregion_structural_clearance_then_risk_score",
                    })
                    existing_axis_s.append(axis_s_sem)
                    semantic_axis_debug["added_count"] += 1
            else:
                semantic_axis_debug["skip_reason"] = "semantic_wall_section_too_narrow_for_artwork_with_edge_margin"
        elif fine_grained_front_policy_enabled:
            max_rect_axes, max_rect_debug = _semantic_solid_max_empty_rect_axes(
                semantic_clear_wall_mask=semantic_clear_wall_mask,
                solid_wall_mask=solid_wall_mask_u8,
                xyz=xyz,
                basis=basis,
                artwork_w_m=artwork_w_m,
                artwork_h_m=artwork_h_m,
                min_edge_margin_m=min_edge_margin_m,
                metric_seed_tolerance_m=float(rule_value(
                    rules,
                    ["candidate_generation", "max_empty_rect_metric_seed_tolerance_m"],
                    0.02,
                )),
                max_axes=max(1, min(3, max_axis)),
            )
            semantic_axis_debug["max_empty_rect_fallback"] = max_rect_debug
            if max_rect_axes:
                existing_axis_s = [float(a.get("axis_s_m", 0.0)) for a in axes]
                min_axis_separation_m = max(0.04, artwork_w_m * 0.10)
                for max_rect_axis in max_rect_axes:
                    axis_s_rect = float(max_rect_axis.get("axis_s_m", 0.0))
                    if any(abs(axis_s_rect - ex) < min_axis_separation_m for ex in existing_axis_s):
                        continue
                    axes.append(max_rect_axis)
                    existing_axis_s.append(axis_s_rect)
                    semantic_axis_debug["added_count"] += 1
                semantic_axis_debug["skip_reason"] = None
                semantic_axis_debug["status"] = "low_safe_coverage_overridden_by_max_empty_rect"
            else:
                semantic_axis_debug["skip_reason"] = "semantic_axis_set_dropped_low_safe_mask_coverage"

        solid_component_debug["semantic_axis_debug"] = semantic_axis_debug

        def _axis_sort_key(axis_item: Dict[str, Any]):
            source = str(axis_item.get("source", ""))
            if source.startswith("solid_component"):
                priority = 3
            elif source.startswith("semantic_clear_subregion"):
                priority = 2
            else:
                priority = 1
            return (priority, int(axis_item.get("component_area_px", 0)), -abs(float(axis_item.get("axis_s_m", 0.0))))

        axes = sorted(axes, key=_axis_sort_key, reverse=True)[:max_axis]

        # HANGING_POLICY_PATCH_V6_BEGIN_STRUCTURAL_GAP_AXES
        # Business rule: a wall with windows/doors is still eligible if there is
        # a fine-grained horizontal gap where artwork vertical edges keep at least
        # artwork_width / 3 distance from structural regions. Coarse center/third
        # axes often hug a window edge, so explicitly add gap-center axes.
        try:
            _hpv6_gap_enabled = bool(rule_value(rules, ["structural_clearance_policy", "generate_gap_center_axes"], True))
            _hpv6_clearance_fraction = float(rule_value(rules, ["structural_clearance_policy", "min_edge_clearance_fraction_of_artwork_width"], 1.0 / 3.0))
            _hpv6_required_struct_clearance_m = float(artwork_w_m) * _hpv6_clearance_fraction
            _hpv6_same_wall_max_dist_m = float(rule_value(rules, ["structural_clearance_policy", "same_wall_plane_max_distance_m"], 0.75))
            _hpv6_gap_debug = {
                "enabled": bool(_hpv6_gap_enabled),
                "required_structural_edge_clearance_m": float(_hpv6_required_struct_clearance_m),
                "same_wall_plane_max_distance_m": float(_hpv6_same_wall_max_dist_m),
                "source": "structural_gap_center_axes_v6",
                "added_count": 0,
                "structural_intervals_expanded_m": [],
                "gaps_m": [],
            }
            if _hpv6_gap_enabled and len(s_vals) > 0:
                _hpv6_wall_s_min = float(np.percentile(s_vals, 5))
                _hpv6_wall_s_max = float(np.percentile(s_vals, 95))
                _hpv6_intervals = []
                for _hpv6_obj in structural_masks:
                    _hpv6_mask = _hpv6_obj.get("mask")
                    if _hpv6_mask is None or int(np.count_nonzero(_hpv6_mask)) <= 0:
                        continue
                    _hpv6_pts = points_from_mask(xyz, _hpv6_mask, max_points=25000)
                    _hpv6_pts = _hpv6_pts[np.isfinite(_hpv6_pts).all(axis=1)] if _hpv6_pts.size else _hpv6_pts
                    if len(_hpv6_pts) < 30:
                        continue
                    _hpv6_plane_dist = np.abs(_hpv6_pts @ np.array(plane.n, np.float32).reshape(3) + float(plane.d))
                    _hpv6_p50 = float(np.percentile(_hpv6_plane_dist, 50)) if _hpv6_plane_dist.size else 999.0
                    if _hpv6_p50 > _hpv6_same_wall_max_dist_m:
                        continue
                    _hpv6_rel = _hpv6_pts - origin.reshape(1, 3)
                    _hpv6_s = _hpv6_rel @ u.reshape(3)
                    _hpv6_s = _hpv6_s[np.isfinite(_hpv6_s)]
                    if _hpv6_s.size < 30:
                        continue
                    _hpv6_s0 = float(np.percentile(_hpv6_s, 5)) - _hpv6_required_struct_clearance_m
                    _hpv6_s1 = float(np.percentile(_hpv6_s, 95)) + _hpv6_required_struct_clearance_m
                    if _hpv6_s1 < _hpv6_wall_s_min or _hpv6_s0 > _hpv6_wall_s_max:
                        continue
                    _hpv6_s0 = max(_hpv6_wall_s_min, _hpv6_s0)
                    _hpv6_s1 = min(_hpv6_wall_s_max, _hpv6_s1)
                    if _hpv6_s1 > _hpv6_s0:
                        _hpv6_intervals.append((_hpv6_s0, _hpv6_s1, str(_hpv6_obj.get("object_id") or _hpv6_obj.get("label") or "structural"), _hpv6_p50))
                        _hpv6_gap_debug["structural_intervals_expanded_m"].append({
                            "object_id": str(_hpv6_obj.get("object_id") or ""),
                            "label": str(_hpv6_obj.get("label") or ""),
                            "s_range_expanded_m": [float(_hpv6_s0), float(_hpv6_s1)],
                            "plane_distance_p50_m": float(_hpv6_p50),
                        })
                _hpv6_intervals_sorted = sorted(_hpv6_intervals, key=lambda x: x[0])
                _hpv6_merged = []
                for _hpv6_i in _hpv6_intervals_sorted:
                    if not _hpv6_merged or _hpv6_i[0] > _hpv6_merged[-1][1]:
                        _hpv6_merged.append([_hpv6_i[0], _hpv6_i[1]])
                    else:
                        _hpv6_merged[-1][1] = max(_hpv6_merged[-1][1], _hpv6_i[1])
                _hpv6_cursor = _hpv6_wall_s_min
                _hpv6_gaps = []
                for _hpv6_s0, _hpv6_s1 in _hpv6_merged:
                    if _hpv6_s0 > _hpv6_cursor:
                        _hpv6_gaps.append((_hpv6_cursor, _hpv6_s0))
                    _hpv6_cursor = max(_hpv6_cursor, _hpv6_s1)
                if _hpv6_cursor < _hpv6_wall_s_max:
                    _hpv6_gaps.append((_hpv6_cursor, _hpv6_wall_s_max))
                _hpv6_existing_axis_values = [float(a.get("axis_s_m", 0.0)) for a in axes]
                for _hpv6_g0, _hpv6_g1 in sorted(_hpv6_gaps, key=lambda g: g[1] - g[0], reverse=True):
                    _hpv6_width = float(_hpv6_g1 - _hpv6_g0)
                    _hpv6_gap_debug["gaps_m"].append([float(_hpv6_g0), float(_hpv6_g1), float(_hpv6_width)])
                    if _hpv6_width < float(artwork_w_m):
                        continue
                    _hpv6_axis_s = float(0.5 * (_hpv6_g0 + _hpv6_g1))
                    if any(abs(_hpv6_axis_s - _old) < max(0.03, float(artwork_w_m) * 0.08) for _old in _hpv6_existing_axis_values):
                        continue
                    axes.append({
                        "axis_s_m": _hpv6_axis_s,
                        "component_id": None,
                        "component_area_px": int(raw_wall_area_px),
                        "source": "structural_gap_center",
                        "structural_gap_s_range_m": [float(_hpv6_g0), float(_hpv6_g1)],
                        "structural_gap_width_m": float(_hpv6_width),
                        "required_structural_edge_clearance_m": float(_hpv6_required_struct_clearance_m),
                    })
                    _hpv6_existing_axis_values.append(_hpv6_axis_s)
                    _hpv6_gap_debug["added_count"] += 1
            solid_component_debug["structural_gap_axis_debug"] = _hpv6_gap_debug
        except Exception as _hpv6_e:
            try:
                solid_component_debug["structural_gap_axis_debug"] = {"enabled": False, "error": str(_hpv6_e)}
            except Exception:
                pass
        # HANGING_POLICY_PATCH_V6_END_STRUCTURAL_GAP_AXES

        if not axes:
            rejected.append({
                "wall_id": wall_id,
                "reason": "no_safe_axis_and_insufficient_solid_component",
                "wall_has_no_safe_candidate": True,
                "semantic_axis_status": semantic_axis_debug.get("status"),
                "geometry_axis_status": "unavailable_insufficient_solid_component",
                "solid_component_debug": solid_component_debug,
                "min_edge_margin_m": float(min_edge_margin_m),
                "required_component_width_m": float(artwork_w_m + 2.0 * min_edge_margin_m),
            })
            if wall_id in wall_debug_registry:
                wall_debug_registry[wall_id]["summary"]["solid_component_debug"] = solid_component_debug
                wall_debug_registry[wall_id]["summary"]["wall_has_no_safe_candidate"] = True
            continue
        else:
            solid_component_debug["fallback_used"] = False
            solid_component_debug["geometry_axis_status"] = "available_safe_component_or_structural_gap"

        # floor absolute height:
        # Use the formal height-to-wall-coordinate conversion instead of
        # the temporary approximation center_t = t_floor + 1.52.
        #
        # Height frame:
        #   h_abs(X) = dot(X, g_hat_cam)
        #
        # Floor point on current wall:
        #   P_floor = wall_origin + wall_v * t_floor
        #
        # Target center absolute height:
        #   center_height_abs_m = dot(P_floor, g_hat_cam) + center_height_above_floor_m
        #
        # Convert absolute height to wall-local coordinate:
        #   center_t_m = (center_height_abs_m - dot(wall_origin, g_hat_cam)) / dot(wall_v, g_hat_cam)
        visible_wall_s_range_m = [float(np.percentile(s_vals, 5)), float(np.percentile(s_vals, 95))]
        s_range_for_floor = visible_wall_s_range_m
        floor_wall_estimation = _estimate_floor_wall_intersection(
            floor_mask=floor_mask,
            xyz=xyz,
            plane=plane,
            basis=basis,
            s_range_m=s_range_for_floor,
            visible_wall_t_vals=t_vals,
            solid_wall_mask=solid_wall_mask_u8,
            normal=normal,
            rules=rules,
        )
        t_floor = float(floor_wall_estimation["t_floor_m"])
        g_hat_np = np.array(basis.g_hat_cam, np.float32)
        wall_origin_np = np.array(basis.origin, np.float32)
        wall_v_np = np.array(basis.v, np.float32)
        dot_v_g = float(np.dot(wall_v_np, g_hat_np))

        if abs(dot_v_g) < 0.85:
            rejected.append({
                "wall_id": wall_id,
                "reason": "height_projection_unstable_dot_v_g",
                "dot_v_g": dot_v_g,
                "floor_wall_estimation": floor_wall_estimation,
            })
            continue

        floor_height_abs_m = float(floor_wall_estimation["floor_height_abs_m"])
        floor_source_trusted = floor_source_is_metric_trusted(floor_wall_estimation.get("source"))
        if wall_id in wall_debug_registry:
            wall_debug_registry[wall_id]["floor_wall_estimation"] = floor_wall_estimation
            wall_debug_registry[wall_id]["summary"]["floor_wall_estimation_source"] = floor_wall_estimation.get("source")
        if floor_source_trusted and not bool(floor_wall_estimation.get("floor_below_camera", floor_height_abs_m < 0.0)):
            rejected.append({
                "wall_id": wall_id,
                "reason": "trusted_floor_direction_invalid_not_below_camera",
                "wall_has_no_safe_candidate": True,
                "floor_wall_estimation": floor_wall_estimation,
                "g_hat_debug": g_debug,
            })
            continue
        if bool((rules.get("room_corner_floor_fit", {}) or {}).get("require_trusted_floor_for_auto_candidate", True)) and not floor_source_trusted:
            rejected.append({"wall_id": wall_id, "reason": "trusted_wall_floor_intersection_required", "wall_has_no_safe_candidate": True, "floor_wall_estimation": floor_wall_estimation})
            continue
        for axis_item in axes:
            axis_s = float(axis_item["axis_s_m"])
            support_context = _estimate_axis_support_context(
                objects,
                xyz,
                basis,
                axis_s,
                artwork_w_m,
                artwork_h_m,
                rules,
                floor_height_abs_m,
                plane=plane,
                floor_source_trusted=floor_source_trusted,
            )
            if not floor_source_trusted:
                support_context.update({
                    "height_mode": "provisional_visual_review_required",
                    "metric_height_finalized": False,
                    "height_is_provisional": True,
                    "requires_frontend_visual_height_selection": True,
                    "requires_visual_height_selection": True,
                    "requires_flux_visual_height_refinement": False,
                    "height_band_base": "unknown",
                    "height_warning": "web_image_metric_scale_or_floor_anchor_untrusted",
                })
            else:
                support_context.setdefault("height_band_base", "estimated_full_wall_or_trusted_floor")
            rule_center_height_abs_m = float(support_context["center_height_abs_m"])
            max_height_rule_deviation_m = float(rule_value(
                rules,
                ["height_rule_consistency", "max_metric_rule_deviation_m"],
                0.10,
            ))
            height_contract = resolve_candidate_center_t(
                rule_center_height_abs_m=rule_center_height_abs_m,
                floor_height_abs_m=floor_height_abs_m,
                wall_origin_dot_g=float(np.dot(wall_origin_np, g_hat_np)),
                dot_v_g=dot_v_g,
                height_mode=str(support_context.get("height_mode", "metric_rule")),
                axis_item=axis_item,
                max_metric_rule_deviation_m=max_height_rule_deviation_m,
            )
            center_t = float(height_contract["resolved_center_t_m"])
            center_height_abs_m = float(height_contract["resolved_center_height_abs_m"])
            center_height_above_floor_m = float(height_contract["resolved_center_height_above_floor_m"])
            support_context["center_height_abs_m"] = center_height_abs_m
            support_context["center_height_above_floor_m"] = center_height_above_floor_m
            support_context["height_contract"] = height_contract
            visible_t_min = float(np.percentile(t_vals, 5))
            visible_t_max = float(np.percentile(t_vals, 95))
            center_t_normalized = float((center_t - visible_t_min) / max(1e-6, visible_t_max - visible_t_min))

            vertical_fit_debug = evaluate_semantic_clear_rect_vertical_fit(
                center_t_m=center_t,
                artwork_height_m=artwork_h_m,
                min_edge_margin_m=min_edge_margin_m,
                component_t_range_m=axis_item.get("component_t_range_m"),
                axis_source=axis_item.get("source"),
            )

            candidate_id = f"cand_{candidate_idx:03d}"
            candidate_idx += 1
            quad = project_quad(basis.origin, basis.u, basis.v, bundle.intr, axis_s, center_t, artwork_w_m, artwork_h_m)
            quad_metrics = _quad_metrics_original(quad, bundle.scale_to_original)
            quad_original_np = np.asarray(scale_points_to_original(quad, bundle.scale_to_original), dtype=np.float32)
            img_w_original, img_h_original = bundle.original_size
            border_margin_px = {
                "left": float(np.min(quad_original_np[:, 0])),
                "right": float(float(img_w_original) - np.max(quad_original_np[:, 0])),
                "top": float(np.min(quad_original_np[:, 1])),
                "bottom": float(float(img_h_original) - np.max(quad_original_np[:, 1])),
            }
            component_s_range = axis_item.get("component_s_range_m")
            edge_margin_debug = {
                "enabled": component_s_range is not None,
                "min_edge_margin_m": float(min_edge_margin_m),
                "required_half_width_m": float(artwork_w_m / 2.0 + min_edge_margin_m),
            }
            if component_s_range is not None:
                comp_s_min_axis = float(component_s_range[0])
                comp_s_max_axis = float(component_s_range[1])
                left_clearance_m = float(axis_s - comp_s_min_axis)
                right_clearance_m = float(comp_s_max_axis - axis_s)
                edge_margin_debug.update({
                    "component_s_range_m": [comp_s_min_axis, comp_s_max_axis],
                    "left_clearance_m": left_clearance_m,
                    "right_clearance_m": right_clearance_m,
                    "min_clearance_m": float(min(left_clearance_m, right_clearance_m)),
                })
            art_mask_poly = polygon_mask((h, w), quad)
            art_area = max(1, int(np.count_nonzero(art_mask_poly)))

            raw_wall_inside = float(np.count_nonzero((art_mask_poly > 0) & (wall_mask > 0))) / art_area
            solid_wall_inside = float(np.count_nonzero((art_mask_poly > 0) & (solid_wall_mask_u8 > 0))) / art_area
            wall_inside = solid_wall_inside

            struct_overlap = 0.0
            for fm in structural_masks:
                struct_overlap += np.count_nonzero((art_mask_poly > 0) & (fm["mask"] > 0)) / art_area

            existing_overlap_all_walls_raw = 0.0
            existing_overlap_on_candidate_wall = 0.0
            removable_existing_overlap_on_candidate_wall = 0.0
            blocking_existing_overlap_on_candidate_wall = 0.0
            existing_ids_on_candidate_wall = []
            removable_existing_ids_on_candidate_wall = []
            blocking_existing_ids_on_candidate_wall = []
            existing_ids_off_candidate_wall_ignored = []
            removable_existing_on_candidate_wall_mask = np.zeros_like(art_mask_poly, dtype=np.uint8)
            blocking_existing_on_candidate_wall_mask = np.zeros_like(art_mask_poly, dtype=np.uint8)
            existing_content_wall_plane_max_dist_m = float(rule_value(rules, ["existing_content_overlap_policy", "wall_plane_max_distance_m"], 0.20))
            for em in existing_masks:
                em_mask_bool = em["mask"] > 0
                overlap_bool = (art_mask_poly > 0) & em_mask_bool
                overlap_count = int(np.count_nonzero(overlap_bool))
                if overlap_count <= 0:
                    continue
                ov_all = float(overlap_count) / art_area
                existing_overlap_all_walls_raw += ov_all
                safety_class = str(em.get("existing_content_safety_class") or _classify_existing_content_object(
                    em,
                    removable_labels=removable_existing_labels,
                    blocking_labels=blocking_existing_labels,
                ))

                # Candidate-wall scope: count only overlap that belongs to the current
                # wall. Prefer direct wall-mask intersection; if the semantic wall mask
                # was carved out by an existing artwork, fall back to checking whether
                # the existing-content points lie near this fitted wall plane.
                candidate_wall_count = int(np.count_nonzero(overlap_bool & (wall_mask > 0)))
                classified_by = "wall_mask_intersection" if candidate_wall_count > 0 else None
                if candidate_wall_count == 0:
                    overlap_u8 = (overlap_bool.astype(np.uint8) * 255)
                    overlap_pts = points_from_mask(xyz, overlap_u8, max_points=12000)
                    overlap_pts = overlap_pts[np.isfinite(overlap_pts).all(axis=1)] if overlap_pts.size else overlap_pts
                    if len(overlap_pts) >= 20:
                        plane_dist = np.abs(overlap_pts @ np.array(plane.n, np.float32).reshape(3) + float(plane.d))
                        if float(np.percentile(plane_dist, 50)) <= existing_content_wall_plane_max_dist_m:
                            candidate_wall_count = overlap_count
                            classified_by = "near_current_wall_plane"
                if candidate_wall_count > 0:
                    candidate_existing_bool = overlap_bool & ((wall_mask > 0) | (candidate_wall_count == overlap_count))
                    candidate_overlap_ratio = float(candidate_wall_count) / art_area
                    existing_overlap_on_candidate_wall += candidate_overlap_ratio
                    existing_info = {
                        "object_id": em.get("object_id"),
                        "label": em.get("label"),
                        "safety_class": safety_class,
                        "overlap_ratio": candidate_overlap_ratio,
                        "classified_by": classified_by,
                    }
                    existing_ids_on_candidate_wall.append(existing_info)
                    if safety_class == "blocking":
                        blocking_existing_on_candidate_wall_mask[candidate_existing_bool] = 255
                        blocking_existing_overlap_on_candidate_wall += candidate_overlap_ratio
                        blocking_existing_ids_on_candidate_wall.append({
                            **existing_info,
                            "action": "hard_reject_if_above_threshold",
                        })
                    else:
                        removable_existing_on_candidate_wall_mask[candidate_existing_bool] = 255
                        removable_existing_overlap_on_candidate_wall += candidate_overlap_ratio
                        removable_existing_ids_on_candidate_wall.append({
                            **existing_info,
                            "action": "cleanup_only_can_count_as_wall_support",
                        })
                else:
                    existing_ids_off_candidate_wall_ignored.append({
                        "object_id": em.get("object_id"),
                        "label": em.get("label"),
                        "safety_class": safety_class,
                        "overlap_ratio_raw": ov_all,
                        "ignored_reason": "not_on_current_candidate_wall",
                    })
            existing_overlap = existing_overlap_on_candidate_wall

            removable_existing_in_art = (art_mask_poly > 0) & (removable_existing_on_candidate_wall_mask > 0)
            removable_existing_ratio = float(np.count_nonzero(removable_existing_in_art)) / art_area
            blocking_existing_in_art = (art_mask_poly > 0) & (blocking_existing_on_candidate_wall_mask > 0)
            blocking_existing_ratio = float(np.count_nonzero(blocking_existing_in_art)) / art_area
            effective_wall_inside_mask = (solid_wall_mask_u8 > 0)
            if allow_cleanup_existing_as_wall_support:
                effective_wall_inside_mask = effective_wall_inside_mask | (removable_existing_on_candidate_wall_mask > 0)
            effective_wall_inside = float(np.count_nonzero((art_mask_poly > 0) & effective_wall_inside_mask)) / art_area

            # art depth, foreground occlusion, opening/recess detection,
            # and target-wall normal consistency.
            art_depth = ray_plane_depth_for_mask(art_mask_poly, bundle.intr, plane)
            valid = (art_mask_poly > 0) & np.isfinite(art_depth)
            valid_count = max(1, int(np.count_nonzero(valid)))
            # HANGING_POLICY_PATCH_V6_BEGIN_REMOVABLE_VISUAL_SUPPORT
            # Only removable flat wall content (old art/posters/labels/hooks) on
            # the target wall can be visual cleanup/inpaint scope. Physical
            # fixtures such as shelves/cabinets are handled by the independent
            # blocking_existing_content_overlap hard gate below and must never
            # be removed from support denominators or counted as wall support.
            _hpv6_wall_proximity_threshold_m = float(rule_value(rules, ["foreground_occlusion_policy", "wall_proximity_threshold_m"], 0.15))
            removable_existing_bool = np.zeros_like(valid, dtype=bool)
            removable_foreground_bool = np.zeros_like(valid, dtype=bool)
            close_wall_foreground_bool = np.zeros_like(valid, dtype=bool)
            _hpv6_removable_existing_ids = []
            _hpv6_removable_foreground_ids = []
            _hpv6_close_wall_foreground_ids = []
            try:
                for _hpv6_em in removable_existing_masks:
                    _hpv6_mask = _hpv6_em.get("mask")
                    if _hpv6_mask is None:
                        continue
                    _hpv6_overlap = valid & (_hpv6_mask > 0)
                    _hpv6_overlap_count = int(np.count_nonzero(_hpv6_overlap))
                    if _hpv6_overlap_count <= 0:
                        continue
                    _hpv6_on_wall_count = int(np.count_nonzero(_hpv6_overlap & (wall_mask > 0)))
                    _hpv6_near_wall = False
                    if _hpv6_on_wall_count <= 0:
                        _hpv6_overlap_u8 = (_hpv6_overlap.astype(np.uint8) * 255)
                        _hpv6_pts = points_from_mask(xyz, _hpv6_overlap_u8, max_points=12000)
                        _hpv6_pts = _hpv6_pts[np.isfinite(_hpv6_pts).all(axis=1)] if _hpv6_pts.size else _hpv6_pts
                        if len(_hpv6_pts) >= 20:
                            _hpv6_dist = np.abs(_hpv6_pts @ np.array(plane.n, np.float32).reshape(3) + float(plane.d))
                            _hpv6_near_wall = bool(float(np.percentile(_hpv6_dist, 50)) <= float(rule_value(rules, ["existing_content_overlap_policy", "wall_plane_max_distance_m"], 0.20)))
                    if _hpv6_on_wall_count > 0 or _hpv6_near_wall:
                        removable_existing_bool |= _hpv6_overlap
                        _hpv6_removable_existing_ids.append({
                            "object_id": _hpv6_em.get("object_id"),
                            "label": _hpv6_em.get("label"),
                            "overlap_ratio": float(_hpv6_overlap_count) / valid_count,
                            "classified_by": "target_wall_cleanup_support_v6",
                        })
                for _hpv6_fm in foreground_masks:
                    _hpv6_mask = _hpv6_fm.get("mask")
                    if _hpv6_mask is None:
                        continue
                    _hpv6_overlap = valid & (_hpv6_mask > 0)
                    _hpv6_overlap_count = int(np.count_nonzero(_hpv6_overlap))
                    if _hpv6_overlap_count <= 0:
                        continue
                    _hpv6_overlap_u8 = (_hpv6_overlap.astype(np.uint8) * 255)
                    _hpv6_pts = points_from_mask(xyz, _hpv6_overlap_u8, max_points=12000)
                    _hpv6_pts = _hpv6_pts[np.isfinite(_hpv6_pts).all(axis=1)] if _hpv6_pts.size else _hpv6_pts
                    _hpv6_p50_dist = None
                    if len(_hpv6_pts) >= 20:
                        _hpv6_dist = np.abs(_hpv6_pts @ np.array(plane.n, np.float32).reshape(3) + float(plane.d))
                        _hpv6_p50_dist = float(np.percentile(_hpv6_dist, 50))
                    if _hpv6_p50_dist is not None and _hpv6_p50_dist <= _hpv6_wall_proximity_threshold_m:
                        close_wall_foreground_bool |= _hpv6_overlap
                        _hpv6_close_wall_foreground_ids.append({
                            "object_id": _hpv6_fm.get("object_id"),
                            "label": _hpv6_fm.get("label"),
                            "overlap_ratio": float(_hpv6_overlap_count) / valid_count,
                            "wall_plane_distance_p50_m": float(_hpv6_p50_dist),
                            "action": "treat_as_close_wall_obstruction",
                        })
                    else:
                        removable_foreground_bool |= _hpv6_overlap
                        _hpv6_removable_foreground_ids.append({
                            "object_id": _hpv6_fm.get("object_id"),
                            "label": _hpv6_fm.get("label"),
                            "overlap_ratio": float(_hpv6_overlap_count) / valid_count,
                            "wall_plane_distance_p50_m": _hpv6_p50_dist,
                            "action": "flux2_cleanup_before_hardpaste",
                        })
            except Exception as _hpv6_e:
                _hpv6_removable_foreground_ids.append({"error": str(_hpv6_e)})
            removable_visual_bool = removable_existing_bool | removable_foreground_bool
            support_valid = valid & (~removable_visual_bool)
            support_valid_count = max(1, int(np.count_nonzero(support_valid)))
            removable_existing_overlap_ratio = float(np.count_nonzero(removable_existing_bool & valid)) / valid_count
            removable_foreground_overlap_ratio = float(np.count_nonzero(removable_foreground_bool & valid)) / valid_count
            close_wall_foreground_overlap_ratio = float(np.count_nonzero(close_wall_foreground_bool & valid)) / valid_count
            removable_visual_overlap_ratio = float(np.count_nonzero(removable_visual_bool & valid)) / valid_count
            # HANGING_POLICY_PATCH_V6_END_REMOVABLE_VISUAL_SUPPORT

            # Foreground occlusion: objects in front of the target wall/art plane.
            fg = (depth < (art_depth - 0.03)) & valid
            foreground_occlusion = float(np.count_nonzero(fg)) / valid_count

            # Depth support:
            # actual depth close to projected wall plane means visible solid wall.
            # actual depth behind projected wall plane means opening/recess/hollow.
            depth_err = np.abs(depth - art_depth)
            art_depth_tol = make_depth_tol_map(art_depth)
            plane_consistent = (depth_err <= art_depth_tol) & support_valid
            plane_consistency_ratio_visible = float(np.count_nonzero(plane_consistent)) / support_valid_count
            plane_consistency_ratio = min(1.0, float(np.count_nonzero(plane_consistent) + np.count_nonzero(removable_visual_bool & valid)) / valid_count)

            # Recompute foreground occlusion after plane consistency is known.
            # Pixels already consistent with the target wall plane are visible wall,
            # not foreground occluders.
            fg = close_wall_foreground_bool & valid
            foreground_occlusion = close_wall_foreground_overlap_ratio

            behind_wall = (depth > (art_depth + art_depth_tol)) & support_valid
            recess_behind_wall_ratio_visible = float(np.count_nonzero(behind_wall)) / support_valid_count
            recess_behind_wall_ratio = recess_behind_wall_ratio_visible

            # Normal consistency:
            # surfaces with depth close to the wall but normals different from the
            # fitted target wall are side returns / angled reveal surfaces / other planes.
            wall_n_np = np.array(basis.n, np.float32)
            normal_alignment = np.abs(np.sum(normal * wall_n_np.reshape(1, 1, 3), axis=-1))
            normal_consistent = (normal_alignment >= normal_consistency_dot_threshold) & support_valid
            normal_consistency_ratio_visible = float(np.count_nonzero(normal_consistent & plane_consistent)) / support_valid_count
            normal_consistency_ratio = min(1.0, float(np.count_nonzero(normal_consistent & plane_consistent) + np.count_nonzero(removable_visual_bool & valid)) / valid_count)

            # Normal mismatch is evaluated only outside foreground occluders.
            non_foreground_valid = support_valid & (~fg)
            normal_mismatch = (normal_alignment < normal_consistency_dot_threshold) & non_foreground_valid
            normal_mismatch_ratio = float(np.count_nonzero(normal_mismatch)) / max(1, int(np.count_nonzero(non_foreground_valid)))

            # Solid target wall support requires both depth and normal agreement.
            solid_wall_support = plane_consistent & normal_consistent
            solid_wall_support_ratio_visible = float(np.count_nonzero(solid_wall_support)) / support_valid_count
            solid_wall_support_ratio = min(1.0, float(np.count_nonzero(solid_wall_support) + np.count_nonzero(removable_visual_bool & valid)) / valid_count)
            effective_wall_support = solid_wall_support
            if allow_cleanup_existing_as_wall_support:
                effective_wall_support = effective_wall_support | (removable_existing_on_candidate_wall_mask > 0)
            effective_wall_support_ratio = float(np.count_nonzero((art_mask_poly > 0) & effective_wall_support)) / valid_count
            cleanup_required = bool(removable_existing_overlap_on_candidate_wall > cleanup_trigger_overlap_ratio)
            safe_wall_support_mask = solid_wall_mask_u8.copy()
            if allow_cleanup_existing_as_wall_support:
                safe_wall_support_mask[removable_existing_on_candidate_wall_mask > 0] = 255
            edge_corner_support_debug = compute_edge_corner_support_debug(
                quad_px=quad,
                safe_wall_support_mask=safe_wall_support_mask,
            )
            wall_support_gate = evaluate_wall_support_gate(
                solid_wall_inside_ratio=solid_wall_inside,
                effective_wall_inside_ratio=effective_wall_inside,
                removable_existing_content_overlap_ratio=removable_existing_ratio,
                opening_or_recess_overlap_ratio=recess_behind_wall_ratio,
                hard_reject_threshold=hard_reject_wall_inside,
                auto_pass_threshold=min_wall_inside,
                opening_recess_used_in_solid_mask=True,
            )

            art_s_min = float(axis_s - artwork_w_m / 2.0)
            art_s_max = float(axis_s + artwork_w_m / 2.0)
            art_t_min = float(center_t - artwork_h_m / 2.0)
            art_t_max = float(center_t + artwork_h_m / 2.0)
            structural_clearance_debug = _candidate_structural_clearance(
                structural_masks=structural_masks,
                xyz=xyz,
                plane=plane,
                basis=basis,
                art_s_min=art_s_min,
                art_s_max=art_s_max,
                art_t_min=art_t_min,
                art_t_max=art_t_max,
                artwork_w_m=artwork_w_m,
                rules=rules,
            )

            reasons = height_contract_gate_reasons(
                height_contract=height_contract,
                semantic_clear_rect_relation=vertical_fit_debug,
            )
            soft_risk_reasons = []
            if fine_grained_front_policy_enabled:
                clearance_has_metric_evidence = structural_clearance_debug.get("min_clearance_m") is not None
                if not structural_clearance_debug.get("passed", False):
                    reasons.append("structural_edge_clearance_below_artwork_width_fraction")
                elif struct_overlap > max_struct_forbidden and not clearance_has_metric_evidence:
                    reasons.append("structural_forbidden_overlap_without_clearance_evidence")
                elif struct_overlap > max_struct_forbidden:
                    # Keep this visible, but do not hard-reject when the wall-local
                    # edge-clearance rule proves the artwork is away from structures.
                    soft_risk_reasons.append("structural_overlap_metric_low_confidence_clearance_passed")
            else:
                if struct_overlap > max_struct_forbidden:
                    reasons.append("structural_forbidden_overlap")

            if blocking_existing_ratio > max_blocking_existing_overlap:
                reasons.append("blocking_existing_content_overlap")
            if wall_support_gate.get("status") == "hard_reject":
                reasons.append("insufficient_safe_wall_support")
            elif wall_support_gate.get("status") == "review":
                soft_risk_reasons.append("safe_wall_support_below_auto_pass_threshold")
            if enforce_edge_corner_support and edge_corner_support_debug.get("status") == "ok":
                edge_support_min = float(edge_corner_support_debug.get("min_edge_support_ratio", 0.0))
                corner_support_min = float(edge_corner_support_debug.get("min_corner_support_ratio", 0.0))
                if edge_support_min < hard_reject_edge_support_ratio:
                    reasons.append("artwork_edge_support_below_hard_reject_threshold")
                elif edge_support_min < min_edge_support_ratio:
                    soft_risk_reasons.append("artwork_edge_support_below_auto_pass_threshold")
                if corner_support_min < hard_reject_corner_support_ratio:
                    reasons.append("artwork_corner_support_below_hard_reject_threshold")
                elif corner_support_min < min_corner_support_ratio:
                    soft_risk_reasons.append("artwork_corner_support_below_auto_pass_threshold")
            # Existing_content is split by safety class: flat visual wall content
            # may be cleaned on the current wall, but blocking physical fixtures
            # (shelves/cabinets/storage) are hard placement blockers.
            if foreground_occlusion > max_occ:
                soft_risk_reasons.append("foreground_occlusion_ratio_too_high")
            if plane_consistency_ratio < min_plane_consistency_ratio:
                soft_risk_reasons.append("plane_support_ratio_below_threshold")
            if recess_behind_wall_ratio > max_recess_behind_wall_ratio:
                soft_risk_reasons.append("opening_or_recess_overlap")
            if normal_consistency_ratio < min_normal_consistency_ratio:
                soft_risk_reasons.append("normal_consistency_ratio_below_threshold")
            if normal_mismatch_ratio > max_normal_mismatch_ratio:
                soft_risk_reasons.append("normal_mismatch_ratio_too_high")
            if effective_wall_support_ratio < min_plane_consistency_ratio:
                soft_risk_reasons.append("solid_wall_support_ratio_below_threshold")
            if edge_margin_debug.get("enabled") and edge_margin_debug.get("min_clearance_m", 0.0) < edge_margin_debug["required_half_width_m"]:
                reasons.append("component_edge_margin_below_threshold")
            if min(border_margin_px.values()) < min_projected_artwork_border_margin_px_original:
                reasons.append("projected_artwork_too_close_to_image_boundary")
            # No minimum projected width/height/area gate. A tiny artwork or
            # far wall is allowed to proceed when placement support is safe.
            if quad_metrics["quad_aspect_ratio_long_over_short"] > max_quad_aspect_ratio_long_over_short:
                reasons.append("projected_quad_aspect_ratio_too_extreme")

            height_penalty = 0.08 if support_context.get("height_mode") != "metric_rule" else 0.0
            quality_score = _clamp01(
                0.25 * effective_wall_inside
                + 0.20 * effective_wall_support_ratio
                + 0.20 * plane_consistency_ratio
                + 0.15 * normal_consistency_ratio
                + 0.10 * (1.0 - recess_behind_wall_ratio)
                + 0.10 * (1.0 - foreground_occlusion)
                - 0.25 * max(0.0, struct_overlap)
                - height_penalty
            )
            if quality_score < high_risk_score_threshold:
                candidate_risk_level = "high"
            elif quality_score < medium_risk_score_threshold or soft_risk_reasons:
                candidate_risk_level = "medium"
            else:
                candidate_risk_level = "low"
            requires_hardpaste_vlm_review = bool(vlm_review_for_high_risk and candidate_risk_level == "high")

            hard = {
                "passed": not reasons,
                "wall_inside_ratio": effective_wall_inside,
                "effective_wall_inside_ratio": effective_wall_inside,
                "raw_wall_inside_ratio": raw_wall_inside,
                "solid_wall_inside_ratio": solid_wall_inside,
                "safe_wall_support_gate_ratio": wall_support_gate.get("safe_wall_support_gate_ratio"),
                "wall_support_gate": wall_support_gate,
                "candidate_reject_stage": (
                    "height_contract_gate" if "metric_height_rule_deviation_exceeds_threshold" in reasons
                    else ("blocking_existing_content_gate" if "blocking_existing_content_overlap" in reasons
                    else ("solid_wall_support_gate" if "insufficient_safe_wall_support" in reasons
                    else ("edge_corner_support_gate" if any(r in reasons for r in ["artwork_edge_support_below_hard_reject_threshold", "artwork_corner_support_below_hard_reject_threshold"]) else None)))
                ),
                "candidate_reject_reason": reasons[0] if reasons else None,
                "hard_gate_reasons": list(dict.fromkeys(wall_support_gate.get("hard_gate_reasons", []) + reasons)),
                "non_solid_reason_breakdown": list(dict.fromkeys(wall_support_gate.get("non_solid_reason_breakdown", []))),
                "opening_or_recess_overlap_ratio": float(recess_behind_wall_ratio),
                "opening_recess_used_in_solid_mask": True,
                "edge_support_debug": edge_corner_support_debug,
                "corner_support_debug": edge_corner_support_debug,
                "edge_corner_support_gate": {
                    "hard_reject_edge_support_ratio": float(hard_reject_edge_support_ratio),
                    "auto_pass_edge_support_ratio": float(min_edge_support_ratio),
                    "hard_reject_corner_support_ratio": float(hard_reject_corner_support_ratio),
                    "auto_pass_corner_support_ratio": float(min_corner_support_ratio),
                    "policy": "below_hard_reject_is_blocked_between_hard_and_auto_pass_is_review",
                },
                "wall_inside_ratio_policy": "solid_wall_plus_current_wall_removable_flat_existing_content_only" if allow_cleanup_existing_as_wall_support else "solid_wall_only",
                "solid_component_debug": solid_component_debug,
                "visible_wall_s_range_m": visible_wall_s_range_m,
                "visible_wall_t_range_m": [float(visible_t_min), float(visible_t_max)],
                "artwork_width_m": float(artwork_w_m),
                "artwork_height_m": float(artwork_h_m),
                "height_rule_consistency": height_contract,
                "semantic_clear_rect_vertical_fit": vertical_fit_debug,
                "structural_forbidden_overlap_ratio": float(struct_overlap),
                "structural_clearance_policy": structural_clearance_debug,
                "soft_risk_reasons": soft_risk_reasons,
                "low_quality_scores_are_soft_risk": bool(fine_grained_front_policy_enabled and allow_low_score_as_high_risk),
                "risk_assessment": {
                    "risk_level": candidate_risk_level,
                    "quality_score": float(quality_score),
                    "high_risk_score_threshold": float(high_risk_score_threshold),
                    "medium_risk_score_threshold": float(medium_risk_score_threshold),
                    "requires_hardpaste_vlm_review": requires_hardpaste_vlm_review,
                    "hardpaste_vlm_review_reason": "low_or_uncertain_geometry_score_after_structural_clearance_pass" if requires_hardpaste_vlm_review else None,
                    "review_input_asset": f"wall_assets/{wall_id}/hard_paste.png",
                },
                "existing_content_overlap_ratio": float(existing_overlap_on_candidate_wall),
                "existing_content_overlap_ratio_all_walls_raw": float(existing_overlap_all_walls_raw),
                "existing_content_off_candidate_wall_ratio_ignored": float(max(0.0, existing_overlap_all_walls_raw - existing_overlap_on_candidate_wall)),
                "max_existing_content_overlap_ratio_deprecated_not_used": float(max_existing),
                "existing_content_ids_on_candidate_wall": existing_ids_on_candidate_wall,
                "existing_content_ids_off_candidate_wall_ignored": existing_ids_off_candidate_wall_ignored,
                "existing_content_policy": "removable_flat_wall_content_cleanup;blocking_physical_fixture_hard_reject",
                "existing_content_overlap_ignored_for_filtering": False,
                "existing_content_as_effective_wall_support": bool(allow_cleanup_existing_as_wall_support and removable_existing_ratio > 0.0),
                "removable_existing_content_overlap_ratio": float(removable_existing_ratio),
                "removable_existing_content_overlap_ratio_candidate_wall_raw": float(removable_existing_overlap_on_candidate_wall),
                "removable_existing_content_ids_on_candidate_wall": removable_existing_ids_on_candidate_wall,
                "blocking_existing_content_overlap_ratio": float(blocking_existing_ratio),
                "blocking_existing_content_overlap_ratio_candidate_wall_raw": float(blocking_existing_overlap_on_candidate_wall),
                "blocking_existing_content_ids_on_candidate_wall": blocking_existing_ids_on_candidate_wall,
                "blocking_existing_content_gate": {
                    "enabled": True,
                    "max_overlap_ratio": float(max_blocking_existing_overlap),
                    "overlap_ratio": float(blocking_existing_ratio),
                    "passed": bool(blocking_existing_ratio <= max_blocking_existing_overlap),
                    "policy": "shelf/cabinet/storage-like physical fixtures are not removable wall clutter and never count as wall support",
                },
                "cleanup": {
                    "enabled": cleanup_required,
                    "cleanup_required": cleanup_required,
                    "stage": "candidate_wall_cleanup_then_render",
                    "candidate_wall_scope_only": True,
                    "cleanup_mask_definition": "art_region_mask ∩ current_wall_mask ∩ removable_existing_content_mask",
                    "protected_scope": "all pixels outside current candidate wall mask plus all blocking existing fixtures",
                    "off_wall_existing_content_protected": True,
                    "blocking_existing_content_protected": True,
                    "mask_bbox_processed": _mask_bbox(removable_existing_on_candidate_wall_mask),
                    "notes": "Only flat wall visual content such as old artwork, posters, labels, hooks or nails may be cleaned and counted as effective wall support. Shelves, cabinets and storage-like fixtures are protected and trigger a hard gate when overlapped.",
                },
                "foreground_occlusion_ratio": foreground_occlusion,
                "foreground_excludes_plane_consistent": True,
                "plane_consistency_ratio": plane_consistency_ratio,
                "recess_behind_wall_ratio": recess_behind_wall_ratio,
                "normal_consistency_ratio": normal_consistency_ratio,
                "normal_mismatch_ratio": normal_mismatch_ratio,
                "solid_wall_support_ratio": effective_wall_support_ratio,
                "raw_solid_wall_support_ratio": solid_wall_support_ratio,
                "effective_wall_support_ratio": effective_wall_support_ratio,
                "normal_consistency_dot_threshold": float(normal_consistency_dot_threshold),
                "min_normal_consistency_ratio": float(min_normal_consistency_ratio),
                "max_normal_mismatch_ratio": float(max_normal_mismatch_ratio),
                "max_plane_depth_error_effective_m": float(max_plane_depth_error_effective),
                "depth_tolerance_relative": float(depth_tolerance_relative),
                "depth_tolerance_max_m": float(depth_tolerance_max_m),
                "art_depth_tolerance_p50_m": float(np.nanpercentile(art_depth_tol[valid], 50)) if np.any(valid) else None,
                "min_plane_consistency_ratio": float(min_plane_consistency_ratio),
                "max_recess_behind_wall_ratio": float(max_recess_behind_wall_ratio),
                "art_depth_rasterized": True,
                "wall_filter_debug": wall_filter_debug,
                "projection_degenerate_guard": {
                    **quad_metrics,
                    "projection_size_gate_enabled": bool(projection_size_gate_enabled),
                    "projection_size_policy": "no_minimum_projected_width_height_or_area_gate",
                    "max_quad_aspect_ratio_long_over_short": float(max_quad_aspect_ratio_long_over_short),
                    "min_projected_artwork_border_margin_px_original": float(min_projected_artwork_border_margin_px_original),
                    "border_margin_px_original": border_margin_px,
                    "edge_margin_debug": edge_margin_debug,
                    "visible_wall_s_range_m": visible_wall_s_range_m,
                    "artwork_width_m": float(artwork_w_m),
                },
            }

            # HANGING_POLICY_PATCH_V6_BEGIN_HARD_DEBUG_FIELDS
            try:
                # P1 safety: people/foreground are cleanup scope, never wall support.
                # Only current-wall removable existing content was already folded
                # into effective_wall_inside above.
                hard["effective_wall_inside_ratio"] = float(effective_wall_inside)
                hard["effective_plane_consistency_ratio"] = float(plane_consistency_ratio)
                hard["effective_normal_consistency_ratio"] = float(normal_consistency_ratio)
                hard["effective_wall_support_ratio"] = float(effective_wall_support_ratio)
                hard["foreground_occlusion_policy"] = {
                    "enabled": True,
                    "wall_proximity_threshold_m": float(_hpv6_wall_proximity_threshold_m),
                    "rule": "foreground/person farther than 15cm from target wall plane is cleanup scope, not placement obstruction",
                    "foreground_occlusion_ratio_close_wall_only": float(foreground_occlusion),
                    "removable_foreground_overlap_ratio": float(removable_foreground_overlap_ratio),
                    "close_wall_foreground_overlap_ratio": float(close_wall_foreground_overlap_ratio),
                    "removable_foreground_ids": _hpv6_removable_foreground_ids,
                    "close_wall_foreground_ids": _hpv6_close_wall_foreground_ids,
                }
                hard["removable_visual_support_policy"] = {
                    "enabled": True,
                    "removable_visual_overlap_ratio": float(removable_visual_overlap_ratio),
                    "removable_existing_overlap_ratio": float(removable_existing_overlap_ratio),
                    "removable_foreground_overlap_ratio": float(removable_foreground_overlap_ratio),
                    "visible_support_valid_ratio_after_excluding_cleanup_scope": float(support_valid_count) / valid_count,
                    "plane_consistency_ratio_visible_only": float(locals().get("plane_consistency_ratio_visible", plane_consistency_ratio)),
                    "normal_consistency_ratio_visible_only": float(locals().get("normal_consistency_ratio_visible", normal_consistency_ratio)),
                    "solid_wall_support_ratio_visible_only": float(locals().get("solid_wall_support_ratio_visible", solid_wall_support_ratio)),
                    "removable_existing_ids_v6": _hpv6_removable_existing_ids,
                }
                hard["foreground_cleanup"] = {
                    "enabled": True,
                    "cleanup_required": bool(removable_foreground_overlap_ratio > 0.001),
                    "stage": "flux2_before_hardpaste",
                    "candidate_wall_scope_only": True,
                    "cleanup_mask_definition": "art_region_mask ∩ current_wall_mask ∩ foreground/person_mask where distance_to_wall_plane_p50 > 0.15m",
                    "object_ids": [x.get("object_id") for x in _hpv6_removable_foreground_ids if isinstance(x, dict)],
                    "overlap_ratio": float(removable_foreground_overlap_ratio),
                }
                if isinstance(hard.get("cleanup"), dict):
                    if removable_foreground_overlap_ratio > 0.001:
                        hard["cleanup"]["cleanup_required"] = True
                        hard["cleanup"]["foreground_cleanup_required"] = True
                        hard["cleanup"]["foreground_cleanup_stage"] = "flux2_before_hardpaste"
                    if removable_existing_overlap_ratio > 0.001:
                        hard["cleanup"]["cleanup_required"] = True
                        hard["cleanup"]["existing_content_cleanup_required"] = True
            except Exception as _hpv6_e:
                hard["foreground_occlusion_policy"] = {"enabled": False, "error": str(_hpv6_e)}
            # HANGING_POLICY_PATCH_V6_END_HARD_DEBUG_FIELDS

            artwork_mount_policy = _compute_artwork_mount_policy_for_candidate(
                rules=rules,
                artwork_analysis=artwork_analysis,
                artwork_has_frame=resolved_artwork_has_frame,
                artwork_has_frame_source=artwork_has_frame_source,
                basis=basis,
                axis_s=axis_s,
                center_t=center_t,
                artwork_w_m=artwork_w_m,
                artwork_h_m=artwork_h_m,
                intr=bundle.intr,
                scale_to_original=bundle.scale_to_original,
                original_size_wh=(int(bundle.original_size[0]), int(bundle.original_size[1])),
            )

            cand = {
                "candidate_id": candidate_id,
                "wall_id": wall_id,
                "axis_id": f"{wall_id}_axis_{candidate_id}",
                "axis": {
                    "axis_s_m": float(axis_s),
                    "solid_component_id": axis_item.get("component_id"),
                    "solid_component_area_px": int(axis_item.get("component_area_px", 0)),
                    "axis_source": axis_item.get("source", "unknown"),
                    "component_s_range_m": axis_item.get("component_s_range_m"),
                    "component_t_range_m": axis_item.get("component_t_range_m"),
                    "component_width_m": axis_item.get("component_width_m"),
                    "component_height_m": axis_item.get("component_height_m"),
                    "candidate_generation_policy": axis_item.get("candidate_generation_policy"),
                    "vertical_policy": axis_item.get("vertical_policy"),
                    "edge_margin_debug": edge_margin_debug,
                },
                "support_context": support_context,
                "height_solution": {
                    "height_mode": support_context.get("height_mode", "metric_rule"),
                    "rule_id": support_context.get("rule_id", "floor_center_152"),
                    "center_height_above_floor_m": float(center_height_above_floor_m),
                    "center_height_abs_m": float(center_height_abs_m),
                    "floor_height_abs_m": float(floor_height_abs_m),
                    "support_top_abs_m": support_context.get("support_top_abs_m"),
                    "bottom_gap_to_support_m": support_context.get("bottom_gap_to_support_m"),
                    "metric_height_finalized": bool(support_context.get("metric_height_finalized", support_context.get("height_mode") == "metric_rule")),
                    "height_is_provisional": bool(support_context.get("height_is_provisional", support_context.get("height_mode") != "metric_rule")),
                    "requires_frontend_visual_height_selection": bool(support_context.get("requires_frontend_visual_height_selection", False)),
                    "requires_visual_height_selection": bool(support_context.get("requires_visual_height_selection", False)),
                    "requires_flux_visual_height_refinement": bool(support_context.get("requires_flux_visual_height_refinement", False)),
                    "flux_visual_refinement": support_context.get("flux_visual_refinement"),
                    "height_band_base": support_context.get("height_band_base", "unknown"),
                    "height_warning": support_context.get("height_warning"),
                    "floor_source_trusted": bool(floor_source_trusted),
                    "room_corner_floor_fit": room_corner_floor_fit_contract(rules),
                    "center_t_m": float(center_t),
                    "center_t_normalized": float(center_t_normalized),
                    "height_rule_consistency": height_contract,
                    "semantic_clear_rect_vertical_fit": vertical_fit_debug,
                    "height_projection": {
                        "formula": "center_t_m=(center_height_abs_m-dot(wall_origin,g_hat_cam))/dot(wall_v,g_hat_cam)",
                        "dot_v_g": float(dot_v_g),
                        "t_floor_m": float(t_floor),
                        "floor_wall_estimation": floor_wall_estimation,
                    },
                    "g_hat_cam_confidence": float(g_conf),
                    "g_hat_debug": g_debug,
                },
                "projection": {
                    "quad_stage": "after_height_rule_before_hard_constraints",
                    "center_px_processed": [float(np.mean(quad[:,0])), float(np.mean(quad[:,1]))],
                    "center_px_original": [float(np.mean(quad[:,0]) * bundle.scale_to_original[0]), float(np.mean(quad[:,1]) * bundle.scale_to_original[1])],
                    "projected_quad_px_processed": [[float(x), float(y)] for x, y in quad],
                    "projected_quad_px_original": scale_points_to_original(quad, bundle.scale_to_original),
                },
                "hard_constraints": hard,
                "reject_reasons": reasons,
                "soft_risk_reasons": soft_risk_reasons,
                "risk_level": candidate_risk_level,
                "requires_hardpaste_vlm_review": requires_hardpaste_vlm_review,
                "score": float(quality_score),
                "causal_light_facts": causal_wall_light.get("light_facts") if isinstance(causal_wall_light, dict) else None,
                "causal_light_semantics": causal_wall_light.get("semantic_light") if isinstance(causal_wall_light, dict) else None,
                "surface_visual_noise_band": causal_wall_light.get("surface_visual_noise_band") if isinstance(causal_wall_light, dict) else None,
                "surface_visual_noise_debug": causal_wall_light.get("surface_visual_noise_debug") if isinstance(causal_wall_light, dict) else None,
                "artwork_wall_area_ratio": _candidate_wall_area_ratio({"hard_constraints": hard}),
                "window_edge_clearance_cm": None,
                "artwork_analysis": artwork_analysis,
                "artwork_has_frame": resolved_artwork_has_frame,
                "artwork_has_frame_source": artwork_has_frame_source,
                "artwork_mount_policy": artwork_mount_policy,
            }
            wall_luminance_mask = wall_mask.copy()
            for semantic_object in objects.values():
                if str(semantic_object.get("role") or "").strip().lower() == "wall":
                    continue
                semantic_mask = semantic_object.get("mask")
                if isinstance(semantic_mask, np.ndarray) and semantic_mask.shape[:2] == wall_luminance_mask.shape[:2]:
                    wall_luminance_mask[semantic_mask > 0] = 0
            wall_styling_metrics = measure_candidate_wall_luminance(
                room_rgb,
                wall_luminance_mask,
                quad,
            )
            cand["wall_mean_luminance"] = wall_styling_metrics.get("wall_mean_luminance")
            cand["is_light_wall"] = bool(wall_styling_metrics.get("is_light_wall"))
            cand["suggest_dark_wallpaper"] = should_suggest_dark_wallpaper(artwork_analysis, wall_styling_metrics)
            cand["suggested_wall_tone_rgb"] = list(artwork_analysis.get("suggested_wall_tone_rgb") or [58, 62, 71])
            cand["wallpaper_suggestion_debug"] = {
                **wall_styling_metrics,
                "artwork_mean_luminance": artwork_analysis.get("artwork_mean_luminance"),
                "artwork_contrast": artwork_analysis.get("artwork_contrast"),
                "is_dark_or_high_contrast": bool(artwork_analysis.get("is_dark_or_high_contrast")),
                "decision_rule": "is_dark_or_high_contrast AND wall_mean_luminance > 0.62",
                "render_requested": False,
            }
            cand["window_edge_clearance_cm"] = _candidate_window_clearance_cm(cand)
            if reasons:
                rejected.append({"candidate_id": candidate_id, "wall_id": wall_id, "reject_reasons": reasons, "soft_risk_reasons": soft_risk_reasons, "hard_constraints": hard})
            else:
                candidates.append(cand)

    # HANGING_POLICY_PATCH_V6_BEGIN_EFFECTIVE_RERANK
    def _hpv6_candidate_policy_score(_cand):
        try:
            _hard = _cand.get("hard_constraints", {}) or {}
            _axis = _cand.get("axis", {}) or {}
            _wall_inside = float(_hard.get("effective_wall_inside_ratio", _hard.get("wall_inside_ratio", 0.0)) or 0.0)
            _plane = float(_hard.get("effective_plane_consistency_ratio", _hard.get("plane_consistency_ratio", 0.0)) or 0.0)
            _normal = float(_hard.get("effective_normal_consistency_ratio", _hard.get("normal_consistency_ratio", 0.0)) or 0.0)
            _solid = float(_hard.get("effective_wall_support_ratio", _hard.get("solid_wall_support_ratio", 0.0)) or 0.0)
            _recess = float(_hard.get("recess_behind_wall_ratio", 0.0) or 0.0)
            _struct = float(_hard.get("structural_forbidden_overlap_ratio", 0.0) or 0.0)
            _quality = max(0.0, min(1.0, min(_wall_inside, _plane, _normal, _solid) - _struct - min(0.25, _recess)))
            _s = float(_axis.get("axis_s_m", 0.0) or 0.0)

            # Rank against the full visible wall, not semantic_clear_wall_mask.
            # This keeps existing artwork as later cleanup scope instead of
            # reintroducing an empty-space reward/penalty through the side door.
            _rng = None
            if isinstance(_hard.get("visible_wall_s_range_m"), list) and len(_hard.get("visible_wall_s_range_m")) == 2:
                _rng = _hard.get("visible_wall_s_range_m")
            elif isinstance(_axis.get("visible_wall_s_range_m"), list) and len(_axis.get("visible_wall_s_range_m")) == 2:
                _rng = _axis.get("visible_wall_s_range_m")
            elif isinstance(_axis.get("component_s_range_m"), list) and len(_axis.get("component_s_range_m")) == 2:
                _rng = _axis.get("component_s_range_m")
            elif isinstance(_axis.get("structural_gap_s_range_m"), list) and len(_axis.get("structural_gap_s_range_m")) == 2:
                _rng = _axis.get("structural_gap_s_range_m")

            _center_score = 0.5
            _corner_score = 0.5
            _corner_clearance_m = None
            _art_w = float(_hard.get("artwork_width_m", _axis.get("artwork_width_m", artwork_w_m)) or artwork_w_m)
            if _rng is not None:
                _s0, _s1 = float(_rng[0]), float(_rng[1])
                _half = max(1e-6, 0.5 * (_s1 - _s0))
                _center = 0.5 * (_s0 + _s1)
                _center_score = max(0.0, min(1.0, 1.0 - abs(_s - _center) / _half))
                _left_art_edge = _s - _art_w / 2.0
                _right_art_edge = _s + _art_w / 2.0
                _corner_clearance_m = min(_left_art_edge - _s0, _s1 - _right_art_edge)
                _corner_score = max(0.0, min(1.0, float(_corner_clearance_m) / max(1e-6, _art_w)))

            _center_w = wall_center_composition_weight
            _corner_w = wall_corner_soft_penalty_weight
            _quality_w = max(0.0, 1.0 - _center_w - _corner_w)
            _score = _quality_w * _quality + _center_w * _center_score + _corner_w * _corner_score
            _cand["score_before_policy_v6"] = float(_cand.get("score", 0.0) or 0.0)
            _cand["score"] = float(max(0.0, min(1.0, _score)))
            _cand.setdefault("ranking_policy", {})["policy_v6"] = {
                "score_formula": "quality_weight*effective_geometry_quality + center_weight*visible_wall_center_composition + corner_weight*visible_wall_corner_clearance",
                "quality_weight": float(_quality_w),
                "center_weight": float(_center_w),
                "corner_weight": float(_corner_w),
                "effective_geometry_quality": float(_quality),
                "wall_center_composition_score": float(_center_score),
                "wall_corner_clearance_score": float(_corner_score),
                "wall_corner_clearance_m": None if _corner_clearance_m is None else float(_corner_clearance_m),
                "visible_wall_s_range_m": _rng,
                "empty_space_bonus_enabled": False,
                "reason": "cleanup-scope people/existing art are not placement obstacles; prefer balanced visible-wall composition and softly demote corner-hugging placements",
            }
        except Exception as _e:
            _cand.setdefault("ranking_policy", {})["policy_v6_error"] = str(_e)
        return _cand
    for _hpv6_c in candidates:
        _hpv6_candidate_policy_score(_hpv6_c)

    for _causal_c in candidates:
        try:
            _components, _band, _delta = _causal_light_for_candidate(_causal_c, {
                "light_facts": _causal_c.get("causal_light_facts"),
                "semantic_light": _causal_c.get("causal_light_semantics"),
            })
            if _components is None:
                continue
            _causal_c["light_components"] = _components
            _causal_c["light_risk_band"] = _band
            if _causal_decision_enabled():
                _causal_c["score_before_light"] = float(_causal_c.get("score", 0.0) or 0.0)
                _causal_c["score"] = _clamp01_local(_causal_c["score_before_light"] + float(_delta or 0.0))
                _causal_c.setdefault("ranking_policy", {})["causal_light"] = {
                    "decision_enabled": True,
                    "light_delta": float(_delta or 0.0),
                    "risk_band": _band,
                    "score_before_light": _causal_c["score_before_light"],
                    "score_after_light": float(_causal_c.get("score", 0.0) or 0.0),
                }
                if _band == "caution":
                    _causal_c.setdefault("soft_risk_reasons", []).append("light_caution")
                elif _band == "concern":
                    _causal_c.setdefault("soft_risk_reasons", []).append("light_concern")
            else:
                _causal_c.setdefault("ranking_policy", {})["causal_light"] = {
                    "decision_enabled": False,
                    "light_delta": float(_delta or 0.0),
                    "risk_band": _band,
                    "score_unchanged": float(_causal_c.get("score", 0.0) or 0.0),
                }
        except Exception as _causal_e:
            _causal_c.setdefault("ranking_policy", {})["causal_light_error"] = str(_causal_e)

    candidates, projected_duplicate_rejections, candidate_diversity_summary = _suppress_projected_duplicate_candidates(candidates, rules)
    rejected.extend(projected_duplicate_rejections)
    # HANGING_POLICY_PATCH_V6_END_EFFECTIVE_RERANK

    candidates.sort(key=lambda x: x["score"], reverse=True)
    selected = candidates[0] if candidates else None

    passed_by_wall = {}
    for cand in candidates:
        passed_by_wall.setdefault(cand["wall_id"], []).append(cand)

    passed_walls = []
    for wall_id, wall_cands in passed_by_wall.items():
        wall_sorted = sorted(wall_cands, key=lambda x: x["score"], reverse=True)
        passed_walls.append({
            "wall_id": wall_id,
            "passed_candidate_count": len(wall_cands),
            "best_candidate_id": wall_sorted[0]["candidate_id"],
            "best_score": float(wall_sorted[0]["score"]),
        })

    passed_walls.sort(key=lambda x: x["best_score"], reverse=True)
    passed_wall_ids = {str(x.get("wall_id")) for x in passed_walls}
    rejected_wall_ids = sorted({str(x.get("wall_id")) for x in rejected if x.get("wall_id") and str(x.get("wall_id")) not in passed_wall_ids})
    rejected_walls = [
        {
            "wall_id": wall_id,
            "wall_has_no_safe_candidate": True,
            "blocked_reason": "no_safe_candidate_after_hard_gates",
            "rejections": [x for x in rejected if str(x.get("wall_id")) == wall_id],
        }
        for wall_id in rejected_wall_ids
    ]

    def _near_miss_candidate_record(rej: Mapping[str, Any]) -> Dict[str, Any] | None:
        hard = rej.get("hard_constraints", {}) if isinstance(rej.get("hard_constraints"), dict) else {}
        reasons_set = {str(x) for x in rej.get("reject_reasons", [])}
        support_ratio = float(hard.get("safe_wall_support_gate_ratio", hard.get("solid_wall_inside_ratio", 0.0)) or 0.0)
        opening_ratio = float(hard.get("opening_or_recess_overlap_ratio", 0.0) or 0.0)
        edge_debug = hard.get("edge_support_debug", {}) if isinstance(hard.get("edge_support_debug"), dict) else {}
        min_edge = float(edge_debug.get("min_edge_support_ratio", 1.0) or 0.0)
        min_corner = float(edge_debug.get("min_corner_support_ratio", 1.0) or 0.0)
        structural_blocked = bool(reasons_set & {
            "structural_edge_clearance_below_artwork_width_fraction",
            "structural_forbidden_overlap",
            "structural_forbidden_overlap_without_clearance_evidence",
            "projected_artwork_too_close_to_image_boundary",
            "component_edge_margin_below_threshold",
            "blocking_existing_content_overlap",
        })
        projection_size_reasons = {
            "projected_artwork_width_too_small",
            "projected_artwork_height_too_small",
            "projected_artwork_area_too_small",
        }
        if reasons_set & projection_size_reasons:
            # Kept only for backwards-compatible outputs generated by older rules;
            # the current code path no longer creates these reject reasons.
            near_reason = "legacy_projection_size_reject_would_now_be_allowed"
        elif support_ratio >= 0.80 and min_edge >= hard_reject_edge_support_ratio and min_corner >= hard_reject_corner_support_ratio and not structural_blocked:
            near_reason = "borderline_support_review_candidate"
        else:
            return None
        if opening_ratio > 0.05 and support_ratio < 0.90:
            return None
        return {
            "candidate_id": rej.get("candidate_id"),
            "wall_id": rej.get("wall_id"),
            "near_miss_reason": near_reason,
            "reject_reasons": list(rej.get("reject_reasons", [])),
            "soft_risk_reasons": list(rej.get("soft_risk_reasons", [])),
            "safe_wall_support_gate_ratio": support_ratio,
            "solid_wall_inside_ratio": hard.get("solid_wall_inside_ratio"),
            "opening_or_recess_overlap_ratio": opening_ratio,
            "min_edge_support_ratio": min_edge,
            "min_corner_support_ratio": min_corner,
            "projection_size_gate_enabled": False,
            "recommended_next_action": "manual_review_or_relaxed_review_mode_not_auto_flux",
        }

    near_miss_candidates = [x for x in (_near_miss_candidate_record(r) for r in rejected) if x is not None]
    near_miss_candidates.sort(key=lambda x: float(x.get("safe_wall_support_gate_ratio", 0.0) or 0.0), reverse=True)

    # Build debug hard-paste assets for every geometrically passed candidate,
    # then select a capped subset for official BFL rendering. v7 order is:
    # original_room -> erase -> hard_paste_after_erase -> flux2.
    original_size_wh = (int(bundle.original_size[0]), int(bundle.original_size[1]))
    stage_b_decor_zone, stage_b_decor_debug = _build_stage_b_decor_zone(
        objects,
        processed_shape_hw=(h, w),
        original_size_wh=original_size_wh,
        wall_items=walls,
    )
    flux_input_files: Dict[str, bytes] = {
        "flux_inputs/original_room.png": _png_bytes(room_rgb),
        "flux_inputs/artwork.png": _png_bytes(art_rgba),
    }
    approved_hardpaste_assets = []
    final_hardpaste_assets = []
    flux_selection_summary: Dict[str, Any] = {}
    all_hardpaste_reviews = []
    # Keep person/foreground cleanup global by default: non-occluding people may
    # still be removed during erase-v1.  The safety fix is to constrain the later
    # FLUX.2 render with a target-local edit mask, not to keep people in the
    # cleaned room.
    remove_foreground_global = bool(rule_value(rules, ["cleanup_policy", "remove_foreground_global"], True))
    remove_mask_dilate_px = int(rule_value(rules, ["cleanup_policy", "remove_mask_dilate_px"], 6))
    flux2_edit_mask_dilate_px = int(rule_value(rules, ["flux_render", "edit_mask_dilate_px"], 24))
    bfl_erase_dilate_pixels_default = int(rule_value(rules, ["cleanup_policy", "bfl_erase_dilate_pixels_default"], 10))
    remove_mask_dilation_policy = {
        "local_export_dilate_px": remove_mask_dilate_px,
        "bfl_erase_api_dilate_pixels_default": bfl_erase_dilate_pixels_default,
        "effective_total_dilate_px_approx": remove_mask_dilate_px + bfl_erase_dilate_pixels_default,
        "intent": "local mask dilation covers semantic-object edges; BFL erase-v1 dilate_pixels smooths server-side erase boundary",
        "debug_note": "If large existing artworks bleed into neighboring wall areas, reduce cleanup_policy.remove_mask_dilate_px or pipeline --erase-dilate-pixels.",
    }
    render_prompt_extra = str(rule_value(rules, ["flux_render", "extra_prompt"], "") or "")
    artwork_has_frame = resolved_artwork_has_frame
    per_wall_bfl_limit = int(rule_value(rules, ["flux_export", "per_wall_bfl_limit"], 2))
    global_final_render_limit = int(rule_value(rules, ["flux_export", "global_final_render_limit"], 4))
    nms_center_distance_fraction = float(rule_value(rules, ["flux_export", "nms_center_distance_fraction_of_artwork_width"], 0.65))
    nms_polygon_iou_threshold = float(rule_value(rules, ["flux_export", "nms_polygon_iou_threshold"], 0.25))
    hardpaste_supersample_scale = int(rule_value(rules, ["hardpaste_quality", "supersample_scale"], 4))

    for cand in candidates:
        wall_id = str(cand.get("wall_id", "unknown_wall"))
        candidate_id = str(cand.get("candidate_id", f"candidate_{len(all_hardpaste_reviews):03d}"))
        asset_id = safe_asset_id(f"{wall_id}__{candidate_id}")
        prefix = f"flux_inputs/items/{asset_id}"
        try:
            quad_orig = cand["projection"]["projected_quad_px_original"]
            quad_int = [[int(round(float(x))), int(round(float(y)))] for x, y in quad_orig]
            xs = [p[0] for p in quad_int]
            ys = [p[1] for p in quad_int]
            bbox_px = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
            mount_policy = cand.get("artwork_mount_policy") if isinstance(cand.get("artwork_mount_policy"), dict) else {}
            if mount_policy.get("precompose_frame_sides"):
                mounted = warp_artwork_mount_to_geometry(
                    room_rgb,
                    art_rgba,
                    front_quad_original=mount_policy.get("front_quad_original") or quad_orig,
                    side_faces_original=mount_policy.get("visible_side_faces_original") or [],
                    edge_sample_px=int(mount_policy.get("edge_sample_px", rule_value(rules, ["artwork_mount", "edge_sample_px"], 8))),
                    side_shading_strength=float(mount_policy.get("side_shading_strength", rule_value(rules, ["artwork_mount", "side_shading_strength"], 0.18))),
                    supersample_scale=hardpaste_supersample_scale,
                    clip_side_faces_to_front_span=bool(mount_policy.get("clip_side_faces_to_front_span", rule_value(rules, ["artwork_mount", "clip_side_faces_to_front_span"], True))),
                    side_face_span_pad_px=float(mount_policy.get("side_face_span_pad_px", rule_value(rules, ["artwork_mount", "side_face_span_pad_px"], 0.35))),
                )
                hard_paste = mounted["image"]
                front_region = mounted["front_mask"]
                frame_side_mask = mounted["side_mask"]
                mount_mask = mounted["mount_mask"]
                # QA/core-SSIM must only anchor on the immutable artwork front.
                # Frame side pixels are allowed to receive BFL contact shadows,
                # so they remain in mount_mask for trace/erase-scope debugging
                # but are intentionally excluded from art_region_mask.png.
                art_region = front_region
                remove_scope_region = mount_mask
                mount_debug = {**(mounted.get("debug") or {}), "policy": mount_policy}
                mount_policy["side_faces_used"] = mounted.get("side_faces_used", [])
                mount_policy["actual_side_mask_area_px"] = int(np.count_nonzero(frame_side_mask))
                mount_policy["actual_mount_mask_area_px"] = int(np.count_nonzero(mount_mask))
            else:
                hard_paste, front_region = warp_artwork_to_quad(room_rgb, art_rgba, quad_orig, supersample_scale=hardpaste_supersample_scale)
                frame_side_mask = np.zeros_like(front_region)
                mount_mask = front_region
                art_region = front_region
                remove_scope_region = mount_mask
                mount_debug = {"mode": "flat_hardpaste", "policy": mount_policy}
            overlay = _draw_overlay(room_rgb.copy(), quad_orig)
            wall_debug = wall_debug_registry.get(wall_id, {})
            wall_mask_for_cleanup = wall_debug.get("filtered_wall_mask")
            if wall_mask_for_cleanup is None:
                wall_mask_for_cleanup = wall_debug.get("wall_mask")
            remove_mask = build_remove_mask_for_hardpaste(
                art_region_mask=remove_scope_region,
                current_wall_mask=wall_mask_for_cleanup,
                existing_masks=removable_existing_masks,
                foreground_masks=foreground_masks,
                original_size_wh=original_size_wh,
                remove_foreground_global=remove_foreground_global,
                dilate_px=remove_mask_dilate_px,
                protect_art_region=False,
            )
            flux2_edit_mask = build_flux2_edit_mask(
                mount_mask=mount_mask,
                original_size_wh=original_size_wh,
                dilate_px=flux2_edit_mask_dilate_px,
            )
            target_quad_for_render = mount_policy.get("front_quad_original") if mount_policy.get("precompose_frame_sides") else quad_orig
            target_quad_original = target_quad_for_render or quad_orig
            target_polygon_px = _quad_to_int_polygon(target_quad_original) or quad_int
            front_bbox_px = _binary_mask_bbox_xyxy(front_region) or bbox_px
            mount_bbox_px = _binary_mask_bbox_xyxy(mount_mask) or front_bbox_px
            stage_b_wall_source = wall_mask_for_cleanup
            if stage_b_wall_source is None and wall_id in objects:
                stage_b_wall_source = objects[wall_id].get("mask")
            if isinstance(stage_b_wall_source, np.ndarray) and np.any(stage_b_wall_source):
                stage_b_wall_zone = _resize_binary_mask(stage_b_wall_source, original_size_wh)
            else:
                stage_b_wall_zone = np.zeros((original_size_wh[1], original_size_wh[0]), np.uint8)
            candidate_stage_b_decor_zone, candidate_stage_b_decor_debug = _scope_stage_b_decor_zone_to_candidate(
                stage_b_decor_zone,
                stage_b_wall_zone,
                front_bbox_px=front_bbox_px,
                base_debug=stage_b_decor_debug,
            )
            stage_b_protect_zone, stage_b_protect_margin_px = _build_stage_b_protect_zone(
                mount_mask,
                front_bbox_px=front_bbox_px,
                candidate=cand,
            )
            stage_b_zone_debug = {
                **candidate_stage_b_decor_debug,
                "wall_id": wall_id,
                "wall_mask_area_px_original": int(np.count_nonzero(stage_b_wall_zone)),
                "protect_mask_area_px_original": int(np.count_nonzero(stage_b_protect_zone)),
                "protect_margin_px": int(stage_b_protect_margin_px),
                "protect_margin_target_m": 0.10,
            }
            cleanup_semantic_types = _cleanup_semantic_types_from_remove_mask(
                remove_mask=remove_mask,
                existing_masks=removable_existing_masks,
                foreground_masks=foreground_masks,
                original_size_wh=original_size_wh,
            )
            has_person_cleanup = bool(cleanup_semantic_types.get("person") or cleanup_semantic_types.get("foreground"))
            has_non_person_cleanup = bool(cleanup_semantic_types.get("non_person_existing_content"))
            cleanup_labels = list(cleanup_semantic_types.get("labels") or [])
            hard = cand.get("hard_constraints", {}) or {}
            if isinstance(hard.get("cleanup"), dict):
                hard["cleanup"]["exported_remove_mask_nonempty"] = bool(cleanup_semantic_types.get("remove_mask_nonempty"))
                hard["cleanup"]["exported_remove_mask_area_px"] = int(cleanup_semantic_types.get("remove_mask_area_px") or 0)
                hard["cleanup"]["cleanup_semantic_types"] = cleanup_semantic_types
            dual_policy = choose_flux2_input_policy(
                cleanup_labels=cleanup_labels,
                has_person_cleanup=has_person_cleanup,
                has_existing_cleanup=has_non_person_cleanup,
            )
            flux2_input_mode = str(dual_policy["flux2_input_mode"])
            cand["artwork_has_frame"] = artwork_has_frame
            cand["flux2_input_mode"] = flux2_input_mode
            cand["flux2_input_policy"] = dual_policy
            cand["bfl_v7_export"] = {
                "erase_order": "original_room_then_hardpaste_after_erase",
                "remove_mask_protects_art_region": False,
                "remove_mask_dilation_policy": remove_mask_dilation_policy,
                "bbox_px": front_bbox_px,
                "front_bbox_px": front_bbox_px,
                "mount_bbox_px": mount_bbox_px,
                "polygon_px": target_polygon_px,
                "front_polygon_px": target_polygon_px,
                "art_region_mask_usage": "front_region_mask_for_core_qa; flux2_edit_mask_path is sent_to_flux2",
                "flux2_edit_mask_path": f"{prefix}/flux2_edit_mask.png",
                "flux2_edit_mask_dilate_px": int(flux2_edit_mask_dilate_px),
                "flux2_mask_required": True,
                "hardpaste_antialias_mode": "supersampled_rgba_lanczos",
                "hardpaste_supersample_scale": int(hardpaste_supersample_scale),
                "art_region_mask_scope": "front_only_for_core_qa",
                "mount_mask_scope": "front_plus_frame_side_for_trace_and_remove_scope",
                "front_region_mask_path": f"{prefix}/front_region_mask.png",
                "frame_side_mask_path": f"{prefix}/frame_side_mask.png",
                "mount_mask_path": f"{prefix}/mount_mask.png",
                "mount_debug_path": f"{prefix}/mount_debug.json",
                "stage_b_decor_zone_mask_path": f"{prefix}/stage_b_decor_zone_mask.png",
                "stage_b_wall_zone_mask_path": f"{prefix}/stage_b_wall_zone_mask.png",
                "stage_b_protect_zone_mask_path": f"{prefix}/stage_b_protect_zone_mask.png",
                "stage_b_zone_debug_path": f"{prefix}/stage_b_zone_debug.json",
                "styling_zone_source": candidate_stage_b_decor_debug.get("styling_zone_source"),
                "artwork_mount_policy": mount_policy,
                "flux2_input_mode": flux2_input_mode,
                "cleanup_semantic_types": cleanup_semantic_types,
                "flux2_input_policy": dual_policy,
            }
            render_prompt = build_render_prompt(
                cand,
                extra_prompt=render_prompt_extra,
                artwork_has_frame=artwork_has_frame,
                artwork_mount_policy=mount_policy,
                flux2_input_mode=flux2_input_mode,
            )
            exit_review = review_hardpaste_exit(hard_paste, cand, rules)
            cand["hardpaste_exit_review"] = exit_review
            cand["approved_for_flux_render"] = bool(exit_review.get("approved_for_flux_render"))
            review_record = {
                "asset_id": asset_id,
                "candidate_id": candidate_id,
                "wall_id": wall_id,
                "approved_for_flux_render": bool(exit_review.get("approved_for_flux_render")),
                "review": exit_review,
                "flux2_input_mode": flux2_input_mode,
                "cleanup_semantic_types": cleanup_semantic_types,
                "flux2_input_policy": dual_policy,
            }
            all_hardpaste_reviews.append(review_record)
            flux_input_files[f"{prefix}/hard_paste.png"] = _png_bytes(hard_paste)
            flux_input_files[f"{prefix}/art_region_mask.png"] = _png_bytes(art_region)
            flux_input_files[f"{prefix}/front_region_mask.png"] = _png_bytes(front_region)
            flux_input_files[f"{prefix}/frame_side_mask.png"] = _png_bytes(frame_side_mask)
            flux_input_files[f"{prefix}/mount_mask.png"] = _png_bytes(mount_mask)
            flux_input_files[f"{prefix}/flux2_edit_mask.png"] = _png_bytes(flux2_edit_mask)
            flux_input_files[f"{prefix}/stage_b_decor_zone_mask.png"] = _png_bytes(candidate_stage_b_decor_zone)
            flux_input_files[f"{prefix}/stage_b_wall_zone_mask.png"] = _png_bytes(stage_b_wall_zone)
            flux_input_files[f"{prefix}/stage_b_protect_zone_mask.png"] = _png_bytes(stage_b_protect_zone)
            flux_input_files[f"{prefix}/stage_b_zone_debug.json"] = json.dumps(stage_b_zone_debug, ensure_ascii=False, indent=2).encode("utf-8")
            flux_input_files[f"{prefix}/mount_debug.json"] = json.dumps(mount_debug, ensure_ascii=False, indent=2).encode("utf-8")
            flux_input_files[f"{prefix}/remove_mask.png"] = _png_bytes(remove_mask)
            flux_input_files[f"{prefix}/debug_overlay.png"] = _png_bytes(overlay)
            flux_input_files[f"{prefix}/candidate.json"] = json.dumps(cand, ensure_ascii=False, indent=2).encode("utf-8")
            flux_input_files[f"{prefix}/render_prompt.txt"] = render_prompt.encode("utf-8")
            if cand["approved_for_flux_render"]:
                approved_hardpaste_assets.append({
                    "asset_id": asset_id,
                    "candidate_id": candidate_id,
                    "wall_id": wall_id,
                    "score": float(cand.get("score", 0.0) or 0.0),
                    "risk_level": cand.get("risk_level"),
                    "original_room_path": "flux_inputs/original_room.png",
                    "artwork_path": "flux_inputs/artwork.png",
                    "hard_paste_path": f"{prefix}/hard_paste.png",
                    "remove_mask_path": f"{prefix}/remove_mask.png",
                    "art_region_mask_path": f"{prefix}/art_region_mask.png",
                    "front_region_mask_path": f"{prefix}/front_region_mask.png",
                    "frame_side_mask_path": f"{prefix}/frame_side_mask.png",
                    "mount_mask_path": f"{prefix}/mount_mask.png",
                    "stage_b_decor_zone_mask_path": f"{prefix}/stage_b_decor_zone_mask.png",
                    "stage_b_wall_zone_mask_path": f"{prefix}/stage_b_wall_zone_mask.png",
                    "stage_b_protect_zone_mask_path": f"{prefix}/stage_b_protect_zone_mask.png",
                    "stage_b_zone_debug_path": f"{prefix}/stage_b_zone_debug.json",
                    "styling_zone_source": candidate_stage_b_decor_debug.get("styling_zone_source"),
                    "flux2_edit_mask_path": f"{prefix}/flux2_edit_mask.png",
                    "flux2_edit_mask_dilate_px": int(flux2_edit_mask_dilate_px),
                    "flux2_mask_required": True,
                    "mount_debug_path": f"{prefix}/mount_debug.json",
                    "candidate_path": f"{prefix}/candidate.json",
                    "render_prompt_path": f"{prefix}/render_prompt.txt",
                    "render_prompt": render_prompt,
                    "target_quad_original": target_quad_original,
                    "polygon_px": target_polygon_px,
                    "front_polygon_px": target_polygon_px,
                    "bbox_px": front_bbox_px,
                    "front_bbox_px": front_bbox_px,
                    "mount_bbox_px": mount_bbox_px,
                    "artwork_has_frame": artwork_has_frame,
                    "artwork_has_frame_source": artwork_has_frame_source,
                    "artwork_analysis": artwork_analysis,
                    "artwork_physical_width_m": hard.get("artwork_width_m"),
                    "artwork_physical_height_m": hard.get("artwork_height_m"),
                    "artwork_mount_policy": mount_policy,
                    "cleanup_provider": "bfl_erase_v1",
                    "render_provider": "bfl_flux2_edit",
                    "erase_order": "original_room_then_hardpaste_after_erase",
                    "remove_mask_protects_art_region": False,
                    "remove_mask_dilation_policy": remove_mask_dilation_policy,
                    "art_region_mask_usage": "front_region_mask_for_core_qa; flux2_edit_mask_path is sent_to_flux2",
                    "art_region_mask_scope": "front_only_for_core_qa",
                    "mount_mask_scope": "front_plus_frame_side_for_trace_and_remove_scope",
                    "hardpaste_antialias_mode": "supersampled_rgba_lanczos",
                    "hardpaste_supersample_scale": int(hardpaste_supersample_scale),
                    "flux2_input_mode": flux2_input_mode,
                    "flux2_input_policy": dual_policy,
                    "cleanup_semantic_types": cleanup_semantic_types,
                    "exit_review": exit_review,
                })
        except Exception as e:
            cand["hardpaste_exit_review"] = {
                "enabled": True,
                "status": "hardpaste_export_error",
                "approved": False,
                "approved_for_flux_render": False,
                "reasons": [str(e)],
            }
            cand["approved_for_flux_render"] = False
            all_hardpaste_reviews.append({
                "asset_id": asset_id,
                "candidate_id": candidate_id,
                "wall_id": wall_id,
                "approved_for_flux_render": False,
                "review": cand["hardpaste_exit_review"],
            })

    final_hardpaste_assets, flux_selection_summary = select_flux_manifest_items(
        approved_hardpaste_assets,
        per_wall_limit=per_wall_bfl_limit,
        global_limit=global_final_render_limit,
        nms_center_distance_fraction=nms_center_distance_fraction,
        nms_polygon_iou_threshold=nms_polygon_iou_threshold,
    )
    selected_asset_ids = {str(x.get("asset_id")) for x in final_hardpaste_assets}
    for review in all_hardpaste_reviews:
        if review.get("approved_for_flux_render") and str(review.get("asset_id")) not in selected_asset_ids:
            review["approved_for_flux_render"] = False
            review["selection_status"] = "approved_but_dropped_by_manifest_cap_or_nms"

    selected_requires_hardpaste_vlm_review = bool(selected and selected.get("requires_hardpaste_vlm_review"))
    selected_exit_code = "OK_SELECTED_HIGH_RISK_VLM_REVIEW_REQUIRED" if selected_requires_hardpaste_vlm_review else "OK_SELECTED"
    if selected and not final_hardpaste_assets:
        selected_exit_code = "OK_NO_FLUX_APPROVED_HARDPASTE_PENDING_REVIEW"

    plan = {
        "status": "ok" if selected else "partial",
        "exit_code": selected_exit_code if selected else ("PARTIAL_REVIEW_CANDIDATES_ONLY" if near_miss_candidates else "PARTIAL_ALL_REJECTED_WITH_REASONS"),
        "pipeline_build": {
            "build_id": HANGING_PIPELINE_BUILD_ID,
            "height_contract": "metric_height_authoritative_exact_footprint_v3",
            "support_contract": "recognized_support_label_specific_clearance_v5",
            "gravity_contract": "camera_up_canonicalized_v2",
            "render_contract": "fullspace_identity_locked_shadow_v2",
            "worker_result_contract": "render_review_is_completed_review_v1",
        },
        "requires_hardpaste_vlm_review": selected_requires_hardpaste_vlm_review,
        "image_space": {
            "processed_size": [w, h],
            "original_size": [bundle.original_size[0], bundle.original_size[1]],
            "scale_to_original": [bundle.scale_to_original[0], bundle.scale_to_original[1]],
            "distortion_policy": {
                "mode": "wide_angle_pinhole_approx" if (bundle.intr.fov_deg or 0) > 70 else "no_undistort",
                "fov_deg": bundle.intr.fov_deg,
                "warnings": ["distortion_unresolved_high_fov"] if (bundle.intr.fov_deg or 0) > 70 else [],
            },
        },
        "selected_candidate": selected,
        "ranked_candidates": candidates,
        "passed_walls": passed_walls,
        "rejected_walls": rejected_walls,
        "near_miss_candidates": near_miss_candidates,
        "final_hardpaste_assets": final_hardpaste_assets,
        "flux_inputs_manifest": "flux_inputs/manifest.json",
        "hardpaste_exit_gate": {
            "enabled": True,
            "scope": "all_passed_candidates_not_best_wall_only",
            "total_hardpaste_count": len(all_hardpaste_reviews),
            "approved_before_manifest_policy_count": len(approved_hardpaste_assets),
            "selected_for_flux_render_count": len(final_hardpaste_assets),
            "blocked_or_pending_count": len([r for r in all_hardpaste_reviews if not r.get("approved_for_flux_render")]),
            "manifest_selection_summary": flux_selection_summary,
            "candidate_diversity_summary": candidate_diversity_summary,
            "reviews": all_hardpaste_reviews,
        },
        "debug": {
            "gravity_reference": {
                "g_hat_cam": [float(x) for x in g_hat],
                "confidence": float(g_conf),
                "source": gravity_source,
                "details": g_debug,
            },
            "rejected_candidates": rejected,
            "near_miss_candidates": near_miss_candidates,
            "auto_multiwall_debug": auto_multiwall_debug,
            "debug_assets": {
                "enabled": True,
                "semantic_wall_debug": "debug_assets/semantic_wall_debug.png",
                "wall_assets_exported": True,
                "rejected_wall_assets_exported": True,
                "client_result_exported": True,
            },
            "warnings": [],
            "semantic_object_inventory": semantic_object_inventory,
            "mvp_cleanup": {
                "enabled": True,
                "stage": "original_room_cleanup_then_hardpaste_then_render",
                "candidate_wall_scope_only": True,
                "cleanup_mask_definition": "(current_wall_mask ∩ removable_existing_content_mask) ∪ global foreground/person mask; blocking shelves/cabinets are protected",
                "foreground_person_global_cleanup_allowed": bool(remove_foreground_global),
                "flux2_edit_mask_definition": "dilate(mount_mask, flux_render.edit_mask_dilate_px) and send to FLUX.2 as edit mask",
                "flux2_edit_mask_dilate_px": int(flux2_edit_mask_dilate_px),
                "off_wall_existing_content_protected": True,
                "erase_provider": "bfl_erase_v1",
                "remove_mask_dilation_policy": remove_mask_dilation_policy,
            },
            "flux": {
                "integrated": False,
                "downstream_contract": "scripts/run_bfl_hanging_flux_pipeline.py reads flux_inputs/manifest.json; erases original_room, hard-pastes onto clean room, then renders selected items only",
                "provider_count": 2,
                "providers": ["bfl_erase_v1", "bfl_flux2_edit"],
                "best_wall_selection": False,
                "per_wall_bfl_limit": per_wall_bfl_limit,
                "global_final_render_limit": global_final_render_limit,
                "art_region_mask_usage": "front_region_mask_for_core_qa; separate flux2_edit_mask is sent_to_flux2",
                "hardpaste_antialias_mode": "supersampled_rgba_lanczos",
                "hardpaste_supersample_scale": int(hardpaste_supersample_scale),
            },
        },
    }

    client_result = build_client_result(plan, candidates, rejected, artwork_w_m=artwork_w_m, artwork_h_m=artwork_h_m)

    blocked_or_pending_reviews = [r for r in all_hardpaste_reviews if not r.get("approved_for_flux_render")]
    exit_review_summary = {
        "total_hardpaste_count": len(all_hardpaste_reviews),
        "approved_before_manifest_policy_count": len(approved_hardpaste_assets),
        "selected_for_flux_render_count": len(final_hardpaste_assets),
        "blocked_or_pending_count": len(blocked_or_pending_reviews),
        "near_miss_count": len(near_miss_candidates),
        "top_near_miss_reason": near_miss_candidates[0].get("near_miss_reason") if near_miss_candidates else None,
        "has_safe_geometry_but_small_projection": False,
        "projection_size_gate_enabled": False,
        "vlm_config_dependency": "review-required candidates are blocked when VLM is missing and exit_review.block_unreviewed_required_vlm=true",
        "manifest_selection_summary": flux_selection_summary,
        "candidate_diversity_summary": candidate_diversity_summary,
    }
    flux_manifest = {
        "version": 2,
        "source": "hanging_mvp",
        "scope": "selected_candidates_after_exit_gate_per_wall_cap_nms_global_cap",
        "cleanup_provider": "bfl_erase_v1",
        "render_provider": "bfl_flux2_edit",
        "semantic_object_inventory": semantic_object_inventory,
        "erase_order": "original_room_then_hardpaste_after_erase",
        "original_room_path": "flux_inputs/original_room.png",
        "artwork_path": "flux_inputs/artwork.png",
        "art_region_mask_usage": "front_region_mask_for_core_qa; per-item flux2_edit_mask_path is sent_to_flux2",
        "flux2_mask_required": True,
        "flux2_mask_sent": True,
        "flux2_edit_mask_definition": "dilate(mount_mask, flux_render.edit_mask_dilate_px)",
        "flux2_edit_mask_dilate_px": int(flux2_edit_mask_dilate_px),
        "render_size_qa_policy": "mask_anchored_core_ssim_and_forbidden_outer_guard",
        "projection_size_gate_enabled": False,
        "projection_size_policy": "no_minimum_projected_width_height_or_area_gate",
        "fallback_product_semantics": "never_silent_replace_failed_flux_with_hardpaste_success",
        "remove_mask_dilation_policy": remove_mask_dilation_policy,
        "per_wall_bfl_limit": per_wall_bfl_limit,
        "global_final_render_limit": global_final_render_limit,
        "nms_center_distance_fraction_of_artwork_width": nms_center_distance_fraction,
        "nms_polygon_iou_threshold": nms_polygon_iou_threshold,
        "selection_summary": flux_selection_summary,
        "items": final_hardpaste_assets,
        "blocked_or_pending_reviews": blocked_or_pending_reviews,
        "near_miss_count": len(near_miss_candidates),
    }

    files = {
        "hanging_plan.json": json.dumps(plan, ensure_ascii=False, indent=2).encode("utf-8"),
        "ranked_candidates.json": json.dumps(candidates, ensure_ascii=False, indent=2).encode("utf-8"),
        "passed_walls.json": json.dumps(passed_walls, ensure_ascii=False, indent=2).encode("utf-8"),
        "rejected_walls.json": json.dumps(rejected_walls, ensure_ascii=False, indent=2).encode("utf-8"),
        "near_miss_candidates.json": json.dumps(near_miss_candidates, ensure_ascii=False, indent=2).encode("utf-8"),
        "client_result.json": json.dumps(client_result, ensure_ascii=False, indent=2).encode("utf-8"),
        "flux_inputs/manifest.json": json.dumps(flux_manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        "flux_inputs/blocked_or_pending_reviews.json": json.dumps(blocked_or_pending_reviews, ensure_ascii=False, indent=2).encode("utf-8"),
        "flux_inputs/exit_review_summary.json": json.dumps(exit_review_summary, ensure_ascii=False, indent=2).encode("utf-8"),
    }
    files.update(flux_input_files)

    try:
        semantic_debug_img = draw_semantic_wall_debug(room_rgb.copy(), semantics.get("objects", []), semantics.get("multiwall_split_debug"), target_space="original")
        files["debug_assets/semantic_wall_debug.png"] = debug_png_bytes(semantic_debug_img)
    except Exception as e:
        files["debug_assets/semantic_wall_debug_error.txt"] = str(e).encode("utf-8")

    best_by_wall = {}
    for cand in candidates:
        wid = cand["wall_id"]
        if wid not in best_by_wall or cand["score"] > best_by_wall[wid]["score"]:
            best_by_wall[wid] = cand

    rejected_by_wall = {}
    for item in rejected:
        rejected_by_wall.setdefault(item.get("wall_id", "unknown_wall"), []).append(item)

    for wall_id, cand in best_by_wall.items():
        quad_orig = cand["projection"]["projected_quad_px_original"]
        hard_paste, art_region = warp_artwork_to_quad(room_rgb, art_rgba, quad_orig, supersample_scale=int(rule_value(rules, ["hardpaste_quality", "supersample_scale"], 4)))
        overlay = _draw_overlay(room_rgb.copy(), quad_orig)
        wall_prefix = f"wall_assets/{wall_id}"
        files[f"{wall_prefix}/hard_paste.png"] = _png_bytes(hard_paste)
        files[f"{wall_prefix}/art_region_mask.png"] = _png_bytes(art_region)
        files[f"{wall_prefix}/debug_overlay.png"] = _png_bytes(overlay)
        files[f"{wall_prefix}/best_candidate.json"] = json.dumps(cand, ensure_ascii=False, indent=2).encode("utf-8")

        wall_debug = wall_debug_registry.get(wall_id, {})
        wall_candidates = [c for c in candidates if c.get("wall_id") == wall_id]
        wall_rejected = rejected_by_wall.get(wall_id, [])
        files[f"{wall_prefix}/candidates.json"] = json.dumps(wall_candidates, ensure_ascii=False, indent=2).encode("utf-8")
        files[f"{wall_prefix}/rejected_candidates.json"] = json.dumps(wall_rejected, ensure_ascii=False, indent=2).encode("utf-8")

        for mask_key in ["wall_mask", "filtered_wall_mask", "semantic_clear_wall_mask", "safe_axis_wall_mask", "solid_wall_mask", "plane_consistency_mask", "normal_consistency_mask", "opening_recess_mask"]:
            if wall_debug.get(mask_key) is not None:
                files[f"{wall_prefix}/{mask_key}.png"] = debug_png_bytes(mask_preview(wall_debug[mask_key]))

        try:
            wall_mask_debug = wall_debug.get("filtered_wall_mask")
            if wall_mask_debug is None:
                wall_mask_debug = wall_debug.get("wall_mask")
            wall_overlay = build_wall_debug_overlay(room_rgb.copy(), wall_mask_debug, wall_debug.get("solid_wall_mask"), cand, wall_id)
            files[f"{wall_prefix}/wall_debug_overlay.png"] = debug_png_bytes(wall_overlay)
        except Exception as e:
            files[f"{wall_prefix}/wall_debug_overlay_error.txt"] = str(e).encode("utf-8")

    for wall_id, items in rejected_by_wall.items():
        if wall_id in best_by_wall:
            continue
        wall_prefix = f"rejected_wall_assets/{wall_id}"
        files[f"{wall_prefix}/reject_summary.json"] = json.dumps({"wall_id": wall_id, "rejected": items}, ensure_ascii=False, indent=2).encode("utf-8")
        wall_debug = wall_debug_registry.get(wall_id, {})
        wall_mask_debug = wall_debug.get("filtered_wall_mask")
        if wall_mask_debug is None:
            wall_mask_debug = wall_debug.get("wall_mask")
        reasons = []
        for item in items:
            if item.get("reason"):
                reasons.append(str(item.get("reason")))
            reasons.extend([str(r) for r in item.get("reject_reasons", []) or []])
        try:
            files[f"{wall_prefix}/debug_overlay.png"] = debug_png_bytes(draw_rejected_wall_overlay(room_rgb.copy(), wall_mask_debug, wall_id, reasons))
        except Exception as e:
            files[f"{wall_prefix}/debug_overlay_error.txt"] = str(e).encode("utf-8")

    return _zip_response(files)
