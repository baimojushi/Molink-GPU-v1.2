from __future__ import annotations

import io
import json
import os
import re
import zipfile
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageOps
import cv2
import numpy as np

from .metric_meta import read_sizes_from_metric3d_zip
from .polygonize import bbox_to_polygon, scale_bbox, scale_polygon
from .vlm_client import parse_with_vlm, parse_artwork_frame_with_vlm
from .debug_assets import draw_semantic_wall_debug, png_bytes
from .sam_refiner import coerce_optional_bool, get_sam_runtime_status, refine_objects_with_sam

API_KEY = os.getenv("API_KEY", "")
ALLOW_UNAUTH_LOCAL = os.getenv("ALLOW_UNAUTH_LOCAL", "").strip().lower() in {"1", "true", "yes", "on"}

app = FastAPI(title="semantic-mvp-bbox-api")

ROLE_BY_LABEL = {
    "wall": "wall",
    "floor": "support",
    "roof": "support",
    "sofa": "support",
    "table": "support",
    "cabinet": "support",
    "window": "forbidden",
    "door": "forbidden",
    "tv": "forbidden",
    "mirror": "forbidden",
    "lamp": "forbidden",
    "track_lighting": "forbidden",
    "arch": "forbidden",
    "opening": "forbidden",
    "existing_art": "existing_content",
    "painting": "existing_content",
    "poster": "existing_content",
    "framed_art": "existing_content",
    "artwork": "existing_content",
    "green_board": "existing_content",
    "notice_board": "existing_content",
    "wall_clutter": "existing_content",
    "wall_label": "existing_content",
    "hook": "existing_content",
    "nail": "existing_content",
    "small_shelf": "existing_content",
    "pedestal": "non_support",
    "plinth": "non_support",
    "display_stand": "non_support",
    "foreground_object": "foreground",
    "person": "foreground",
    "human": "foreground",
    "people": "foreground",
    "plant": "foreground",
    "chair": "foreground",
}

LABEL_ALIASES = {
    "ceiling": "roof",
    "ceilings": "roof",
    "roof ceiling": "roof",
    "wall art": "existing_art",
    "wall_art": "existing_art",
    "art": "existing_art",
    "artwork": "existing_art",
    "painting": "existing_art",
    "poster": "existing_art",
    "framed painting": "existing_art",
    "framed artwork": "existing_art",
    "frame": "framed_art",
    "picture": "existing_art",
    "green board": "green_board",
    "notice board": "notice_board",
    "label": "wall_label",
    "sign": "wall_label",
    "signage": "wall_label",
    "person": "person",
    "people": "person",
    "human": "person",
    "man": "person",
    "woman": "person",
    "visitor": "person",
}

EXISTING_CONTENT_LABELS = {
    "existing_art", "painting", "poster", "framed_art", "wall_clutter",
    "wall_label", "hook", "nail", "small_shelf", "green_board", "notice_board",
}
FOREGROUND_LABELS = {"person", "human", "people", "foreground_object", "plant", "chair"}


def _truthy_env(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default)
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def _causal_light_fields_enabled() -> bool:
    return _truthy_env("MOLINK_CAUSAL_ENABLED", "0") and _truthy_env("MOLINK_CAUSAL_VLM_FIELDS", "0")


def _coerce_optional_float(value: Any, *, lo: float = 0.0, hi: float = 1.0) -> float | None:
    try:
        if value is None or value == "":
            return None
        return max(lo, min(hi, float(value)))
    except Exception:
        return None


def _normalize_light_payload(raw: Any) -> dict[str, Any] | None:
    if not _causal_light_fields_enabled() or not isinstance(raw, dict):
        return None

    def pick(name: str, allowed: set[str]) -> str | None:
        value = str(raw.get(name) or "").strip().lower()
        return value if value in allowed else None

    out = {
        "theta": pick("theta", {"same", "opposite", "perpendicular", "none"}),
        "exterior_brightness_class": pick("exterior_brightness_class", {"bright_direct", "overcast", "artificial"}),
        "ambient_light_mode": pick("ambient_light_mode", {"gallery_ambient", "soft_diffuse", "direct_dramatic", "flat_artificial", "mixed"}),
        "shadow_presence": pick("shadow_presence", {"none", "subtle", "moderate", "dramatic"}),
        "temporal_light_quality": pick("temporal_light_quality", {"static", "dynamic"}),
        "ceiling_height": pick("ceiling_height", {"low", "standard", "high", "dramatic"}),
        "landscape_competition_score": _coerce_optional_float(raw.get("landscape_competition_score")),
    }
    if out.get("landscape_competition_score") is not None:
        out["landscape_competition_scope"] = "scene"
    if isinstance(raw.get("surface_active_light"), bool):
        out["surface_active_light"] = raw.get("surface_active_light")
        out["surface_active_light_scope"] = "scene"
    else:
        text = str(raw.get("surface_active_light") or "").strip().lower()
        out["surface_active_light"] = True if text in {"true", "1", "yes"} else False if text in {"false", "0", "no"} else None
        if out["surface_active_light"] is not None:
            out["surface_active_light_scope"] = "scene"
    return {k: v for k, v in out.items() if v is not None}


def _normalize_wall_light_extras(raw: Any) -> dict[str, Any]:
    if not _causal_light_fields_enabled() or not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    score = _coerce_optional_float(raw.get("landscape_competition_score"))
    if score is not None:
        out["landscape_competition_score"] = score
        out["landscape_competition_scope"] = "wall"
    sal = raw.get("surface_active_light")
    if isinstance(sal, bool):
        out["surface_active_light"] = sal
        out["surface_active_light_scope"] = "wall"
    else:
        text = str(sal or "").strip().lower()
        if text in {"true", "1", "yes"}:
            out["surface_active_light"] = True
            out["surface_active_light_scope"] = "wall"
        elif text in {"false", "0", "no"}:
            out["surface_active_light"] = False
            out["surface_active_light_scope"] = "wall"
    return out


def _copy_wall_light_extras(src: dict[str, Any], dst: dict[str, Any]) -> None:
    for k, v in _normalize_wall_light_extras(src).items():
        dst[k] = v


_INSTANCE_SUFFIX_RE = re.compile(r"^(?P<base>.+?)[_\-\s]+\d{1,4}$")


def _label_lookup_candidates(label: str) -> list[str]:
    raw = str(label or "foreground_object").strip().lower().replace("-", "_")
    raw = re.sub(r"\s+", "_", raw)
    candidates = [raw, raw.replace("_", " ")]
    m = _INSTANCE_SUFFIX_RE.match(raw)
    if m:
        base = m.group("base")
        candidates.extend([base, base.replace("_", " ")])
    out: list[str] = []
    for item in candidates:
        if item and item not in out:
            out.append(item)
    return out


def _canonical_label_and_role(label: str, role: str | None) -> tuple[str, str]:
    candidates = _label_lookup_candidates(label)
    raw_label = candidates[0] if candidates else "foreground_object"
    canonical = raw_label
    for candidate in candidates:
        aliased = LABEL_ALIASES.get(candidate)
        if aliased:
            canonical = aliased
            break
        candidate_key = candidate.replace(" ", "_")
        if candidate_key in ROLE_BY_LABEL or candidate_key in EXISTING_CONTENT_LABELS or candidate_key in FOREGROUND_LABELS:
            canonical = candidate_key
            break
    raw_role = str(role or "").strip().lower()
    inferred = ROLE_BY_LABEL.get(canonical, "foreground")
    if canonical in EXISTING_CONTENT_LABELS:
        inferred = "existing_content"
    elif canonical in {"person", "human", "people"}:
        canonical = "person"
        inferred = "foreground"
    if raw_role in {"wall", "support", "forbidden", "foreground", "non_support", "existing_content"}:
        # Existing-content/person labels override VLM role mistakes because cleanup
        # depends on object inventory recall. Suffix-style detector labels such as
        # painting_01/framed_art_02/person_01 are canonicalized before this gate.
        if canonical in EXISTING_CONTENT_LABELS or raw_role == "existing_content":
            if raw_role == "existing_content" and canonical not in EXISTING_CONTENT_LABELS:
                # Preserve role recall even if a provider emits a new wall-attached
                # label name that is not yet in the configured class list.
                return canonical, "existing_content"
            return canonical, "existing_content"
        if canonical == "person" or raw_role == "foreground":
            return canonical, "foreground"
        return canonical, raw_role
    return canonical, inferred


def _build_object_inventory(objects):
    counts_by_role: dict[str, int] = {}
    counts_by_label: dict[str, int] = {}
    removable = []
    foreground = []
    walls = []
    for obj in objects:
        role = str(obj.get("role") or "unknown")
        label = str(obj.get("label") or "unknown")
        counts_by_role[role] = counts_by_role.get(role, 0) + 1
        counts_by_label[label] = counts_by_label.get(label, 0) + 1
        record = {
            "object_id": obj.get("object_id"),
            "label": label,
            "role": role,
            "bbox_xyxy_original": obj.get("bbox_xyxy_original"),
            "bbox_xyxy_processed": obj.get("bbox_xyxy_processed"),
            "confidence": obj.get("confidence"),
            "source": obj.get("source"),
        }
        if role == "existing_content":
            removable.append(record)
        elif role == "foreground":
            foreground.append(record)
        elif role == "wall":
            walls.append(record)
    status = "ok"
    warnings = []
    if not removable and not foreground:
        status = "empty_removable_and_foreground_inventory"
        warnings.append("semantic_object_inventory_empty: no existing_art/person/wall_attached objects detected; do not interpret empty cleanup masks as proof that cleanup is unnecessary")
    elif not removable:
        status = "no_existing_content_detected"
    return {
        "status": status,
        "method": "full_scene_vlm_object_inventory_plus_optional_sam_refine",
        "per_wall_high_res_crop_used": False,
        "counts_by_role": counts_by_role,
        "counts_by_label": counts_by_label,
        "removable_existing_content": removable,
        "foreground_objects": foreground,
        "walls": walls,
        "warnings": warnings,
    }


def _clip_box_xyxy(box, width, height):
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(float(width - 1), x1))
    x2 = max(0.0, min(float(width - 1), x2))
    y1 = max(0.0, min(float(height - 1), y1))
    y2 = max(0.0, min(float(height - 1), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]



def _valid_vlm_bbox_values(bbox):
    if not bbox or len(bbox) != 4:
        return None
    try:
        vals = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(vals)):
        return None
    return vals

def _infer_vlm_bbox_coordinate_space(objects, original_w, original_h, vlm_image_meta: Optional[Dict[str, Any]] = None):
    """Infer whether VLM boxes are original pixels, sent-image pixels, or normalized coordinates.

    The OpenAI-compatible client may resize large images before sending them to
    the VLM. Prompting and normalization must use that sent-image size; otherwise
    boxes from high-resolution photos are projected into the upper-left region of
    the original image. Some providers still return 0..1000 visual coordinates,
    so keep the normalized-grid guard as a second path.
    """
    boxes = []
    for obj in objects or []:
        bbox = obj.get("bbox_xyxy") or obj.get("bbox") or []
        vals = _valid_vlm_bbox_values(bbox)
        if vals is None:
            continue
        boxes.append(vals)

    debug = {
        'enabled': True,
        'source_coordinate_space': 'original_pixels',
        'sample_count': len(boxes),
        'reason': 'no_valid_vlm_boxes' if not boxes else 'pixel_space_default',
    }
    if not boxes:
        return 'original_pixels', debug

    xs = [v for x1, _, x2, _ in boxes for v in (x1, x2)]
    ys = [v for _, y1, _, y2 in boxes for v in (y1, y2)]
    max_x = float(max(xs))
    max_y = float(max(ys))
    min_x = float(min(xs))
    min_y = float(min(ys))
    max_coord = max(max_x, max_y)
    vlm_image_meta = vlm_image_meta or {}
    sent_size = vlm_image_meta.get("sent_size") if isinstance(vlm_image_meta, dict) else None
    try:
        sent_w = int(sent_size[0]) if isinstance(sent_size, (list, tuple)) and len(sent_size) >= 2 else int(original_w)
        sent_h = int(sent_size[1]) if isinstance(sent_size, (list, tuple)) and len(sent_size) >= 2 else int(original_h)
    except Exception:
        sent_w, sent_h = int(original_w), int(original_h)
    sent_w = max(1, sent_w)
    sent_h = max(1, sent_h)

    debug.update({
        'raw_min_x': min_x,
        'raw_max_x': max_x,
        'raw_min_y': min_y,
        'raw_max_y': max_y,
        'original_size': [int(original_w), int(original_h)],
        'vlm_image_meta': vlm_image_meta,
        'sent_size': [int(sent_w), int(sent_h)],
    })

    if max_coord <= 1.5 and min_x >= -0.05 and min_y >= -0.05:
        debug.update({
            'source_coordinate_space': 'normalized_0_1',
            'reason': 'all_vlm_bbox_coords_are_unit_interval_like',
            'scale_x_to_original': float(original_w),
            'scale_y_to_original': float(original_h),
        })
        return 'normalized_0_1', debug

    normalized_1000_like = (min_x >= -20.0 and min_y >= -20.0 and max_x <= 1024.0 and max_y <= 1024.0 and max_coord >= 10.0)
    y_exceeds_original = max_y > float(original_h) * 1.03 and original_h < 980
    looks_1000_tall = original_h < 900 and max_y >= 900.0
    high_res_image_1000_grid = (float(original_w) > 1400.0 or float(original_h) > 1400.0) and (sent_w > 1100 or sent_h > 1100)
    compact_1000_grid_on_original = (
        normalized_1000_like
        and max_coord < max(1.0, min(float(original_w), float(original_h)) * 0.5)
    )

    if normalized_1000_like and (
        y_exceeds_original
        or looks_1000_tall
        or high_res_image_1000_grid
        or compact_1000_grid_on_original
    ):
        debug.update({
            'source_coordinate_space': 'normalized_0_1000',
            'reason': 'vlm_boxes_match_0_1000_coordinate_grid_not_original_or_sent_pixels',
            'scale_x_to_original': float(original_w) / 1000.0,
            'scale_y_to_original': float(original_h) / 1000.0,
        })
        return 'normalized_0_1000', debug

    sent_differs_from_original = (abs(float(sent_w) - float(original_w)) > 1.0 or abs(float(sent_h) - float(original_h)) > 1.0)
    boxes_fit_sent_image = (min_x >= -20.0 and min_y >= -20.0 and max_x <= float(sent_w) * 1.03 and max_y <= float(sent_h) * 1.03)
    if sent_differs_from_original and boxes_fit_sent_image:
        debug.update({
            'source_coordinate_space': 'sent_image_pixels',
            'reason': 'vlm_boxes_match_resized_request_image_pixels',
            'scale_x_to_original': float(original_w) / float(sent_w),
            'scale_y_to_original': float(original_h) / float(sent_h),
        })
        return 'sent_image_pixels', debug

    # Dynamic VLM internal size inference: VLM may internally resize images
    # (e.g. qwen3-vl resizes 2048×1536 → 1536×1152). Infer the actual pixel
    # space from bbox coordinate ranges.
    inferred_vlm_w = max(1, int(round(max_x)))
    inferred_vlm_h = max(1, int(round(max_y)))
    # Check if coords roughly match a 4:3 or 3:4 aspect ratio and are
    # substantially smaller than original but larger than 1000-grid
    aspect_orig = float(original_w) / max(1, float(original_h))
    aspect_inferred = float(inferred_vlm_w) / max(1, float(inferred_vlm_h))
    aspect_sent = float(sent_w) / max(1, float(sent_h))
    # If inferred aspect ratio matches original within 10%, treat max_coords
    # as the VLM's actual pixel space
    aspect_match = abs(aspect_inferred - aspect_orig) < 0.10
    coords_between_1000_and_original = (
        max_coord > 1024.0 and
        max_coord < min(float(original_w), float(original_h)) * 0.95
    )
    if aspect_match and coords_between_1000_and_original:
        # VLM used an internal pixel space — compute actual scale
        scale_x = float(original_w) / float(max(1, inferred_vlm_w))
        scale_y = float(original_h) / float(max(1, inferred_vlm_h))
        debug.update({
            'source_coordinate_space': 'sent_image_pixels',
            'reason': f'dynamic_inferred_vlm_internal_pixel_space_{inferred_vlm_w}x{inferred_vlm_h}',
            'scale_x_to_original': scale_x,
            'scale_y_to_original': scale_y,
            'inferred_vlm_internal_size': [inferred_vlm_w, inferred_vlm_h],
        })
        return 'sent_image_pixels', debug

    debug.update({
        'scale_x_to_original': 1.0,
        'scale_y_to_original': 1.0,
    })
    return 'original_pixels', debug

def _normalize_vlm_bbox_to_original_pixels(bbox, original_w, original_h, coordinate_space, vlm_image_meta: Optional[Dict[str, Any]] = None):
    vals = _valid_vlm_bbox_values(bbox)
    if vals is None:
        return []
    x1, y1, x2, y2 = vals
    if coordinate_space == 'normalized_0_1':
        box = [x1 * float(original_w), y1 * float(original_h), x2 * float(original_w), y2 * float(original_h)]
    elif coordinate_space == 'normalized_0_1000':
        box = [x1 / 1000.0 * float(original_w), y1 / 1000.0 * float(original_h), x2 / 1000.0 * float(original_w), y2 / 1000.0 * float(original_h)]
    elif coordinate_space == 'sent_image_pixels':
        vlm_image_meta = vlm_image_meta or {}
        sent_size = vlm_image_meta.get("sent_size") if isinstance(vlm_image_meta, dict) else None
        try:
            sent_w = float(sent_size[0]) if isinstance(sent_size, (list, tuple)) and len(sent_size) >= 2 else float(original_w)
            sent_h = float(sent_size[1]) if isinstance(sent_size, (list, tuple)) and len(sent_size) >= 2 else float(original_h)
        except Exception:
            sent_w, sent_h = float(original_w), float(original_h)
        box = [
            x1 / max(1.0, sent_w) * float(original_w),
            y1 / max(1.0, sent_h) * float(original_h),
            x2 / max(1.0, sent_w) * float(original_w),
            y2 / max(1.0, sent_h) * float(original_h),
        ]
    else:
        box = [x1, y1, x2, y2]
    return _clip_box_xyxy(box, original_w, original_h)

def _load_metric_geometry_from_zip(zip_bytes: bytes):
    if not zip_bytes:
        return None, None, 'metric3d_zip_missing'
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as z:
            normal = np.load(io.BytesIO(z.read('normal_xyz.npy')))
            depth = np.load(io.BytesIO(z.read('depth_m.npy')))
        return normal, depth, None
    except Exception as e:
        return None, None, f'metric_geometry_load_failed: {e}'

def _estimate_g_hat_from_normals(normal: np.ndarray):
    if normal is None:
        return np.array([0.0, -1.0, 0.0], np.float32), 0.0, {'reason': 'normal_missing'}
    h, w, _ = normal.shape
    lower = np.zeros((h, w), dtype=bool)
    lower[int(h * 0.55):, :] = True
    cand = lower & np.isfinite(normal).all(axis=-1) & (normal[..., 1] < -0.75) & (np.abs(normal[..., 2]) < 0.45)
    if int(np.count_nonzero(cand)) < 500:
        cand = np.isfinite(normal).all(axis=-1) & (normal[..., 1] < -0.75) & (np.abs(normal[..., 2]) < 0.45)
    if int(np.count_nonzero(cand)) < 100:
        return np.array([0.0, -1.0, 0.0], np.float32), 0.0, {'reason': 'insufficient_floor_like_normals', 'count': int(np.count_nonzero(cand))}
    vecs = normal[cand].astype(np.float32)
    mean = vecs.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm < 1e-6:
        return np.array([0.0, -1.0, 0.0], np.float32), 0.0, {'reason': 'degenerate_mean'}
    g = mean / norm
    spread = float(np.mean(np.linalg.norm(vecs - g.reshape(1, 3), axis=1)))
    conf = max(0.0, min(1.0, 1.0 - min(spread, 1.0)))
    return g.astype(np.float32), conf, {'count': int(np.count_nonzero(cand)), 'spread_l2': spread}

def _rect_mask(h, w, bbox_xyxy):
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    m = np.zeros((h, w), np.uint8)
    if x2 > x1 and y2 > y1:
        m[y1:y2 + 1, x1:x2 + 1] = 255
    return m

def _box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 1e-6 else 0.0

def _name_wall_proposals_left_to_right(props):
    n = len(props)
    if n == 1:
        props[0]['object_id'] = 'wall_front_01'
        props[0]['position_hint'] = 'front'
        return props
    if n == 2:
        names = ['wall_left_01', 'wall_right_01']
    elif n == 3:
        names = ['wall_left_01', 'wall_front_01', 'wall_right_01']
    else:
        names = [f'wall_{i+1:02d}' for i in range(n)]
    for p, nm in zip(props, names):
        p['object_id'] = nm
        p['position_hint'] = nm.replace('wall_', '').rsplit('_', 1)[0]
    return props

def _is_wall_obj(obj):
    label = str(obj.get('label', ''))
    return obj.get('role') == 'wall' or label == 'wall' or label.startswith('wall_')

def _box_area_xyxy(box):
    if not box or len(box) != 4:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def _box_intersection_xyxy(a, b):
    if not a or not b or len(a) != 4 or len(b) != 4:
        return 0.0, None
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0, None
    return (ix2 - ix1) * (iy2 - iy1), [ix1, iy1, ix2, iy2]

def _wall_x_coverage_ratio(walls, processed_w):
    intervals = []
    for wall in walls:
        box = wall.get('bbox_xyxy_processed') or wall.get('bbox_xyxy') or []
        if len(box) != 4:
            continue
        x1, _, x2, _ = _clip_box_xyxy(box, processed_w, 10**9)
        if x2 > x1:
            intervals.append([x1, x2])
    if not intervals:
        return 0.0, []
    intervals.sort()
    merged = []
    for x1, x2 in intervals:
        if not merged or x1 > merged[-1][1]:
            merged.append([x1, x2])
        else:
            merged[-1][1] = max(merged[-1][1], x2)
    total = sum(x2 - x1 for x1, x2 in merged)
    return float(total / max(1.0, float(processed_w))), merged

def _sanitize_wall_bboxes_against_edge_structures(objects, processed_w, processed_h, sx, sy):
    """Trim wall rectangles when a door/window/opening occupies a wall edge.

    VLM may return a wall bbox that includes a door panel at the edge. The
    hanging stage subtracts forbidden masks later, but a rectangle that still
    names the door as a wall is confusing and can seed bad wall proposals. This
    conservative trim only adjusts left/right edges when the forbidden object is
    tall and edge-attached; interior windows/openings are left for fine-grained
    clearance checks.
    """
    structural_labels = {'opening', 'arch', 'door', 'window', 'tv', 'mirror', 'lamp', 'track_lighting'}
    structures = [o for o in objects if str(o.get('label')) in structural_labels or o.get('role') == 'forbidden']
    out = []
    trims = []
    for obj in objects:
        if not _is_wall_obj(obj):
            out.append(obj)
            continue
        box = obj.get('bbox_xyxy_processed') or obj.get('bbox_xyxy') or []
        if len(box) != 4:
            out.append(obj)
            continue
        wx1, wy1, wx2, wy2 = _clip_box_xyxy(box, processed_w, processed_h)
        orig_box = [wx1, wy1, wx2, wy2]
        wall_w = max(1.0, wx2 - wx1)
        wall_h = max(1.0, wy2 - wy1)
        min_keep_w = max(55.0, processed_w * 0.045)
        changed_by = []
        for st in structures:
            sb = st.get('bbox_xyxy_processed') or st.get('bbox_xyxy') or []
            if len(sb) != 4:
                continue
            sx1, sy1, sx2, sy2 = _clip_box_xyxy(sb, processed_w, processed_h)
            inter_area, inter = _box_intersection_xyxy([wx1, wy1, wx2, wy2], [sx1, sy1, sx2, sy2])
            if not inter:
                continue
            ix1, iy1, ix2, iy2 = inter
            inter_h_ratio = (iy2 - iy1) / wall_h
            inter_w_ratio = (ix2 - ix1) / wall_w
            struct_area = max(1.0, _box_area_xyxy([sx1, sy1, sx2, sy2]))
            struct_inside_ratio = inter_area / struct_area
            # Only edge-attached tall structures trim the wall rectangle. This
            # catches doors and full-height reveals, without deleting normal
            # wall area around small interior decorations.
            if inter_h_ratio < 0.55 or struct_inside_ratio < 0.30 or inter_w_ratio < 0.04:
                continue
            margin = 2.0
            center_x = 0.5 * (sx1 + sx2)
            touches_left_edge = ix1 <= wx1 + wall_w * 0.08
            touches_right_edge = ix2 >= wx2 - wall_w * 0.08
            if touches_right_edge or center_x > wx1 + wall_w * 0.62:
                new_wx2 = min(wx2, sx1 - margin)
                if new_wx2 - wx1 >= min_keep_w:
                    wx2 = new_wx2
                    changed_by.append({
                        'object_id': st.get('object_id'),
                        'label': st.get('label'),
                        'trim': 'right_edge_to_structure_left',
                        'structure_box': [sx1, sy1, sx2, sy2],
                    })
            elif touches_left_edge or center_x < wx1 + wall_w * 0.38:
                new_wx1 = max(wx1, sx2 + margin)
                if wx2 - new_wx1 >= min_keep_w:
                    wx1 = new_wx1
                    changed_by.append({
                        'object_id': st.get('object_id'),
                        'label': st.get('label'),
                        'trim': 'left_edge_to_structure_right',
                        'structure_box': [sx1, sy1, sx2, sy2],
                    })
        if changed_by:
            obj2 = dict(obj)
            new_box = [float(wx1), float(wy1), float(wx2), float(wy2)]
            obj2['bbox_xyxy_processed'] = new_box
            new_box_original = [float(new_box[0] / sx), float(new_box[1] / sy), float(new_box[2] / sx), float(new_box[3] / sy)] if sx > 0 and sy > 0 else list(new_box)
            obj2['bbox_xyxy_original'] = new_box_original
            # The public semantic payload declares coordinate_space=processed
            # whenever metric3d_zip is present. Keep polygon in processed
            # coordinates and retain polygon_original separately for any later
            # seed/debug reuse. A previous version only updated polygon, which
            # could leave downstream consumers with mixed coordinate spaces after
            # wall-edge trimming.
            obj2['polygon'] = bbox_to_polygon(new_box, processed_w, processed_h)
            obj2['polygon_original'] = bbox_to_polygon(new_box_original, int(max(1, processed_w / sx)) if sx > 0 else processed_w, int(max(1, processed_h / sy)) if sy > 0 else processed_h)
            obj2['source'] = str(obj2.get('source', 'vlm_bbox_polygon')) + '+edge_structural_trim'
            obj2['notes'] = (str(obj2.get('notes', '')) + ' | edge structural trim applied: forbidden door/window/opening clipped from wall bbox').strip()
            out.append(obj2)
            trims.append({
                'wall_id': obj.get('object_id'),
                'old_bbox_xyxy_processed': orig_box,
                'new_bbox_xyxy_processed': new_box,
                'changed_by': changed_by,
            })
        else:
            out.append(obj)
    return out, trims


def _build_artwork_analysis_not_requested(reason: str) -> dict[str, Any]:
    return {
        "enabled": False,
        "status": "not_requested",
        "source": "semantic_service",
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
        "reason": reason,
    }


def _parse_artwork_analysis_for_semantics(
    artwork_image: Optional[UploadFile],
    artwork_vlm_enable: Optional[bool],
    vlm_mode: str,
) -> dict[str, Any]:
    enabled = coerce_optional_bool(artwork_vlm_enable, "ARTWORK_VLM_ENABLE", True)
    if not enabled:
        return _build_artwork_analysis_not_requested("artwork_vlm_disabled")
    if artwork_image is None:
        return _build_artwork_analysis_not_requested("artwork_image_not_supplied")
    try:
        artwork_bytes = artwork_image.file.read()
        if not artwork_bytes:
            return _build_artwork_analysis_not_requested("artwork_image_empty")
        art_img = ImageOps.exif_transpose(Image.open(io.BytesIO(artwork_bytes))).convert("RGB")
        art_w, art_h = art_img.size
        analysis = parse_artwork_frame_with_vlm(artwork_bytes, vlm_mode, int(art_w), int(art_h))
        if isinstance(analysis, dict):
            analysis.setdefault("enabled", True)
            analysis.setdefault("source", "semantic_artwork_vlm")
            analysis.setdefault("artwork_image_size", [int(art_w), int(art_h)])
            return analysis
        return _build_artwork_analysis_not_requested("artwork_vlm_returned_non_object")
    except Exception as exc:
        return {
            **_build_artwork_analysis_not_requested(f"artwork_frame_vlm_failed: {exc}"),
            "enabled": True,
            "status": "error",
            "source": "semantic_artwork_vlm",
        }

def _make_wall_id_for_bbox(box, processed_w, index):
    x1, _, x2, _ = box
    cx = 0.5 * (x1 + x2)
    if cx < processed_w * 0.34:
        hint = 'left'
    elif cx > processed_w * 0.66:
        hint = 'right'
    else:
        hint = 'front'
    return f'wall_{hint}_auto_{index:02d}', hint

def _geometry_wall_candidates_full_scene(objects, processed_w, processed_h, sx, sy, metric_zip_bytes):
    debug = {'enabled': True, 'method': 'full_scene_normal_cluster', 'candidates': [], 'warnings': []}
    normal, depth, geo_err = _load_metric_geometry_from_zip(metric_zip_bytes)
    if geo_err is not None or normal is None:
        debug['reason'] = 'geometry_unavailable'
        debug['warnings'].append(geo_err)
        return [], debug

    g_hat, g_conf, g_debug = _estimate_g_hat_from_normals(normal)
    debug['g_hat_confidence'] = float(g_conf)
    debug['g_hat_debug'] = g_debug

    h, w = normal.shape[:2]
    roi = np.ones((h, w), dtype=bool)
    exclude_labels = {'opening', 'arch', 'door', 'window', 'tv', 'mirror', 'lamp', 'track_lighting'}
    exclude_roles = {'support', 'forbidden', 'foreground', 'non_support'}
    exclude = np.zeros((h, w), np.uint8)
    for obj in objects:
        label = str(obj.get('label'))
        role = str(obj.get('role'))
        # Keep existing_art/wall_clutter as wall-mounted removable content; do
        # not let it punch holes in the wall proposal.
        if label in exclude_labels or role in exclude_roles:
            bb = obj.get('bbox_xyxy_processed') or obj.get('bbox_xyxy')
            if bb and len(bb) == 4:
                exclude = cv2.bitwise_or(exclude, _rect_mask(h, w, bb))
    roi = roi & (exclude == 0)

    finite = np.isfinite(normal).all(axis=-1)
    if depth is not None:
        finite = finite & np.isfinite(depth)

    ndotg = np.abs(np.sum(normal * g_hat.reshape(1, 1, 3), axis=-1))
    vertical = finite & roi & (ndotg <= 0.58)

    nproj = normal - (np.sum(normal * g_hat.reshape(1, 1, 3), axis=-1, keepdims=True) * g_hat.reshape(1, 1, 3))
    nproj_norm = np.linalg.norm(nproj, axis=-1)
    valid = vertical & (nproj_norm > 1e-4)

    valid_count = int(np.count_nonzero(valid))
    debug['valid_vertical_pixels'] = valid_count
    if valid_count < 1500:
        debug['reason'] = 'insufficient_valid_vertical_pixels'
        return [], debug

    angle = np.arctan2(nproj[..., 0], nproj[..., 2])
    bins = np.floor(((angle + np.pi) / (2 * np.pi)) * 18).astype(np.int32)
    bins = np.clip(bins, 0, 17)

    proposals = []
    min_area = max(2500, int(processed_h * processed_w * 0.0028))
    min_w = max(55, int(processed_w * 0.045))
    min_h = max(80, int(processed_h * 0.16))
    k3 = np.ones((3, 3), np.uint8)
    k11 = np.ones((11, 11), np.uint8)

    for b in range(18):
        bm = ((bins == b) & valid).astype(np.uint8) * 255
        if int(np.count_nonzero(bm)) < min_area:
            continue
        bm = cv2.morphologyEx(bm, cv2.MORPH_OPEN, k3)
        bm = cv2.morphologyEx(bm, cv2.MORPH_CLOSE, k11)
        num_labels, labels_cc, stats, cents = cv2.connectedComponentsWithStats((bm > 0).astype(np.uint8), connectivity=8)
        for cid in range(1, num_labels):
            area = int(stats[cid, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            xx = int(stats[cid, cv2.CC_STAT_LEFT])
            yy = int(stats[cid, cv2.CC_STAT_TOP])
            ww = int(stats[cid, cv2.CC_STAT_WIDTH])
            hh = int(stats[cid, cv2.CC_STAT_HEIGHT])
            if ww < min_w or hh < min_h:
                continue
            if (yy + hh) < int(processed_h * 0.35):
                continue
            fill_ratio = float(area) / max(1.0, float(ww * hh))
            if fill_ratio < 0.035:
                continue
            comp_mask = (labels_cc == cid)
            mean_vec = nproj[comp_mask].mean(axis=0)
            mean_angle_deg = float(np.degrees(np.arctan2(float(mean_vec[0]), float(mean_vec[2]) + 1e-9)))
            bbox_proc = [float(xx), float(yy), float(xx + ww), float(yy + hh)]
            oid, hint = _make_wall_id_for_bbox(bbox_proc, processed_w, len(proposals))
            proposal = {
                'object_id': oid,
                'label': 'wall',
                'role': 'wall',
                'bbox_xyxy_processed': bbox_proc,
                'bbox_xyxy_original': [float(bbox_proc[0] / sx), float(bbox_proc[1] / sy), float(bbox_proc[2] / sx), float(bbox_proc[3] / sy)],
                'polygon': bbox_to_polygon(bbox_proc, processed_w, processed_h),
                'confidence': 0.72,
                'source': 'semantic_full_scene_geometry_wall_proposal',
                'notes': 'geometry-added wall proposal from full-scene normal clustering; used to recover VLM-missed side walls behind foreground/objects',
                'position_hint': hint,
                'mean_angle_deg': mean_angle_deg,
                'area_px': area,
                'fill_ratio': fill_ratio,
                'cx': float(xx + ww / 2.0),
            }
            proposals.append(proposal)
            debug['candidates'].append({
                'object_id': oid,
                'bbox_xyxy_processed': bbox_proc,
                'area_px': area,
                'normal_bin': int(b),
                'fill_ratio': fill_ratio,
                'mean_angle_deg': mean_angle_deg,
                'position_hint': hint,
            })

    proposals = sorted(proposals, key=lambda x: (-x['area_px'], x['cx']))
    dedup = []
    for cand in proposals:
        keep = True
        for kept in dedup:
            inter_area, _ = _box_intersection_xyxy(cand['bbox_xyxy_processed'], kept['bbox_xyxy_processed'])
            if inter_area / max(1.0, min(_box_area_xyxy(cand['bbox_xyxy_processed']), _box_area_xyxy(kept['bbox_xyxy_processed']))) > 0.70:
                keep = False
                break
        if keep:
            dedup.append(cand)
    dedup = sorted(dedup, key=lambda x: x['cx'])[:6]
    debug['kept_count'] = len(dedup)
    if not dedup:
        debug['reason'] = 'no_full_scene_geometry_wall_candidates'
    return dedup, debug

def _append_missing_geometry_walls(objects, geometry_walls, processed_w):
    existing_walls = [o for o in objects if _is_wall_obj(o)]
    added = []
    out = list(objects)
    for cand in geometry_walls:
        cbox = cand.get('bbox_xyxy_processed') or []
        c_area = max(1.0, _box_area_xyxy(cbox))
        duplicate = False
        mostly_covered = False
        for wall in existing_walls:
            wbox = wall.get('bbox_xyxy_processed') or wall.get('bbox_xyxy') or []
            inter_area, _ = _box_intersection_xyxy(cbox, wbox)
            if inter_area <= 0:
                continue
            if inter_area / c_area >= 0.62:
                mostly_covered = True
            if _box_iou(cbox, wbox) > 0.72:
                duplicate = True
        if duplicate or mostly_covered:
            continue
        # Prefer candidates that recover a missing horizontal span. A very small
        # overlap with an existing wall is fine; a candidate extending into a
        # large uncovered side region should be added.
        old_cov, _ = _wall_x_coverage_ratio(existing_walls + added, processed_w)
        new_cov, _ = _wall_x_coverage_ratio(existing_walls + added + [cand], processed_w)
        if new_cov - old_cov < 0.05 and existing_walls:
            continue
        cand2 = dict(cand)
        cand2['object_id'] = cand.get('object_id') or f'wall_geometry_auto_{len(added):02d}'
        out.insert(len(existing_walls) + len(added), cand2)
        added.append(cand2)
    return out, added

def _should_augment_wall_set(walls, objects, processed_w):
    cov, intervals = _wall_x_coverage_ratio(walls, processed_w)
    max_right = max((float((w.get('bbox_xyxy_processed') or w.get('bbox_xyxy') or [0, 0, 0, 0])[2]) for w in walls), default=0.0)
    right_third_content = False
    for obj in objects:
        if obj.get('role') in {'existing_content', 'foreground'}:
            box = obj.get('bbox_xyxy_processed') or obj.get('bbox_xyxy') or []
            if len(box) == 4 and float(box[2]) > processed_w * 0.62:
                right_third_content = True
                break
    need = bool(cov < 0.70 or max_right < processed_w * 0.82 or (right_third_content and max_right < processed_w * 0.90))
    return need, {'coverage_ratio': cov, 'intervals': intervals, 'max_right_x': max_right, 'right_third_content': right_third_content}

def _semantic_multiwall_postprocess(objects, processed_w, processed_h, sx, sy, metric_zip_bytes):
    debug = {
        'enabled': True,
        'method': 'vlm_first_plus_geometry_cropped_wall_proposals',
        'before_wall_count': 0,
        'after_wall_count': 0,
        'used': False,
        'created': [],
        'warnings': [],
    }

    wall_objs = [o for o in objects if _is_wall_obj(o)]
    debug['before_wall_count'] = len(wall_objs)

    objects, trim_debug = _sanitize_wall_bboxes_against_edge_structures(objects, processed_w, processed_h, sx, sy)
    if trim_debug:
        debug['edge_structural_trim'] = trim_debug
    wall_objs = [o for o in objects if _is_wall_obj(o)]

    needs_augment, coverage_debug = _should_augment_wall_set(wall_objs, objects, processed_w)
    debug['semantic_wall_coverage_debug'] = coverage_debug

    if needs_augment or not wall_objs:
        geometry_walls, geometry_debug = _geometry_wall_candidates_full_scene(objects, processed_w, processed_h, sx, sy, metric_zip_bytes)
        debug['full_scene_geometry_augment_debug'] = geometry_debug
        if geometry_walls:
            objects_aug, added = _append_missing_geometry_walls(objects, geometry_walls, processed_w)
            if added:
                objects = objects_aug
                wall_objs = [o for o in objects if _is_wall_obj(o)]
                debug['used'] = True
                debug['reason'] = 'full_scene_geometry_added_missing_wall_planes'
                debug['created'] = [
                    {
                        'object_id': o.get('object_id'),
                        'bbox_xyxy_processed': o.get('bbox_xyxy_processed'),
                        'position_hint': o.get('position_hint'),
                        'source': o.get('source'),
                    }
                    for o in added
                ]
                debug['after_wall_count'] = len(wall_objs)
                return objects, debug
        if len(wall_objs) >= 2:
            debug['after_wall_count'] = len(wall_objs)
            debug['used'] = bool(trim_debug)
            debug['reason'] = 'semantic_multiwall_preserved_after_trim_geometry_no_add'
            return objects, debug
        if not wall_objs:
            debug['reason'] = 'no_wall_to_split_or_augment'
            return objects, debug

    if len(wall_objs) >= 2:
        debug['after_wall_count'] = len(wall_objs)
        debug['used'] = bool(trim_debug)
        debug['reason'] = 'vlm_multiwall_preserved_after_edge_trim' if trim_debug else 'vlm_already_returned_multiple_walls'
        return objects, debug

    coarse_wall = wall_objs[0]
    bbox = coarse_wall.get('bbox_xyxy_processed') or coarse_wall.get('bbox_xyxy') or []
    if len(bbox) != 4:
        debug['reason'] = 'wall_bbox_missing'
        return objects, debug

    x1, y1, x2, y2 = _clip_box_xyxy(bbox, processed_w, processed_h)
    bw = x2 - x1
    bh = y2 - y1
    if bw < processed_w * 0.45 or bh < processed_h * 0.22:
        debug['reason'] = 'wall_bbox_not_broad_enough'
        debug['after_wall_count'] = 1
        return objects, debug

    normal, depth, geo_err = _load_metric_geometry_from_zip(metric_zip_bytes)
    if geo_err is not None or normal is None:
        debug['reason'] = 'geometry_unavailable'
        debug['warnings'].append(geo_err)
        debug['after_wall_count'] = 1
        return objects, debug

    g_hat, g_conf, g_debug = _estimate_g_hat_from_normals(normal)
    debug['g_hat_confidence'] = float(g_conf)
    debug['g_hat_debug'] = g_debug

    h, w = normal.shape[:2]
    roi = _rect_mask(h, w, [x1, y1, x2, y2]) > 0

    exclude_labels = {'opening', 'arch', 'door', 'window', 'tv', 'mirror', 'lamp', 'track_lighting'}
    exclude = np.zeros((h, w), np.uint8)
    for obj in objects:
        if str(obj.get('label')) in exclude_labels:
            bb = obj.get('bbox_xyxy_processed') or obj.get('bbox_xyxy')
            if bb and len(bb) == 4:
                exclude = cv2.bitwise_or(exclude, _rect_mask(h, w, bb))
    roi = roi & (exclude == 0)

    finite = np.isfinite(normal).all(axis=-1)
    if depth is not None:
        finite = finite & np.isfinite(depth)

    ndotg = np.abs(np.sum(normal * g_hat.reshape(1, 1, 3), axis=-1))
    vertical = finite & roi & (ndotg <= 0.55)

    nproj = normal - (np.sum(normal * g_hat.reshape(1, 1, 3), axis=-1, keepdims=True) * g_hat.reshape(1, 1, 3))
    nproj_norm = np.linalg.norm(nproj, axis=-1)
    valid = vertical & (nproj_norm > 1e-4)

    if int(np.count_nonzero(valid)) < 1000:
        debug['reason'] = 'insufficient_valid_vertical_pixels'
        debug['after_wall_count'] = 1
        return objects, debug

    angle = np.arctan2(nproj[..., 0], nproj[..., 2])
    bins = np.floor(((angle + np.pi) / (2 * np.pi)) * 12).astype(np.int32)
    bins = np.clip(bins, 0, 11)

    candidates = []
    min_area = max(1800, int(processed_h * processed_w * 0.003))
    k3 = np.ones((3, 3), np.uint8)
    k9 = np.ones((9, 9), np.uint8)

    for b in range(12):
        bm = ((bins == b) & valid).astype(np.uint8) * 255
        if int(np.count_nonzero(bm)) < min_area:
            continue
        bm = cv2.morphologyEx(bm, cv2.MORPH_OPEN, k3)
        bm = cv2.morphologyEx(bm, cv2.MORPH_CLOSE, k9)
        num_labels, labels_cc, stats, cents = cv2.connectedComponentsWithStats((bm > 0).astype(np.uint8), connectivity=8)
        for cid in range(1, num_labels):
            area = int(stats[cid, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            xx = int(stats[cid, cv2.CC_STAT_LEFT])
            yy = int(stats[cid, cv2.CC_STAT_TOP])
            ww = int(stats[cid, cv2.CC_STAT_WIDTH])
            hh = int(stats[cid, cv2.CC_STAT_HEIGHT])
            if ww < max(35, int(processed_w * 0.05)) or hh < max(50, int(processed_h * 0.10)):
                continue
            if (yy + hh) < int(processed_h * 0.35):
                continue

            comp_mask = (labels_cc == cid)
            mean_vec = nproj[comp_mask].mean(axis=0)
            mean_angle_deg = float(np.degrees(np.arctan2(float(mean_vec[0]), float(mean_vec[2]) + 1e-9)))
            bbox_proc = [float(xx), float(yy), float(xx + ww), float(yy + hh)]
            bbox_orig = [float(bbox_proc[0] / sx), float(bbox_proc[1] / sy), float(bbox_proc[2] / sx), float(bbox_proc[3] / sy)]
            candidates.append({
                'bbox_xyxy_processed': bbox_proc,
                'bbox_xyxy_original': bbox_orig,
                'polygon': bbox_to_polygon(bbox_proc, processed_w, processed_h),
                'confidence': max(0.50, min(0.85, float(coarse_wall.get('confidence', 0.6)) * 0.95)),
                'source': 'semantic_vlm_plus_geometry_proposal',
                'notes': 'geometry-cropped wall proposal inside coarse VLM wall bbox',
                'area_px': area,
                'cx': float(xx + ww / 2.0),
                'mean_angle_deg': mean_angle_deg,
            })

    candidates = sorted(candidates, key=lambda x: (-x['area_px'], x['cx']))
    dedup = []
    for cand in candidates:
        keep = True
        for kept in dedup:
            if _box_iou(cand['bbox_xyxy_processed'], kept['bbox_xyxy_processed']) > 0.75:
                keep = False
                break
        if keep:
            dedup.append(cand)

    dedup = sorted(dedup, key=lambda x: x['cx'])
    dedup = dedup[:4]

    if len(dedup) < 2:
        debug['reason'] = 'geometry_cropped_proposals_lt2'
        debug['after_wall_count'] = 1
        return objects, debug

    dedup = _name_wall_proposals_left_to_right(dedup)

    split_walls = []
    for cand in dedup:
        split_walls.append({
            'object_id': cand['object_id'],
            'label': 'wall',
            'role': 'wall',
            'bbox_xyxy_original': cand['bbox_xyxy_original'],
            'bbox_xyxy_processed': cand['bbox_xyxy_processed'],
            'polygon': cand['polygon'],
            'confidence': float(cand['confidence']),
            'source': cand['source'],
            'notes': cand['notes'],
            'position_hint': cand.get('position_hint'),
            'mean_angle_deg': cand['mean_angle_deg'],
        })
        debug['created'].append({
            'object_id': cand['object_id'],
            'bbox_xyxy_processed': cand['bbox_xyxy_processed'],
            'mean_angle_deg': cand['mean_angle_deg'],
            'position_hint': cand.get('position_hint'),
        })

    new_objects = [o for o in objects if o is not coarse_wall]
    new_objects = split_walls + new_objects
    debug['used'] = True
    debug['after_wall_count'] = len(split_walls)
    return new_objects, debug


def _objects_have_original_geometry(objects):
    for obj in objects or []:
        if obj.get("bbox_xyxy_original") or obj.get("polygon_original"):
            return True
    return False


def _clip_polygon_to_size(poly, width, height):
    out = []
    for pt in poly or []:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        try:
            x = max(0.0, min(float(width - 1), float(pt[0])))
            y = max(0.0, min(float(height - 1), float(pt[1])))
        except (TypeError, ValueError):
            continue
        out.append([x, y])
    return out


def _remap_objects_to_processed_space(objects, processed_w, processed_h, sx, sy):
    """Map semantic objects into the Metric3D processed coordinate space.

    Contract: when the response declares ``coordinate_space=processed``, every
    public ``polygon`` must also be in processed pixels. Keep original-image
    geometry in ``polygon_original``/``bbox_xyxy_original`` only. Hanging builds
    masks directly from ``polygon``; mixed coordinate spaces make side walls
    disappear or over-expand.
    """
    remapped = []
    for obj in objects or []:
        obj2 = dict(obj)

        bbox_original = obj2.get("bbox_xyxy_original") or obj2.get("bbox_xyxy") or []
        bbox_processed_existing = obj2.get("bbox_xyxy_processed") or []
        if len(bbox_original) == 4:
            try:
                bbox_original = [float(v) for v in bbox_original]
                bbox_processed = _clip_box_xyxy(scale_bbox(bbox_original, sx, sy), processed_w, processed_h)
                obj2["bbox_xyxy_original"] = bbox_original
                obj2["bbox_xyxy_processed"] = bbox_processed
            except Exception:
                bbox_processed = bbox_processed_existing if len(bbox_processed_existing) == 4 else []
        elif len(bbox_processed_existing) == 4:
            bbox_processed = _clip_box_xyxy([float(v) for v in bbox_processed_existing], processed_w, processed_h)
            obj2["bbox_xyxy_processed"] = bbox_processed
            if sx > 0 and sy > 0:
                obj2["bbox_xyxy_original"] = [
                    float(bbox_processed[0] / sx),
                    float(bbox_processed[1] / sy),
                    float(bbox_processed[2] / sx),
                    float(bbox_processed[3] / sy),
                ]
        else:
            bbox_processed = []

        polygon_original = obj2.get("polygon_original") or obj2.get("polygon") or []
        polygon_processed = []
        if polygon_original:
            try:
                polygon_original = [[float(x), float(y)] for x, y in polygon_original]
                polygon_processed = _clip_polygon_to_size(scale_polygon(polygon_original, sx, sy), processed_w, processed_h)
                obj2["polygon_original"] = polygon_original
                obj2["polygon"] = polygon_processed
            except Exception:
                polygon_processed = []

        if not polygon_processed and len(bbox_processed) == 4:
            obj2["polygon"] = bbox_to_polygon(bbox_processed, processed_w, processed_h)
            if sx > 0 and sy > 0:
                original_w = int(max(1, round(float(processed_w) / float(sx))))
                original_h = int(max(1, round(float(processed_h) / float(sy))))
                obj2["polygon_original"] = bbox_to_polygon(
                    [
                        float(bbox_processed[0] / sx),
                        float(bbox_processed[1] / sy),
                        float(bbox_processed[2] / sx),
                        float(bbox_processed[3] / sy),
                    ],
                    original_w,
                    original_h,
                )
            else:
                obj2["polygon_original"] = obj2["polygon"]

        if obj2.get("source"):
            source = str(obj2.get("source"))
            if "+metric3d_aligned" not in source:
                obj2["source"] = source + "+metric3d_aligned"
        remapped.append(obj2)
    return remapped


def _build_vlm_objects_in_original_space(raw, original_w, original_h, bbox_coordinate_space, vlm_image_meta: Optional[Dict[str, Any]] = None):
    """Convert VLM boxes to canonical object records in original image space.

    This keeps VLM/SAM independent from Metric3D. A later alignment step maps
    bbox/polygon into the Metric3D processed coordinate system.
    """
    out = []
    objs = raw.get("objects", [])
    for i, obj in enumerate(objs):
        raw_label = str(obj.get("label", "foreground_object"))
        label, role = _canonical_label_and_role(raw_label, obj.get("role"))
        bbox = obj.get("bbox_xyxy") or obj.get("bbox") or []
        if len(bbox) != 4:
            continue
        bbox_raw = [float(x) for x in bbox]
        bbox_orig = _normalize_vlm_bbox_to_original_pixels(bbox_raw, original_w, original_h, bbox_coordinate_space, vlm_image_meta)
        if len(bbox_orig) != 4:
            continue
        poly_orig = bbox_to_polygon(bbox_orig, original_w, original_h)
        source = "vlm_bbox_polygon" if "mock" not in str(obj.get("notes", "")) else "mock_bbox_polygon"
        notes = str(obj.get("notes", ""))
        if bbox_coordinate_space != 'original_pixels':
            source += "+bbox_normalized"
            notes = (notes + f" | bbox normalized from {bbox_coordinate_space} to original pixels").strip()
        record = {
            "object_id": obj.get("object_id") or f"{label}_{i:02d}",
            "label": label,
            "role": role,
            "raw_label": raw_label,
            "raw_role": obj.get("role"),
            "bbox_xyxy_vlm_raw": bbox_raw,
            "bbox_coordinate_space": bbox_coordinate_space,
            "bbox_xyxy_original": bbox_orig,
            "bbox_xyxy_processed": bbox_orig,
            "polygon": poly_orig,
            "polygon_original": poly_orig,
            "confidence": float(obj.get("confidence", 0.5)),
            "source": source,
            "notes": notes,
        }
        _copy_wall_light_extras(obj, record)
        out.append(record)
    return out

def require_auth(a):
    if ALLOW_UNAUTH_LOCAL:
        return
    if not API_KEY:
        raise HTTPException(500, "API_KEY is not configured; set ALLOW_UNAUTH_LOCAL=1 for local development")
    if a != f"Bearer {API_KEY}":
        raise HTTPException(401, "Unauthorized")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "api": "semantic-mvp-bbox-api",
        "mode": os.getenv("VLM_PROVIDER", "mock_or_auto"),
        "sam": get_sam_runtime_status(),
    }

@app.post("/parse_semantics_mvp")
async def parse_semantics_mvp(
    room_image: UploadFile = File(...),
    metric3d_zip: Optional[UploadFile] = File(None),
    artwork_image: Optional[UploadFile] = File(None),
    scene_profile: str = Form("gallery"),
    vlm_mode: str = Form("auto"),
    artwork_vlm_enable: Optional[bool] = Form(None),
    sam_enable: Optional[bool] = Form(None),
    sam_refine_labels: str = Form(""),
    sam_max_side: Optional[int] = Form(None),
    semantic_seed_json: Optional[str] = Form(None),
    return_debug_assets: bool = Form(False),
    authorization: Optional[str] = Header(None),
):
    require_auth(authorization)
    image_bytes = await room_image.read()
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    image_np = np.array(img)
    original_w, original_h = img.size

    metric_zip_bytes = None
    if metric3d_zip is not None:
        metric_zip_bytes = await metric3d_zip.read()
        meta_info = read_sizes_from_metric3d_zip(metric_zip_bytes, (original_w, original_h))
    else:
        meta_info = {
            "processed_size": [original_w, original_h],
            "original_size": [original_w, original_h],
            "meta": {},
            "warning": "metric3d_zip_not_provided"
        }

    processed_w, processed_h = meta_info["processed_size"]
    meta_original_w, meta_original_h = meta_info["original_size"]
    sx = float(processed_w) / float(meta_original_w or original_w)
    sy = float(processed_h) / float(meta_original_h or original_h)

    seed_payload = None
    raw = {}
    bbox_coordinate_debug = {"enabled": True, "source_coordinate_space": "seed_payload"}
    sam_debug: Dict[str, Any] = {"enabled": False, "reason": "semantic_seed_json_supplied_skip_sam"}
    pipeline_stage = "final_metric_aligned" if metric_zip_bytes else "raw_original"
    seed_debug: Dict[str, Any] = {
        "used": False,
        "input_coordinate_space": None,
        "skip_vlm": False,
        "skip_sam": False,
    }

    artwork_analysis = _build_artwork_analysis_not_requested("artwork_image_not_supplied_or_artwork_vlm_disabled")

    if semantic_seed_json and str(semantic_seed_json).strip():
        try:
            seed_payload = json.loads(semantic_seed_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(400, f"semantic_seed_json is not valid JSON: {exc}")
        seed_objects = seed_payload.get("objects", [])
        if not isinstance(seed_objects, list):
            raise HTTPException(400, "semantic_seed_json.objects must be a list")
        out = _remap_objects_to_processed_space(seed_objects, processed_w, processed_h, sx, sy)
        bbox_coordinate_debug = seed_payload.get("vlm_bbox_coordinate_debug") or bbox_coordinate_debug
        sam_debug = seed_payload.get("sam_debug") or sam_debug
        seed_artwork = seed_payload.get("artwork_analysis")
        if isinstance(seed_artwork, dict):
            artwork_analysis = dict(seed_artwork)
        else:
            artwork_analysis = _build_artwork_analysis_not_requested("semantic_seed_json_without_artwork_analysis")

        seed_debug = {
            "used": True,
            "input_coordinate_space": seed_payload.get("coordinate_space"),
            "input_stage": seed_payload.get("semantic_pipeline_stage"),
            "input_processed_size": seed_payload.get("processed_size"),
            "input_original_size": seed_payload.get("original_size"),
            "object_count": len(seed_objects),
            "skip_vlm": True,
            "skip_sam": True,
            "alignment": {
                "processed_size": [int(processed_w), int(processed_h)],
                "original_size": [int(meta_original_w), int(meta_original_h)],
                "scale_x": float(sx),
                "scale_y": float(sy),
            },
        }
    else:
        artwork_analysis = _parse_artwork_analysis_for_semantics(artwork_image, artwork_vlm_enable, vlm_mode)

        # VLM sees the original image. Some providers still return normalized
        # visual coordinates, so normalize before any SAM or Metric3D alignment.
        raw = parse_with_vlm(image_bytes, scene_profile, vlm_mode, original_w, original_h)
        objs = raw.get("objects", [])
        vlm_image_meta = raw.get("_vlm_image_meta") if isinstance(raw, dict) else None
        bbox_coordinate_space, bbox_coordinate_debug = _infer_vlm_bbox_coordinate_space(objs, original_w, original_h, vlm_image_meta)
        original_space_objects = _build_vlm_objects_in_original_space(raw, original_w, original_h, bbox_coordinate_space, vlm_image_meta)

        # Optional SAM now runs only in original-image space. This lets callers
        # start /parse_semantics_mvp without metric3d_zip while Metric3D is still
        # computing. A later call with semantic_seed_json remaps these polygons
        # to Metric3D processed space and skips VLM/SAM entirely.
        sam_enabled = coerce_optional_bool(sam_enable, "SAM_ENABLE", False)
        original_space_objects, sam_debug = refine_objects_with_sam(
            image_np,
            original_space_objects,
            processed_size=(original_w, original_h),
            original_size=(original_w, original_h),
            enabled=sam_enabled,
            refine_labels_csv=sam_refine_labels,
            max_side=sam_max_side,
        )
        out = _remap_objects_to_processed_space(original_space_objects, processed_w, processed_h, sx, sy)
        seed_debug = {
            "used": False,
            "input_coordinate_space": None,
            "object_count": len(original_space_objects),
            "skip_vlm": False,
            "skip_sam": False,
            "alignment": {
                "processed_size": [int(processed_w), int(processed_h)],
                "original_size": [int(meta_original_w), int(meta_original_h)],
                "scale_x": float(sx),
                "scale_y": float(sy),
            },
        }

    if metric_zip_bytes:
        out, multiwall_split_debug = _semantic_multiwall_postprocess(out, processed_w, processed_h, sx, sy, metric_zip_bytes)
    else:
        multiwall_split_debug = {
            "enabled": False,
            "used": False,
            "reason": "metric3d_zip_not_provided_geometry_deferred",
            "before_wall_count": len([o for o in out if _is_wall_obj(o)]),
            "after_wall_count": len([o for o in out if _is_wall_obj(o)]),
            "warnings": ["geometry_postprocess_deferred_until_metric3d_zip_available"],
        }
    object_inventory = _build_object_inventory(out)

    warnings = []
    if raw.get("warning"):
        warnings.append(raw.get("warning"))
    if seed_payload:
        warnings.extend(seed_payload.get("warnings", []) or [])
    if meta_info.get("warning"):
        warnings.append(meta_info.get("warning"))
    if bbox_coordinate_debug.get("source_coordinate_space") not in {None, "original_pixels", "seed_payload"}:
        warnings.append(f"vlm_bbox_coordinate_space_normalized:{bbox_coordinate_debug.get('source_coordinate_space')}")
    warnings.extend(object_inventory.get("warnings", []))
    # Preserve order while deduplicating.
    warnings = list(dict.fromkeys([str(w) for w in warnings if w]))

    light_payload = _normalize_light_payload((seed_payload or raw or {}).get("light"))

    semantic_payload = {
        "status": "ok",
        "coordinate_space": "processed" if metric_zip_bytes else "original",
        "semantic_pipeline_stage": pipeline_stage,
        "geometry_deferred": not bool(metric_zip_bytes),
        "scene_profile": scene_profile,
        "processed_size": [processed_w, processed_h],
        "original_size": [meta_original_w, meta_original_h],
        "objects": out,
        "object_inventory": object_inventory,
        "semantic_object_inventory_status": object_inventory.get("status"),
        "vlm_bbox_coordinate_debug": bbox_coordinate_debug,
        "multiwall_split_debug": multiwall_split_debug,
        "sam_debug": sam_debug,
        "semantic_seed_debug": seed_debug,
        "artwork_analysis": artwork_analysis,
        "warnings": warnings
    }
    if light_payload:
        semantic_payload["light"] = light_payload

    if return_debug_assets:
        debug_img = draw_semantic_wall_debug(image_np, out, multiwall_split_debug, target_space="original")
        files = {
            "semantics.json": json.dumps(semantic_payload, ensure_ascii=False, indent=2).encode("utf-8"),
            "semantic_wall_debug.png": png_bytes(debug_img),
        }
        return _zip_response(files, filename="semantics_mvp_debug.zip")

    return JSONResponse(semantic_payload)


def _zip_response(files: Dict[str, bytes], filename: str = "semantics_mvp_debug.zip") -> Response:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, content in files.items():
            z.writestr(name, content)
    return Response(buf.getvalue(), media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{filename}"'})
