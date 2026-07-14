from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np

_INSTANCE_SUFFIX_RE = re.compile(r"^(?P<base>.+?)[_\-\s]+\d{1,4}$")

_LABEL_ALIASES = {
    "ceiling": "roof",
    "ceilings": "roof",
    "roof ceiling": "roof",
    "wall art": "existing_art",
    "wall_art": "existing_art",
    "art": "existing_art",
    "artwork": "existing_art",
    "picture": "existing_art",
    "framed painting": "framed_art",
    "framed_artwork": "framed_art",
    "framed artwork": "framed_art",
    "frame": "framed_art",
    "green board": "green_board",
    "notice board": "notice_board",
    "label": "wall_label",
    "sign": "wall_label",
    "wall sign": "wall_label",
    "small sign": "small_sign",
    "signage": "wall_label",
    "people": "person",
    "human": "person",
    "man": "person",
    "woman": "person",
    "visitor": "person",
    "wall mounted shelf": "wall_mounted_shelf",
    "wall-mounted shelf": "wall_mounted_shelf",
    "built in shelf": "built_in_shelf",
    "built-in shelf": "built_in_shelf",
    "book shelf": "bookshelf",
    "counter fixture": "counter_fixture",
}


_KNOWN_CANONICAL_LABELS = {
    "wall", "floor", "roof", "sofa", "table", "cabinet", "shelf", "wall_mounted_shelf", "built_in_shelf", "storage", "counter_fixture", "counter", "appliance", "bookcase", "bookshelf", "rack", "window", "door", "tv", "mirror",
    "lamp", "track_lighting", "arch", "opening", "existing_art", "painting", "poster",
    "framed_art", "artwork", "green_board", "notice_board", "wall_clutter", "wall_label",
    "hook", "nail", "small_shelf", "pedestal", "plinth", "display_stand",
    "foreground_object", "person", "plant", "chair", "sideboard", "small_sign", "signage", "sticker",
}


def _label_lookup_candidates(label: Any) -> list[str]:
    raw = str(label or "unknown").strip().lower().replace("-", "_")
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


def canonical_semantic_label(label: Any) -> str:
    candidates = _label_lookup_candidates(label)
    if not candidates:
        return "unknown"
    for candidate in candidates:
        aliased = _LABEL_ALIASES.get(candidate)
        if aliased:
            return aliased
        key = candidate.replace(" ", "_")
        if key in _KNOWN_CANONICAL_LABELS:
            if key in {"human", "people"}:
                return "person"
            return key
    return candidates[0]


def label_matches(label: Any, labels: set[str]) -> bool:
    canonical = canonical_semantic_label(label)
    normalized_labels = {canonical_semantic_label(x) for x in labels}
    return canonical in normalized_labels


def semantic_label_matches(label: Any, labels: set[str]) -> bool:
    """Backward-compatible alias used by hanging_mvp.service."""
    return label_matches(label, labels)


def _coerce_polygon(poly: Any) -> List[List[float]]:
    out: List[List[float]] = []
    if not isinstance(poly, (list, tuple)):
        return out
    for pt in poly:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        try:
            x = float(pt[0])
            y = float(pt[1])
        except (TypeError, ValueError):
            continue
        if np.isfinite(x) and np.isfinite(y):
            out.append([x, y])
    return out


def _clip_bbox_xyxy(box: Any, width: int, height: int) -> List[float]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return []
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except (TypeError, ValueError):
        return []
    if not all(np.isfinite([x1, y1, x2, y2])):
        return []
    x1 = max(0.0, min(float(width - 1), x1))
    x2 = max(0.0, min(float(width - 1), x2))
    y1 = max(0.0, min(float(height - 1), y1))
    y2 = max(0.0, min(float(height - 1), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 <= x1 or y2 <= y1:
        return []
    return [x1, y1, x2, y2]


def _bbox_to_polygon_xyxy(box: List[float]) -> List[List[float]]:
    x1, y1, x2, y2 = box
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _polygon_bbox(poly: List[List[float]]) -> List[float]:
    if not poly:
        return []
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


def _polygon_out_of_bounds(poly: List[List[float]], width: int, height: int) -> bool:
    if not poly:
        return False
    tol = max(4.0, float(max(width, height)) * 0.02)
    return any(x < -tol or y < -tol or x > float(width - 1) + tol or y > float(height - 1) + tol for x, y in poly)


def _bbox_area(box: List[float]) -> float:
    if len(box) != 4:
        return 0.0
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def _bbox_iou(a: List[float], b: List[float]) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    ix1 = max(float(a[0]), float(b[0]))
    iy1 = max(float(a[1]), float(b[1]))
    ix2 = min(float(a[2]), float(b[2]))
    iy2 = min(float(a[3]), float(b[3]))
    inter = _bbox_area([ix1, iy1, ix2, iy2])
    union = _bbox_area(a) + _bbox_area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def _polygon_bbox_mismatch(poly: List[List[float]], bbox_processed: List[float], width: int, height: int) -> bool:
    """Detect the common failure where polygon is still original-size pixels.

    SAM polygons may be tighter than detector boxes, so this is intentionally
    conservative: only treat the polygon as suspicious if its bbox is far bigger
    than the processed bbox, mostly non-overlapping, or materially outside the
    processed image.
    """
    if not poly or len(bbox_processed) != 4:
        return False
    pbox = _polygon_bbox(poly)
    if len(pbox) != 4:
        return False
    if _polygon_out_of_bounds(poly, width, height):
        return True
    bw = max(1.0, bbox_processed[2] - bbox_processed[0])
    bh = max(1.0, bbox_processed[3] - bbox_processed[1])
    pw = max(1.0, pbox[2] - pbox[0])
    ph = max(1.0, pbox[3] - pbox[1])
    if (pw > bw * 1.45 or ph > bh * 1.45) and _bbox_iou(pbox, bbox_processed) < 0.75:
        return True
    if _bbox_iou(pbox, bbox_processed) < 0.10:
        return True
    return False


def _scale_polygon(poly: List[List[float]], sx: float, sy: float, width: int, height: int) -> List[List[float]]:
    out: List[List[float]] = []
    for x, y in poly:
        out.append([
            max(0.0, min(float(width - 1), float(x) * sx)),
            max(0.0, min(float(height - 1), float(y) * sy)),
        ])
    return out


def mask_from_polygon(size_hw: Tuple[int, int], polygon: List[List[float]]) -> np.ndarray:
    h, w = size_hw
    mask = np.zeros((h, w), np.uint8)
    poly = _coerce_polygon(polygon)
    if not poly:
        return mask
    pts = np.array([[int(round(x)), int(round(y))] for x, y in poly], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def _mask_from_semantic_geometry(obj: Dict[str, Any], semantics: Dict[str, Any], size_hw: Tuple[int, int]) -> Tuple[np.ndarray, List[List[float]], str, Dict[str, Any]]:
    h, w = size_hw
    debug: Dict[str, Any] = {"warnings": []}
    bbox_processed = _clip_bbox_xyxy(obj.get("bbox_xyxy_processed"), w, h)
    poly = _coerce_polygon(obj.get("polygon") or obj.get("points") or [])

    polygon_bad = bool(poly) and _polygon_bbox_mismatch(poly, bbox_processed, w, h)
    debug["polygon_out_of_bounds"] = bool(poly) and _polygon_out_of_bounds(poly, w, h)
    debug["polygon_bbox_mismatch"] = polygon_bad

    if poly and not polygon_bad:
        mask = mask_from_polygon(size_hw, poly)
        if np.count_nonzero(mask) > 0:
            return mask, poly, "polygon", debug
        debug["warnings"].append("polygon_mask_empty")

    if bbox_processed:
        poly_bbox = _bbox_to_polygon_xyxy(bbox_processed)
        mask = mask_from_polygon(size_hw, poly_bbox)
        if np.count_nonzero(mask) > 0:
            if polygon_bad:
                debug["warnings"].append("polygon_rejected_used_bbox_xyxy_processed")
            return mask, poly_bbox, "bbox_xyxy_processed", debug

    poly_original = _coerce_polygon(obj.get("polygon_original") or [])
    original_size = semantics.get("original_size") if isinstance(semantics, dict) else None
    processed_size = semantics.get("processed_size") if isinstance(semantics, dict) else None
    try:
        orig_w = float(original_size[0]) if isinstance(original_size, (list, tuple)) and len(original_size) >= 2 else 0.0
        orig_h = float(original_size[1]) if isinstance(original_size, (list, tuple)) and len(original_size) >= 2 else 0.0
        proc_w = float(processed_size[0]) if isinstance(processed_size, (list, tuple)) and len(processed_size) >= 2 else float(w)
        proc_h = float(processed_size[1]) if isinstance(processed_size, (list, tuple)) and len(processed_size) >= 2 else float(h)
    except Exception:
        orig_w = orig_h = 0.0
        proc_w, proc_h = float(w), float(h)
    if poly_original and orig_w > 0 and orig_h > 0:
        scaled = _scale_polygon(poly_original, proc_w / orig_w, proc_h / orig_h, w, h)
        mask = mask_from_polygon(size_hw, scaled)
        if np.count_nonzero(mask) > 0:
            debug["warnings"].append("used_scaled_polygon_original")
            return mask, scaled, "polygon_original_scaled", debug

    empty = np.zeros((h, w), np.uint8)
    debug["warnings"].append("semantic_geometry_empty")
    return empty, poly, "empty", debug


def load_semantic_masks(semantics: Dict[str, Any], size_hw: Tuple[int, int]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for obj in semantics.get("objects", []):
        oid = obj.get("object_id") or f"obj_{len(out):03d}"
        mask, poly, mask_source, mask_debug = _mask_from_semantic_geometry(obj, semantics, size_hw)
        raw_label = obj.get("label", "unknown")
        record = {
            "object_id": oid,
            "label": canonical_semantic_label(raw_label),
            "raw_label": raw_label,
            "role": str(obj.get("role", "unknown")).strip().lower(),
            "confidence": float(obj.get("confidence", 1.0)),
            "mask": mask,
            "polygon": poly,
            "bbox_xyxy_processed": obj.get("bbox_xyxy_processed"),
            "bbox_xyxy_original": obj.get("bbox_xyxy_original"),
            "mask_source": mask_source,
            "mask_debug": mask_debug,
            "semantic_source": obj.get("source"),
            "sam": obj.get("sam") if isinstance(obj.get("sam"), dict) else None,
        }
        if mask_debug.get("warnings"):
            record["warnings"] = list(mask_debug.get("warnings") or [])
        for extra_key in ("surface_active_light", "surface_active_light_scope",
                          "landscape_competition_score", "landscape_competition_scope"):
            if obj.get(extra_key) is not None:
                record[extra_key] = obj.get(extra_key)
        out[oid] = record
    return out

def union_masks(items: Iterable[Dict[str, Any]]) -> np.ndarray:
    items = list(items)
    if not items:
        return np.zeros((1, 1), np.uint8)
    h, w = items[0]["mask"].shape
    out = np.zeros((h, w), np.uint8)
    for it in items:
        out = cv2.bitwise_or(out, it["mask"])
    return out

def masks_by_label(objects: Dict[str, Dict[str, Any]], labels: set[str]) -> List[Dict[str, Any]]:
    return [o for o in objects.values() if label_matches(o.get("label"), labels) or label_matches(o.get("raw_label"), labels)]

def masks_by_role(objects: Dict[str, Dict[str, Any]], role: str) -> List[Dict[str, Any]]:
    role_norm = str(role or "").strip().lower()
    return [o for o in objects.values() if str(o.get("role", "")).strip().lower() == role_norm]
