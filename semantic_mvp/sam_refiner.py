from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

_TRUE_VALUES = {"1", "true", "yes", "y", "on", "enable", "enabled"}
_FALSE_VALUES = {"0", "false", "no", "n", "off", "disable", "disabled"}

@dataclass
class SamRuntimeStatus:
    enabled_by_env: bool
    backend: str
    model_type: str
    checkpoint: str
    device: str
    available: bool
    loaded: bool
    error: str = ""

_PREDICTOR = None
_PREDICTOR_KEY: Tuple[str, str, str] | None = None
_PREDICTOR_ERROR = ""

def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in _TRUE_VALUES:
        return True
    if s in _FALSE_VALUES:
        return False
    return default

def coerce_optional_bool(value: Optional[bool], env_name: str, default: bool = False) -> bool:
    if value is not None:
        return bool(value)
    return env_bool(env_name, default)

def _default_device() -> str:
    forced = os.getenv("SAM_DEVICE", "").strip()
    if forced:
        return forced
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def _checkpoint_path() -> str:
    return os.getenv("SAM_CHECKPOINT", "/opt/molink-cache/sam/sam_vit_b_01ec64.pth").strip()

def get_sam_runtime_status() -> Dict[str, Any]:
    checkpoint = _checkpoint_path()
    backend = os.getenv("SAM_BACKEND", "sam1").strip().lower() or "sam1"
    model_type = os.getenv("SAM_MODEL_TYPE", "vit_b").strip() or "vit_b"
    available = bool(checkpoint and os.path.exists(checkpoint))
    return SamRuntimeStatus(
        enabled_by_env=env_bool("SAM_ENABLE", False),
        backend=backend,
        model_type=model_type,
        checkpoint=checkpoint,
        device=_default_device(),
        available=available,
        loaded=_PREDICTOR is not None,
        error=_PREDICTOR_ERROR,
    ).__dict__

def _load_sam1_predictor():
    global _PREDICTOR, _PREDICTOR_KEY, _PREDICTOR_ERROR
    backend = os.getenv("SAM_BACKEND", "sam1").strip().lower() or "sam1"
    model_type = os.getenv("SAM_MODEL_TYPE", "vit_b").strip() or "vit_b"
    checkpoint = _checkpoint_path()
    device = _default_device()
    key = (backend, model_type, checkpoint)
    if _PREDICTOR is not None and _PREDICTOR_KEY == key:
        return _PREDICTOR
    _PREDICTOR_ERROR = ""
    if backend not in {"sam1", "segment_anything", "sam"}:
        raise RuntimeError(f"Unsupported SAM_BACKEND={backend!r}. V1 ships sam1/segment_anything adapter.")
    if not checkpoint or not os.path.exists(checkpoint):
        raise RuntimeError(f"SAM checkpoint not found: {checkpoint}")
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except Exception as e:
        _PREDICTOR_ERROR = f"segment_anything import failed: {e}"
        raise RuntimeError(_PREDICTOR_ERROR)
    try:
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        _PREDICTOR = predictor
        _PREDICTOR_KEY = key
        return predictor
    except Exception as e:
        _PREDICTOR_ERROR = f"SAM load failed: {e}"
        raise


def warmup_sam_predictor() -> bool:
    """Pre-load SAM predictor at startup time.

    Returns True if loaded successfully, False if SAM is disabled or unavailable.
    Raises on hard failure when SAM is explicitly enabled but cannot load.
    """
    if not env_bool("SAM_ENABLE", False):
        return False
    checkpoint = _checkpoint_path()
    if not checkpoint or not os.path.exists(checkpoint):
        _PREDICTOR_ERROR = f"SAM checkpoint not found: {checkpoint}"
        raise RuntimeError(_PREDICTOR_ERROR)
    _load_sam1_predictor()
    return True


def _clip_box(box: Iterable[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in list(box)[:4]]
    x1 = max(0.0, min(float(width - 1), x1)); x2 = max(0.0, min(float(width - 1), x2))
    y1 = max(0.0, min(float(height - 1), y1)); y2 = max(0.0, min(float(height - 1), y2))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return [x1, y1, x2, y2]

def _resize_for_sam(image_rgb: np.ndarray, max_side: int) -> Tuple[np.ndarray, float, float]:
    h, w = image_rgb.shape[:2]
    if max_side <= 0 or max(h, w) <= max_side:
        return image_rgb, 1.0, 1.0
    scale = float(max_side) / float(max(h, w))
    nw = max(1, int(round(w * scale))); nh = max(1, int(round(h * scale)))
    return cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_AREA), float(nw) / float(w), float(nh) / float(h)

def mask_to_polygon(mask: np.ndarray, max_points: int = 96) -> List[List[float]]:
    if mask is None or mask.size == 0 or int(np.count_nonzero(mask)) == 0:
        return []
    m = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3 or cv2.contourArea(contour) < 4:
        return []
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, max(1.0, 0.006 * peri), True).reshape(-1, 2)
    if len(approx) > max_points:
        approx = approx[:: int(np.ceil(len(approx) / float(max_points)))]
    return [[float(x), float(y)] for x, y in approx]

def _bbox_area(box: List[float]) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))

def _mask_bbox(mask: np.ndarray) -> List[float] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]

def default_refine_label_set() -> set[str]:
    labels = os.getenv(
        "SAM_REFINE_LABELS",
        "sofa,floor,table,cabinet,bar,window,door,tv,mirror,lamp,track_lighting,arch,opening,"
        "existing_art,wall_clutter,wall_label,hook,nail,small_shelf,foreground_object,plant,chair,"
        "pedestal,plinth,display_stand",
    )
    out = {x.strip() for x in labels.split(",") if x.strip()}
    # Floor is a metric geometry anchor. Keep it in the refinement set even if
    # a deployment narrows SAM_REFINE_LABELS; an explicit empty force variable
    # remains available as an operator opt-out.
    forced = os.getenv("SAM_FORCE_REFINE_LABELS", "floor")
    out.update({x.strip() for x in forced.split(",") if x.strip()})
    if env_bool("SAM_REFINE_WALL", False):
        out.add("wall")
    return out

def refine_objects_with_sam(
    image_rgb: np.ndarray,
    objects: List[Dict[str, Any]],
    processed_size: Tuple[int, int],
    original_size: Tuple[int, int],
    enabled: Optional[bool] = None,
    refine_labels_csv: str = "",
    max_side: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    effective_enabled = bool(enabled) if enabled is not None else env_bool("SAM_ENABLE", False)
    debug: Dict[str, Any] = {
        "enabled": effective_enabled,
        "runtime": get_sam_runtime_status(),
        "refined_count": 0,
        "skipped_count": 0,
        "fallback_count": 0,
        "items": [],
    }
    if not effective_enabled:
        debug["reason"] = "sam_disabled"
        return objects, debug
    refine_labels = {x.strip() for x in refine_labels_csv.split(",") if x.strip()} if refine_labels_csv.strip() else default_refine_label_set()
    forced_refine_labels = {
        x.strip() for x in os.getenv("SAM_FORCE_REFINE_LABELS", "floor").split(",") if x.strip()
    }
    refine_labels.update(forced_refine_labels)
    debug["refine_labels"] = sorted(refine_labels)
    debug["forced_refine_labels"] = sorted(forced_refine_labels)
    try:
        predictor = _load_sam1_predictor()
    except Exception as e:
        debug["enabled"] = False
        debug["reason"] = "sam_unavailable_fallback_to_bbox"
        debug["error"] = str(e)
        debug["runtime"] = get_sam_runtime_status()
        return objects, debug
    original_w, original_h = original_size
    processed_w, processed_h = processed_size
    max_side = int(max_side or os.getenv("SAM_MAX_IMAGE_SIDE", "1536") or 1536)
    sam_img, sam_sx, sam_sy = _resize_for_sam(image_rgb, max_side)
    predictor.set_image(sam_img)
    debug["sam_image_size"] = [int(sam_img.shape[1]), int(sam_img.shape[0])]
    debug["sam_scale_from_original"] = [float(sam_sx), float(sam_sy)]
    refined: List[Dict[str, Any]] = []
    for obj in objects:
        label = str(obj.get("label", "")); oid = str(obj.get("object_id", label))
        if label not in refine_labels:
            debug["skipped_count"] += 1; refined.append(obj); continue
        bbox_orig = obj.get("bbox_xyxy_original") or obj.get("bbox_xyxy") or []
        if len(bbox_orig) != 4:
            debug["fallback_count"] += 1; refined.append(obj)
            debug["items"].append({"object_id": oid, "label": label, "status": "fallback", "reason": "bbox_missing"})
            continue
        bbox_orig = _clip_box(bbox_orig, original_w, original_h)
        bbox_sam = np.array([bbox_orig[0] * sam_sx, bbox_orig[1] * sam_sy, bbox_orig[2] * sam_sx, bbox_orig[3] * sam_sy], dtype=np.float32)
        try:
            masks, scores, _ = predictor.predict(box=bbox_sam, multimask_output=True)
            if masks is None or len(masks) == 0:
                raise RuntimeError("empty SAM masks")
            best_idx = int(np.argmax(scores)) if scores is not None and len(scores) else 0
            mask_sam = masks[best_idx].astype(np.uint8) * 255
            score = float(scores[best_idx]) if scores is not None and len(scores) else None
            mask_orig = cv2.resize(mask_sam, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
            mask_proc = cv2.resize(mask_orig, (processed_w, processed_h), interpolation=cv2.INTER_NEAREST)
            bbox_proc = obj.get("bbox_xyxy_processed") or [bbox_orig[0] * processed_w / original_w, bbox_orig[1] * processed_h / original_h, bbox_orig[2] * processed_w / original_w, bbox_orig[3] * processed_h / original_h]
            bbox_proc = _clip_box(bbox_proc, processed_w, processed_h)
            bbox_area = max(1.0, _bbox_area(bbox_proc))
            mask_area = float(np.count_nonzero(mask_proc)); area_ratio = mask_area / bbox_area
            min_ratio = float(os.getenv("SAM_MIN_MASK_TO_BOX_RATIO_WALL" if label == "wall" else "SAM_MIN_MASK_TO_BOX_RATIO", "0.20" if label == "wall" else "0.015"))
            poly_proc = mask_to_polygon(mask_proc, max_points=int(os.getenv("SAM_MAX_POLYGON_POINTS", "96")))
            poly_orig = mask_to_polygon(mask_orig, max_points=int(os.getenv("SAM_MAX_POLYGON_POINTS", "96")))
            if len(poly_proc) < 3 or area_ratio < min_ratio:
                raise RuntimeError(f"invalid SAM polygon/area ratio: points={len(poly_proc)} area_ratio={area_ratio:.4f} min={min_ratio:.4f}")
            obj2 = dict(obj)
            obj2["polygon"] = poly_proc
            obj2["polygon_original"] = poly_orig
            obj2["source"] = "vlm_bbox_sam_polygon"
            obj2["sam"] = {
                "enabled": True,
                "score": score,
                "mask_area_px_processed": int(mask_area),
                "mask_to_bbox_area_ratio": float(area_ratio),
                "mask_bbox_xyxy_processed": _mask_bbox(mask_proc),
                "backend": os.getenv("SAM_BACKEND", "sam1"),
                "model_type": os.getenv("SAM_MODEL_TYPE", "vit_b"),
            }
            refined.append(obj2); debug["refined_count"] += 1
            debug["items"].append({"object_id": oid, "label": label, "status": "refined", "score": score, "area_ratio": float(area_ratio)})
        except Exception as e:
            obj2 = dict(obj); obj2["sam"] = {"enabled": True, "fallback_to_bbox": True, "error": str(e)}
            refined.append(obj2); debug["fallback_count"] += 1
            debug["items"].append({"object_id": oid, "label": label, "status": "fallback", "reason": str(e)})
    debug["runtime"] = get_sam_runtime_status()
    return refined, debug
