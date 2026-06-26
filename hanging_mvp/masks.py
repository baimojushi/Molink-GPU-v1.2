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
    "signage": "wall_label",
    "people": "person",
    "human": "person",
    "man": "person",
    "woman": "person",
    "visitor": "person",
}

_KNOWN_CANONICAL_LABELS = {
    "wall", "floor", "roof", "sofa", "table", "cabinet", "window", "door", "tv", "mirror",
    "lamp", "track_lighting", "arch", "opening", "existing_art", "painting", "poster",
    "framed_art", "artwork", "green_board", "notice_board", "wall_clutter", "wall_label",
    "hook", "nail", "small_shelf", "pedestal", "plinth", "display_stand",
    "foreground_object", "person", "plant", "chair",
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


def mask_from_polygon(size_hw: Tuple[int, int], polygon: List[List[float]]) -> np.ndarray:
    h, w = size_hw
    mask = np.zeros((h, w), np.uint8)
    if not polygon:
        return mask
    pts = np.array([[int(round(x)), int(round(y))] for x, y in polygon], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask

def load_semantic_masks(semantics: Dict[str, Any], size_hw: Tuple[int, int]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for obj in semantics.get("objects", []):
        oid = obj.get("object_id") or f"obj_{len(out):03d}"
        poly = obj.get("polygon") or obj.get("points") or []
        mask = mask_from_polygon(size_hw, poly)
        raw_label = obj.get("label", "unknown")
        record = {
            "object_id": oid,
            "label": canonical_semantic_label(raw_label),
            "raw_label": raw_label,
            "role": str(obj.get("role", "unknown")).strip().lower(),
            "confidence": float(obj.get("confidence", 1.0)),
            "mask": mask,
            "polygon": poly,
        }
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
