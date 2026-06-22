from __future__ import annotations

import io
from typing import Any, Dict, Iterable, Tuple

import cv2
import numpy as np
from PIL import Image


def png_bytes(arr: np.ndarray) -> bytes:
    if arr.ndim == 2:
        img = Image.fromarray(arr.astype(np.uint8))
    else:
        img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def mask_preview(mask: np.ndarray) -> np.ndarray:
    if mask is None:
        return np.zeros((1, 1), dtype=np.uint8)
    return ((mask > 0).astype(np.uint8) * 255)


def overlay_mask(
    room_rgb: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 220, 0),
    alpha: float = 0.32,
) -> np.ndarray:
    img = room_rgb.copy().astype(np.float32)
    if mask is None:
        return img.astype(np.uint8)
    m = mask > 0
    if m.shape[:2] != img.shape[:2]:
        m = cv2.resize(m.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    img[m] = img[m] * (1.0 - alpha) + color_arr * alpha
    return np.clip(img, 0, 255).astype(np.uint8)


def _draw_label(img: np.ndarray, x: int, y: int, text: str, color: Tuple[int, int, int] = (255, 255, 255)) -> None:
    h, w = img.shape[:2]
    x = max(0, min(w - 1, int(x)))
    y = max(14, min(h - 2, int(y)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (x, y - th - 8), (min(w - 1, x + tw + 8), min(h - 1, y + 4)), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 4, y - 4), font, scale, color, thickness, cv2.LINE_AA)


def draw_candidate_overlay(room_rgb: np.ndarray, candidate: Dict[str, Any], title: str | None = None) -> np.ndarray:
    img = room_rgb.copy()
    quad = candidate.get("projection", {}).get("projected_quad_px_original")
    if quad:
        pts = np.array(quad, np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 0), thickness=6)
        c = pts.mean(axis=0).astype(int)
        cv2.line(img, (int(c[0]), max(0, int(c[1]) - 180)), (int(c[0]), min(img.shape[0]-1, int(c[1]) + 180)), (0, 255, 0), 4)
        label = title or f"{candidate.get('candidate_id')} score={float(candidate.get('score', 0.0)):.2f}"
        _draw_label(img, int(c[0]) + 8, int(c[1]), label, (255, 255, 255))
    return img


def draw_rejected_wall_overlay(
    room_rgb: np.ndarray,
    wall_mask: np.ndarray | None,
    wall_id: str,
    reasons: Iterable[str],
) -> np.ndarray:
    img = overlay_mask(room_rgb, wall_mask, color=(255, 0, 0), alpha=0.28)
    if wall_mask is not None:
        m = wall_mask
        if m.shape[:2] != img.shape[:2]:
            m = cv2.resize((m > 0).astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        ys, xs = np.where(m > 0)
    else:
        ys, xs = [], []
    if len(xs):
        x, y = int(xs.min()), int(max(14, ys.min()))
    else:
        x, y = 12, 28
    reason_text = ",".join(list(reasons)[:3]) if reasons else "rejected"
    _draw_label(img, x, y, f"{wall_id}: {reason_text}", (255, 80, 80))
    return img


def build_wall_debug_overlay(
    room_rgb: np.ndarray,
    wall_mask: np.ndarray | None,
    solid_wall_mask: np.ndarray | None,
    candidate: Dict[str, Any] | None = None,
    wall_id: str = "",
) -> np.ndarray:
    img = room_rgb.copy()
    if wall_mask is not None:
        img = overlay_mask(img, wall_mask, color=(0, 120, 255), alpha=0.18)
    if solid_wall_mask is not None:
        img = overlay_mask(img, solid_wall_mask, color=(0, 255, 0), alpha=0.30)
    if candidate is not None:
        img = draw_candidate_overlay(img, candidate, title=f"{wall_id} {candidate.get('candidate_id')}")
    elif wall_id:
        _draw_label(img, 12, 28, wall_id, (255, 255, 255))
    return img
