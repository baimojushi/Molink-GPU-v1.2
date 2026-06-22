from __future__ import annotations

import io
from typing import Any, Dict, Iterable, List, Tuple

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


def _box_for_space(obj: Dict[str, Any], target_space: str = "original") -> List[float] | None:
    if target_space == "original":
        box = obj.get("bbox_xyxy_original") or obj.get("bbox_xyxy")
    else:
        box = obj.get("bbox_xyxy_processed") or obj.get("bbox_xyxy")
    if not box or len(box) != 4:
        return None
    return [float(v) for v in box]


def _draw_label(img: np.ndarray, x: int, y: int, text: str, color: Tuple[int, int, int]) -> None:
    h, w = img.shape[:2]
    x = max(0, min(w - 1, int(x)))
    y = max(12, min(h - 2, int(y)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (x, y - th - 8), (min(w - 1, x + tw + 8), min(h - 1, y + 4)), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 4, y - 4), font, scale, color, thickness, cv2.LINE_AA)


def draw_semantic_wall_debug(
    room_rgb: np.ndarray,
    objects: Iterable[Dict[str, Any]],
    multiwall_split_debug: Dict[str, Any] | None = None,
    target_space: str = "original",
) -> np.ndarray:
    img = room_rgb.copy()
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    wall_labels = {"wall"}
    structural_labels = {"opening", "arch", "door", "window", "tv", "mirror", "lamp", "track_lighting"}
    existing_labels = {"existing_art", "wall_clutter", "wall_label", "hook", "nail", "small_shelf"}
    support_labels = {"floor", "sofa", "table", "cabinet"}

    for obj in objects or []:
        label = str(obj.get("label", "object"))
        role = str(obj.get("role", ""))
        object_id = str(obj.get("object_id", label))
        source = str(obj.get("source", ""))
        conf = obj.get("confidence")
        box = _box_for_space(obj, target_space=target_space)
        if box is None:
            continue

        x1, y1, x2, y2 = [int(round(v)) for v in box]
        if label in wall_labels or role == "wall":
            color = (0, 220, 0)
            thick = 4
        elif label in structural_labels or role == "forbidden":
            color = (255, 0, 0)
            thick = 3
        elif label in existing_labels or role == "existing_content":
            color = (255, 220, 0)
            thick = 3
        elif label in support_labels or role == "support":
            color = (0, 180, 255)
            thick = 2
        else:
            color = (0, 120, 255)
            thick = 2

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        text = object_id
        if conf is not None:
            try:
                text += f" {float(conf):.2f}"
            except Exception:
                pass
        if source:
            text += f" {source[:18]}"
        if "mean_angle_deg" in obj:
            try:
                text += f" {float(obj['mean_angle_deg']):.0f}deg"
            except Exception:
                pass
        _draw_label(img, x1, max(14, y1 - 4), text, color)

    if multiwall_split_debug:
        method = str(multiwall_split_debug.get("method", ""))
        used = multiwall_split_debug.get("used")
        after = multiwall_split_debug.get("after_wall_count")
        header = f"multiwall: used={used} walls={after} {method}"
        _draw_label(img, 12, 28, header, (255, 255, 255))

    return img
