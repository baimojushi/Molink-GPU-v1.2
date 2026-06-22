from __future__ import annotations

from typing import List, Tuple

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))

def bbox_to_polygon(bbox_xyxy: List[float], width: int, height: int) -> List[List[float]]:
    x1, y1, x2, y2 = bbox_xyxy
    x1 = clamp(x1, 0, width - 1)
    x2 = clamp(x2, 0, width - 1)
    y1 = clamp(y1, 0, height - 1)
    y2 = clamp(y2, 0, height - 1)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

def scale_bbox(bbox: List[float], sx: float, sy: float) -> List[float]:
    return [float(bbox[0] * sx), float(bbox[1] * sy), float(bbox[2] * sx), float(bbox[3] * sy)]

def scale_polygon(poly: List[List[float]], sx: float, sy: float) -> List[List[float]]:
    return [[float(x * sx), float(y * sy)] for x, y in poly]
