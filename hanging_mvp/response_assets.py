from __future__ import annotations

import io
import zipfile
from typing import Dict

import cv2
import numpy as np
from fastapi.responses import Response
from PIL import Image

def _png_bytes(arr: np.ndarray) -> bytes:
    if arr.ndim == 2:
        img = Image.fromarray(arr)
    else:
        img = Image.fromarray(arr.astype(np.uint8))
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()


def _draw_overlay(room_rgb: np.ndarray, quad_orig: list[list[float]]) -> np.ndarray:
    pts = np.array(quad_orig, np.int32)
    cv2.polylines(room_rgb, [pts], isClosed=True, color=(255, 0, 0), thickness=6)
    c = pts.mean(axis=0).astype(int)
    half_len = max(24, int(round(room_rgb.shape[0] * 0.25)))
    cv2.line(
        room_rgb,
        (int(c[0]), max(0, int(c[1]) - half_len)),
        (int(c[0]), min(room_rgb.shape[0] - 1, int(c[1]) + half_len)),
        (0, 255, 0),
        max(2, int(round(room_rgb.shape[0] / 360))),
    )
    return room_rgb


def _zip_response(files: Dict[str, bytes]) -> Response:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, content in files.items():
            z.writestr(name, content)
    return Response(buf.getvalue(), media_type="application/zip", headers={"Content-Disposition": 'attachment; filename="hanging_mvp_result.zip"'})
