from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from .schemas import Intrinsics

@dataclass
class Metric3DBundle:
    meta: Dict[str, Any]
    depth_m: np.ndarray
    normal_xyz: np.ndarray
    intr: Intrinsics
    processed_size: Tuple[int, int]
    original_size: Tuple[int, int]
    scale_to_original: Tuple[float, float]

def _read_npy(z: zipfile.ZipFile, name: str) -> np.ndarray:
    with z.open(name) as f:
        return np.load(io.BytesIO(f.read()))

def _get_intr(meta: Dict[str, Any]) -> Intrinsics:
    raw = meta.get("selected_intrinsics") or meta.get("intrinsics")
    if not raw:
        raise ValueError("Metric3D meta.json missing selected_intrinsics/intrinsics")
    for k in ["fx", "fy", "cx", "cy"]:
        if k not in raw:
            raise ValueError(f"selected_intrinsics missing {k}")
    return Intrinsics(
        fx=float(raw["fx"]),
        fy=float(raw["fy"]),
        cx=float(raw["cx"]),
        cy=float(raw["cy"]),
        source=str(raw.get("source", "")),
        fov_deg=float(raw["fov_deg"]) if raw.get("fov_deg") is not None else None,
    )

def load_metric3d_zip(path_or_bytes) -> Metric3DBundle:
    if isinstance(path_or_bytes, (bytes, bytearray)):
        zf = zipfile.ZipFile(io.BytesIO(path_or_bytes), "r")
    else:
        zf = zipfile.ZipFile(path_or_bytes, "r")
    with zf as z:
        meta = json.loads(z.read("meta.json").decode("utf-8"))
        depth = _read_npy(z, "depth_m.npy").astype(np.float32)
        normal = _read_npy(z, "normal_xyz.npy").astype(np.float32)
    intr = _get_intr(meta)
    image_meta = meta.get("image", {})
    processed = tuple(image_meta.get("processed_size") or meta.get("processed_size") or [depth.shape[1], depth.shape[0]])
    original = tuple(image_meta.get("original_size") or meta.get("original_size") or processed)
    sx = float(original[0]) / float(processed[0])
    sy = float(original[1]) / float(processed[1])
    return Metric3DBundle(
        meta=meta,
        depth_m=depth,
        normal_xyz=normal,
        intr=intr,
        processed_size=(int(processed[0]), int(processed[1])),
        original_size=(int(original[0]), int(original[1])),
        scale_to_original=(sx, sy),
    )
