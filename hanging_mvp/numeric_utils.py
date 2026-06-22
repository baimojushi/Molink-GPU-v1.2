from __future__ import annotations

from typing import Dict, Optional

import numpy as np

def _safe_percentile(values: np.ndarray, pct: float, default: Optional[float] = None) -> Optional[float]:
    try:
        arr = np.asarray(values, dtype=np.float32)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return default
        return float(np.percentile(arr, pct))
    except Exception:
        return default


def _mask_bbox(mask: np.ndarray) -> Optional[Dict[str, int]]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return {
        "xmin": int(xs.min()),
        "ymin": int(ys.min()),
        "xmax": int(xs.max()),
        "ymax": int(ys.max()),
    }


def _clamp01(value: float) -> float:
    try:
        v = float(value)
    except Exception:
        return 0.0
    return float(max(0.0, min(1.0, v)))
