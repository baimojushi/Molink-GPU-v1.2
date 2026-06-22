from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

Point = Tuple[float, float]
Polygon = List[Point]

@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    source: str = ""
    fov_deg: Optional[float] = None

@dataclass
class WallPlane:
    n: Tuple[float, float, float]
    d: float

@dataclass
class WallBasis:
    origin: Tuple[float, float, float]
    u: Tuple[float, float, float]
    v: Tuple[float, float, float]
    n: Tuple[float, float, float]
    g_hat_cam: Tuple[float, float, float]

@dataclass
class Candidate:
    candidate_id: str
    wall_id: str
    axis_s_m: float
    center_t_m: float
    center_height_m: Optional[float]
    projected_quad_px_processed: List[List[float]]
    projected_quad_px_original: List[List[float]]
    reject_reasons: List[str]
    hard_constraints: Dict[str, Any]
    score: float = 0.0

def to_dict(x: Any) -> Any:
    if hasattr(x, "__dataclass_fields__"):
        return asdict(x)
    return x
