from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

@dataclass
class SemanticObject:
    object_id: str
    label: str
    role: str
    bbox_xyxy_processed: List[float]
    bbox_xyxy_original: List[float]
    polygon: List[List[float]]
    confidence: float
    source: str
    notes: str = ""

def to_dict(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return dict(obj)
