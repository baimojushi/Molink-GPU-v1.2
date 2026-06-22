from __future__ import annotations

import json
from typing import Any, Dict

from fastapi import HTTPException

def parse_json(s: str, name: str) -> Dict[str, Any]:
    try:
        return json.loads(s) if s else {}
    except Exception as e:
        raise HTTPException(400, f"{name} must be valid JSON: {e}")


def rule_value(rules: Dict[str, Any], path: list[str], default):
    cur = rules
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    if isinstance(cur, dict) and "value" in cur:
        return cur["value"]
    return cur
