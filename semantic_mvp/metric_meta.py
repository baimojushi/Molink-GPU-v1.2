from __future__ import annotations

import io
import json
import zipfile
from typing import Any, Dict, Tuple

def read_sizes_from_metric3d_zip(zip_bytes: bytes, fallback_original: Tuple[int, int]) -> Dict[str, Any]:
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
            meta = json.loads(z.read("meta.json").decode("utf-8"))
        processed = meta.get("processed_size") or meta.get("image", {}).get("processed_size")
        original = meta.get("original_size") or meta.get("image", {}).get("original_size")
        if not processed:
            # fallback based on depth if needed omitted in MVP semantic service
            processed = fallback_original
        if not original:
            original = fallback_original
        return {
            "meta": meta,
            "processed_size": [int(processed[0]), int(processed[1])],
            "original_size": [int(original[0]), int(original[1])]
        }
    except Exception as e:
        return {
            "meta": {},
            "processed_size": [int(fallback_original[0]), int(fallback_original[1])],
            "original_size": [int(fallback_original[0]), int(fallback_original[1])],
            "warning": f"metric3d_zip_meta_read_failed: {e}"
        }
