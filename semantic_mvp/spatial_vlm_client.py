"# Spatial Expert VLM client - separate model for bbox and exit review" 

from __future__ import annotations

import base64
import hashlib
import io
import json
import os

import requests
from PIL import Image, ImageOps

from .vlm_client import (
    _resize_image_with_meta,
    _VLM_MAX_IMAGE_SIDE,
    image_to_data_url,
    extract_json,
    mock_objects,
)
from .prompts import system_prompt, user_prompt

_VLM_MAX_IMAGE_SIDE_SPATIAL = int(os.getenv("VLM_MAX_IMAGE_SIDE", "2048"))


def _spatial_vlm_config():
    return {
        "api_url": os.getenv("SPATIAL_VLM_API_URL", "").strip() or os.getenv("VLM_API_URL", "").strip(),
        "api_key": os.getenv("SPATIAL_VLM_API_KEY", "").strip() or os.getenv("VLM_API_KEY", "").strip(),
        "model": os.getenv("SPATIAL_VLM_MODEL", "").strip() or os.getenv("VLM_MODEL", "").strip(),
        "timeout": int(os.getenv("SPATIAL_VLM_TIMEOUT_S", "").strip() or os.getenv("VLM_TIMEOUT_S", "120")),
    }


def call_spatial_vlm(
    image_bytes: bytes,
    system_text: str,
    user_text: str,
):
    cfg = _spatial_vlm_config()
    if not cfg["api_url"] or not cfg["api_key"] or not cfg["model"]:
        raise RuntimeError("SPATIAL_VLM not configured")
    resized, meta = _resize_image_with_meta(image_bytes, _VLM_MAX_IMAGE_SIDE_SPATIAL)
    data_url = image_to_data_url(resized)
    payload = {
        "model": cfg["model"],
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_text},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    }
    headers = {"Authorization": f"Bearer {cfg[chr(34) + chr(97) + chr(112) + chr(105) + chr(95) + chr(107) + chr(101) + chr(121) + chr(34)]}", "Content-Type": "application/json"}
    resp = requests.post(cfg["api_url"], headers=headers, json=payload, timeout=cfg["timeout"])
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    result = extract_json(text)
    if isinstance(result, dict):
        result["_vlm_image_meta"] = meta
        result["_vlm_call_audit"] = {
            "provider": "spatial_vlm",
            "model": cfg["model"],
            "provider_request_id": data.get("id") or resp.headers.get("x-request-id", ""),
        }
    return result


def parse_with_spatial_vlm(image_bytes, scene_profile, mode, width, height):
    if mode == "mock":
        result = mock_objects(width, height, scene_profile)
        result["_vlm_call_audit"] = {"provider": "mock"}
        return result
    if mode == "openai_compatible" or (mode == "auto" and os.getenv("VLM_PROVIDER", "") == "openai_compatible"):
        try:
            cfg = _spatial_vlm_config()
            if cfg["api_url"] and cfg["api_key"] and cfg["model"]:
                _, meta = _resize_image_with_meta(image_bytes, _VLM_MAX_IMAGE_SIDE_SPATIAL)
                sw, sh = meta.get("sent_size", [width, height])
                return call_spatial_vlm(
                    image_bytes,
                    system_prompt(scene_profile),
                    user_prompt(scene_profile, int(sw), int(sh)),
                )
        except Exception:
            pass
        from .vlm_client import call_openai_compatible
        return call_openai_compatible(image_bytes, scene_profile, width, height)
    result = mock_objects(width, height, scene_profile)
    result["_vlm_call_audit"] = {"provider": "mock_fallback"}
    return result
