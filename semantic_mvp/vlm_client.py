from __future__ import annotations

import base64
import io
import json
import os
import re
from typing import Any, Dict, List

import requests
from PIL import Image, ImageOps

from .prompts import artwork_frame_system_prompt, artwork_frame_user_prompt, system_prompt, user_prompt

_MIME_MAP = {"jpeg": "image/jpeg", "jpg": "image/jpeg", "png": "image/png", "webp": "image/webp", "gif": "image/gif"}
_VLM_MAX_IMAGE_SIDE = int(os.getenv("VLM_MAX_IMAGE_SIDE", "2048"))

def _detect_mime(image_bytes: bytes) -> str:
    # imghdr was removed in Python 3.13.  Use PIL for the small set of formats
    # sent to VLM providers and fall back to JPEG for unknown bytes.
    try:
        fmt = str(Image.open(io.BytesIO(image_bytes)).format or "").strip().lower()
    except Exception:
        fmt = ""
    return _MIME_MAP.get(fmt or "", "image/jpeg")

def _resize_image_with_meta(image_bytes: bytes, max_side: int) -> tuple[bytes, Dict[str, Any]]:
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    original_w, original_h = img.size
    sent_w, sent_h = original_w, original_h
    resized_for_vlm = False
    if original_w > max_side or original_h > max_side:
        resized_for_vlm = True
        if original_w > original_h:
            sent_w, sent_h = max_side, int(original_h * max_side / original_w)
        else:
            sent_w, sent_h = int(original_w * max_side / original_h), max_side
        img = img.resize((sent_w, sent_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue(), {
        "original_size": [int(original_w), int(original_h)],
        "sent_size": [int(sent_w), int(sent_h)],
        "max_side": int(max_side),
        "resized_for_vlm": bool(resized_for_vlm),
        "scale_x_sent_to_original": float(original_w) / float(max(1, sent_w)),
        "scale_y_sent_to_original": float(original_h) / float(max(1, sent_h)),
    }


def _resize_image(image_bytes: bytes, max_side: int) -> bytes:
    return _resize_image_with_meta(image_bytes, max_side)[0]

def image_to_data_url(image_bytes: bytes, mime: str = "") -> str:
    mime = mime if mime else _detect_mime(image_bytes)
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))

def mock_objects(width: int, height: int, scene_profile: str) -> Dict[str, Any]:
    # Coarse smoke-test fallback: one wall upper area + one floor lower area.
    return {
        "objects": [
            {
                "label": "wall",
                "role": "wall",
                "bbox_xyxy": [0.05 * width, 0.10 * height, 0.95 * width, 0.72 * height],
                "confidence": 0.45,
                "notes": "mock coarse wall"
            },
            {
                "label": "floor",
                "role": "support",
                "bbox_xyxy": [0, 0.68 * height, width, height],
                "confidence": 0.45,
                "notes": "mock coarse floor"
            }
        ]
    }

def call_openai_compatible_with_prompts(
    image_bytes: bytes,
    system_text: str,
    user_text: str,
    *,
    timeout: int = 120,
) -> Dict[str, Any]:
    api_url = os.getenv("VLM_API_URL", "").strip()
    api_key = os.getenv("VLM_API_KEY", "").strip()
    model = os.getenv("VLM_MODEL", "").strip()
    if not api_url or not api_key or not model:
        raise RuntimeError("VLM_API_URL, VLM_API_KEY and VLM_MODEL are required for openai_compatible mode")

    resized, vlm_image_meta = _resize_image_with_meta(image_bytes, _VLM_MAX_IMAGE_SIDE)
    data_url = image_to_data_url(resized)
    payload = {
        "model": model,
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
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    result = extract_json(text)
    if isinstance(result, dict):
        result["_vlm_image_meta"] = vlm_image_meta
    return result


def call_openai_compatible(image_bytes: bytes, scene_profile: str, width: int, height: int) -> Dict[str, Any]:
    # The provider sees the resized request image, so ask for coordinates in that
    # image space and map them back to original pixels in semantic_mvp.service.
    _, vlm_image_meta = _resize_image_with_meta(image_bytes, _VLM_MAX_IMAGE_SIDE)
    sent_w, sent_h = vlm_image_meta.get("sent_size", [width, height])
    return call_openai_compatible_with_prompts(
        image_bytes,
        system_prompt(scene_profile),
        user_prompt(scene_profile, int(sent_w), int(sent_h)),
    )


def _coerce_rgb_triplet(value: Any) -> list[int]:
    if not isinstance(value, list) or len(value) < 3:
        return [0, 0, 0]
    out = []
    for v in value[:3]:
        try:
            out.append(int(max(0, min(255, round(float(v))))))
        except Exception:
            out.append(0)
    return out


def normalize_artwork_frame_result(raw: Dict[str, Any], *, source: str = "vlm") -> Dict[str, Any]:
    has_frame_raw = raw.get("has_frame")
    if isinstance(has_frame_raw, bool):
        has_frame = has_frame_raw
    elif has_frame_raw is None:
        has_frame = None
    elif str(has_frame_raw).strip().lower() in {"true", "yes", "1"}:
        has_frame = True
    elif str(has_frame_raw).strip().lower() in {"false", "no", "0"}:
        has_frame = False
    else:
        has_frame = None
    try:
        confidence = float(raw.get("confidence", 0.0) or 0.0)
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    frame_style = str(raw.get("frame_style") or ("visible_outer_frame" if has_frame else "none" if has_frame is False else "uncertain"))
    ratios_raw = raw.get("frame_width_ratio") if isinstance(raw.get("frame_width_ratio"), dict) else {}
    ratios = {}
    for edge in ("left", "right", "top", "bottom"):
        try:
            ratios[edge] = float(max(0.0, min(0.5, float(ratios_raw.get(edge, 0.0) or 0.0))))
        except Exception:
            ratios[edge] = 0.0
    colors_raw = raw.get("dominant_edge_colors_rgb") if isinstance(raw.get("dominant_edge_colors_rgb"), dict) else {}
    colors = {edge: _coerce_rgb_triplet(colors_raw.get(edge)) for edge in ("left", "right", "top", "bottom")}
    should_precompose = bool(
        has_frame is True
        and confidence >= float(os.getenv("ARTWORK_FRAME_MIN_CONFIDENCE", "0.72"))
        and frame_style in {"visible_outer_frame", "canvas_wrap", "frame", "framed"}
    )
    return {
        "enabled": True,
        "source": source,
        "status": "ok" if has_frame is not None else "uncertain",
        "has_frame": has_frame,
        "confidence": float(confidence),
        "frame_style": frame_style,
        "frame_width_ratio": ratios,
        "dominant_edge_colors_rgb": colors,
        "should_precompose_frame_sides": should_precompose,
        "reason": str(raw.get("reason") or ""),
    }


def parse_artwork_frame_with_vlm(image_bytes: bytes, mode: str, width: int, height: int) -> Dict[str, Any]:
    mode_norm = str(mode or "auto").strip().lower()
    if mode_norm == "mock":
        forced = os.getenv("ARTWORK_FRAME_MOCK_HAS_FRAME", "").strip().lower()
        if forced in {"1", "true", "yes", "y", "on"}:
            confidence = float(os.getenv("ARTWORK_FRAME_MOCK_CONFIDENCE", "0.95"))
            return normalize_artwork_frame_result(
                {
                    "has_frame": True,
                    "confidence": confidence,
                    "frame_style": os.getenv("ARTWORK_FRAME_MOCK_STYLE", "visible_outer_frame"),
                    "reason": "forced by ARTWORK_FRAME_MOCK_HAS_FRAME for deterministic integration testing in mock mode",
                },
                source="mock_forced",
            )
        if forced in {"0", "false", "no", "n", "off"}:
            confidence = float(os.getenv("ARTWORK_FRAME_MOCK_CONFIDENCE", "0.95"))
            return normalize_artwork_frame_result(
                {"has_frame": False, "confidence": confidence, "frame_style": "none", "reason": "forced no-frame by ARTWORK_FRAME_MOCK_HAS_FRAME in mock mode"},
                source="mock_forced",
            )
        return {
            "enabled": True,
            "source": "mock",
            "status": "uncertain",
            "has_frame": None,
            "confidence": 0.0,
            "frame_style": "uncertain",
            "frame_width_ratio": {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0},
            "dominant_edge_colors_rgb": {"left": [0, 0, 0], "right": [0, 0, 0], "top": [0, 0, 0], "bottom": [0, 0, 0]},
            "should_precompose_frame_sides": False,
            "reason": "mock mode does not inspect artwork frame unless ARTWORK_FRAME_MOCK_HAS_FRAME is set",
        }
    if mode_norm == "openai_compatible" or (mode_norm == "auto" and os.getenv("VLM_PROVIDER", "").strip() == "openai_compatible"):
        raw = call_openai_compatible_with_prompts(
            image_bytes,
            artwork_frame_system_prompt(),
            artwork_frame_user_prompt(width, height),
        )
        return normalize_artwork_frame_result(raw, source="openai_compatible")
    return {
        "enabled": True,
        "source": "auto",
        "status": "uncertain",
        "has_frame": None,
        "confidence": 0.0,
        "frame_style": "uncertain",
        "frame_width_ratio": {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0},
        "dominant_edge_colors_rgb": {"left": [0, 0, 0], "right": [0, 0, 0], "top": [0, 0, 0], "bottom": [0, 0, 0]},
        "should_precompose_frame_sides": False,
        "reason": "no artwork VLM provider configured",
    }


def parse_with_vlm(image_bytes: bytes, scene_profile: str, mode: str, width: int, height: int) -> Dict[str, Any]:
    if mode == "mock":
        return mock_objects(width, height, scene_profile)
    if mode == "openai_compatible":
        return call_openai_compatible(image_bytes, scene_profile, width, height)
    # auto
    if os.getenv("VLM_PROVIDER", "").strip() == "openai_compatible":
        try:
            return call_openai_compatible(image_bytes, scene_profile, width, height)
        except Exception as e:
            return {
                **mock_objects(width, height, scene_profile),
                "warning": f"vlm_failed_fallback_to_mock: {e}"
            }
    return {
        **mock_objects(width, height, scene_profile),
        "warning": "no_vlm_provider_configured_fallback_to_mock"
    }
