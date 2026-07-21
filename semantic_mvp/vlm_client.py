from __future__ import annotations

import base64
import colorsys
import hashlib
import io
import json
import os
import re
from typing import Any, Dict, List, Mapping

import requests
from PIL import Image, ImageOps

from .prompts import (
    artwork_frame_system_prompt,
    artwork_frame_user_prompt,
    styling_analysis_system_prompt,
    styling_analysis_user_prompt,
    system_prompt,
    user_prompt,
)

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
        # Bind diagnostics to the exact provider call.  Do not regenerate a
        # "raw VLM response" later: the model is non-deterministic and a retry
        # can return a different object count (the production 422 had 18
        # objects, while the later diagnostic retry returned a 2-object mock).
        # These fields contain no authorization data.
        result["_vlm_call_audit"] = {
            "provider": "openai_compatible",
            "model": model,
            "provider_request_id": data.get("id") or resp.headers.get("x-request-id") or resp.headers.get("request-id"),
            "prompt_sha256": hashlib.sha256((system_text + "\n" + user_text).encode("utf-8")).hexdigest(),
            "provider_message_sha256": hashlib.sha256(str(text).encode("utf-8")).hexdigest(),
        }
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
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return [0, 0, 0]
    out = []
    for v in value[:3]:
        try:
            out.append(int(max(0, min(255, round(float(v))))))
        except Exception:
            out.append(0)
    return out


_STYLING_STYLE_REGISTERS = {
    "minimalist_modern", "mid_century", "classical_ornate", "industrial",
    "coastal_natural", "eclectic_mixed", "uncertain",
}
_STYLING_TEXTURES = {
    "smooth_flat", "heavy_impasto", "graphic_hard_edge",
    "soft_atmospheric", "metallic_industrial", "uncertain",
}
_STYLING_TENSION_LEVELS = {"low", "medium", "high"}
_STYLING_LINE_DIRECTIONS = {"horizontal", "vertical", "diagonal", "radial", "none"}


def _styling_enum(value: Any, allowed: set[str], fallback: str) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in allowed else fallback


def _styling_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalize_styling_analysis(
    raw: Mapping[str, Any],
    *,
    source: str,
    soft_furnishing_requested: bool,
    wallpaper_tone_rgb: list[int] | None,
) -> Dict[str, Any]:
    accents: list[list[int]] = []
    if soft_furnishing_requested:
        for value in raw.get("accent_colors_abstracted_rgb", []) or []:
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                accents.append(_coerce_rgb_triplet(value))
            if len(accents) >= 2:
                break
    neutral = _coerce_rgb_triplet(raw.get("neutral_pairing_rgb") or [132, 128, 120])
    suggested = wallpaper_tone_rgb or _coerce_rgb_triplet(raw.get("suggested_wall_tone_rgb") or [58, 62, 71])
    return {
        "enabled": True,
        "status": "ok",
        "source": source,
        "is_abstract": _styling_bool(raw.get("is_abstract")),
        "style_register": _styling_enum(raw.get("style_register"), _STYLING_STYLE_REGISTERS, "uncertain"),
        "texture_character": _styling_enum(raw.get("texture_character"), _STYLING_TEXTURES, "uncertain"),
        "accent_colors_abstracted_rgb": accents,
        "neutral_pairing_rgb": neutral,
        "visual_tension_level": _styling_enum(raw.get("visual_tension_level"), _STYLING_TENSION_LEVELS, "medium"),
        "dominant_line_direction": _styling_enum(raw.get("dominant_line_direction"), _STYLING_LINE_DIRECTIONS, "none"),
        "suggested_wall_tone_rgb": suggested,
        "soft_furnishing_fields_skipped": not bool(soft_furnishing_requested),
        "wallpaper_target_was_fixed_input": bool(wallpaper_tone_rgb),
    }


def _abstract_rgb(rgb: list[int], *, lightness_delta: float = 0.0, saturation_scale: float = 0.78) -> list[int]:
    r, g, b = [max(0.0, min(1.0, float(v) / 255.0)) for v in rgb[:3]]
    h, lightness, saturation = colorsys.rgb_to_hls(r, g, b)
    lightness = max(0.0, min(1.0, lightness + lightness_delta))
    saturation = max(0.0, min(1.0, saturation * saturation_scale))
    rr, gg, bb = colorsys.hls_to_rgb(h, lightness, saturation)
    return [int(round(rr * 255)), int(round(gg * 255)), int(round(bb * 255))]


def _algorithmic_styling_analysis(
    image_bytes: bytes,
    *,
    soft_furnishing_requested: bool,
    wallpaper_tone_rgb: list[int] | None,
    reason: str,
) -> Dict[str, Any]:
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    image.thumbnail((192, 192), Image.Resampling.LANCZOS)
    quantized = image.quantize(colors=8, method=Image.Quantize.MEDIANCUT).convert("RGB")
    counts = sorted((quantized.getcolors(maxcolors=256) or []), key=lambda item: item[0], reverse=True)
    palette = [list(color) for _, color in counts[:8]] or [[128, 128, 128]]

    def luma(rgb: list[int]) -> float:
        return (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) / 255.0

    def saturation(rgb: list[int]) -> float:
        return colorsys.rgb_to_hls(*(float(v) / 255.0 for v in rgb[:3]))[2]

    contrast_base = luma(wallpaper_tone_rgb) if wallpaper_tone_rgb else 0.5
    ranked_accents = sorted(palette, key=lambda c: (saturation(c), abs(luma(c) - contrast_base)), reverse=True)
    accents = [
        _abstract_rgb(ranked_accents[0], lightness_delta=0.04),
        _abstract_rgb(ranked_accents[min(1, len(ranked_accents) - 1)], lightness_delta=-0.02),
    ] if soft_furnishing_requested else []
    avg = [int(round(sum(c[i] for c in palette) / len(palette))) for i in range(3)]
    neutral = _abstract_rgb(avg, saturation_scale=0.18)
    darkest = min(palette, key=luma)
    suggested = wallpaper_tone_rgb or _abstract_rgb(darkest, lightness_delta=min(0.0, 0.24 - luma(darkest)), saturation_scale=0.52)
    max_sat = max(saturation(c) for c in palette)
    contrast = max(luma(c) for c in palette) - min(luma(c) for c in palette)
    tension = "high" if max_sat > 0.62 or contrast > 0.62 else ("medium" if max_sat > 0.30 or contrast > 0.35 else "low")
    return {
        "enabled": True,
        "status": "fallback_algorithmic",
        "source": "algorithmic_palette_fallback",
        "is_abstract": False,
        "style_register": "uncertain",
        "texture_character": "uncertain",
        "accent_colors_abstracted_rgb": accents,
        "neutral_pairing_rgb": neutral,
        "visual_tension_level": tension,
        "dominant_line_direction": "none",
        "suggested_wall_tone_rgb": suggested,
        "soft_furnishing_fields_skipped": not bool(soft_furnishing_requested),
        "wallpaper_target_was_fixed_input": bool(wallpaper_tone_rgb),
        "fallback_reason": reason,
    }


def parse_styling_analysis_with_vlm(
    image_bytes: bytes,
    mode: str,
    *,
    soft_furnishing_requested: bool,
    wallpaper_recolor: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run one optional artwork VLM request, with a deterministic palette fallback."""
    wallpaper_recolor = wallpaper_recolor if isinstance(wallpaper_recolor, Mapping) else {}
    wallpaper_tone = None
    if wallpaper_recolor.get("enabled"):
        wallpaper_tone = _coerce_rgb_triplet(wallpaper_recolor.get("tone_rgb") or [58, 62, 71])
    fallback = lambda reason: _algorithmic_styling_analysis(
        image_bytes,
        soft_furnishing_requested=soft_furnishing_requested,
        wallpaper_tone_rgb=wallpaper_tone,
        reason=reason,
    )
    mode_norm = str(mode or "auto").strip().lower()
    use_remote = mode_norm == "openai_compatible" or (
        mode_norm == "auto" and os.getenv("VLM_PROVIDER", "").strip() == "openai_compatible"
    )
    if not use_remote:
        return fallback("styling_vlm_not_configured_or_mock_mode")
    try:
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
        width, height = image.size
        raw = call_openai_compatible_with_prompts(
            image_bytes,
            styling_analysis_system_prompt(),
            styling_analysis_user_prompt(
                int(width),
                int(height),
                soft_furnishing_requested=bool(soft_furnishing_requested),
                wallpaper_tone_rgb=wallpaper_tone,
            ),
            timeout=int(os.getenv("STYLING_VLM_TIMEOUT_S", "120")),
        )
        return _normalize_styling_analysis(
            raw,
            source="openai_compatible_styling_vlm",
            soft_furnishing_requested=bool(soft_furnishing_requested),
            wallpaper_tone_rgb=wallpaper_tone,
        )
    except Exception as exc:
        return fallback(f"styling_vlm_failed: {exc}")


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
    surface_style_raw = str(raw.get("artwork_surface_style") or raw.get("surface_style") or raw.get("material_style") or "uncertain").strip().lower()
    surface_aliases = {
        "fine": "refined_paper_or_delicate",
        "delicate": "refined_paper_or_delicate",
        "refined": "refined_paper_or_delicate",
        "paper": "refined_paper_or_delicate",
        "paper_painting": "refined_paper_or_delicate",
        "chinese_paper": "refined_paper_or_delicate",
        "guohua": "refined_paper_or_delicate",
        "国画": "refined_paper_or_delicate",
        "rough": "rough_or_canvas",
        "coarse": "rough_or_canvas",
        "canvas": "rough_or_canvas",
        "canvas_wrap": "rough_or_canvas",
        "fabric": "rough_or_canvas",
        "oil_painting": "rough_or_canvas",
        "acrylic": "rough_or_canvas",
        "impasto": "rough_or_canvas",
        "scroll": "chinese_hanging_scroll",
        "hanging_scroll": "chinese_hanging_scroll",
        "chinese_scroll": "chinese_hanging_scroll",
        "guohua_hanging_scroll": "chinese_hanging_scroll",
        "vertical_scroll": "chinese_hanging_scroll",
        "立轴": "chinese_hanging_scroll",
        "国画立轴": "chinese_hanging_scroll",
    }
    artwork_surface_style = surface_aliases.get(surface_style_raw, surface_style_raw)
    if artwork_surface_style not in {"refined_paper_or_delicate", "rough_or_canvas", "chinese_hanging_scroll", "uncertain"}:
        artwork_surface_style = "uncertain"
    policy_raw = str(raw.get("side_thickness_policy") or "").strip().lower()
    if policy_raw in {"none", "no", "flat", "skip", "0"} or artwork_surface_style == "chinese_hanging_scroll":
        side_thickness_policy = "none"
    elif policy_raw in {"double", "thick", "2x", "two_x"} or artwork_surface_style == "rough_or_canvas":
        side_thickness_policy = "double"
    else:
        side_thickness_policy = "normal"
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
    mood_raw = str(raw.get("mood_register") or "").strip().lower()
    mood_register = mood_raw if mood_raw in {"calm", "contemplative", "dynamic", "melancholic", "joyful"} else None
    try:
        visual_complexity = raw.get("visual_complexity")
        visual_complexity = None if visual_complexity is None or visual_complexity == "" else max(0.0, min(1.0, float(visual_complexity)))
    except Exception:
        visual_complexity = None
    return {
        "enabled": True,
        "source": source,
        "status": "ok" if has_frame is not None else "uncertain",
        "has_frame": has_frame,
        "confidence": float(confidence),
        "frame_style": frame_style,
        "artwork_surface_style": artwork_surface_style,
        "side_thickness_policy": side_thickness_policy,
        "frame_width_ratio": ratios,
        "dominant_edge_colors_rgb": colors,
        "should_precompose_frame_sides": should_precompose,
        "mood_register": mood_register,
        "visual_complexity": visual_complexity,
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
            "mood_register": None,
            "visual_complexity": None,
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
        "mood_register": None,
        "visual_complexity": None,
        "reason": "no artwork VLM provider configured",
    }


def parse_with_vlm(image_bytes: bytes, scene_profile: str, mode: str, width: int, height: int) -> Dict[str, Any]:
    if mode == "mock":
        result = mock_objects(width, height, scene_profile)
        result["_vlm_call_audit"] = {"provider": "mock", "model": None, "provider_request_id": None}
        return result
    if mode == "openai_compatible":
        return call_openai_compatible(image_bytes, scene_profile, width, height)
    # auto
    if os.getenv("VLM_PROVIDER", "").strip() == "openai_compatible":
        try:
            return call_openai_compatible(image_bytes, scene_profile, width, height)
        except Exception as e:
            result = {
                **mock_objects(width, height, scene_profile),
                "warning": f"vlm_failed_fallback_to_mock: {e}"
            }
            result["_vlm_call_audit"] = {"provider": "mock_fallback", "model": None, "provider_request_id": None}
            return result
    result = {
        **mock_objects(width, height, scene_profile),
        "warning": "no_vlm_provider_configured_fallback_to_mock"
    }
    result["_vlm_call_audit"] = {"provider": "mock_fallback", "model": None, "provider_request_id": None}
    return result
