from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def _rgb_luma(rgb: np.ndarray) -> np.ndarray:
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return np.tensordot(np.asarray(rgb, dtype=np.float32), weights, axes=([-1], [0]))


def analyze_artwork_luminance(art_rgba: np.ndarray) -> dict[str, Any]:
    """Return the phase-4 artwork luminance contract from valid alpha pixels."""
    arr = np.asarray(art_rgba)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return {
            "artwork_mean_luminance": None,
            "artwork_contrast": None,
            "is_dark_or_high_contrast": False,
            "suggested_wall_tone_rgb": [58, 62, 71],
            "wallpaper_metric_status": "invalid_artwork_array",
        }
    rgb = arr[:, :, :3].astype(np.float32)
    valid = arr[:, :, 3] > 8 if arr.shape[2] >= 4 else np.ones(arr.shape[:2], dtype=bool)
    if int(np.count_nonzero(valid)) < 16:
        return {
            "artwork_mean_luminance": None,
            "artwork_contrast": None,
            "is_dark_or_high_contrast": False,
            "suggested_wall_tone_rgb": [58, 62, 71],
            "suggested_wall_tone_source": "default_conservative_tone",
            "wallpaper_metric_status": "insufficient_valid_alpha_pixels",
        }
    luma = _rgb_luma(rgb)
    values = luma[valid]
    mean_luminance = float(np.mean(values) / 255.0)
    p5, p95 = [float(v) for v in np.percentile(values, [5, 95])]
    contrast = float((p95 - p5) / 255.0)
    is_dark_or_high_contrast = bool(mean_luminance < 0.42 or contrast > 0.55)

    dark_cutoff = float(np.percentile(values, 35))
    dark_pixels = rgb[valid & (luma <= dark_cutoff)]
    if dark_pixels.size == 0:
        dark_rgb = np.array([58.0, 62.0, 71.0], dtype=np.float32)
    else:
        dark_rgb = np.median(dark_pixels, axis=0)
    neutral_anchor = np.array([58.0, 62.0, 71.0], dtype=np.float32)
    tone = 0.35 * dark_rgb + 0.65 * neutral_anchor
    tone_luma = float(_rgb_luma(tone.reshape(1, 1, 3))[0, 0] / 255.0)
    if tone_luma > 0.36:
        tone *= 0.36 / max(tone_luma, 1e-6)
    tone = np.clip(np.rint(tone), 0, 255).astype(np.uint8).tolist()
    return {
        "artwork_mean_luminance": round(mean_luminance, 4),
        "artwork_contrast": round(contrast, 4),
        "is_dark_or_high_contrast": is_dark_or_high_contrast,
        "suggested_wall_tone_rgb": [int(v) for v in tone],
        "suggested_wall_tone_source": "algorithmic_abstracted_dark_palette",
        "wallpaper_metric_status": "ok",
    }


def measure_candidate_wall_luminance(
    room_rgb: np.ndarray,
    wall_mask: np.ndarray,
    artwork_quad_px: Sequence[Sequence[float]] | None,
    *,
    exclusion_margin_px: int | None = None,
) -> dict[str, Any]:
    """Measure wall luminance after excluding a dilated candidate artwork quad."""
    wall = np.asarray(wall_mask) > 0
    height, width = wall.shape[:2]
    room = np.asarray(room_rgb)
    if room.shape[:2] != (height, width):
        room = np.asarray(Image.fromarray(room.astype(np.uint8)).resize((width, height), Image.Resampling.BILINEAR))
    room = room[:, :, :3].astype(np.float32)

    margin = int(exclusion_margin_px) if exclusion_margin_px is not None else max(4, int(round(min(width, height) * 0.012)))
    exclusion_image = Image.new("L", (width, height), 0)
    points: list[tuple[float, float]] = []
    if artwork_quad_px is not None:
        try:
            points = [(float(point[0]), float(point[1])) for point in list(artwork_quad_px)]
        except (TypeError, ValueError, IndexError):
            points = []
    if len(points) >= 3:
        ImageDraw.Draw(exclusion_image).polygon(points, fill=255)
        exclusion_image = exclusion_image.filter(ImageFilter.MaxFilter(size=2 * max(0, margin) + 1))
    exclusion = np.asarray(exclusion_image, dtype=np.uint8) > 0
    sampled = wall & (~exclusion)
    sample_count = int(np.count_nonzero(sampled))
    if sample_count < 32:
        return {
            "wall_mean_luminance": None,
            "is_light_wall": False,
            "wall_luminance_sample_px": sample_count,
            "artwork_exclusion_margin_px": int(margin),
            "wallpaper_metric_status": "insufficient_wall_pixels_after_artwork_exclusion",
        }
    mean_luminance = float(np.mean(_rgb_luma(room)[sampled]) / 255.0)
    return {
        "wall_mean_luminance": round(mean_luminance, 4),
        "is_light_wall": bool(mean_luminance > 0.62),
        "wall_luminance_sample_px": sample_count,
        "artwork_exclusion_margin_px": int(margin),
        "wallpaper_metric_status": "ok",
    }


def should_suggest_dark_wallpaper(
    artwork_analysis: Mapping[str, Any] | None,
    wall_analysis: Mapping[str, Any] | None,
) -> bool:
    artwork = artwork_analysis if isinstance(artwork_analysis, Mapping) else {}
    wall = wall_analysis if isinstance(wall_analysis, Mapping) else {}
    return bool(artwork.get("is_dark_or_high_contrast")) and bool(wall.get("is_light_wall"))
