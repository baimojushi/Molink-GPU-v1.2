from __future__ import annotations

import base64
import io
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import requests
import cv2
from PIL import Image, ImageFilter
import numpy as np

from .base import ImageEditRequest, ImageEditResult
from .bfl import BflImageEditProvider


APIYI_GPT_IMAGE2_VIP_PROVIDER = "apiyi_gpt_image2_vip"
APIYI_GPT_IMAGE2_VIP_ALIASES = {"apiyi_gpt_image2_vip", "gpt_image2_vip", "gpt-image-2-vip", "apiyi"}

# API易 gpt-image-2-vip image-edit endpoint accepts size=auto or one of these 30 fixed sizes.
VIP_SIZES: tuple[str, ...] = (
    "1280x1280", "848x1280", "1280x848", "960x1280", "1280x960",
    "1024x1280", "1280x1024", "720x1280", "1280x720", "1280x544",
    "2048x2048", "1360x2048", "2048x1360", "1536x2048", "2048x1536",
    "1632x2048", "2048x1632", "1152x2048", "2048x1152", "2048x864",
    "2880x2880", "2336x3520", "3520x2336", "2480x3312", "3312x2480",
    "2560x3216", "3216x2560", "2160x3840", "3840x2160", "3840x1632",
)
VIP_ALLOWED_SIZES: tuple[str, ...] = ("auto", *VIP_SIZES)


def _env_truthy(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() not in {"", "0", "false", "no", "off"}


def _env_optional(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def _mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"


def _validate_size(size: str | None) -> str | None:
    if not size:
        return None
    normalized = size.strip().lower()
    if normalized not in VIP_ALLOWED_SIZES:
        raise ValueError(f"unsupported gpt-image-2-vip size={size!r}; expected auto or one of {VIP_SIZES}")
    return normalized


def _decode_b64_image(value: str) -> bytes:
    data = value.strip()
    if data.startswith("data:"):
        _, data = data.split(",", 1)
    return base64.b64decode(data)


def _write_apiyi_image(data: Mapping[str, Any], output_path: Path, *, timeout_s: float) -> str:
    images = data.get("data")
    if not isinstance(images, list) or not images:
        raise RuntimeError(f"APIYI image response has no data[0]: {json.dumps(data, ensure_ascii=False)[:2000]}")
    first = images[0] if isinstance(images[0], dict) else {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if first.get("b64_json"):
        output_path.write_bytes(_decode_b64_image(str(first["b64_json"])))
        return "b64_json"
    if first.get("url"):
        resp = requests.get(str(first["url"]), timeout=timeout_s)
        resp.raise_for_status()
        output_path.write_bytes(resp.content)
        return "url"
    raise RuntimeError(f"APIYI image response has neither b64_json nor url: {json.dumps(data, ensure_ascii=False)[:2000]}")


def _resize_to_match(source_path: Path, output_path: Path) -> dict[str, Any]:
    with Image.open(source_path) as source, Image.open(output_path) as generated:
        source_size = tuple(source.size)
        generated_size = tuple(generated.size)
        if source_size == generated_size:
            return {"resized": False, "source_size": list(source_size), "generated_size": list(generated_size)}
        resized = generated.convert("RGB").resize(source_size, Image.Resampling.LANCZOS)
        resized.save(output_path)
    return {"resized": True, "source_size": list(source_size), "generated_size": list(generated_size)}


def _maybe_compress_upload(path: Path, tmp_dir: Path, *, max_bytes: int, enabled: bool) -> Path:
    if not enabled or path.stat().st_size <= max_bytes:
        return path
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out = tmp_dir / f"{path.stem}_apiyi_upload.jpg"
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        quality = 88
        scale = 1.0
        buf = io.BytesIO()
        for _ in range(8):
            candidate = rgb
            if scale < 0.999:
                w, h = rgb.size
                candidate = rgb.resize((max(16, int(w * scale)), max(16, int(h * scale))), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            candidate.save(buf, format="JPEG", quality=quality, optimize=True)
            if len(buf.getvalue()) <= max_bytes or quality <= 72:
                out.write_bytes(buf.getvalue())
                return out
            quality -= 5
            scale *= 0.85
        out.write_bytes(buf.getvalue())
    return out


def _image_size_wh(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return tuple(int(v) for v in img.size)


def _round_coord(value: float) -> int | float:
    rounded = round(float(value), 2)
    if abs(rounded - round(rounded)) < 1e-6:
        return int(round(rounded))
    return rounded


def _scale_bbox_for_sent_image(value: Any, *, sx: float, sy: float) -> list[int | float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        return [
            _round_coord(float(value[0]) * sx),
            _round_coord(float(value[1]) * sy),
            _round_coord(float(value[2]) * sx),
            _round_coord(float(value[3]) * sy),
        ]
    except Exception:
        return None


def _scale_polygon_for_sent_image(value: Any, *, sx: float, sy: float) -> list[list[int | float]] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    out: list[list[int | float]] = []
    try:
        for pt in list(value)[:4]:
            if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                return None
            out.append([_round_coord(float(pt[0]) * sx), _round_coord(float(pt[1]) * sy)])
    except Exception:
        return None
    return out


def _build_full_space_coordinate_prompt(
    metadata: Mapping[str, Any],
    *,
    original_size_wh: tuple[int, int],
    sent_size_wh: tuple[int, int],
) -> tuple[str, dict[str, Any]]:
    """Build a remote coordinate lock in the exact pixels of the uploaded image.

    The API易 image-edit endpoint does not expose a mask field.  For textured
    walls we intentionally send the full room instead of a crop, so the only
    remote edit boundary is text.  The model sees the compressed upload, not the
    original file, therefore all prompt coordinates must be scaled to that upload
    size.
    """
    ow, oh = int(original_size_wh[0]), int(original_size_wh[1])
    sw, sh = int(sent_size_wh[0]), int(sent_size_wh[1])
    sx = float(sw) / max(1.0, float(ow))
    sy = float(sh) / max(1.0, float(oh))
    front_polygon = _scale_polygon_for_sent_image(
        metadata.get("front_polygon_px") or metadata.get("polygon_px") or metadata.get("target_quad_original"),
        sx=sx,
        sy=sy,
    )
    mount_polygon = _scale_polygon_for_sent_image(
        metadata.get("mount_polygon_px") or metadata.get("mount_quad_original"),
        sx=sx,
        sy=sy,
    )
    edit_polygon = mount_polygon or front_polygon
    front_bbox = _scale_bbox_for_sent_image(metadata.get("front_bbox_px") or metadata.get("bbox_px"), sx=sx, sy=sy)
    mount_bbox = _scale_bbox_for_sent_image(metadata.get("mount_bbox_px"), sx=sx, sy=sy)
    edit_bbox = mount_bbox or front_bbox

    matrix: dict[str, Any] = {
        "coordinate_space": "pixels_in_the_exact_compressed_image_1_sent_to_api",
        "image_1_sent_size_px": [sw, sh],
        "source_original_size_px": [ow, oh],
        "scale_from_original_to_sent": [sx, sy],
        "front_polygon_px": front_polygon,
        "front_bbox_px": front_bbox,
        "edit_polygon_px": edit_polygon,
        "edit_bbox_px": edit_bbox,
    }
    # Drop empty fields from the text payload but keep trace context explicit.
    compact = {k: v for k, v in matrix.items() if v is not None}
    matrix_json = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
    text = (
        "\n\nFULL-SPACE REMOTE COORDINATE LOCK. 图1是完整房间图，不是局部裁剪。"
        "以下坐标全部是你实际收到的压缩后图1像素坐标，不是原图坐标，也不是归一化坐标："
        f"{matrix_json}。"
        "只允许在 edit_polygon_px / edit_bbox_px 所限定的挂画对象、边侧、接触阴影和极窄融合边缘内做渲染；"
        "该坐标范围以外的墙面纹理、肌理、光影分界、门窗、地面、天花、家具、结构线、镜头畸变和构图必须保持不变。"
        "不要移动、缩放、拉伸、重画、复制或替换挂画。不要矫正透视畸变、镜头畸变、墙面线条、房间结构或相机视角；"
        "保持当前照片中的自然透视和畸变。"
    )
    return text, {**matrix, "prompt_json": compact, "enabled": True}


def _mask_bbox(mask: Image.Image, image_size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    gray = mask.convert("L")
    if gray.size != image_size:
        gray = gray.resize(image_size, Image.Resampling.NEAREST)
    return gray.getbbox()


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    padding_px: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    width, height = image_size
    pad = max(0, int(padding_px))
    return (
        max(0, left - pad),
        max(0, top - pad),
        min(width, right + pad),
        min(height, bottom + pad),
    )


def _save_mask_local_edit_inputs(
    *,
    image_path: Path,
    mask_path: Path,
    tmp_dir: Path,
    padding_px: int,
) -> dict[str, Any]:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as image, Image.open(mask_path) as mask:
        image_rgb = image.convert("RGB")
        image_size = tuple(image_rgb.size)
        mask_gray = mask.convert("L")
        if mask_gray.size != image_size:
            mask_gray = mask_gray.resize(image_size, Image.Resampling.NEAREST)
        bbox_raw = _mask_bbox(mask_gray, image_size)
        if bbox_raw is None:
            raise ValueError(f"mask has no editable pixels: {mask_path}")
        bbox = _expand_bbox(bbox_raw, image_size, padding_px=padding_px)
        crop_path = tmp_dir / f"{image_path.stem}_masked_edit_crop.png"
        mask_crop_path = tmp_dir / f"{image_path.stem}_masked_edit_mask_crop.png"
        image_rgb.crop(bbox).save(crop_path)
        mask_gray.crop(bbox).save(mask_crop_path)
    return {
        "mode": "mask_local_composite",
        "crop_path": crop_path,
        "mask_crop_path": mask_crop_path,
        "bbox": list(bbox),
        "raw_bbox": list(bbox_raw),
        "padding_px": int(padding_px),
        "full_size": list(image_size),
        "crop_size": [bbox[2] - bbox[0], bbox[3] - bbox[1]],
    }


def _composite_masked_crop(
    *,
    original_path: Path,
    generated_crop_path: Path,
    mask_crop_path: Path,
    bbox: tuple[int, int, int, int],
    output_path: Path,
    feather_px: int,
) -> dict[str, Any]:
    with Image.open(original_path) as original, Image.open(generated_crop_path) as generated, Image.open(mask_crop_path) as mask:
        base = original.convert("RGB")
        crop_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
        generated_rgb = generated.convert("RGB")
        generated_size = tuple(generated_rgb.size)
        if generated_size != crop_size:
            generated_rgb = generated_rgb.resize(crop_size, Image.Resampling.LANCZOS)
        alpha = mask.convert("L")
        if alpha.size != crop_size:
            alpha = alpha.resize(crop_size, Image.Resampling.NEAREST)
        if feather_px > 0:
            alpha = alpha.filter(ImageFilter.GaussianBlur(radius=float(feather_px)))
        base.paste(generated_rgb, (bbox[0], bbox[1]), alpha)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        base.save(output_path)
    return {
        "composited": True,
        "bbox": list(bbox),
        "crop_size": list(crop_size),
        "generated_size": list(generated_size),
        "feather_px": int(feather_px),
    }


def _coerce_optional_path(value: Any) -> Path | None:
    if not value:
        return None
    try:
        path = Path(str(value))
    except Exception:
        return None
    return path if path.exists() else None


def _crop_mask_to_bbox(mask_path: Path, *, image_size: tuple[int, int], bbox: tuple[int, int, int, int], tmp_dir: Path, stem: str) -> Path:
    with Image.open(mask_path) as mask:
        m = mask.convert("L")
        if tuple(m.size) != image_size:
            m = m.resize(image_size, Image.Resampling.NEAREST)
        out = tmp_dir / f"{stem}_core_mask_crop.png"
        m.crop(bbox).save(out)
        return out


def _as_alpha(mask: Image.Image, size: tuple[int, int], *, feather_px: int = 0) -> np.ndarray:
    alpha = mask.convert("L")
    if tuple(alpha.size) != size:
        alpha = alpha.resize(size, Image.Resampling.NEAREST)
    if feather_px > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=float(feather_px)))
    return np.asarray(alpha, dtype=np.float32) / 255.0


def _low_frequency_luma_delta(original: Image.Image, generated: Image.Image, *, blur_px: int, max_delta: float, strength: float) -> np.ndarray:
    """Return a clamped low-frequency achromatic luminance delta field.

    The field is derived from generated-vs-original lighting but applied as a
    neutral lightness change only, so intrinsic artwork/frame colors are
    preserved and the model cannot recolor the object.
    """
    radius = max(1, int(blur_px))
    orig_blur = np.asarray(original.convert("RGB").filter(ImageFilter.GaussianBlur(radius=float(radius))), dtype=np.float32)
    gen_blur = np.asarray(generated.convert("RGB").filter(ImageFilter.GaussianBlur(radius=float(radius))), dtype=np.float32)
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    orig_luma = np.tensordot(orig_blur, weights, axes=([-1], [0]))
    gen_luma = np.tensordot(gen_blur, weights, axes=([-1], [0]))
    delta = gen_luma - orig_luma
    clamp = float(max(0.0, max_delta))
    if clamp > 0:
        delta = np.clip(delta, -clamp, clamp)
    return delta[..., None] * float(strength)


def _darken_only_luma_transfer(original_arr: np.ndarray, generated_arr: np.ndarray, *, max_darkening: float) -> np.ndarray:
    """Apply only the darkening part of generated shadow to original pixels.

    This is used in the narrow guard ring around the artwork object.  It keeps
    contact shadow if the model made that ring darker, but refuses bright
    generated wall/halo pixels that would create a white seam on the frame edge.
    """
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    orig_luma = np.tensordot(original_arr, weights, axes=([-1], [0]))
    gen_luma = np.tensordot(generated_arr, weights, axes=([-1], [0]))
    dark_delta = np.minimum(gen_luma - orig_luma, 0.0)
    clamp = float(max(0.0, max_darkening))
    if clamp > 0:
        dark_delta = np.clip(dark_delta, -clamp, 0.0)
    return np.clip(original_arr + dark_delta[..., None], 0, 255)


def _alpha_bbox(alpha: np.ndarray, *, threshold: float = 0.2) -> tuple[int, int, int, int] | None:
    arr = np.asarray(alpha)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    ys, xs = np.where(arr > float(threshold))
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _expand_bbox_xyxy(
    bbox: tuple[int, int, int, int],
    size_wh: tuple[int, int],
    *,
    pad_px: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    w, h = int(size_wh[0]), int(size_wh[1])
    p = max(0, int(pad_px))
    return max(0, x0 - p), max(0, y0 - p), min(w, x1 + p), min(h, y1 + p)


def _dilate_alpha(alpha: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return np.asarray(alpha, dtype=np.float32)
    arr = np.asarray(alpha, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    k = max(1, int(radius_px) * 2 + 1)
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate((np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8), kernel, iterations=1).astype(np.float32) / 255.0


def _to_gray_float(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float32)


def _estimate_anchor_similarity(
    *,
    original_crop: Image.Image,
    generated_crop: Image.Image,
    anchor_alpha: np.ndarray,
    max_shift_px: int,
    min_score: float,
    scale_min: float,
    scale_max: float,
    scale_steps: int,
) -> dict[str, Any]:
    """Estimate original->generated similarity transform from the artwork anchor.

    GPT Image may move and slightly scale the pasted artwork inside the crop.
    Translation-only correction is not enough: a hard-locked texture pasted at
    the old scale can cover the generated contact shadow.  This searches a
    small uniform scale range and returns an affine matrix that maps hard-paste
    pixels into the generated crop coordinate system.
    """
    max_shift = max(0, int(max_shift_px))
    bbox = _alpha_bbox(anchor_alpha, threshold=0.2)
    if bbox is None or max_shift <= 0:
        return {
            "applied": False,
            "reason": "missing_anchor_or_search_disabled",
            "dx": 0,
            "dy": 0,
            "scale": 1.0,
            "score": None,
            "matrix_original_to_generated": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        }
    w, h = original_crop.size
    x0, y0, x1, y1 = _expand_bbox_xyxy(bbox, (w, h), pad_px=2)
    bw, bh = x1 - x0, y1 - y0
    if bw < 6 or bh < 6:
        return {
            "applied": False,
            "reason": "anchor_too_small",
            "dx": 0,
            "dy": 0,
            "scale": 1.0,
            "score": None,
            "anchor_bbox": [x0, y0, x1, y1],
            "matrix_original_to_generated": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        }

    orig_gray = _to_gray_float(original_crop)
    gen_gray = _to_gray_float(generated_crop)
    template0 = orig_gray[y0:y1, x0:x1]
    if float(np.std(template0)) < 3.0:
        return {
            "applied": False,
            "reason": "anchor_low_texture",
            "dx": 0,
            "dy": 0,
            "scale": 1.0,
            "score": None,
            "anchor_bbox": [x0, y0, x1, y1],
            "matrix_original_to_generated": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        }

    s_min = max(0.75, float(scale_min))
    s_max = min(1.35, float(scale_max))
    if s_max < s_min:
        s_min, s_max = s_max, s_min
    steps = max(1, int(scale_steps))
    scales = [1.0] if steps == 1 else [float(v) for v in np.linspace(s_min, s_max, steps)]
    if 1.0 not in scales:
        scales.append(1.0)
    scales = sorted(set(round(float(v), 4) for v in scales))

    sx0 = max(0, x0 - max_shift)
    sy0 = max(0, y0 - max_shift)
    sx1 = min(w, x1 + max_shift)
    sy1 = min(h, y1 + max_shift)
    search = gen_gray[sy0:sy1, sx0:sx1]
    best: dict[str, Any] | None = None

    for scale in scales:
        tw = max(3, int(round(float(bw) * float(scale))))
        th = max(3, int(round(float(bh) * float(scale))))
        if tw > search.shape[1] or th > search.shape[0]:
            continue
        if tw == bw and th == bh:
            template = template0
        else:
            template = cv2.resize(template0, (tw, th), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
        if float(np.std(template)) < 3.0:
            continue
        result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(result)
        rec = {
            "score": float(max_val),
            "scale": float(scale),
            "gen_x0": int(sx0 + max_loc[0]),
            "gen_y0": int(sy0 + max_loc[1]),
            "template_w": int(tw),
            "template_h": int(th),
        }
        if best is None or float(rec["score"]) > float(best["score"]):
            best = rec

    if best is None:
        return {
            "applied": False,
            "reason": "search_region_too_small",
            "dx": 0,
            "dy": 0,
            "scale": 1.0,
            "score": None,
            "anchor_bbox": [x0, y0, x1, y1],
            "matrix_original_to_generated": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        }

    scale = float(best["scale"])
    gen_x0 = int(best["gen_x0"])
    gen_y0 = int(best["gen_y0"])
    tx = float(gen_x0) - scale * float(x0)
    ty = float(gen_y0) - scale * float(y0)
    raw_dx = gen_x0 - x0
    raw_dy = gen_y0 - y0
    scale_delta = abs(scale - 1.0)
    applied = bool(float(best["score"]) >= float(min_score) and (abs(raw_dx) > 0 or abs(raw_dy) > 0 or scale_delta >= 0.003))
    return {
        "applied": applied,
        "reason": None if applied else "score_below_threshold_or_identity_transform",
        "dx": int(raw_dx) if applied else 0,
        "dy": int(raw_dy) if applied else 0,
        "raw_dx": int(raw_dx),
        "raw_dy": int(raw_dy),
        "scale": float(scale if applied else 1.0),
        "raw_scale": float(scale),
        "score": float(best["score"]),
        "min_score": float(min_score),
        "max_shift_px": int(max_shift),
        "scale_search_min": float(s_min),
        "scale_search_max": float(s_max),
        "scale_search_steps": int(steps),
        "anchor_bbox": [int(x0), int(y0), int(x1), int(y1)],
        "matched_bbox": [int(gen_x0), int(gen_y0), int(gen_x0 + int(best["template_w"])), int(gen_y0 + int(best["template_h"]))],
        "matrix_original_to_generated": [[float(scale), 0.0, float(tx)], [0.0, float(scale), float(ty)]] if applied else [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "resampling": "lanczos4_for_locked_rgb_and_linear_antialias_for_masks",
    }




def _translate_rgb_array(image: Image.Image, *, dx: int, dy: int, fill_image: Image.Image) -> Image.Image:
    if dx == 0 and dy == 0:
        return image.convert("RGB")
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    fill = np.asarray(fill_image.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    matrix = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
    shifted = cv2.warpAffine(arr, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    valid = cv2.warpAffine(np.ones((h, w), dtype=np.uint8) * 255, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    shifted[valid == 0] = fill[valid == 0]
    return Image.fromarray(shifted, mode="RGB")


def _warp_rgb_similarity(image: Image.Image, *, matrix: list[list[float]], size_wh: tuple[int, int]) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    m = np.array(matrix, dtype=np.float32)
    out_w, out_h = int(size_wh[0]), int(size_wh[1])
    warped = cv2.warpAffine(
        arr,
        m,
        (out_w, out_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return Image.fromarray(warped, mode="RGB")


def _warp_alpha_similarity(alpha: np.ndarray, *, matrix: list[list[float]], size_wh: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(alpha, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    m = np.array(matrix, dtype=np.float32)
    out_w, out_h = int(size_wh[0]), int(size_wh[1])
    warped = cv2.warpAffine(
        np.clip(arr, 0.0, 1.0),
        m,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return np.clip(warped.astype(np.float32), 0.0, 1.0)


def _correct_generated_wall_rgb(
    *,
    original_crop: Image.Image,
    generated_crop: Image.Image,
    edit_alpha: np.ndarray,
    object_alpha: np.ndarray,
    exclusion_px: int,
    sample_max_delta: float,
    max_rgb_delta: float,
    strength: float,
    apply_to_wall_only: bool = False,
) -> tuple[Image.Image, dict[str, Any]]:
    """Match generated wall RGB to the hard-paste wall while preserving shadows.

    The RGB comparison deliberately excludes the artwork object, frame, and
    likely projection/shadow pixels.  A constant robust median delta is then
    applied to the generated crop, which removes crop-paste wall patches while
    retaining local shadow contrast.
    """
    orig = np.asarray(original_crop.convert("RGB"), dtype=np.float32)
    gen = np.asarray(generated_crop.convert("RGB"), dtype=np.float32)
    edit = np.asarray(edit_alpha, dtype=np.float32)
    obj_dilated = _dilate_alpha(object_alpha, int(exclusion_px))

    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    orig_luma = np.tensordot(orig, weights, axes=([-1], [0]))
    gen_luma = np.tensordot(gen, weights, axes=([-1], [0]))
    luma_delta = gen_luma - orig_luma
    rgb_delta_abs = np.max(np.abs(gen - orig), axis=2)

    sample_mask = (
        (edit > 0.15)
        & (obj_dilated < 0.05)
        & (np.abs(luma_delta) <= float(sample_max_delta))
        & (rgb_delta_abs <= float(sample_max_delta) * 2.0)
    )
    sample_count = int(np.count_nonzero(sample_mask))
    if sample_count < 32:
        return generated_crop.convert("RGB"), {
            "applied": False,
            "reason": "insufficient_clean_wall_samples",
            "sample_count": sample_count,
            "exclusion_px": int(exclusion_px),
            "sample_max_delta": float(sample_max_delta),
        }

    orig_med = np.median(orig[sample_mask], axis=0)
    gen_med = np.median(gen[sample_mask], axis=0)
    delta = (orig_med - gen_med) * float(strength)
    clamp = float(max(0.0, max_rgb_delta))
    if clamp > 0:
        delta = np.clip(delta, -clamp, clamp)
    corrected_full = np.clip(gen + delta.reshape(1, 1, 3), 0, 255)
    if apply_to_wall_only:
        # Apply the median wall color offset only to wall pixels inside the
        # editable crop.  The artwork, frame, mat, and model-rendered object
        # pixels remain exactly as returned by GPT Image, avoiding a second
        # locked-texture paste or any frame/core repaint in post.
        wall_apply_alpha = np.clip(edit * (1.0 - np.clip(object_alpha, 0.0, 1.0)), 0.0, 1.0)[..., None]
        corrected = gen * (1.0 - wall_apply_alpha) + corrected_full * wall_apply_alpha
        apply_scope = "wall_pixels_only_inside_edit_mask_object_unchanged"
    else:
        corrected = corrected_full
        apply_scope = "entire_generated_crop"
    return Image.fromarray(corrected.astype(np.uint8), mode="RGB"), {
        "applied": True,
        "sample_count": sample_count,
        "exclusion_px": int(exclusion_px),
        "sample_max_delta": float(sample_max_delta),
        "max_rgb_delta": float(max_rgb_delta),
        "strength": float(strength),
        "orig_wall_median_rgb": [float(v) for v in orig_med],
        "generated_wall_median_rgb": [float(v) for v in gen_med],
        "applied_delta_rgb": [float(v) for v in delta],
        "apply_scope": apply_scope,
    }


def _composite_wall_rgb_only_crop(
    *,
    original_path: Path,
    generated_crop_path: Path,
    edit_mask_crop_path: Path,
    object_mask_crop_path: Path | None,
    bbox: tuple[int, int, int, int],
    output_path: Path,
    edge_feather_px: int,
    wall_sample_exclusion_px: int,
    wall_sample_max_delta: float,
    wall_rgb_max_delta: float,
    wall_rgb_correction_strength: float,
) -> dict[str, Any]:
    """Paste the GPT Image crop back after wall-only RGB correction.

    This mode intentionally does *not* estimate relocation, warp a hard-paste
    locked texture, rebuild the frame/core, or run an object edge guard.  The
    provider-rendered crop is treated as the visual source of truth; post only
    removes clean-wall RGB mismatch by applying a robust median delta to wall
    pixels outside the object mask.
    """
    with Image.open(original_path) as original, Image.open(generated_crop_path) as generated, Image.open(edit_mask_crop_path) as edit_mask:
        base = original.convert("RGB")
        crop_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
        original_crop = base.crop(bbox).convert("RGB")
        generated_rgb = generated.convert("RGB")
        generated_size = tuple(generated_rgb.size)
        if generated_size != crop_size:
            generated_rgb = generated_rgb.resize(crop_size, Image.Resampling.LANCZOS)

        edit_alpha = _as_alpha(edit_mask, crop_size, feather_px=edge_feather_px)
        if object_mask_crop_path is not None and object_mask_crop_path.exists():
            with Image.open(object_mask_crop_path) as object_mask:
                object_alpha = _as_alpha(object_mask, crop_size, feather_px=0)
            object_mask_available = True
        else:
            object_alpha = np.zeros((crop_size[1], crop_size[0]), dtype=np.float32)
            object_mask_available = False

        generated_rgb, wall_rgb_trace = _correct_generated_wall_rgb(
            original_crop=original_crop,
            generated_crop=generated_rgb,
            edit_alpha=edit_alpha,
            object_alpha=object_alpha,
            exclusion_px=wall_sample_exclusion_px,
            sample_max_delta=wall_sample_max_delta,
            max_rgb_delta=wall_rgb_max_delta,
            strength=wall_rgb_correction_strength,
            apply_to_wall_only=True,
        )

        alpha = Image.fromarray(np.clip(edit_alpha * 255.0, 0, 255).astype(np.uint8), mode="L")
        base.paste(generated_rgb, (bbox[0], bbox[1]), alpha)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        base.save(output_path)

    return {
        "composited": True,
        "mode": "gpt_render_preserved_wall_rgb_only_composite",
        "bbox": list(bbox),
        "crop_size": list(crop_size),
        "generated_size": list(generated_size),
        "object_mask_available": object_mask_available,
        "locked_texture_pasted": False,
        "hardpaste_frame_core_repaste": False,
        "anchor_relocation_applied": False,
        "wall_rgb_correction": wall_rgb_trace,
        "edge_feather_px": int(edge_feather_px),
    }


def _composite_photometric_crop(
    *,
    original_path: Path,
    generated_crop_path: Path,
    edit_mask_crop_path: Path,
    object_mask_crop_path: Path | None,
    core_mask_crop_path: Path | None,
    bbox: tuple[int, int, int, int],
    output_path: Path,
    edge_feather_px: int,
    core_feather_px: int,
    core_protection_px: int,
    photometric_blur_px: int,
    photometric_max_delta: float,
    photometric_strength: float,
    relocation_search_px: int,
    relocation_min_score: float,
    relocation_scale_min: float,
    relocation_scale_max: float,
    relocation_scale_steps: int,
    wall_sample_exclusion_px: int,
    wall_sample_max_delta: float,
    wall_rgb_max_delta: float,
    wall_rgb_correction_strength: float,
    object_edge_guard_px: int,
) -> dict[str, Any]:
    with Image.open(original_path) as original, Image.open(generated_crop_path) as generated, Image.open(edit_mask_crop_path) as edit_mask:
        base = original.convert("RGB")
        crop_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
        original_crop = base.crop(bbox).convert("RGB")
        generated_rgb = generated.convert("RGB")
        generated_size = tuple(generated_rgb.size)
        if generated_size != crop_size:
            generated_rgb = generated_rgb.resize(crop_size, Image.Resampling.LANCZOS)

        edit_alpha = _as_alpha(edit_mask, crop_size, feather_px=edge_feather_px)
        if object_mask_crop_path is not None and object_mask_crop_path.exists():
            with Image.open(object_mask_crop_path) as object_mask:
                object_alpha = _as_alpha(object_mask, crop_size, feather_px=0)
            object_mask_available = True
        else:
            object_alpha = edit_alpha.copy()
            object_mask_available = False
        if core_mask_crop_path is not None and core_mask_crop_path.exists():
            with Image.open(core_mask_crop_path) as core_mask:
                core_alpha = _as_alpha(core_mask, crop_size, feather_px=core_feather_px)
            core_mask_available = True
        else:
            core_alpha = np.clip(object_alpha * edit_alpha, 0.0, 1.0)
            core_mask_available = False

        # Estimate both translation and scale, then move the locked hard-paste
        # texture into the provider-rendered coordinate system.  This avoids the
        # previous failure mode where a translation-only lock covered the GPT
        # Image shadow when the provider also scaled the artwork slightly.
        anchor_alpha = core_alpha if core_mask_available else object_alpha
        relocation_trace = _estimate_anchor_similarity(
            original_crop=original_crop,
            generated_crop=generated_rgb,
            anchor_alpha=anchor_alpha,
            max_shift_px=relocation_search_px,
            min_score=relocation_min_score,
            scale_min=relocation_scale_min,
            scale_max=relocation_scale_max,
            scale_steps=relocation_scale_steps,
        )
        transform_matrix = relocation_trace.get("matrix_original_to_generated") or [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        locked_rgb = _warp_rgb_similarity(original_crop, matrix=transform_matrix, size_wh=crop_size)
        locked_object_alpha = _warp_alpha_similarity(object_alpha, matrix=transform_matrix, size_wh=crop_size)

        wall_exclusion_alpha = np.maximum(object_alpha, locked_object_alpha)
        generated_rgb, wall_rgb_trace = _correct_generated_wall_rgb(
            original_crop=original_crop,
            generated_crop=generated_rgb,
            edit_alpha=edit_alpha,
            object_alpha=wall_exclusion_alpha,
            exclusion_px=wall_sample_exclusion_px,
            sample_max_delta=wall_sample_max_delta,
            max_rgb_delta=wall_rgb_max_delta,
            strength=wall_rgb_correction_strength,
        )

        orig_arr = np.asarray(original_crop, dtype=np.float32)
        locked_arr = np.asarray(locked_rgb, dtype=np.float32)
        generated_arr = np.asarray(generated_rgb, dtype=np.float32)
        delta = _low_frequency_luma_delta(
            locked_rgb,
            generated_rgb,
            blur_px=photometric_blur_px,
            max_delta=photometric_max_delta,
            strength=photometric_strength,
        )
        # Apply the render's reasonable frame/object lightness change to the
        # locked texture itself.  This keeps a slightly grayer, more realistic
        # frame while preserving artwork identity and intrinsic color.
        photometric_arr = np.clip(locked_arr + delta, 0, 255)

        object_w_2d = np.clip(locked_object_alpha * edit_alpha, 0.0, 1.0)
        object_w = object_w_2d[..., None]

        object_guard_alpha = _dilate_alpha(locked_object_alpha, int(object_edge_guard_px))
        guard_w_2d = np.clip((object_guard_alpha - locked_object_alpha) * edit_alpha, 0.0, 1.0)
        guard_w = guard_w_2d[..., None]
        guard_shadow_arr = _darken_only_luma_transfer(
            orig_arr,
            generated_arr,
            max_darkening=photometric_max_delta,
        )

        shadow_w_2d = np.clip(edit_alpha * (1.0 - object_guard_alpha), 0.0, 1.0)
        shadow_w = shadow_w_2d[..., None]
        combined_w = np.clip(object_w + guard_w + shadow_w, 0.0, 1.0)
        out_arr = (
            orig_arr * (1.0 - combined_w)
            + photometric_arr * object_w
            + guard_shadow_arr * guard_w
            + generated_arr * shadow_w
        )

        out_crop = Image.fromarray(np.clip(out_arr, 0, 255).astype(np.uint8), mode="RGB")
        base.paste(out_crop, (bbox[0], bbox[1]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        base.save(output_path)

    return {
        "composited": True,
        "mode": "photometric_intrinsic_color_preserving_shadow_composite",
        "bbox": list(bbox),
        "crop_size": list(crop_size),
        "generated_size": list(generated_size),
        "object_mask_available": object_mask_available,
        "core_mask_available": core_mask_available,
        "hardpaste_core_pixel_restore": False,
        "core_treatment": "low_frequency_achromatic_luminance_transfer_only",
        "edge_treatment": "generated_shadow_pixels_preserved_after_similarity_lock_and_wall_rgb_correction_with_object_edge_guard",
        "object_edge_guard_px": int(object_edge_guard_px),
        "guard_treatment": "darken_only_luma_transfer_in_narrow_object_edge_ring",
        "locked_texture_transform": "hardpaste_texture_warped_to_provider_translation_and_scale",
        "locked_texture_resampling": "cv2.INTER_LANCZOS4_rgb_and_linear_alpha",
        "anchor_relocation": relocation_trace,
        "wall_rgb_correction": wall_rgb_trace,
        "edge_feather_px": int(edge_feather_px),
        "core_feather_px": int(core_feather_px),
        "core_protection_px": int(core_protection_px),
        "photometric_blur_px": int(photometric_blur_px),
        "photometric_max_delta": float(photometric_max_delta),
        "photometric_strength": float(photometric_strength),
    }


@dataclass(frozen=True)
class ApiyiGptImage2Client:
    api_key: str | None = None
    base_url: str = "https://api.apiyi.com/v1"
    model: str = "gpt-image-2-vip"
    timeout_s: float = 300.0
    max_retries: int = 3
    response_format: str | None = None
    size: str | None = "auto"
    compress_inputs: bool = True
    max_input_bytes: int = 10_000_000
    resize_to_input_size: bool = True
    include_references: bool = False
    mask_edit_mode: str = "photometric_composite"
    mask_crop_padding_px: int = 48
    mask_composite_feather_px: int = 2
    photometric_blur_px: int = 18
    photometric_max_delta: float = 32.0
    photometric_strength: float = 1.0
    core_mask_feather_px: int = 1
    core_protection_px: int = 3
    relocation_search_px: int = 24
    relocation_min_score: float = 0.38
    relocation_scale_min: float = 0.92
    relocation_scale_max: float = 1.08
    relocation_scale_steps: int = 17
    wall_sample_exclusion_px: int = 12
    wall_sample_max_delta: float = 18.0
    wall_rgb_max_delta: float = 24.0
    wall_rgb_correction_strength: float = 1.0
    object_edge_guard_px: int = 2
    allow_unmasked_fallback: bool = False

    @classmethod
    def from_env(cls) -> "ApiyiGptImage2Client":
        api_key = (
            os.getenv("APIYI_API_KEY", "").strip()
            or os.getenv("YI_API_KEY", "").strip()
            or os.getenv("GPT_IMAGE2_API_KEY", "").strip()
            or None
        )
        # The API易 gpt-image-2-vip edit endpoint returns b64_json by default.
        # Keep response_format absent unless a deployment explicitly routes to image2_OSS.
        response_format = _env_optional("APIYI_GPT_IMAGE_RESPONSE_FORMAT")
        size = _validate_size(os.getenv("APIYI_GPT_IMAGE_SIZE", "auto"))
        return cls(
            api_key=api_key,
            base_url=os.getenv("APIYI_API_BASE", "https://api.apiyi.com/v1").rstrip("/"),
            model=os.getenv("APIYI_GPT_IMAGE_MODEL", "gpt-image-2-vip").strip() or "gpt-image-2-vip",
            timeout_s=max(30.0, float(os.getenv("APIYI_GPT_IMAGE_TIMEOUT_S", "300"))),
            max_retries=max(1, int(os.getenv("APIYI_GPT_IMAGE_MAX_RETRIES", "3"))),
            response_format=response_format,
            size=size,
            compress_inputs=_env_truthy("APIYI_GPT_IMAGE_COMPRESS_INPUTS", "1"),
            max_input_bytes=max(128_000, int(os.getenv("APIYI_GPT_IMAGE_MAX_INPUT_BYTES", "10000000"))),
            resize_to_input_size=_env_truthy("APIYI_GPT_IMAGE_RESIZE_TO_INPUT", "1"),
            include_references=_env_truthy("APIYI_GPT_IMAGE_INCLUDE_REFERENCES", "1"),
            mask_edit_mode=os.getenv("APIYI_GPT_IMAGE_MASK_EDIT_MODE", "photometric_composite").strip().lower() or "photometric_composite",
            mask_crop_padding_px=max(0, int(os.getenv("APIYI_GPT_IMAGE_MASK_CROP_PADDING_PX", "48"))),
            mask_composite_feather_px=max(0, int(os.getenv("APIYI_GPT_IMAGE_MASK_COMPOSITE_FEATHER_PX", "2"))),
            photometric_blur_px=max(1, int(os.getenv("APIYI_GPT_IMAGE_PHOTOMETRIC_BLUR_PX", "18"))),
            photometric_max_delta=max(0.0, float(os.getenv("APIYI_GPT_IMAGE_PHOTOMETRIC_MAX_DELTA", "32"))),
            photometric_strength=max(0.0, float(os.getenv("APIYI_GPT_IMAGE_PHOTOMETRIC_STRENGTH", "1.0"))),
            core_mask_feather_px=max(0, int(os.getenv("APIYI_GPT_IMAGE_CORE_MASK_FEATHER_PX", "1"))),
            core_protection_px=max(0, int(os.getenv("APIYI_GPT_IMAGE_CORE_PROTECTION_PX", "3"))),
            relocation_search_px=max(0, int(os.getenv("APIYI_GPT_IMAGE_RELOCATION_SEARCH_PX", "24"))),
            relocation_min_score=max(0.0, float(os.getenv("APIYI_GPT_IMAGE_RELOCATION_MIN_SCORE", "0.38"))),
            relocation_scale_min=float(os.getenv("APIYI_GPT_IMAGE_RELOCATION_SCALE_MIN", "0.92")),
            relocation_scale_max=float(os.getenv("APIYI_GPT_IMAGE_RELOCATION_SCALE_MAX", "1.08")),
            relocation_scale_steps=max(1, int(os.getenv("APIYI_GPT_IMAGE_RELOCATION_SCALE_STEPS", "17"))),
            wall_sample_exclusion_px=max(0, int(os.getenv("APIYI_GPT_IMAGE_WALL_SAMPLE_EXCLUSION_PX", "12"))),
            wall_sample_max_delta=max(0.0, float(os.getenv("APIYI_GPT_IMAGE_WALL_SAMPLE_MAX_DELTA", "18"))),
            wall_rgb_max_delta=max(0.0, float(os.getenv("APIYI_GPT_IMAGE_WALL_RGB_MAX_DELTA", "24"))),
            wall_rgb_correction_strength=max(0.0, float(os.getenv("APIYI_GPT_IMAGE_WALL_RGB_CORRECTION_STRENGTH", "1.0"))),
            object_edge_guard_px=max(0, int(os.getenv("APIYI_GPT_IMAGE_OBJECT_EDGE_GUARD_PX", "2"))),
            allow_unmasked_fallback=_env_truthy("APIYI_GPT_IMAGE_ALLOW_UNMASKED_FALLBACK", "0"),
        )

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError("APIYI_API_KEY, YI_API_KEY or GPT_IMAGE2_API_KEY is required for APIYI GPT Image provider")
        return {"Authorization": f"Bearer {self.api_key}"}

    def edit(self, request: ImageEditRequest) -> ImageEditResult:
        if not request.prompt:
            raise ValueError("gpt-image-2-vip edit requires request.prompt")

        image = Path(request.image)
        out_dir = request.output_dir or image.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (request.output_name or f"{image.stem}_gpt_image2_vip.png")
        raw_out_path = out_dir / f"{out_path.stem}.apiyi_raw.png"
        references = [Path(p) for p in request.references]
        requested_references = references
        references = requested_references if self.include_references else []
        upload_tmp = out_dir / ".apiyi_upload_tmp"
        upload_tmp.mkdir(parents=True, exist_ok=True)

        mask_trace: dict[str, Any] = {
            "mask_path": str(request.mask) if request.mask else None,
            "mask_parameter_supported_by_endpoint": False,
            "mask_sent_as_multipart_field": False,
            "pixel_coordinate_matrix_supported_by_endpoint": False,
            "pixel_coordinate_matrix_sent": False,
            "remote_edit_region_control": "none_api_has_no_mask_or_coordinate_matrix_field",
            "mask_enforcement": "none",
        }
        target_for_api = image
        output_scope = "whole_image"
        composite_trace: dict[str, Any] = {"composited": False}
        object_mask_source = (
            request.metadata.get("render_object_mask_path")
            or request.metadata.get("mount_mask_path")
            or request.metadata.get("art_region_mask_path")
            or request.metadata.get("qa_art_mask_path")
            or request.metadata.get("front_region_mask_path")
        )
        object_mask_path = _coerce_optional_path(object_mask_source)
        core_mask_source = (
            request.metadata.get("qa_art_mask_path")
            or request.metadata.get("front_region_mask_path")
            or request.metadata.get("art_region_mask_path")
        )
        core_mask_path = _coerce_optional_path(core_mask_source)

        normalized_mask_edit_mode = self.mask_edit_mode.replace("-", "_")
        full_space_coordinate_prompt_requested = bool(request.metadata.get("full_space_coordinate_prompt_requested"))
        if request.mask and full_space_coordinate_prompt_requested:
            target_for_api = image
            output_scope = "full_space_coordinate_prompt"
            mask_trace.update({
                "mask_enforcement": "remote_full_space_coordinate_prompt_no_local_crop",
                "mask_enforced_locally": False,
                "local_crop_disabled": True,
                "remote_coordinate_matrix_sent": True,
                "pixel_coordinate_matrix_sent": True,
                "remote_edit_region_control": "prompt_side_compressed_pixel_coordinate_matrix",
                "full_space_image_sent": True,
                "mask_path": str(request.mask),
                "reason": request.metadata.get("full_space_render_reason") or "wall_texture_gate_recommended_full_space_render",
            })
        elif request.mask and normalized_mask_edit_mode in {
            "photometric_composite",
            "photometric_core_edge_composite",
            "structure_preserving_composite",
            "wall_rgb_only",
            "wall_rgb_correction_only",
            "render_wall_rgb_only",
            "local_composite",
            "crop_composite",
            "masked_crop",
        }:
            local = _save_mask_local_edit_inputs(
                image_path=image,
                mask_path=Path(request.mask),
                tmp_dir=upload_tmp,
                padding_px=self.mask_crop_padding_px,
            )
            target_for_api = Path(local["crop_path"])
            photometric_mode = normalized_mask_edit_mode in {
                "photometric_composite",
                "photometric_core_edge_composite",
                "structure_preserving_composite",
            }
            wall_rgb_only_mode = normalized_mask_edit_mode in {
                "wall_rgb_only",
                "wall_rgb_correction_only",
                "render_wall_rgb_only",
            }
            if wall_rgb_only_mode:
                output_scope = "mask_wall_rgb_only_composite"
            else:
                output_scope = "mask_photometric_composite" if photometric_mode else "mask_local_composite"
            object_mask_crop_path = None
            if (photometric_mode or wall_rgb_only_mode) and object_mask_path is not None:
                object_mask_crop_path = _crop_mask_to_bbox(
                    object_mask_path,
                    image_size=tuple(local["full_size"]),
                    bbox=tuple(int(v) for v in local["bbox"]),
                    tmp_dir=upload_tmp,
                    stem=f"{image.stem}_object",
                )
            core_mask_crop_path = None
            if photometric_mode and core_mask_path is not None:
                core_mask_crop_path = _crop_mask_to_bbox(
                    core_mask_path,
                    image_size=tuple(local["full_size"]),
                    bbox=tuple(int(v) for v in local["bbox"]),
                    tmp_dir=upload_tmp,
                    stem=image.stem,
                )
            mask_trace.update({
                "mask_enforcement": (
                    "local_crop_then_wall_rgb_only_composite_no_locked_texture"
                    if wall_rgb_only_mode
                    else ("local_crop_then_photometric_artwork_lock_and_shadow_composite" if photometric_mode else "local_crop_then_mask_composite")
                ),
                "mask_enforced_locally": True,
                "local_coordinate_bbox_xyxy": local["bbox"],
                "remote_coordinate_matrix_sent": False,
                "edit_crop_path": str(local["crop_path"]),
                "mask_crop_path": str(local["mask_crop_path"]),
                "object_mask_path": str(object_mask_path) if object_mask_path is not None else None,
                "object_mask_crop_path": str(object_mask_crop_path) if object_mask_crop_path is not None else None,
                "object_mask_source": str(object_mask_source) if object_mask_source else None,
                "core_mask_path": str(core_mask_path) if core_mask_path is not None else None,
                "core_mask_crop_path": str(core_mask_crop_path) if core_mask_crop_path is not None else None,
                "core_mask_source": str(core_mask_source) if core_mask_source else None,
                "mask_bbox": local["bbox"],
                "mask_raw_bbox": local["raw_bbox"],
                "mask_crop_padding_px": local["padding_px"],
                "mask_crop_size": local["crop_size"],
                "full_size": local["full_size"],
                "hardpaste_core_pixel_restore": False,
            })
        elif request.mask:
            if not self.allow_unmasked_fallback:
                raise ValueError(
                    "gpt-image-2-vip endpoint has no mask field; set APIYI_GPT_IMAGE_MASK_EDIT_MODE=photometric_composite "
                    "or APIYI_GPT_IMAGE_ALLOW_UNMASKED_FALLBACK=1 for explicit diagnostics"
                )
            mask_trace.update({"mask_enforcement": "disabled_unmasked_fallback", "mask_enforced_locally": False})
        elif not self.allow_unmasked_fallback:
            raise ValueError(
                "final gpt-image-2-vip render requires a non-empty local mask for bounded edit; "
                "set APIYI_GPT_IMAGE_ALLOW_UNMASKED_FALLBACK=1 only for diagnostics"
            )

        target_upload_path = _maybe_compress_upload(target_for_api, upload_tmp, max_bytes=self.max_input_bytes, enabled=self.compress_inputs)
        reference_upload_paths = [
            _maybe_compress_upload(ref, upload_tmp, max_bytes=self.max_input_bytes, enabled=self.compress_inputs)
            for ref in references
        ]
        upload_paths = [target_upload_path, *reference_upload_paths]

        coordinate_prompt = ""
        coordinate_prompt_trace: dict[str, Any] = {"enabled": False}
        if output_scope == "full_space_coordinate_prompt":
            try:
                coordinate_prompt, coordinate_prompt_trace = _build_full_space_coordinate_prompt(
                    request.metadata,
                    original_size_wh=_image_size_wh(target_for_api),
                    sent_size_wh=_image_size_wh(target_upload_path),
                )
            except Exception as exc:
                coordinate_prompt_trace = {"enabled": False, "error": str(exc)}
                coordinate_prompt = (
                    "\n\nFULL-SPACE REMOTE COORDINATE LOCK failed to build numeric coordinates; "
                    "use the visible hard-pasted artwork as the only editable object and do not modify any other room pixels. "
                    "不要矫正透视畸变、镜头畸变、墙面线条或相机视角。"
                )
            mask_trace["remote_coordinate_matrix"] = coordinate_prompt_trace

        prompt = self._build_prompt(
            request.prompt,
            reference_count=len(references),
            output_scope=output_scope,
            coordinate_prompt=coordinate_prompt,
        )
        data: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
        }
        size = _validate_size(str(request.metadata.get("size") or self.size or "auto"))
        if size:
            data["size"] = size
        if self.response_format:
            data["response_format"] = self.response_format

        files = []
        handles = []
        started = time.time()
        response_json: Mapping[str, Any] | None = None
        response_source = "unknown"
        resize_trace: dict[str, Any] = {"resized": False}
        try:
            for idx, p in enumerate(upload_paths, start=1):
                handle = p.open("rb")
                handles.append(handle)
                files.append(("image", (p.name or f"image_{idx}.png", handle, _mime_type(p))))

            response_json = self._post_images_edit(data=data, files=files)
            response_source = _write_apiyi_image(response_json, raw_out_path, timeout_s=self.timeout_s)

            if output_scope in {"mask_local_composite", "mask_photometric_composite", "mask_wall_rgb_only_composite"}:
                crop_mask = Path(mask_trace["mask_crop_path"])
                bbox = tuple(int(v) for v in mask_trace["mask_bbox"])
                if self.resize_to_input_size:
                    resize_trace = _resize_to_match(target_for_api, raw_out_path)
                if output_scope == "mask_photometric_composite":
                    object_crop_value = mask_trace.get("object_mask_crop_path")
                    core_crop_value = mask_trace.get("core_mask_crop_path")
                    composite_trace = _composite_photometric_crop(
                        original_path=image,
                        generated_crop_path=raw_out_path,
                        edit_mask_crop_path=crop_mask,
                        object_mask_crop_path=Path(str(object_crop_value)) if object_crop_value else None,
                        core_mask_crop_path=Path(str(core_crop_value)) if core_crop_value else None,
                        bbox=bbox,  # type: ignore[arg-type]
                        output_path=out_path,
                        edge_feather_px=self.mask_composite_feather_px,
                        core_feather_px=self.core_mask_feather_px,
                        core_protection_px=self.core_protection_px,
                        photometric_blur_px=self.photometric_blur_px,
                        photometric_max_delta=self.photometric_max_delta,
                        photometric_strength=self.photometric_strength,
                        relocation_search_px=self.relocation_search_px,
                        relocation_min_score=self.relocation_min_score,
                        relocation_scale_min=self.relocation_scale_min,
                        relocation_scale_max=self.relocation_scale_max,
                        relocation_scale_steps=self.relocation_scale_steps,
                        wall_sample_exclusion_px=self.wall_sample_exclusion_px,
                        wall_sample_max_delta=self.wall_sample_max_delta,
                        wall_rgb_max_delta=self.wall_rgb_max_delta,
                        wall_rgb_correction_strength=self.wall_rgb_correction_strength,
                        object_edge_guard_px=self.object_edge_guard_px,
                    )
                elif output_scope == "mask_wall_rgb_only_composite":
                    object_crop_value = mask_trace.get("object_mask_crop_path")
                    composite_trace = _composite_wall_rgb_only_crop(
                        original_path=image,
                        generated_crop_path=raw_out_path,
                        edit_mask_crop_path=crop_mask,
                        object_mask_crop_path=Path(str(object_crop_value)) if object_crop_value else None,
                        bbox=bbox,  # type: ignore[arg-type]
                        output_path=out_path,
                        edge_feather_px=self.mask_composite_feather_px,
                        wall_sample_exclusion_px=self.wall_sample_exclusion_px,
                        wall_sample_max_delta=self.wall_sample_max_delta,
                        wall_rgb_max_delta=self.wall_rgb_max_delta,
                        wall_rgb_correction_strength=self.wall_rgb_correction_strength,
                    )
                else:
                    composite_trace = _composite_masked_crop(
                        original_path=image,
                        generated_crop_path=raw_out_path,
                        mask_crop_path=crop_mask,
                        bbox=bbox,  # type: ignore[arg-type]
                        output_path=out_path,
                        feather_px=self.mask_composite_feather_px,
                    )
            else:
                if self.resize_to_input_size:
                    resize_trace = _resize_to_match(image, raw_out_path)
                raw_out_path.replace(out_path)
        finally:
            for handle in handles:
                handle.close()
            try:
                if upload_tmp.exists():
                    for child in upload_tmp.glob("*"):
                        child.unlink(missing_ok=True)
                    upload_tmp.rmdir()
            except Exception:
                pass

        trace = {
            "provider": APIYI_GPT_IMAGE2_VIP_PROVIDER,
            "capability": "gpt_image2_vip_image_edit",
            "endpoint": "/v1/images/edits",
            "base_url": self.base_url,
            "model": self.model,
            "size": size,
            "response_format": self.response_format,
            "response_source": response_source,
            "elapsed_s": round(time.time() - started, 3),
            "output_path": str(out_path),
            "raw_output_path": str(raw_out_path),
            "reference_count": len(references),
            "requested_reference_count": len(requested_references),
            "include_references": self.include_references,
            "references_dropped_by_default": len(requested_references) - len(references),
            "payload_image_keys": ["image"] * len(upload_paths),
            "input_upload_count": len(upload_paths),
            "input_upload_paths": [str(p) for p in upload_paths],
            "target_image_sent_to_api": str(target_for_api),
            "target_image_role": "图1",
            "output_scope": output_scope,
            "mask_sent_to_flux2": False,
            "flux2_mask_sent": False,
            "mask_local_composite": output_scope in {"mask_local_composite", "mask_photometric_composite", "mask_wall_rgb_only_composite"},
            "mask_photometric_composite": output_scope == "mask_photometric_composite",
            "mask_wall_rgb_only_composite": output_scope == "mask_wall_rgb_only_composite",
            "full_space_coordinate_prompt": output_scope == "full_space_coordinate_prompt",
            "mask_enforced_locally": output_scope in {"mask_local_composite", "mask_photometric_composite", "mask_wall_rgb_only_composite"},
            "mask_contract": (
                "API易 gpt-image-2-vip image-edit endpoint has no multipart mask field. "
                "For pure-color walls the adapter enforces bounded repaint locally by cropping the mask bbox and compositing/QA outside the remote request. "
                "For textured candidate walls the adapter sends the full room image and uses a prompt-side pixel coordinate matrix in the exact compressed upload coordinate space. "
                "In photometric crop mode, artwork core pixels are not hard-restored from hard-paste; "
                "they receive only low-frequency exposure/color transfer from the generated crop, while direct generated pixels are limited to the edge/shadow region. "
                "In wall_rgb_only mode, the GPT Image crop is preserved and only wall RGB outside the object mask is corrected."
            ),
            "remote_coordinate_matrix_sent": output_scope == "full_space_coordinate_prompt",
            "mask_adapter": mask_trace,
            "local_composite": composite_trace,
            "apiyi_images_edit_request": True,
            "apiyi_form_fields_sent": sorted(data.keys()),
            "resize_to_input_size": resize_trace,
            "metadata": {k: v for k, v in dict(request.metadata).items() if k not in {"prompt"}},
        }
        return ImageEditResult(image=out_path, trace=trace)

    def _post_images_edit(self, *, data: Mapping[str, Any], files: Iterable[Any]) -> Mapping[str, Any]:
        url = f"{self.base_url}/images/edits"
        file_list = list(files)
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                for _, file_tuple in file_list:
                    try:
                        file_tuple[1].seek(0)
                    except Exception:
                        pass
                resp = requests.post(url, headers=self._headers(), data=dict(data), files=file_list, timeout=self.timeout_s)
                if resp.status_code in {429, 500, 502, 503, 504} and attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.max_retries - 1:
                    break
                time.sleep(2 ** attempt)
        raise RuntimeError(f"APIYI GPT Image request failed after {self.max_retries} attempts: {last_error}")

    @staticmethod
    def _build_prompt(prompt: str, *, reference_count: int, output_scope: str, coordinate_prompt: str = "") -> str:
        if output_scope in {"mask_local_composite", "mask_photometric_composite", "mask_wall_rgb_only_composite"}:
            if output_scope == "mask_photometric_composite":
                prefix = (
                    "修改图1。图1是完整房间画面中挂画区域的局部裁剪。"
                    "图1中的挂画位置、挂画尺寸、透视、裁切、边框/画框外观、边框厚度、边框颜色、白衬/卡纸、画芯内容、人物/图案结构都必须保持不变。"
                    "只允许根据环境做低频、非着色性的光影和明度叠加；不得改变固有色，不得改变任何边框、白衬、画芯或图案的本色。"
                    "不要修改挂画以外的区域，不要改动墙面、窗户、地面或房间结构。"
                    "不要新增、移动、缩放、复制、替换、重绘或重新设计画作，不要把画芯重新生成成另一幅画，也不要把边框改成新的黑边、白边或装饰框。"
                    "如果不确定，请尽量保持现有像素不变。输出保持图1的构图比例，便于回贴到原图。"
                )
            elif output_scope == "mask_wall_rgb_only_composite":
                prefix = (
                    "修改图1。图1是完整房间画面中挂画区域的局部裁剪。"
                    "以图1中已经硬贴好的挂画为准，只做环境光、接触阴影、投影和墙面融合渲染。"
                    "不要重绘、替换、移动、压扁、缩放或重新设计挂画、画框、白衬/卡纸或画芯内容。"
                    "允许墙面上的自然投影和明暗过渡，但不要改变房间结构。"
                    "输出保持图1的构图比例，便于回贴到原图。"
                )
            else:
                prefix = (
                    "修改图1。图1是完整房间画面中挂画区域的局部裁剪。"
                    "保持图1内部挂画位置、挂画尺寸、透视、画芯内容、边框/画框外观、边框固有色和裁切稳定。"
                    "只允许最小化的光照一致性调整，不要改变画作和边框的固有颜色，不要改动挂画以外的区域。"
                    "不要新增、移动、缩放、复制、替换画作，不要改变画芯图案，不要改边框颜色、厚度、样式或开口。"
                    "不要矫正透视畸变、镜头畸变、墙面线条或相机视角。输出保持图1的构图比例，便于回贴到原图。"
                )
        elif output_scope == "full_space_coordinate_prompt":
            prefix = (
                "修改图1。图1是完整房间画面，不是局部裁剪；输出也必须保持完整房间构图和图1比例。"
                "以图1中已经硬贴好的挂画为唯一权威目标，只做自然环境光、接触阴影、投影和边缘融合渲染。"
                "不要重绘、替换、移动、压扁、缩放、拉伸、复制或重新设计挂画、画框、白衬/卡纸或画芯内容。"
                "坐标矩阵限定范围之外的墙面肌理、纹理、光影分界、窗户、地面、天花、家具、结构线和房间内容都必须保持不变。"
                "不要矫正透视畸变、镜头畸变、墙面线条、房间结构或相机视角；保留原照片的自然透视和畸变。"
            )
        else:
            prefix = (
                "修改图1。以图1作为唯一权威画布和最终构图。"
                "保持房间结构、墙面、挂画位置、挂画尺寸、透视、边框/画框外观、边框固有色、画芯内容和画面裁切稳定。"
                "只允许对挂画对象本身进行最小化的低频光影/明度调整，不要改动固有色，也不要改动挂画以外的区域。"
                "不要新增、移动、缩放、复制、替换画作，不要改变边框样式或画芯图案，不要重绘非挂画区域。"
                "不要矫正透视畸变、镜头畸变、墙面线条或相机视角。"
            )
        if reference_count:
            prefix += (
                "图2及后续图片仅作为环境光方向、整体明暗、墙面材质和背景纹理参考；"
                "如果参考图中没有挂画或挂画位置为空，这是正常的 erase 后空间参考，不代表要删除、移动或替换图1中的挂画。"
                "不得让参考图覆盖图1的构图、画作位置、边框形状或画芯内容。"
            )
        if "不要矫正透视畸变" not in prefix:
            prefix += "不要矫正透视畸变、镜头畸变、墙面线条、房间结构或相机视角；保持原照片的自然透视和畸变。"
        return f"{prefix}{coordinate_prompt}\n\n原始渲染指令：{prompt}"


class ApiyiGptImage2VipRenderProvider:
    """Hybrid provider: keep BFL erase, render final image through APIYI gpt-image-2-vip.

    API易 gpt-image-2-vip image-edit endpoint does not expose a mask parameter.
    This adapter keeps the product's bounded-edit contract by using BFL for the
    cleanup mask, then using local mask crop/composite for the final render mask.
    """

    def __init__(
        self,
        *,
        cleanup_provider: BflImageEditProvider | None = None,
        render_client: ApiyiGptImage2Client | None = None,
    ) -> None:
        self.cleanup_provider = cleanup_provider or BflImageEditProvider.from_env()
        self.render_client = render_client or ApiyiGptImage2Client.from_env()

    @classmethod
    def from_env(cls) -> "ApiyiGptImage2VipRenderProvider":
        return cls(cleanup_provider=BflImageEditProvider.from_env(), render_client=ApiyiGptImage2Client.from_env())

    def erase(self, request: ImageEditRequest) -> ImageEditResult:
        return self.cleanup_provider.erase(request)

    def edit(self, request: ImageEditRequest) -> ImageEditResult:
        return self.render_client.edit(request)
