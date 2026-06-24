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

        orig_arr = np.asarray(original_crop, dtype=np.float32)
        delta = _low_frequency_luma_delta(
            original_crop,
            generated_rgb,
            blur_px=photometric_blur_px,
            max_delta=photometric_max_delta,
            strength=photometric_strength,
        )
        photometric_arr = np.clip(orig_arr + delta, 0, 255)

        # The visible artwork object is geometry / intrinsic-color locked: it
        # receives only low-frequency achromatic lightness transfer, so GPT
        # Image cannot squash, recolor, or repaint the pasted artwork.
        object_w = np.clip(object_alpha * edit_alpha, 0.0, 1.0)[..., None]

        # Outside the artwork object but still inside the edit mask, keep the
        # model-generated wall/contact-shadow pixels. This is where GPT Image
        # usually produces the best cast shadow and wall integration, and it
        # must not be pasted back to the hard-paste wall.
        shadow_w = np.clip(edit_alpha - object_alpha, 0.0, 1.0)[..., None]
        out_arr = (
            orig_arr * (1.0 - np.clip(object_w + shadow_w, 0.0, 1.0))
            + photometric_arr * object_w
            + np.asarray(generated_rgb, dtype=np.float32) * shadow_w
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
        "edge_treatment": "generated_shadow_pixels_preserved_outside_artwork_object_within_edit_mask",
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
            mask_edit_mode=os.getenv("APIYI_GPT_IMAGE_MASK_EDIT_MODE", "photometric_composite").strip().lower() or "photometric_composite",
            mask_crop_padding_px=max(0, int(os.getenv("APIYI_GPT_IMAGE_MASK_CROP_PADDING_PX", "48"))),
            mask_composite_feather_px=max(0, int(os.getenv("APIYI_GPT_IMAGE_MASK_COMPOSITE_FEATHER_PX", "2"))),
            photometric_blur_px=max(1, int(os.getenv("APIYI_GPT_IMAGE_PHOTOMETRIC_BLUR_PX", "18"))),
            photometric_max_delta=max(0.0, float(os.getenv("APIYI_GPT_IMAGE_PHOTOMETRIC_MAX_DELTA", "32"))),
            photometric_strength=max(0.0, float(os.getenv("APIYI_GPT_IMAGE_PHOTOMETRIC_STRENGTH", "1.0"))),
            core_mask_feather_px=max(0, int(os.getenv("APIYI_GPT_IMAGE_CORE_MASK_FEATHER_PX", "1"))),
            core_protection_px=max(0, int(os.getenv("APIYI_GPT_IMAGE_CORE_PROTECTION_PX", "3"))),
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

        if request.mask and self.mask_edit_mode in {
            "photometric_composite",
            "photometric_core_edge_composite",
            "structure_preserving_composite",
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
            photometric_mode = self.mask_edit_mode in {
                "photometric_composite",
                "photometric_core_edge_composite",
                "structure_preserving_composite",
            }
            output_scope = "mask_photometric_composite" if photometric_mode else "mask_local_composite"
            object_mask_crop_path = None
            if photometric_mode and object_mask_path is not None:
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
                "mask_enforcement": "local_crop_then_photometric_artwork_lock_and_shadow_composite" if photometric_mode else "local_crop_then_mask_composite",
                "mask_enforced_locally": True,
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

        upload_paths = [
            _maybe_compress_upload(target_for_api, upload_tmp, max_bytes=self.max_input_bytes, enabled=self.compress_inputs),
            *[
                _maybe_compress_upload(ref, upload_tmp, max_bytes=self.max_input_bytes, enabled=self.compress_inputs)
                for ref in references
            ],
        ]

        prompt = self._build_prompt(request.prompt, reference_count=len(references), output_scope=output_scope)
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

            if output_scope in {"mask_local_composite", "mask_photometric_composite"}:
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
            "mask_local_composite": output_scope in {"mask_local_composite", "mask_photometric_composite"},
            "mask_photometric_composite": output_scope == "mask_photometric_composite",
            "mask_enforced_locally": output_scope in {"mask_local_composite", "mask_photometric_composite"},
            "mask_contract": (
                "API易 gpt-image-2-vip image-edit endpoint has no multipart mask field; "
                "the adapter enforces bounded repaint by cropping the mask bbox and compositing locally. "
                "In photometric mode, artwork core pixels are not hard-restored from hard-paste; "
                "they receive only low-frequency exposure/color transfer from the generated crop, while direct generated pixels are limited to the edge/shadow region."
            ),
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
    def _build_prompt(prompt: str, *, reference_count: int, output_scope: str) -> str:
        if output_scope in {"mask_local_composite", "mask_photometric_composite"}:
            if output_scope == "mask_photometric_composite":
                prefix = (
                    "修改图1。图1是完整房间画面中挂画区域的局部裁剪。"
                    "图1中的挂画位置、挂画尺寸、透视、裁切、边框/画框外观、边框厚度、边框颜色、白衬/卡纸、画芯内容、人物/图案结构都必须保持不变。"
                    "只允许根据环境做低频、非着色性的光影和明度叠加；不得改变固有色，不得改变任何边框、白衬、画芯或图案的本色。"
                    "不要修改挂画以外的区域，不要改动墙面、窗户、地面或房间结构。"
                    "不要新增、移动、缩放、复制、替换、重绘或重新设计画作，不要把画芯重新生成成另一幅画，也不要把边框改成新的黑边、白边或装饰框。"
                    "如果不确定，请尽量保持现有像素不变。输出保持图1的构图比例，便于回贴到原图。"
                )
            else:
                prefix = (
                    "修改图1。图1是完整房间画面中挂画区域的局部裁剪。"
                    "保持图1内部挂画位置、挂画尺寸、透视、画芯内容、边框/画框外观、边框固有色和裁切稳定。"
                    "只允许最小化的光照一致性调整，不要改变画作和边框的固有颜色，不要改动挂画以外的区域。"
                    "不要新增、移动、缩放、复制、替换画作，不要改变画芯图案，不要改边框颜色、厚度、样式或开口。"
                    "输出保持图1的构图比例，便于回贴到原图。"
                )
        else:
            prefix = (
                "修改图1。以图1作为唯一权威画布和最终构图。"
                "保持房间结构、墙面、挂画位置、挂画尺寸、透视、边框/画框外观、边框固有色、画芯内容和画面裁切稳定。"
                "只允许对挂画对象本身进行最小化的低频光影/明度调整，不要改动固有色，也不要改动挂画以外的区域。"
                "不要新增、移动、缩放、复制、替换画作，不要改变边框样式或画芯图案，不要重绘非挂画区域。"
            )
        if reference_count:
            prefix += "图2及后续图片仅作为光照、材质、背景纹理或艺术品身份参考，不得覆盖图1构图。"
        return f"{prefix}\n\n原始渲染指令：{prompt}"


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
