from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageOps


def load_rgba(path_or_bytes) -> np.ndarray:
    if isinstance(path_or_bytes, (bytes, bytearray)):
        import io
        img = Image.open(io.BytesIO(path_or_bytes))
    else:
        img = Image.open(path_or_bytes)
    img = ImageOps.exif_transpose(img).convert("RGBA")
    return np.array(img)


def alpha_blend_rgba(base_rgb: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    if base_rgb.shape[2] == 4:
        base_rgb = base_rgb[:, :, :3]
    alpha = overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0
    out = overlay_rgba[:, :, :3].astype(np.float32) * alpha + base_rgb.astype(np.float32) * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def _effective_supersample_scale(width: int, height: int, requested: int, max_canvas_side: int = 8192) -> int:
    scale = max(1, int(requested or 1))
    if scale <= 1:
        return 1
    max_dim = max(int(width), int(height), 1)
    return max(1, min(scale, max(1, int(max_canvas_side) // max_dim)))


def _quad_bbox_xyxy(points, canvas_w: int, canvas_h: int, pad: int = 3) -> tuple[int, int, int, int] | None:
    pts = np.array(points, np.float32).reshape(-1, 2)
    if pts.size == 0 or not np.isfinite(pts).all():
        return None
    x0 = max(0, int(np.floor(float(np.min(pts[:, 0])))) - int(pad))
    y0 = max(0, int(np.floor(float(np.min(pts[:, 1])))) - int(pad))
    x1 = min(int(canvas_w), int(np.ceil(float(np.max(pts[:, 0])))) + int(pad) + 1)
    y1 = min(int(canvas_h), int(np.ceil(float(np.max(pts[:, 1])))) + int(pad) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _downsample_rgba_premultiplied(rgba_hi: np.ndarray, out_size_wh: tuple[int, int]) -> np.ndarray:
    """Downsample an RGBA image without dark/colored halos around alpha edges."""
    out_w, out_h = int(out_size_wh[0]), int(out_size_wh[1])
    if rgba_hi.shape[1] == out_w and rgba_hi.shape[0] == out_h:
        return rgba_hi.astype(np.uint8, copy=False)
    arr = rgba_hi.astype(np.float32)
    alpha = arr[:, :, 3:4] / 255.0
    premul = arr[:, :, :3] * alpha
    # AREA gives stable sub-pixel coverage for small projected artworks.
    premul_small = cv2.resize(premul, (out_w, out_h), interpolation=cv2.INTER_AREA)
    alpha_small = cv2.resize(alpha, (out_w, out_h), interpolation=cv2.INTER_AREA)
    if alpha_small.ndim == 2:
        alpha_small = alpha_small[:, :, None]
    rgb_small = np.zeros_like(premul_small)
    np.divide(premul_small, np.maximum(alpha_small, 1e-6), out=rgb_small, where=alpha_small > 1e-6)
    out = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    out[:, :, :3] = np.clip(rgb_small, 0, 255).astype(np.uint8)
    out[:, :, 3] = np.clip(alpha_small[:, :, 0] * 255.0, 0, 255).astype(np.uint8)
    return out


def _clip_polygon_with_halfplane(poly: list[list[float]], normal: np.ndarray, threshold: float, *, eps: float = 1e-6) -> list[list[float]]:
    """Sutherland-Hodgman clipping. Keeps points where dot(p, normal) >= threshold."""
    if len(poly) < 3:
        return []
    out: list[list[float]] = []
    prev = np.array(poly[-1], np.float32)
    prev_val = float(np.dot(prev, normal) - threshold)
    prev_inside = prev_val >= -eps
    for cur_list in poly:
        cur = np.array(cur_list, np.float32)
        cur_val = float(np.dot(cur, normal) - threshold)
        cur_inside = cur_val >= -eps
        if cur_inside != prev_inside:
            denom = prev_val - cur_val
            if abs(denom) > eps:
                t = prev_val / denom
                inter = prev + t * (cur - prev)
                out.append([float(inter[0]), float(inter[1])])
        if cur_inside:
            out.append([float(cur[0]), float(cur[1])])
        prev, prev_val, prev_inside = cur, cur_val, cur_inside
    return out


def _side_face_front_span_clip_polygon(side_quad: list[list[float]], *, pad_px: float = 0.35) -> list[list[float]] | None:
    """Clip side-face pixels to the span of the locked front edge.

    Side faces are built as [back_a, back_b, front_b, front_a]. In oblique
    perspective the projected back edge can be a few pixels longer than the
    front edge; when warped directly this appears as a black frame line sticking
    beyond the visible painting.  This clip keeps the side texture inside the
    finite front-edge span while still allowing physical thickness outward.
    """
    q = np.array(side_quad, np.float32)
    if q.shape[0] < 4 or not np.isfinite(q).all():
        return None
    front_a = q[3]
    front_b = q[2]
    edge = front_b - front_a
    edge_len = float(np.linalg.norm(edge))
    if edge_len < 1e-3:
        return None
    tangent = edge / edge_len
    poly = [[float(p[0]), float(p[1])] for p in q]
    # s = dot(p - front_a, tangent). Keep -pad <= s <= edge_len + pad.
    low = float(np.dot(front_a, tangent) - float(pad_px))
    high = float(np.dot(front_a, tangent) + edge_len + float(pad_px))
    poly = _clip_polygon_with_halfplane(poly, tangent, low)
    if len(poly) < 3:
        return None
    poly = _clip_polygon_with_halfplane(poly, -tangent, -high)
    if len(poly) < 3:
        return None
    return poly


def _warp_rgba_overlay_local(
    texture_rgba: np.ndarray,
    dst_quad: list[list[float]],
    canvas_size_hw: tuple[int, int],
    *,
    interpolation: int = cv2.INTER_LANCZOS4,
    supersample_scale: int = 4,
    clip_polygon: Optional[list[list[float]]] = None,
    bbox_pad_px: int = 3,
    max_roi_side_px: int = 8192,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    """Warp texture into a local ROI at high resolution, returning RGBA overlay."""
    h, w = int(canvas_size_hw[0]), int(canvas_size_hw[1])
    th, tw = texture_rgba.shape[:2]
    if th <= 0 or tw <= 0:
        return None, None
    all_pts = list(dst_quad)
    if clip_polygon:
        all_pts += list(clip_polygon)
    bbox = _quad_bbox_xyxy(all_pts, w, h, pad=bbox_pad_px)
    if bbox is None:
        return None, None
    x0, y0, x1, y1 = bbox
    roi_w, roi_h = x1 - x0, y1 - y0
    scale = _effective_supersample_scale(roi_w, roi_h, supersample_scale, max_canvas_side=max_roi_side_px)

    src = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], np.float32)
    dst = np.array(dst_quad, np.float32)
    dst_local = dst - np.array([[x0, y0]], np.float32)

    if scale > 1:
        hi_w, hi_h = int(roi_w * scale), int(roi_h * scale)
        H = cv2.getPerspectiveTransform(src, dst_local * float(scale))
        warped_hi = cv2.warpPerspective(
            texture_rgba,
            H,
            (hi_w, hi_h),
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        if clip_polygon:
            clip = np.zeros((hi_h, hi_w), np.uint8)
            pts = (np.array(clip_polygon, np.float32) - np.array([[x0, y0]], np.float32)) * float(scale)
            cv2.fillPoly(clip, [np.round(pts).astype(np.int32)], 255)
            warped_hi[:, :, 3] = np.minimum(warped_hi[:, :, 3], clip)
        overlay = _downsample_rgba_premultiplied(warped_hi, (roi_w, roi_h))
    else:
        H = cv2.getPerspectiveTransform(src, dst_local)
        overlay = cv2.warpPerspective(
            texture_rgba,
            H,
            (roi_w, roi_h),
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        if clip_polygon:
            clip = np.zeros((roi_h, roi_w), np.uint8)
            pts = np.array(clip_polygon, np.float32) - np.array([[x0, y0]], np.float32)
            cv2.fillPoly(clip, [np.round(pts).astype(np.int32)], 255)
            overlay[:, :, 3] = np.minimum(overlay[:, :, 3], clip)
    return overlay.astype(np.uint8, copy=False), bbox


def warp_artwork_to_quad(
    room_rgb: np.ndarray,
    artwork_rgba: np.ndarray,
    target_quad_original: list[list[float]],
    artwork_mask: Optional[np.ndarray] = None,
    *,
    supersample_scale: int = 4,
    interpolation: int = cv2.INTER_LANCZOS4,
) -> tuple[np.ndarray, np.ndarray]:
    """Perspective-warp artwork into a target quad with anti-aliased edges.

    The warp is performed only in a local high-resolution ROI.  This permits a
    larger effective supersampling factor for small projected artworks without
    upscaling/downscaling the whole room image, so the room background stays
    unchanged and the pasted artwork edges are less jagged.
    """
    h, w = room_rgb.shape[:2]
    overlay, bbox = _warp_rgba_overlay_local(
        artwork_rgba,
        target_quad_original,
        (h, w),
        interpolation=interpolation,
        supersample_scale=supersample_scale,
        bbox_pad_px=max(3, int(supersample_scale or 1)),
    )
    if overlay is None or bbox is None:
        return room_rgb[:, :, :3].copy(), np.zeros((h, w), np.uint8)
    x0, y0, x1, y1 = bbox
    out = room_rgb[:, :, :3].copy()
    out[y0:y1, x0:x1] = alpha_blend_rgba(out[y0:y1, x0:x1], overlay)
    art_mask = np.zeros((h, w), np.uint8)
    art_mask[y0:y1, x0:x1] = (overlay[:, :, 3] > 8).astype(np.uint8) * 255
    return out, art_mask


def save_rgb(path: str, arr: np.ndarray) -> None:
    Image.fromarray(arr).save(path)


def _warp_rgba_to_quad(
    canvas_rgba: np.ndarray,
    texture_rgba: np.ndarray,
    dst_quad: list[list[float]],
    *,
    interpolation: int = cv2.INTER_LANCZOS4,
    supersample_scale: int = 4,
    clip_polygon: Optional[list[list[float]]] = None,
) -> np.ndarray:
    h, w = canvas_rgba.shape[:2]
    overlay, bbox = _warp_rgba_overlay_local(
        texture_rgba,
        dst_quad,
        (h, w),
        interpolation=interpolation,
        supersample_scale=supersample_scale,
        clip_polygon=clip_polygon,
        bbox_pad_px=max(3, int(supersample_scale or 1)),
    )
    if overlay is None or bbox is None:
        return canvas_rgba
    x0, y0, x1, y1 = bbox
    out = canvas_rgba.copy()
    roi = out[y0:y1, x0:x1]
    alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
    out_rgb = overlay[:, :, :3].astype(np.float32) * alpha + roi[:, :, :3].astype(np.float32) * (1.0 - alpha)
    out_alpha = np.maximum(roi[:, :, 3:4], overlay[:, :, 3:4])
    out[y0:y1, x0:x1] = np.concatenate([np.clip(out_rgb, 0, 255).astype(np.uint8), out_alpha.astype(np.uint8)], axis=2)
    return out


def _edge_side_texture(artwork_rgba: np.ndarray, source_edge: str, sample_px: int, *, side_shading_strength: float = 0.18, average_depth_px: int | None = None) -> np.ndarray:
    ah, aw = artwork_rgba.shape[:2]
    sample = max(1, min(int(sample_px or 1), max(1, min(aw, ah) // 2)))
    edge = str(source_edge or "right").lower()
    if edge == "left":
        tex = artwork_rgba[:, :sample, :].copy()
    elif edge == "right":
        tex = artwork_rgba[:, max(0, aw - sample):, :].copy()
    elif edge == "top":
        tex = artwork_rgba[:sample, :, :].copy()
    elif edge == "bottom":
        tex = artwork_rgba[max(0, ah - sample):, :, :].copy()
    else:
        tex = artwork_rgba[:, max(0, aw - sample):, :].copy()

    # Side-face color must come from the original artwork outer edge itself,
    # not from a warped hard-paste strip and not from a deep strip that can
    # accidentally include inner mat / liner pixels after small downscales.
    # We therefore estimate a single intrinsic edge color from only the
    # outermost edge band of the original artwork, then build a solid side-face
    # texture from that mean color and apply deterministic shading on top.
    avg_depth = max(1, min(int(average_depth_px or max(1, min(2, sample))), sample))
    if edge == "left":
        avg_band = artwork_rgba[:, :avg_depth, :].copy()
    elif edge == "right":
        avg_band = artwork_rgba[:, max(0, aw - avg_depth):, :].copy()
    elif edge == "top":
        avg_band = artwork_rgba[:avg_depth, :, :].copy()
    elif edge == "bottom":
        avg_band = artwork_rgba[max(0, ah - avg_depth):, :, :].copy()
    else:
        avg_band = artwork_rgba[:, max(0, aw - avg_depth):, :].copy()

    band = avg_band.astype(np.float32)
    alpha = band[:, :, 3:4] / 255.0
    weight = np.maximum(alpha, 1e-6)
    denom = np.sum(weight)
    if denom <= 1e-6:
        mean_rgb = np.mean(band[:, :, :3].reshape(-1, 3), axis=0)
        mean_alpha = 255.0
    else:
        mean_rgb = np.sum(band[:, :, :3] * weight, axis=(0, 1)) / denom
        mean_alpha = float(np.sum(band[:, :, 3:4]) / np.maximum(band.shape[0] * band.shape[1], 1))

    solid = np.zeros_like(tex, dtype=np.float32)
    solid[:, :, :3] = mean_rgb.reshape(1, 1, 3)
    solid[:, :, 3] = mean_alpha
    tex = solid
    shade_strength = max(0.0, min(0.65, float(side_shading_strength or 0.0)))
    # Product lighting convention for deterministic side precompose:
    # keep side faces overall darker than the artwork front, but make the
    # edge adjacent to the visible front face slightly brighter than the
    # outer/wall-contact edge. This avoids relying on an implicit left-to-right
    # gradient whose physical meaning changes by source edge.
    if edge in {"left", "right"}:
        steps = tex.shape[1]
        if edge == "right":
            # right strip columns run artwork-interior -> outer edge.
            values = np.linspace(1.0, 1.0 - shade_strength, max(1, steps), dtype=np.float32)
        else:
            # left strip columns run outer edge -> artwork-interior.
            values = np.linspace(1.0 - shade_strength, 1.0, max(1, steps), dtype=np.float32)
        shade = values.reshape(1, steps, 1)
    else:
        steps = tex.shape[0]
        if edge == "bottom":
            # bottom strip rows run artwork-interior -> outer edge.
            values = np.linspace(1.0, 1.0 - shade_strength, max(1, steps), dtype=np.float32)
        else:
            # top strip rows run outer edge -> artwork-interior.
            values = np.linspace(1.0 - shade_strength, 1.0, max(1, steps), dtype=np.float32)
        shade = values.reshape(steps, 1, 1)
    tex[:, :, :3] *= shade
    return np.clip(tex, 0, 255).astype(np.uint8)


def _mask_from_quad(size_hw: tuple[int, int], quad: list[list[float]]) -> np.ndarray:
    h, w = size_hw
    mask = np.zeros((h, w), np.uint8)
    pts = np.array(quad, np.float32)
    if pts.shape[0] >= 3 and np.isfinite(pts).all():
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 255)
    return mask


def _mask_from_polygon(size_hw: tuple[int, int], polygon: list[list[float]]) -> np.ndarray:
    h, w = size_hw
    mask = np.zeros((h, w), np.uint8)
    pts = np.array(polygon, np.float32)
    if pts.shape[0] >= 3 and np.isfinite(pts).all():
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 255)
    return mask


def warp_artwork_mount_to_geometry(
    room_rgb: np.ndarray,
    artwork_rgba: np.ndarray,
    front_quad_original: list[list[float]],
    side_faces_original: list[dict],
    *,
    edge_sample_px: int = 8,
    edge_average_depth_px: int | None = None,
    side_shading_strength: float = 0.18,
    supersample_scale: int = 4,
    clip_side_faces_to_front_span: bool = True,
    side_face_span_pad_px: float = 0.35,
) -> dict:
    """Warp deterministic artwork front face plus visible frame side faces.

    The side faces use pixels sampled from the corresponding source edge of the
    artwork image. `source_edge` in each side-face record refers to artwork UV
    space, not image-screen left/right after projection.
    """
    h, w = room_rgb.shape[:2]
    canvas = np.zeros((h, w, 4), dtype=np.uint8)
    canvas[:, :, :3] = room_rgb[:, :, :3]
    canvas[:, :, 3] = 255
    side_mask = np.zeros((h, w), np.uint8)
    used_faces = []

    # Draw side faces before the front face, so the artwork front remains the
    # authoritative core image and cannot be covered by side texture.
    for face in side_faces_original or []:
        quad = face.get("quad") if isinstance(face, dict) else None
        if not quad or len(quad) < 4:
            continue
        edge = str(face.get("source_edge") or face.get("edge") or "right")
        clip_polygon = None
        if clip_side_faces_to_front_span:
            clip_polygon = _side_face_front_span_clip_polygon(quad, pad_px=side_face_span_pad_px)
            if clip_polygon is None or len(clip_polygon) < 3:
                continue
        tex = _edge_side_texture(
            artwork_rgba,
            edge,
            int(edge_sample_px),
            side_shading_strength=side_shading_strength,
            average_depth_px=edge_average_depth_px,
        )
        canvas = _warp_rgba_to_quad(canvas, tex, quad, supersample_scale=supersample_scale, clip_polygon=clip_polygon)
        fm = _mask_from_polygon((h, w), clip_polygon) if clip_polygon else _mask_from_quad((h, w), quad)
        side_mask = cv2.bitwise_or(side_mask, fm)
        used = dict(face)
        used["source_edge"] = edge
        if clip_polygon:
            used["clip_polygon_original"] = [[float(p[0]), float(p[1])] for p in clip_polygon]
            used["side_face_clip_policy"] = "front_edge_span_limited"
            used["side_face_span_pad_px"] = float(side_face_span_pad_px)
        used_faces.append(used)

    front_paste, front_mask = warp_artwork_to_quad(
        canvas[:, :, :3],
        artwork_rgba,
        front_quad_original,
        supersample_scale=supersample_scale,
    )
    image = front_paste
    side_mask[front_mask > 0] = 0
    mount_mask = cv2.bitwise_or(front_mask, side_mask)
    return {
        "image": image,
        "front_mask": ((front_mask > 0).astype(np.uint8) * 255),
        "side_mask": ((side_mask > 0).astype(np.uint8) * 255),
        "mount_mask": ((mount_mask > 0).astype(np.uint8) * 255),
        "side_faces_used": used_faces,
        "debug": {
            "mode": "precomposed_frame_sides",
            "side_face_count": len(used_faces),
            "front_mask_area_px": int(np.count_nonzero(front_mask)),
            "side_mask_area_px": int(np.count_nonzero(side_mask)),
            "mount_mask_area_px": int(np.count_nonzero(mount_mask)),
            "edge_sample_px": int(edge_sample_px),
            "edge_average_depth_px": int(edge_average_depth_px if edge_average_depth_px is not None else max(1, min(2, int(edge_sample_px) if int(edge_sample_px) > 0 else 1))),
            "side_texture_policy": "solid_texture_from_original_artwork_outer_edge_average_color",
            "side_shading_strength": float(side_shading_strength),
            "side_shading_policy": "side overall darker; artwork-front-adjacent edge brighter than outer/wall-contact edge",
            "clip_side_faces_to_front_span": bool(clip_side_faces_to_front_span),
            "side_face_span_pad_px": float(side_face_span_pad_px),
            "supersample_mode": "local_roi_rgba_premultiplied_downsample",
            "supersample_scale": int(supersample_scale),
        },
    }
