from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class CandidateFluxAsset:
    asset_id: str
    prefix: str
    hard_paste: np.ndarray
    art_region_mask: np.ndarray
    remove_mask: np.ndarray
    debug_overlay: np.ndarray
    render_prompt: str
    manifest_entry: dict[str, Any]


def safe_asset_id(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value).strip())
    return value.strip("_") or "candidate"


def resize_mask_to_original(mask: np.ndarray, original_size_wh: tuple[int, int]) -> np.ndarray:
    ow, oh = original_size_wh
    if mask.shape[:2] == (oh, ow):
        out = mask.copy()
    else:
        out = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)
    return ((out > 0).astype(np.uint8) * 255)


def union_original_masks(items: Iterable[Mapping[str, Any]], original_size_wh: tuple[int, int]) -> np.ndarray:
    ow, oh = original_size_wh
    out = np.zeros((oh, ow), np.uint8)
    for item in items:
        mask = item.get("mask")
        if mask is None:
            continue
        out = cv2.bitwise_or(out, resize_mask_to_original(mask, original_size_wh))
    return out


def build_remove_mask_for_hardpaste(
    *,
    art_region_mask: np.ndarray,
    current_wall_mask: np.ndarray | None,
    existing_masks: Iterable[Mapping[str, Any]],
    foreground_masks: Iterable[Mapping[str, Any]],
    original_size_wh: tuple[int, int],
    remove_foreground_global: bool = True,
    dilate_px: int = 6,
    protect_art_region: bool = True,
) -> np.ndarray:
    """Build a candidate-scoped erase mask.

    Existing wall content is limited to the candidate wall.  Foreground objects
    are similarly constrained to the candidate wall area so that off-wall
    furniture (chair, sofa, table) is not unnecessarily erased.  People and
    objects in front of the wall still safely overlap the wall mask and are
    correctly included in the erase scope.

    For the legacy hard_paste -> erase order, protect_art_region=True avoids
    erasing the newly pasted artwork. For the corrected v7 order
    original_room -> erase -> hard_paste, protect_art_region must be False so
    old artwork/wall clutter underneath the target artwork area is removed.

    Note on dilation: this function may apply local morphological dilation
    before remove_mask.png is exported. The BFL erase-v1 call can then apply
    its own server-side dilate_pixels value. The combined expansion is
    intentionally documented in rules.cleanup_policy.remove_mask_dilation_policy
    because it affects boundary debugging around large existing artworks.
    """
    ow, oh = original_size_wh
    remove = np.zeros((oh, ow), np.uint8)
    existing = union_original_masks(existing_masks, original_size_wh)
    if current_wall_mask is not None:
        wall = resize_mask_to_original(current_wall_mask, original_size_wh)
        existing = cv2.bitwise_and(existing, wall)
    remove = cv2.bitwise_or(remove, existing)
    if remove_foreground_global:
        fg = union_original_masks(foreground_masks, original_size_wh)
        # Constrain foreground erase to the candidate wall area, matching the
        # behaviour for existing-wall-content masks above.  This prevents
        # foreground objects (chair, sofa, table) that sit away from the
        # target wall from being unnecessarily erased.  People in front of
        # the wall still safely overlap the wall mask.
        if current_wall_mask is not None:
            fg = cv2.bitwise_and(fg, wall)
        remove = cv2.bitwise_or(remove, fg)
    if dilate_px > 0 and np.count_nonzero(remove):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        remove = cv2.dilate(remove, k)
    if protect_art_region:
        protected_art = ((art_region_mask > 0).astype(np.uint8) * 255)
        if protected_art.shape[:2] != (oh, ow):
            protected_art = cv2.resize(protected_art, (ow, oh), interpolation=cv2.INTER_NEAREST)
        remove[protected_art > 0] = 0
    return ((remove > 0).astype(np.uint8) * 255)




def build_flux2_edit_mask(
    *,
    mount_mask: np.ndarray,
    original_size_wh: tuple[int, int],
    dilate_px: int = 24,
) -> np.ndarray:
    """Build the mandatory FLUX.2 edit-region mask.

    The mask is deliberately scoped to the mounted artwork/front-frame region
    plus a small halo for contact shadow and edge blending.  Erase may clean
    global foreground/person masks before hard-paste, but final FLUX.2 must not
    receive an unconstrained whole-room edit request.
    """
    ow, oh = original_size_wh
    mask = resize_mask_to_original(mount_mask, original_size_wh)
    if dilate_px > 0 and np.count_nonzero(mask):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(dilate_px) + 1, 2 * int(dilate_px) + 1))
        mask = cv2.dilate(mask, k)
    if mask.shape[:2] != (oh, ow):
        mask = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)
    return ((mask > 0).astype(np.uint8) * 255)


def _coerce_int_bbox(value: Any) -> list[int] | None:
    if not isinstance(value, Sequence) or len(value) < 4:
        return None
    try:
        return [int(round(float(value[i]))) for i in range(4)]
    except Exception:
        return None


def _coerce_int_polygon(value: Any) -> list[list[int]] | None:
    if not isinstance(value, Sequence) or len(value) < 4:
        return None
    pts: list[list[int]] = []
    try:
        for pt in value[:4]:
            if not isinstance(pt, Sequence) or len(pt) < 2:
                return None
            pts.append([int(round(float(pt[0]))), int(round(float(pt[1])))])
    except Exception:
        return None
    return pts


def _quad_bbox_and_polygon(candidate: Mapping[str, Any]) -> tuple[list[int] | None, list[list[int]] | None, list[int] | None]:
    # Prefer the export-time, mask-derived front bbox/polygon.  For framed
    # artwork, candidate.projection may describe the wall/back-face anchor,
    # while hard-paste QA locks the immutable visible front face.
    export = candidate.get("bfl_v7_export") if isinstance(candidate.get("bfl_v7_export"), Mapping) else {}
    bbox = _coerce_int_bbox(export.get("front_bbox_px") or export.get("bbox_px") or candidate.get("front_bbox_px") or candidate.get("bbox_px"))
    mount_bbox = _coerce_int_bbox(export.get("mount_bbox_px") or candidate.get("mount_bbox_px"))
    polygon = _coerce_int_polygon(export.get("front_polygon_px") or export.get("polygon_px") or candidate.get("front_polygon_px") or candidate.get("polygon_px"))
    if bbox is not None or polygon is not None:
        return bbox, polygon, mount_bbox

    projection = candidate.get("projection", {}) if isinstance(candidate.get("projection"), Mapping) else {}
    quad = projection.get("projected_quad_px_original") or candidate.get("target_quad_original")
    polygon = _coerce_int_polygon(quad)
    if polygon is None:
        return None, None, mount_bbox
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
    return bbox, polygon, mount_bbox


def _bool_candidate_value(candidate: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = candidate.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def build_render_prompt(
    candidate: Mapping[str, Any],
    *,
    extra_prompt: str | None = None,
    artwork_has_frame: bool | None = None,
    artwork_mount_policy: Mapping[str, Any] | None = None,
    flux2_input_mode: str | None = None,
) -> str:
    """Build a full-context render prompt with an immutable artwork anchor.

    Geometry and artwork identity are hard constraints.  Lighting, contact
    shadow, frame/side material response, edge occlusion and low-frequency scene
    adaptation are positive requirements.  The prompt intentionally does not
    ask a full-frame provider to behave like a pixel-locked local inpaint.
    """
    risk = candidate.get("risk_level") or candidate.get("hard_constraints", {}).get("risk_assessment", {}).get("risk_level")
    bbox, polygon, mount_bbox = _quad_bbox_and_polygon(candidate)
    frame_known = _bool_candidate_value(candidate, "artwork_has_frame", False) if artwork_has_frame is None else bool(artwork_has_frame)
    input_mode = str(flux2_input_mode or candidate.get("flux2_input_mode") or "single_after_erase_hardpaste")

    if input_mode == "dual_erased_room_plus_target":
        input_contract = (
            "Image 1 is the cleaned hard-paste target and is the authoritative final composition. "
            "Image 2 is the already-erased clean room reference and may be used only to infer wall material, ambient light, and background continuity. "
            "Never restore removed people, old art, labels, hooks, shelves, or clutter from Image 2. "
        )
    elif input_mode == "dual_original_plus_target":
        input_contract = (
            "Legacy dual-original input is not authoritative. Treat Image 1 as the only cleaned target composition. "
        )
    else:
        input_contract = "Image 1 is the cleaned hard-paste target and is the authoritative final composition. "

    region_parts = []
    if bbox is not None:
        region_parts.append(f"front_bbox_px={bbox}")
    if polygon is not None:
        region_parts.append(f"front_polygon_px={polygon}")
    if mount_bbox is not None and mount_bbox != bbox:
        region_parts.append(f"mount_bbox_px={mount_bbox}")
    region_text = "; ".join(region_parts) if region_parts else "the visible pasted artwork footprint in Image 1"

    mount_policy = artwork_mount_policy if isinstance(artwork_mount_policy, Mapping) else (candidate.get("artwork_mount_policy") if isinstance(candidate.get("artwork_mount_policy"), Mapping) else {})
    visible_side_faces = mount_policy.get("visible_side_faces_original") if isinstance(mount_policy.get("visible_side_faces_original"), Sequence) else []
    visible_side_edges = [
        str(face.get("source_edge"))
        for face in visible_side_faces
        if isinstance(face, Mapping) and face.get("source_edge") in {"left", "right", "top", "bottom"}
    ]
    hidden_side_edges = [edge for edge in ("left", "right", "top", "bottom") if edge not in visible_side_edges]
    visible_side_text = ", ".join(visible_side_edges) if visible_side_edges else "the already visible side faces"
    hidden_side_text = ", ".join(hidden_side_edges) if hidden_side_edges else "the remaining perimeter"

    if mount_policy.get("precompose_frame_sides"):
        frame_rule = (
            "The hard-paste already contains deterministic 3D frame side faces with the correct visible thickness geometry. "
            f"Only these side faces are visible and allowed to remain visible: {visible_side_text}. "
            f"Do not invent a perimeter border or thickness strip on hidden edges such as {hidden_side_text}. "
            "Do not add, redraw, thicken, thin, recolor, restyle, or replace the existing frame/side geometry, and do not add mat board. "
            "Relight the existing side faces and frame material naturally, but do not alter their geometry, opening, or thickness. "
        )
    elif frame_known:
        frame_rule = (
            "The artwork image contains a visible frame or border. Keep that frame exactly. Do not add mat board. "
            "Preserve the existing physical frame geometry, opening, and identity. "
            "You may change only its physically plausible illumination, highlight, roughness appearance, and contact shadow. "
        )
    else:
        frame_rule = (
            "Do not invent a decorative frame or mat. If the artwork is frameless, retain a subtle plausible canvas/object edge only where supported by the current geometry. "
        )

    base = (
        "Task: turn the hard-pasted artwork into a convincing physically mounted object using the complete room context. "
        f"{input_contract}"
        f"PROTECTED ARTWORK ANCHOR: {region_text}. "
        "Immutable geometry: keep the artwork center, four projected corners, outer footprint, scale, aspect ratio, crop, and perspective fixed. "
        "Immutable identity: keep the painting composition, subjects, symbols, text, and high-frequency structure recognizable and unchanged. "
        "STRICT SIZE LOCK: Do not change the painting core size. Do not move, resize, rotate, stretch, duplicate, replace, or redesign the artwork. "
        f"{frame_rule}"
        "ALLOWED CHANGES: physically plausible low-frequency exposure and color-temperature adaptation on the artwork core; "
        "frame and visible-side relighting; material highlights and roughness response; narrow edge occlusion; wall bounce; and contact/cast shadow around the mount. "
        "REQUIRED RESULT: create a visible but restrained contact shadow consistent with the room lights, make frame/side lighting agree with the wall, "
        "and remove the flat pasted-on appearance. A near-no-op result is not acceptable. "
        "Preserve room identity, architecture, furniture geometry, camera viewpoint, lens perspective, and existing structural lines. "
        "Do not redesign the room or introduce new decorations during this render pass. "
        "Do not correct perspective distortion, lens distortion, wall lines, room structure, or camera viewpoint. "
        "FINAL SIZE LOCK AFTER FRAME RULE: keep the painting core footprint and projected boundary fixed after all frame and lighting work. "
    )
    metadata = f" Candidate id: {candidate.get('candidate_id')}; wall id: {candidate.get('wall_id')}; risk level: {risk}; flux2_input_mode: {input_mode}."
    if extra_prompt:
        return f"{base} {extra_prompt.strip()} {metadata}"
    return f"{base} {metadata}"


def build_stage_b_prompt(
    soft_furnishing: bool,
    wallpaper: Mapping[str, Any] | None = None,
    styling_analysis: Mapping[str, Any] | None = None,
    artwork_analysis: Mapping[str, Any] | None = None,
) -> str:
    """Build one Stage B prompt from structured styling and request switches."""
    styling_analysis = styling_analysis if isinstance(styling_analysis, Mapping) else {}
    artwork_analysis = artwork_analysis if isinstance(artwork_analysis, Mapping) else {}
    wallpaper = wallpaper if isinstance(wallpaper, Mapping) else {}
    wallpaper_enabled = bool(wallpaper.get("enabled"))
    if not soft_furnishing and not wallpaper_enabled:
        return ""

    parts = [
        "Stage B full-space styling edit. Use the complete room image as the creative canvas; no spatial edit mask is supplied.",
        "IMMUTABLE ARTWORK ANCHOR: keep the mounted artwork center, four corners, footprint, scale, perspective, composition, and frame identity fixed. Do not move, replace, duplicate, or repaint it.",
        "Preserve doors, windows, wall/ceiling/floor geometry, major furniture geometry, architectural lines, camera viewpoint, and perspective.",
        "Create a clearly visible but coherent styling result. A no-op or near-identical return is not acceptable.",
    ]
    next_number = 4
    if soft_furnishing:
        tension = str(styling_analysis.get("visual_tension_level") or "medium")
        try:
            max_artwork_m = max(float(artwork_analysis.get("physical_width_m") or 0.0), float(artwork_analysis.get("physical_height_m") or 0.0))
        except Exception:
            max_artwork_m = 0.0
        if tension == "high" or max_artwork_m >= 1.20:
            count_text = "two or three"
        elif tension == "low" and (max_artwork_m <= 0.60 or max_artwork_m == 0.0):
            count_text = "one"
        else:
            count_text = "one or two"
        accents = styling_analysis.get("accent_colors_abstracted_rgb") or []
        neutral = styling_analysis.get("neutral_pairing_rgb") or [132, 128, 120]
        texture = str(styling_analysis.get("texture_character") or "uncertain")
        style = str(styling_analysis.get("style_register") or "uncertain")
        line = str(styling_analysis.get("dominant_line_direction") or "none")
        material_words = {
            "smooth_flat": "matte ceramic, smooth linen, or restrained lacquer",
            "heavy_impasto": "textured linen, tactile ceramic, or woven fibers",
            "graphic_hard_edge": "clean geometric forms with crisp but non-identical line relationships",
            "soft_atmospheric": "soft linen, translucent glass, or gentle natural fibers",
            "metallic_industrial": "brushed metal, dark ceramic, or restrained industrial textiles",
            "uncertain": "quiet natural materials with low visual noise",
        }.get(texture, "quiet natural materials with low visual noise")
        style_words = {
            "minimalist_modern": "minimalist modern",
            "mid_century": "restrained mid-century",
            "classical_ornate": "warm classical",
            "industrial": "refined industrial",
            "coastal_natural": "coastal natural",
            "eclectic_mixed": "controlled eclectic",
            "uncertain": "stylistically neutral",
        }.get(style, "stylistically neutral")
        palette_text = f"abstracted accent RGB values {accents[:2]} and neutral RGB {neutral}"
        parts.extend([
            f"{next_number}. Add {count_text} restrained small accessories anywhere in the complete room where a physically valid floor, sofa, chair, shelf, or tabletop support exists. Keep the artwork as the dominant focal point.",
            f"{next_number + 1}. Allowed categories are cushions, a small plant or floral arrangement, a small object, a table runner, or a small rug. Do not add or replace large furniture.",
            f"{next_number + 2}. Use a {style_words} vocabulary and materials such as {material_words}. Coordinate through {palette_text}.",
            f"{next_number + 3}. Do not copy the artwork's image, recognizable pattern, composition, figures, symbols, or color-block layout onto any accessory. Only use abstracted color, texture, shape, and line relationships.",
        ])
        next_number += 4
        if line in {"horizontal", "vertical", "diagonal", "radial"}:
            parts.append(
                f"{next_number}. Where an existing textile support surface makes it natural, use only a subtle {line} texture direction; never reproduce the artwork's actual lines or layout."
            )
            next_number += 1
        if bool(styling_analysis.get("is_abstract")):
            parts.append(f"{next_number}. Because the artwork is abstract, avoid decorative accumulation; use only the minimum pieces needed for support.")
            next_number += 1
    if wallpaper_enabled:
        tone = wallpaper.get("tone_rgb")
        if not (isinstance(tone, Sequence) and len(tone) >= 3):
            tone = [58, 62, 71]
        try:
            rgb = [max(0, min(255, int(round(float(v))))) for v in list(tone)[:3]]
        except (TypeError, ValueError):
            rgb = [58, 62, 71]
        parts.extend([
            f"{next_number}. Recolor only the existing wall surfaces to rgb({rgb[0]}, {rgb[1]}, {rgb[2]}), preserving each wall's illumination gradients, shadows, seams, reflections, and texture instead of applying a flat fill.",
            f"{next_number + 1}. Keep baseboards, corners, switches, outlets, seams, and material grain unchanged. Add no wallpaper pattern, molding, graphic, or wall decoration.",
        ])
        next_number += 2
        if soft_furnishing:
            parts.append(
                f"{next_number}. Keep the new accessories harmonious with rgb({rgb[0]}, {rgb[1]}, {rgb[2]}) without reducing the artwork's visual priority."
            )
    if artwork_analysis.get("has_frame") is True:
        parts.append("Final artwork anchor: preserve the complete visible physical frame, opening, thickness geometry, footprint, and artwork identity; do not move or redesign it.")
    else:
        parts.append("Final artwork anchor: preserve the complete artwork footprint and identity; do not move it or invent a frame or mat.")
    return "\n".join(parts)


def _entry_center_and_width(entry: Mapping[str, Any]) -> tuple[float, float, float]:
    bbox = entry.get("bbox_px")
    if isinstance(bbox, Sequence) and len(bbox) >= 4:
        x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
        return 0.5 * (x0 + x1), 0.5 * (y0 + y1), max(1.0, abs(x1 - x0))
    quad = entry.get("polygon_px") or entry.get("target_quad_original")
    if isinstance(quad, Sequence) and len(quad) >= 4:
        xs = [float(p[0]) for p in quad[:4] if isinstance(p, Sequence) and len(p) >= 2]
        ys = [float(p[1]) for p in quad[:4] if isinstance(p, Sequence) and len(p) >= 2]
        if xs and ys:
            return sum(xs) / len(xs), sum(ys) / len(ys), max(1.0, max(xs) - min(xs))
    return 0.0, 0.0, 1.0


def _entry_polygon(entry: Mapping[str, Any]) -> list[list[int]] | None:
    polygon = _coerce_int_polygon(entry.get("front_polygon_px") or entry.get("polygon_px") or entry.get("target_quad_original"))
    if polygon is not None:
        return polygon
    bbox = _coerce_int_bbox(entry.get("front_bbox_px") or entry.get("bbox_px"))
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _bbox_iou(a: list[int] | None, b: list[int] | None) -> float:
    if a is None or b is None:
        return 0.0
    ax0, ay0, ax1, ay1 = [float(v) for v in a[:4]]
    bx0, by0, bx1, by1 = [float(v) for v in b[:4]]
    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = inter_w * inter_h
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def _polygon_iou_px(poly_a: list[list[int]] | None, poly_b: list[list[int]] | None) -> float:
    if poly_a is None or poly_b is None:
        return 0.0
    arr_a = np.array(poly_a[:4], np.float32)
    arr_b = np.array(poly_b[:4], np.float32)
    if arr_a.shape[0] < 3 or arr_b.shape[0] < 3 or not (np.isfinite(arr_a).all() and np.isfinite(arr_b).all()):
        return 0.0
    x0 = int(np.floor(min(float(arr_a[:, 0].min()), float(arr_b[:, 0].min()))))
    y0 = int(np.floor(min(float(arr_a[:, 1].min()), float(arr_b[:, 1].min()))))
    x1 = int(np.ceil(max(float(arr_a[:, 0].max()), float(arr_b[:, 0].max()))))
    y1 = int(np.ceil(max(float(arr_a[:, 1].max()), float(arr_b[:, 1].max()))))
    width = max(1, x1 - x0 + 3)
    height = max(1, y1 - y0 + 3)
    if width * height > 12_000_000:
        return _bbox_iou([int(arr_a[:,0].min()), int(arr_a[:,1].min()), int(arr_a[:,0].max()), int(arr_a[:,1].max())], [int(arr_b[:,0].min()), int(arr_b[:,1].min()), int(arr_b[:,0].max()), int(arr_b[:,1].max())])
    mask_a = np.zeros((height, width), np.uint8)
    mask_b = np.zeros((height, width), np.uint8)
    offset = np.array([x0 - 1, y0 - 1], np.float32)
    cv2.fillPoly(mask_a, [np.round(arr_a - offset).astype(np.int32)], 255)
    cv2.fillPoly(mask_b, [np.round(arr_b - offset).astype(np.int32)], 255)
    union = int(np.count_nonzero((mask_a > 0) | (mask_b > 0)))
    if union <= 0:
        return 0.0
    inter = int(np.count_nonzero((mask_a > 0) & (mask_b > 0)))
    return float(inter / union)


def _entry_polygon_iou(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    return _polygon_iou_px(_entry_polygon(a), _entry_polygon(b))


def select_flux_manifest_items(
    entries: Sequence[Mapping[str, Any]],
    *,
    per_wall_limit: int = 2,
    global_limit: int = 4,
    nms_center_distance_fraction: float = 0.65,
    nms_polygon_iou_threshold: float = 0.25,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply per-wall cap, same-wall NMS, and global cap before BFL render.

    This is intentionally an export-stage policy; it prevents a single wall from
    flooding the official BFL batch while preserving debug assets for all reviewed
    candidates in the zip.
    """
    per_wall_limit = max(1, int(per_wall_limit))
    global_limit = max(1, int(global_limit))
    by_wall: dict[str, list[dict[str, Any]]] = {}
    for raw in entries:
        item = dict(raw)
        by_wall.setdefault(str(item.get("wall_id") or "unknown_wall"), []).append(item)

    selected: list[dict[str, Any]] = []
    rejected_by_policy: list[dict[str, Any]] = []
    for wall_id, wall_entries in by_wall.items():
        wall_sorted = sorted(wall_entries, key=lambda e: float(e.get("score", 0.0) or 0.0), reverse=True)
        kept_for_wall: list[dict[str, Any]] = []
        for item in wall_sorted:
            cx, cy, width = _entry_center_and_width(item)
            too_close_to = None
            nms_debug: dict[str, Any] = {}
            for kept in kept_for_wall:
                kx, ky, kw = _entry_center_and_width(kept)
                threshold = max(12.0, min(width, kw) * float(nms_center_distance_fraction))
                center_distance = float(((cx - kx) ** 2 + (cy - ky) ** 2) ** 0.5)
                polygon_iou = _entry_polygon_iou(item, kept)
                if center_distance < threshold or polygon_iou > float(nms_polygon_iou_threshold):
                    too_close_to = kept
                    nms_debug = {
                        "drop_reason": "projected_polygon_iou" if polygon_iou > float(nms_polygon_iou_threshold) else "projected_center_distance",
                        "center_distance_px": center_distance,
                        "center_distance_threshold_px": float(threshold),
                        "projected_polygon_iou": float(polygon_iou),
                        "nms_polygon_iou_threshold": float(nms_polygon_iou_threshold),
                    }
                    break
            if too_close_to is not None:
                rejected_by_policy.append({
                    "asset_id": item.get("asset_id"),
                    "candidate_id": item.get("candidate_id"),
                    "wall_id": wall_id,
                    "status": "dropped_by_same_wall_nms",
                    "kept_asset_id": too_close_to.get("asset_id"),
                    **nms_debug,
                })
                continue
            if len(kept_for_wall) >= per_wall_limit:
                rejected_by_policy.append({
                    "asset_id": item.get("asset_id"),
                    "candidate_id": item.get("candidate_id"),
                    "wall_id": wall_id,
                    "status": "dropped_by_per_wall_limit",
                    "per_wall_limit": per_wall_limit,
                })
                continue
            item["manifest_selection"] = {
                "selected_for_bfl": True,
                "per_wall_limit": per_wall_limit,
                "global_limit": global_limit,
                "nms_center_distance_fraction": float(nms_center_distance_fraction),
                "nms_polygon_iou_threshold": float(nms_polygon_iou_threshold),
            }
            kept_for_wall.append(item)
        selected.extend(kept_for_wall)

    selected = sorted(selected, key=lambda e: float(e.get("score", 0.0) or 0.0), reverse=True)
    final = selected[:global_limit]
    dropped_global = selected[global_limit:]
    for item in dropped_global:
        rejected_by_policy.append({
            "asset_id": item.get("asset_id"),
            "candidate_id": item.get("candidate_id"),
            "wall_id": item.get("wall_id"),
            "status": "dropped_by_global_limit",
            "global_limit": global_limit,
        })
    summary = {
        "policy": "per_wall_cap_then_same_wall_center_or_polygon_nms_then_global_cap",
        "input_count": len(entries),
        "selected_count": len(final),
        "per_wall_limit": per_wall_limit,
        "global_limit": global_limit,
        "nms_center_distance_fraction": float(nms_center_distance_fraction),
        "nms_polygon_iou_threshold": float(nms_polygon_iou_threshold),
        "selected_by_wall": {},
        "dropped_count": len(rejected_by_policy),
        "dropped": rejected_by_policy,
    }
    for item in final:
        summary["selected_by_wall"].setdefault(str(item.get("wall_id") or "unknown_wall"), 0)
        summary["selected_by_wall"][str(item.get("wall_id") or "unknown_wall")] += 1
    return [dict(x) for x in final], summary


def select_public_candidate_after_exit_gate(
    candidates: Sequence[Mapping[str, Any]],
    final_manifest_items: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Resolve the public winner strictly from final approved manifest items."""
    for item in final_manifest_items:
        candidate_id = str(item.get("candidate_id") or "")
        wall_id = str(item.get("wall_id") or "")
        for raw in candidates:
            if (
                str(raw.get("candidate_id") or "") == candidate_id
                and str(raw.get("wall_id") or "") == wall_id
                and bool(raw.get("approved_for_flux_render"))
            ):
                return dict(raw)
    return None
