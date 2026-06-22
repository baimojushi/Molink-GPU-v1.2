from __future__ import annotations

from typing import Any, Mapping

import cv2
import numpy as np

from .masks import canonical_semantic_label


def evaluate_wall_support_gate(
    *,
    solid_wall_inside_ratio: float,
    effective_wall_inside_ratio: float,
    removable_existing_content_overlap_ratio: float,
    opening_or_recess_overlap_ratio: float,
    hard_reject_threshold: float = 0.90,
    auto_pass_threshold: float = 0.97,
    opening_recess_used_in_solid_mask: bool = True,
) -> dict[str, Any]:
    """Evaluate one primary safe-wall-support gate.

    Opening/recess evidence is intentionally an explanation field when the
    solid-wall mask already excludes holes. Existing art or wall-attached
    clutter may contribute to the effective ratio only after it has been
    classified as removable content on the current fitted wall plane.
    Foreground/person masks must never be added as wall support.
    """
    solid = float(max(0.0, min(1.0, solid_wall_inside_ratio)))
    effective = float(max(0.0, min(1.0, effective_wall_inside_ratio)))
    removable = float(max(0.0, min(1.0, removable_existing_content_overlap_ratio)))
    opening = float(max(0.0, min(1.0, opening_or_recess_overlap_ratio)))
    ratio = effective if removable > 0.0 else solid
    ratio_source = "solid_wall_plus_current_wall_removable_existing_content" if removable > 0.0 else "solid_wall_mask"

    if ratio < float(hard_reject_threshold):
        status = "hard_reject"
    elif ratio < float(auto_pass_threshold):
        status = "review"
    else:
        status = "pass"

    reasons: list[str] = []
    review_reasons: list[str] = []
    explanation_reasons: list[str] = []
    if status == "hard_reject":
        reasons.append("insufficient_safe_wall_support")
    elif status == "review":
        review_reasons.append("safe_wall_support_below_auto_pass_threshold")

    # Final P1 contract: if the solid-wall mask already excludes openings and
    # recesses, their overlap is diagnostic context for the support ratio rather
    # than a second independent hard gate. This keeps tuning focused on the
    # single support threshold and prevents duplicate reject reasons.
    if opening > 0.0:
        if opening_recess_used_in_solid_mask:
            explanation_reasons.append("overlaps_opening_or_recess")
        elif opening > 0.05:
            reasons.append("overlaps_opening_or_recess")
        else:
            explanation_reasons.append("minor_opening_or_recess_overlap_below_gate")

    return {
        "status": status,
        "safe_wall_support_gate_ratio": ratio,
        "safe_wall_support_gate_ratio_source": ratio_source,
        "solid_wall_inside_ratio": solid,
        "effective_wall_inside_ratio": effective,
        "removable_existing_content_overlap_ratio": removable,
        "hard_reject_threshold": float(hard_reject_threshold),
        "auto_pass_threshold": float(auto_pass_threshold),
        "opening_or_recess_overlap_ratio": opening,
        "opening_recess_used_in_solid_mask": bool(opening_recess_used_in_solid_mask),
        "opening_or_recess_is_explanation_only": bool(opening_recess_used_in_solid_mask),
        "hard_gate_reasons": reasons,
        "review_gate_reasons": review_reasons,
        "non_solid_reason_breakdown": explanation_reasons,
    }


def semantic_axis_policy(
    *,
    semantic_wall_mask: np.ndarray,
    safe_axis_wall_mask: np.ndarray,
    min_safe_coverage: float = 0.50,
) -> dict[str, Any]:
    semantic = semantic_wall_mask > 0
    safe = safe_axis_wall_mask > 0
    semantic_px = int(np.count_nonzero(semantic))
    safe_px = int(np.count_nonzero(semantic & safe))
    coverage = float(safe_px) / max(1, semantic_px)
    allowed = semantic_px > 0 and coverage >= float(min_safe_coverage)
    return {
        "status": "allowed" if allowed else "dropped_low_safe_coverage",
        "allowed": bool(allowed),
        "semantic_wall_area_px": semantic_px,
        "safe_axis_wall_area_px": safe_px,
        "semantic_axis_safe_mask_coverage": coverage,
        "min_semantic_axis_safe_mask_coverage": float(min_safe_coverage),
        "policy": "drop_entire_semantic_axis_set_when_low_coverage_do_not_clip_s_range",
    }


def _sample_polyline_mask(shape_hw: tuple[int, int], p0: np.ndarray, p1: np.ndarray, thickness_px: int) -> np.ndarray:
    out = np.zeros(shape_hw, np.uint8)
    a = tuple(int(round(float(v))) for v in p0[:2])
    b = tuple(int(round(float(v))) for v in p1[:2])
    cv2.line(out, a, b, 255, max(1, int(thickness_px)), lineType=cv2.LINE_8)
    return out


def _support_ratio(sample_mask: np.ndarray, safe_mask: np.ndarray) -> float:
    sample = sample_mask > 0
    denom = int(np.count_nonzero(sample))
    if denom <= 0:
        return 0.0
    return float(np.count_nonzero(sample & (safe_mask > 0))) / denom


def compute_edge_corner_support_debug(
    *,
    quad_px: np.ndarray,
    safe_wall_support_mask: np.ndarray,
    edge_thickness_px: int = 3,
    corner_radius_px: int = 4,
) -> dict[str, Any]:
    """Measure whether all projected artwork edges/corners remain on safe wall.

    This is a lightweight geometry diagnostic. It is intentionally anchored to
    the already-projected quad and does not run any image detector.
    """
    quad = np.asarray(quad_px, dtype=np.float32).reshape(-1, 2)[:4]
    if quad.shape != (4, 2):
        return {"status": "unavailable_invalid_quad"}
    h, w = safe_wall_support_mask.shape[:2]
    edges: list[dict[str, Any]] = []
    for idx in range(4):
        sample = _sample_polyline_mask((h, w), quad[idx], quad[(idx + 1) % 4], edge_thickness_px)
        edges.append({"edge_index": idx, "support_ratio": _support_ratio(sample, safe_wall_support_mask)})
    corners: list[dict[str, Any]] = []
    for idx, point in enumerate(quad):
        sample = np.zeros((h, w), np.uint8)
        cv2.circle(sample, (int(round(float(point[0]))), int(round(float(point[1])))), max(1, int(corner_radius_px)), 255, -1)
        corners.append({"corner_index": idx, "support_ratio": _support_ratio(sample, safe_wall_support_mask)})
    return {
        "status": "ok",
        "edge_thickness_px": int(edge_thickness_px),
        "corner_radius_px": int(corner_radius_px),
        "edges": edges,
        "corners": corners,
        "min_edge_support_ratio": min((x["support_ratio"] for x in edges), default=0.0),
        "min_corner_support_ratio": min((x["support_ratio"] for x in corners), default=0.0),
    }


def choose_flux2_input_policy(
    *,
    cleanup_labels: list[str] | tuple[str, ...],
    has_person_cleanup: bool,
    has_existing_cleanup: bool,
    risky_existing_art_labels: set[str] | None = None,
) -> dict[str, Any]:
    """Choose a safe FLUX.2 reference policy.

    Raw original-room dual references are never selected. A cleaned-room dual
    reference is allowed only for wall-attached non-art cleanup without people.
    """
    # cleanup_labels are expected to be canonical, but normalize both sides
    # here so aliases such as "wall art" cannot silently bypass art-reference rules.
    risky = {
        canonical_semantic_label(x)
        for x in (risky_existing_art_labels or {"existing_art", "painting", "poster", "framed_art"})
        if str(x or "").strip()
    }
    labels = sorted({canonical_semantic_label(x) for x in cleanup_labels if str(x or "").strip()})
    existing_art_present = any(label in risky for label in labels)
    clearable_non_art_present = bool(labels) and not existing_art_present

    if has_person_cleanup:
        mode = "single_after_erase_hardpaste"
        reason = "person_or_foreground_cleanup_prohibits_dual_reference"
    elif existing_art_present:
        mode = "single_after_erase_hardpaste"
        reason = "existing_art_cleanup_prohibits_reference_content_reflow"
    elif has_existing_cleanup and clearable_non_art_present:
        mode = "dual_erased_room_plus_target"
        reason = "cleaned_room_reference_allowed_for_wall_attached_non_art_cleanup"
    else:
        mode = "single_after_erase_hardpaste"
        reason = "single_target_default"

    return {
        "flux2_input_mode": mode,
        "reason": reason,
        "raw_original_dual_allowed": False,
        "dual_reference_when_allowed": "erased_room_only",
        "cleanup_labels": labels,
        "has_person_cleanup": bool(has_person_cleanup),
        "has_existing_cleanup": bool(has_existing_cleanup),
        "existing_art_present": bool(existing_art_present),
        "clearable_wall_attached_non_art_present": bool(clearable_non_art_present),
    }
