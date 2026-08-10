from __future__ import annotations

from typing import Any, Dict, Iterable, List

# ===== 业务端/LLM 友好字段映射（§7.3）=====

RULE_LABELS = {
    "sofa_center_135": "沙发上方居中（画心离地约135cm）",
    "sofa_bottom_gap_15": "沙发上方留15cm下边距",
    "table_center_135": "边柜/茶几上方居中",
    "cabinet_center_135": "柜体上方居中",
    "counter_center_135": "吧台/操作台上方居中",
    "counter_fixture_center_135": "吧台/操作台上方居中",
    "sideboard_center_135": "边柜上方居中",
    "floor_center_152": "悬空墙面标准高度（画心离地约152cm）",
    "unknown_provisional_visual_review_required": "临时预览高度，建议现场确认",
}

LABEL_ZH = {
    "door": "门", "window": "窗", "arch": "拱", "opening": "开口",
    "sofa": "沙发", "table": "边柜/茶几", "cabinet": "柜体",
    "counter": "吧台/操作台", "counter_fixture": "吧台/操作台", "sideboard": "边柜",
    "bar": "吧台",
    "floor": "悬空", "unknown": "未知支撑",
}

SOFT_RISK_ZH = {
    "structural_overlap_metric_low_confidence_clearance_passed":
        "结构间距判断依赖低置信度度量数据，已通过但建议人工复核",
    "safe_wall_support_below_auto_pass_threshold":
        "实墙支撑覆盖率低于自动通过阈值",
    "artwork_edge_support_below_auto_pass_threshold":
        "画框边缘实墙支撑低于自动通过阈值",
    "artwork_corner_support_below_auto_pass_threshold":
        "画框四角实墙支撑低于自动通过阈值",
    "foreground_occlusion_ratio_too_high":
        "前景遮挡比例偏高，渲染前将处理遮挡",
    "plane_support_ratio_below_threshold":
        "透视平面一致性低于阈值",
    "opening_or_recess_overlap":
        "存在凹陷或开口区域与画框区域重叠",
    "normal_consistency_ratio_below_threshold":
        "法线一致性低于阈值，可能存在非平整壁面",
    "normal_mismatch_ratio_too_high":
        "法线错配比例偏高",
    "solid_wall_support_ratio_below_threshold":
        "实墙固体支撑比例低于阈值",
    "plane_support_ratio_below_threshold_v6":
        "透视平面一致性低于阈值",
    "light_caution":
        "候选墙光线条件需留意，已在效果图说明中标注光线影响",
    "light_concern":
        "候选墙存在较明显光线干扰，已作为排序参考",
    "light_deprioritized":
        "候选墙的背景安静度或景观竞争已作为排序参考",
    "light_quiet_backdrop":
        "候选墙面纹理会与作品表达竞争，已作为背景安静度参考",
    "light_landscape_competition":
        "候选墙主要视线方向存在景观竞争，已作为尺度关系参考",
}

NAIL_OFFSET_CM = 3  # 挂钩至画心默认偏移，可由 rules 配置覆盖


def _to_cm(value_m: Any) -> Any:
    try:
        if value_m is None:
            return None
        return round(float(value_m) * 100)
    except (TypeError, ValueError):
        return None


def _cleanup_info(hard: Dict[str, Any]) -> Dict[str, Any]:
    cleanup = hard.get("cleanup", {}) if isinstance(hard.get("cleanup"), dict) else {}
    fg = hard.get("foreground_cleanup", {}) if isinstance(hard.get("foreground_cleanup"), dict) else {}
    art_req = bool(cleanup.get("existing_content_cleanup_required") or cleanup.get("cleanup_required"))
    person_req = bool(fg.get("cleanup_required"))
    if art_req:
        label = "候选墙有既有挂画，渲染前将自动清除"
    elif person_req:
        label = "前景人物将由渲染引擎预清除"
    else:
        label = "无需预清除"
    return {
        "existing_art_cleanup_required": art_req,
        "foreground_person_cleanup_required": person_req,
        "any_cleanup_required": art_req or person_req,
        "cleanup_scope_label": label,
        "cleanup_provider": "bfl_erase_v1",
        "cleanup_policy_note": "清除范围限于候选墙内，其他墙面挂画不受影响",
    }


def _structural_clearance(hard: Dict[str, Any], artwork_w_m: float) -> Dict[str, Any]:
    sc = hard.get("structural_clearance_policy", {}) if isinstance(hard.get("structural_clearance_policy"), dict) else {}
    required_cm = _to_cm(sc.get("required_clearance_m", (artwork_w_m or 0.0) / 3.0)) or 0
    min_cm = _to_cm(sc.get("min_clearance_m"))
    elements = []
    for obj in sc.get("considered_objects", []) or []:
        if obj.get("skipped"):
            continue
        clr_cm = _to_cm(obj.get("edge_clearance_m", 0)) or 0
        label = str(obj.get("label", ""))
        elements.append({
            "label": label,
            "label_zh": LABEL_ZH.get(label, label),
            "relative_position": obj.get("relative_position", ""),
            "edge_clearance_cm": clr_cm,
            "sufficient": clr_cm >= required_cm,
        })
    blocking = [e["label_zh"] for e in elements if not e["sufficient"]]
    return {
        "passed": bool(sc.get("passed", True)),
        "required_clearance_cm": required_cm,
        "min_clearance_cm": min_cm,
        "elements": elements,
        "blocking_elements": blocking,
    }


def _axis_centering(candidate: Dict[str, Any]) -> str:
    ax = candidate.get("axis", {}) if isinstance(candidate.get("axis"), dict) else {}
    s_range = ax.get("component_s_range_m")
    axis_s = ax.get("axis_s_m")
    try:
        if s_range and axis_s is not None:
            lo, hi = float(s_range[0]), float(s_range[1])
            width = hi - lo
            if width > 1e-6:
                rel = (float(axis_s) - lo) / width
                if rel < 0.4:
                    return "偏左"
                if rel > 0.6:
                    return "偏右"
                return "居中"
    except (TypeError, ValueError, IndexError):
        pass
    return "居中"


def _composition_context(candidate: Dict[str, Any], artwork_h_m: float) -> Dict[str, Any]:
    sc = candidate.get("support_context", {}) if isinstance(candidate.get("support_context"), dict) else {}
    height = candidate.get("height_solution", {}) if isinstance(candidate.get("height_solution"), dict) else {}
    lbl = str(sc.get("label", height.get("height_band_base", "unknown")) or "unknown")
    rule_id = str(sc.get("rule_id", height.get("rule_id", "")) or "")
    c_t = height.get("center_t_normalized")
    try:
        c_t = float(c_t) if c_t is not None else 0.55
    except (TypeError, ValueError):
        c_t = 0.55
    c_h_cm = _to_cm(height.get("center_height_above_floor_m"))
    art_h_cm = _to_cm(artwork_h_m) or 0
    bottom_gap = sc.get("bottom_gap_to_support_m", height.get("bottom_gap_to_support_m"))
    t_label = "视觉中心区" if 0.42 <= c_t <= 0.66 else ("偏高" if c_t > 0.66 else "偏低")
    above_zh = LABEL_ZH.get(lbl, lbl)
    return {
        "above_furniture_label": lbl,
        "above_furniture_label_zh": "悬空墙面" if lbl == "floor" else (f"{above_zh}上方" if lbl in ("sofa", "table", "cabinet", "counter", "counter_fixture", "sideboard", "bar") else above_zh),
        "height_rule_label": RULE_LABELS.get(rule_id, rule_id),
        "bottom_gap_to_furniture_cm": _to_cm(bottom_gap),
        "center_t_label": t_label,
        "axis_centering": _axis_centering(candidate),
        "bottom_edge_cm": (c_h_cm - art_h_cm // 2) if c_h_cm is not None else None,
        "top_edge_cm": (c_h_cm + art_h_cm // 2) if c_h_cm is not None else None,
    }


def _geometry_confidence(hard: Dict[str, Any]) -> Dict[str, Any]:
    solid = hard.get("effective_wall_support_ratio",
                     hard.get("solid_wall_support_ratio", hard.get("solid_wall_inside_ratio")))
    plane = hard.get("effective_plane_consistency_ratio", hard.get("plane_consistency_ratio"))
    try:
        solid = float(solid) if solid is not None else None
    except (TypeError, ValueError):
        solid = None
    try:
        plane = float(plane) if plane is not None else None
    except (TypeError, ValueError):
        plane = None
    ref = solid if solid is not None else 0.0
    if ref >= 0.85:
        overall = "高"
    elif ref >= 0.70:
        overall = "中"
    else:
        overall = "低"
    return {
        "solid_wall_ratio": round(solid, 3) if solid is not None else None,
        "plane_consistency_ratio": round(plane, 3) if plane is not None else None,
        "overall_label": overall,
        "depth_method": "metric3d_vit_giant2",
    }


def _soft_risk_info(candidate: Dict[str, Any], risk_level: str) -> Dict[str, Any] | None:
    reasons = candidate.get("soft_risk_reasons", []) or []
    has_light_soft_risk = any(str(r).startswith("light_") for r in reasons)
    if risk_level not in ("medium", "high") and not has_light_soft_risk:
        return None
    zh = [SOFT_RISK_ZH.get(r, r) for r in reasons]
    if not zh:
        return None
    return {"has_soft_risk": True, "soft_risk_reasons_zh": list(dict.fromkeys(zh))}


def _risk_level(candidate: Dict[str, Any]) -> str:
    # Prefer the service-side risk assessment because it now separates hard
    # structural clearance from soft geometry/ranking scores.
    explicit = candidate.get("risk_level")
    if explicit in {"low", "medium", "high"}:
        return explicit
    hard = candidate.get("hard_constraints", {})
    risk = hard.get("risk_assessment", {}) if isinstance(hard.get("risk_assessment", {}), dict) else {}
    explicit = risk.get("risk_level")
    if explicit in {"low", "medium", "high"}:
        return explicit

    score = float(candidate.get("score", 0.0))
    # HANGING_POLICY_PATCH_V6_CLIENT_USE_EFFECTIVE_RATIOS
    plane = float(hard.get("effective_plane_consistency_ratio", hard.get("plane_consistency_ratio", 1.0)) or 0.0)
    solid = float(hard.get("effective_wall_support_ratio", hard.get("solid_wall_support_ratio", hard.get("solid_wall_inside_ratio", 1.0))) or 0.0)
    foreground = float(hard.get("foreground_occlusion_ratio", 0.0) or 0.0)
    recess = float(hard.get("recess_behind_wall_ratio", 0.0) or 0.0)

    if score >= 0.88 and plane >= 0.88 and solid >= 0.80 and foreground <= 0.05 and recess <= 0.08:
        return "low"
    if score >= 0.70 and plane >= 0.75 and foreground <= 0.15:
        return "medium"
    return "high"


def _light_decision_enabled_for_candidate(candidate: Dict[str, Any]) -> bool:
    ranking = candidate.get("ranking_policy", {}) if isinstance(candidate.get("ranking_policy"), dict) else {}
    causal = ranking.get("causal_light", {}) if isinstance(ranking.get("causal_light"), dict) else {}
    return bool(causal.get("decision_enabled"))


def _reason_tags(candidate: Dict[str, Any]) -> List[str]:
    hard = candidate.get("hard_constraints", {})
    tags = []
    clearance = hard.get("structural_clearance_policy", {}) if isinstance(hard.get("structural_clearance_policy", {}), dict) else {}
    if clearance.get("passed"):
        tags.append("门窗结构安全间距通过")
    if float(hard.get("effective_wall_inside_ratio", hard.get("wall_inside_ratio", 0.0)) or 0.0) >= 0.90:
        tags.append("墙面范围匹配")
    cleanup = hard.get("cleanup", {}) if isinstance(hard.get("cleanup", {}), dict) else {}
    if cleanup.get("cleanup_required"):
        tags.append("需先清除候选墙已有挂画")
    if float(hard.get("effective_plane_consistency_ratio", hard.get("plane_consistency_ratio", 0.0)) or 0.0) >= 0.85:
        tags.append("透视深度稳定")
    if float(hard.get("effective_wall_support_ratio", hard.get("solid_wall_support_ratio", hard.get("solid_wall_inside_ratio", 0.0))) or 0.0) >= 0.80:
        tags.append("实墙支撑充足")
    if float(hard.get("foreground_occlusion_ratio", 1.0) or 1.0) <= 0.05:
        tags.append("无遮挡")
    # HANGING_POLICY_PATCH_V6_CLIENT_REASON_TAGS
    fg_cleanup = hard.get("foreground_cleanup", {}) if isinstance(hard, dict) else {}
    if isinstance(fg_cleanup, dict) and fg_cleanup.get("cleanup_required"):
        tags.append("人物遮挡交由Flux2预清除")
    cleanup = hard.get("cleanup", {}) if isinstance(hard, dict) else {}
    if isinstance(cleanup, dict) and cleanup.get("existing_content_cleanup_required"):
        tags.append("需先清除候选墙已有挂画")
    height = candidate.get("height_solution", {})
    height_consistency = height.get("height_rule_consistency", {}) if isinstance(height, dict) else {}
    height_rule_consistent = bool(height_consistency.get("metric_height_rule_consistent", True))
    if height.get("height_mode") == "metric_rule" and height_rule_consistent:
        tags.append("高度符合规则")
    elif height.get("height_mode") == "metric_rule":
        tags.append("高度偏离规则，需人工确认")
    elif height.get("requires_frontend_visual_height_selection") or height.get("requires_visual_height_selection"):
        tags.append("高度为临时预览，需确认")
    risk = hard.get("risk_assessment", {}) if isinstance(hard.get("risk_assessment", {}), dict) else {}
    if candidate.get("requires_hardpaste_vlm_review") or risk.get("requires_hardpaste_vlm_review"):
        tags.append("需硬贴图VLM复核")
    light_tags = []
    if _light_decision_enabled_for_candidate(candidate):
        band = candidate.get("light_risk_band")
        if band == "caution":
            light_tags.append("light_caution")
        elif band == "concern":
            light_tags.append("light_concern")
        penalty_band = candidate.get("light_penalty_band") or (candidate.get("light_components") or {}).get("penalty_band")
        comps = candidate.get("light_components") if isinstance(candidate.get("light_components"), dict) else {}
        if penalty_band in {"caution", "concern"}:
            light_tags.append("light_deprioritized")
            try:
                if float(comps.get("P_backdrop") or 0.0) >= 0.05:
                    light_tags.append("light_quiet_backdrop")
            except Exception:
                pass
            try:
                if float(comps.get("P_view") or 0.0) >= 0.05:
                    light_tags.append("light_landscape_competition")
            except Exception:
                pass
    base_tags = list(dict.fromkeys(tags))
    light_tags = list(dict.fromkeys(light_tags))
    if light_tags:
        keep_base = max(0, 5 - len(light_tags))
        return list(dict.fromkeys(base_tags[:keep_base] + light_tags))[:5] or ["候选位置可用"]
    return base_tags[:5] or ["候选位置可用"]


def _normalized_artwork_bbox(candidate: Dict[str, Any], original_size: Any) -> tuple[Any, str | None]:
    """Return an original-image normalized artwork bbox for delivery presentation.

    The hard-paste front mask is the most accurate source.  Candidates that did
    not reach that export step fall back to their projected quadrilateral.
    """
    try:
        width, height = float(original_size[0]), float(original_size[1])
    except (TypeError, ValueError, IndexError):
        return None, None
    if width <= 0 or height <= 0:
        return None, None

    export = candidate.get("bfl_v7_export") if isinstance(candidate.get("bfl_v7_export"), dict) else {}
    bbox = export.get("front_bbox_px") or export.get("bbox_px")
    source = "front_bbox_px"
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        projection = candidate.get("projection") if isinstance(candidate.get("projection"), dict) else {}
        quad = projection.get("projected_quad_px_original")
        if not isinstance(quad, (list, tuple)) or len(quad) < 3:
            return None, None
        try:
            xs = [float(point[0]) for point in quad]
            ys = [float(point[1]) for point in quad]
        except (TypeError, ValueError, IndexError):
            return None, None
        bbox = [min(xs), min(ys), max(xs), max(ys)]
        source = "projected_quad_px_original"

    try:
        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    except (TypeError, ValueError):
        return None, None
    x1, x2 = sorted((max(0.0, min(width, x1)), max(0.0, min(width, x2))))
    y1, y2 = sorted((max(0.0, min(height, y1)), max(0.0, min(height, y2))))
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None, None
    return [round(x1 / width, 6), round(y1 / height, 6), round(x2 / width, 6), round(y2 / height, 6)], source


def _candidate_to_plan(candidate: Dict[str, Any], rank: int, base_asset_prefix: str = "",
                       artwork_w_m: float = 0.0, artwork_h_m: float = 0.0,
                       original_size: Any = None) -> Dict[str, Any]:
    wall_id = candidate.get("wall_id", "")
    cand_id = candidate.get("candidate_id", "")
    height = candidate.get("height_solution", {})
    projection = candidate.get("projection", {})
    hard = candidate.get("hard_constraints", {}) if isinstance(candidate.get("hard_constraints"), dict) else {}
    center_height_m = height.get("center_height_above_floor_m")
    provisional_center_height_cm = round(float(center_height_m) * 100) if center_height_m is not None else None
    height_consistency = height.get("height_rule_consistency", {}) if isinstance(height.get("height_rule_consistency"), dict) else {}
    height_rule_consistent = bool(height_consistency.get("metric_height_rule_consistent", True))
    metric_height_finalized = bool(
        height.get("metric_height_finalized", height.get("height_mode") == "metric_rule")
        and height_rule_consistent
    )
    center_height_cm = provisional_center_height_cm if metric_height_finalized else None

    risk_level = _risk_level(candidate)
    composition = _composition_context(candidate, artwork_h_m)
    rule_id = str(height.get("rule_id", "") or "")
    height_rule_label = RULE_LABELS.get(rule_id, rule_id)

    # install_reference 增强字段
    bottom_edge_cm = composition.get("bottom_edge_cm")
    top_edge_cm = composition.get("top_edge_cm")
    nail_height_cm = (center_height_cm + NAIL_OFFSET_CM) if center_height_cm is not None else None

    artwork_bbox_norm, artwork_bbox_source = _normalized_artwork_bbox(candidate, original_size)
    plan = {
        "plan_id": f"{wall_id}_{cand_id}",
        "rank": rank,
        "wall_id": wall_id,
        "candidate_id": cand_id,
        "title": "推荐挂在该墙面" if rank == 1 else "备选挂画方案",
        "score": round(float(candidate.get("score", 0.0)) * 100),
        "risk_level": risk_level,
        "requires_hardpaste_vlm_review": bool(candidate.get("requires_hardpaste_vlm_review") or (candidate.get("hard_constraints", {}).get("risk_assessment", {}) or {}).get("requires_hardpaste_vlm_review")),
        "reason_tags": _reason_tags(candidate),
        "asset_paths": {
            "hard_paste": f"{base_asset_prefix}wall_assets/{wall_id}/hard_paste.png",
            "debug_overlay": f"{base_asset_prefix}wall_assets/{wall_id}/debug_overlay.png",
            "art_region_mask": f"{base_asset_prefix}wall_assets/{wall_id}/art_region_mask.png",
        },
        "install_reference": {
            "center_height_cm": center_height_cm,
            "provisional_center_height_cm": None if metric_height_finalized else provisional_center_height_cm,
            "height_rule": height.get("rule_id"),
            "height_mode": height.get("height_mode"),
            "center_px_original": projection.get("center_px_original"),
            "projected_quad_px_original": projection.get("projected_quad_px_original"),
            "artwork_bbox_norm": artwork_bbox_norm,
            "artwork_bbox_source": artwork_bbox_source,
            "wall_mean_luminance": candidate.get("wall_mean_luminance"),
            "suggest_dark_wallpaper": bool(candidate.get("suggest_dark_wallpaper")),
            "suggested_wall_tone_rgb": candidate.get("suggested_wall_tone_rgb"),
            "requires_manual_height_adjustment": bool(
                height.get("requires_frontend_visual_height_selection", False)
                or not height_rule_consistent
            ),
            "requires_flux_visual_height_refinement": bool(height.get("requires_flux_visual_height_refinement", False)),
            "metric_height_finalized": metric_height_finalized,
            "height_band_base": height.get("height_band_base", "unknown"),
            "height_warning": height.get("height_warning"),
            "height_rule_consistency": height_consistency,
            "center_t_normalized": height.get("center_t_normalized"),
            "flux_visual_refinement": height.get("flux_visual_refinement"),
            # —— §7.3.7 增强字段 ——
            "height_rule_label": height_rule_label,
            "bottom_edge_cm": bottom_edge_cm,
            "top_edge_cm": top_edge_cm,
            "bottom_gap_to_furniture_cm": composition.get("bottom_gap_to_furniture_cm"),
            "nail_height_cm": nail_height_cm,
        },
        # —— §7.3.1~7.3.5 一级细化字段 ——
        "cleanup_info": _cleanup_info(hard),
        "structural_clearance": _structural_clearance(hard, artwork_w_m),
        "composition_context": composition,
        "geometry_confidence": _geometry_confidence(hard),
        "soft_risk_info": _soft_risk_info(candidate, risk_level),
        "debug_summary": {
            "hard_constraints": candidate.get("hard_constraints", {}),
            "axis": candidate.get("axis", {}),
            "support_context": candidate.get("support_context", {}),
        },
    }
    if candidate.get("light_components") is not None:
        plan["light_components"] = candidate.get("light_components")
    if candidate.get("light_risk_band") is not None:
        plan["light_risk_band"] = candidate.get("light_risk_band")
    if candidate.get("light_penalty_band") is not None:
        plan["light_penalty_band"] = candidate.get("light_penalty_band")
    if candidate.get("score_before_light") is not None:
        plan["score_before_light"] = round(float(candidate.get("score_before_light", 0.0)) * 100)
    if candidate.get("causal_light_semantics"):
        plan["light_semantics"] = candidate.get("causal_light_semantics")
    if candidate.get("causal_light_facts"):
        plan["light_facts"] = candidate.get("causal_light_facts")
    if candidate.get("window_edge_clearance_cm") is not None:
        plan.setdefault("composition_context", {})["window_edge_clearance_cm"] = candidate.get("window_edge_clearance_cm")
    if candidate.get("artwork_wall_area_ratio") is not None:
        plan.setdefault("composition_context", {})["artwork_wall_area_ratio"] = candidate.get("artwork_wall_area_ratio")
    if candidate.get("artwork_analysis"):
        plan["artwork_analysis"] = candidate.get("artwork_analysis")
    return plan


def _group_rejected(rejected: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in rejected or []:
        wall_id = item.get("wall_id", "unknown_wall")
        grouped.setdefault(wall_id, []).append(item)
    return grouped


def build_client_result(
    plan: Dict[str, Any],
    ranked_candidates: List[Dict[str, Any]],
    rejected_candidates: List[Dict[str, Any]],
    base_asset_prefix: str = "",
    artwork_w_m: float = 0.0,
    artwork_h_m: float = 0.0,
) -> Dict[str, Any]:
    final_manifest_items = plan.get("final_hardpaste_assets")
    if isinstance(final_manifest_items, list):
        approved_by_key = {
            (str(candidate.get("wall_id") or ""), str(candidate.get("candidate_id") or "")): candidate
            for candidate in ranked_candidates
            if bool(candidate.get("approved_for_flux_render"))
        }
        public_candidates = []
        for item in final_manifest_items:
            key = (str(item.get("wall_id") or ""), str(item.get("candidate_id") or ""))
            candidate = approved_by_key.get(key)
            if candidate is not None and candidate not in public_candidates:
                public_candidates.append(candidate)
    else:
        # Compatibility for callers that predate the exit-review manifest.
        public_candidates = list(ranked_candidates)
    image_space = plan.get("image_space") if isinstance(plan.get("image_space"), dict) else {}
    original_size = image_space.get("original_size")
    plans = [
        _candidate_to_plan(c, i + 1, base_asset_prefix=base_asset_prefix,
                           artwork_w_m=artwork_w_m, artwork_h_m=artwork_h_m,
                           original_size=original_size)
        for i, c in enumerate(public_candidates)
    ]

    rejected_grouped = _group_rejected(rejected_candidates)
    passed_wall_ids = {str(c.get("wall_id", "")) for c in public_candidates if c.get("wall_id")}
    not_recommended = []
    for wall_id, items in rejected_grouped.items():
        if str(wall_id) in passed_wall_ids:
            continue
        reasons = []
        for it in items:
            if it.get("reason"):
                reasons.append(str(it["reason"]))
            for r in it.get("reject_reasons", []) or []:
                reasons.append(str(r))
        reasons = list(dict.fromkeys(reasons))[:5]

        # §7.3.6 增强：定位主阻挡物、所处门控与可重试提示
        first = items[0] if items else {}
        hard = first.get("hard_constraints", {}) if isinstance(first.get("hard_constraints"), dict) else {}
        sc = hard.get("structural_clearance_policy", {}) if isinstance(hard.get("structural_clearance_policy"), dict) else {}
        blocking = sc.get("blocking_objects", []) or []
        primary = blocking[0] if blocking else None
        primary_label_zh = LABEL_ZH.get(str(primary.get("label", "")), primary.get("label")) if primary else None
        primary_clearance_cm = _to_cm(primary.get("edge_clearance_m")) if primary else None
        required_cm = _to_cm(sc.get("required_clearance_m"))
        rejection_stage = first.get("candidate_reject_stage") or (
            "structural_clearance" if blocking else (first.get("reject_reasons", [""])[0] if first.get("reject_reasons") else "geometry_gate")
        )
        solid_ratio = hard.get("solid_wall_inside_ratio")
        can_retry_hint = None
        if primary and required_cm:
            can_retry_hint = f"将此幅换成更窄的作品，或重新拍摄以减少{primary_label_zh or '结构物'}占比"

        not_recommended.append({
            "wall_id": wall_id,
            "title": "该墙面暂不推荐",
            "reasons": reasons or ["未通过几何硬限制"],
            "rejection_stage": rejection_stage,
            "primary_blocker_label_zh": primary_label_zh,
            "primary_blocker_edge_clearance_cm": primary_clearance_cm,
            "required_clearance_cm": required_cm,
            "solid_wall_ratio": round(float(solid_ratio), 3) if isinstance(solid_ratio, (int, float)) else None,
            "can_retry_hint": can_retry_hint,
            "debug_asset_paths": {
                "reject_summary": f"{base_asset_prefix}rejected_wall_assets/{wall_id}/reject_summary.json",
                "debug_overlay": f"{base_asset_prefix}rejected_wall_assets/{wall_id}/debug_overlay.png",
            },
        })

    return {
        "status": plan.get("status"),
        "exit_code": plan.get("exit_code"),
        "pipeline_build": plan.get("pipeline_build"),
        # Public-safe infrastructure classification.  Raw provider exceptions
        # remain in diagnostic artifacts and are never copied to the business payload.
        "exit_review_failure": plan.get("exit_review_failure"),
        "public_candidate_policy": "final_manifest_approved_for_flux_render_only" if isinstance(final_manifest_items, list) else "legacy_ranked_candidates",
        "best_plan": plans[0] if plans else None,
        "alternative_plans": plans[1:],
        "not_recommended": not_recommended,
        "assets_manifest": {
            "semantic_wall_debug": f"{base_asset_prefix}debug_assets/semantic_wall_debug.png",
            "hanging_plan": f"{base_asset_prefix}hanging_plan.json",
            "ranked_candidates": f"{base_asset_prefix}ranked_candidates.json",
            "passed_walls": f"{base_asset_prefix}passed_walls.json",
            "rejected_walls": f"{base_asset_prefix}rejected_walls.json",
        },
        "business_server_payload_version": "v2.0-geometry-enriched-llm-ready",
    }
