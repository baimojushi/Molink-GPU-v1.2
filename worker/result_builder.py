"""worker/result_builder.py —— 组装上报业务端的 result 消息 payload。

合并：client_result.json（best_plan + alternative_plans，已含 §7.3 细化字段）
    + BFL 渲染 final 图 URL（按 wall_id/candidate_id 匹配）
    + partial_review hardpaste 预览图兜底 URL
    + not_recommended
→ §10.2 约定的 result payload。
"""
from __future__ import annotations

import os
from typing import Any, Dict, List


PARTIAL_REVIEW_SUFFIXES = (
    "__frame_side_hardpaste_preview",
    "__hardpaste_preview",
)


def _truthy_env(name: str, default: str = "1") -> bool:
    value = os.getenv(name, default)
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def _normalize_partial_asset_id(value: Any) -> str:
    text = str(value or "")
    for suffix in PARTIAL_REVIEW_SUFFIXES:
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def _status_from_exit(
    exit_code: str,
    production_usable_count: int,
    has_partial: bool,
    deliverable_count: int = 0,
    promoted_partial_count: int = 0,
) -> str:
    code = str(exit_code or "")
    if code.startswith("FAILED"):
        return "failed"
    if code == "PARTIAL_ALL_REJECTED_WITH_REASONS":
        return "no_safe_wall"
    if production_usable_count > 0:
        return "succeeded" if not has_partial else "succeeded_partial"
    # final_hd 只来自真实 Flux 输出。hardpaste preview 仍可保留到 partial_review，
    # 默认不再提升为交付图，避免把擦除后的硬贴图当成效果图交付。
    if deliverable_count > 0:
        return "succeeded_partial" if (has_partial or promoted_partial_count > 0) else "succeeded"
    if has_partial:
        return "render_review"
    return "failed"


def _index_partials(partial_images: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """按标准 asset_id 索引 partial_review 图。

    兼容旧字段：旧 r2_upload.py 会把 `<asset_id>__hardpaste_preview` 放进 asset_id；
    新 r2_upload.py 会把标准 id 放进 asset_id，原文件 stem 放进 partial_asset_id。
    """
    index: Dict[str, Dict[str, Any]] = {}
    for img in partial_images or []:
        raw_id = img.get("asset_id") or img.get("partial_asset_id") or img.get("review_asset_id")
        asset_id = _normalize_partial_asset_id(raw_id)
        if not asset_id or not img.get("url"):
            continue
        index.setdefault(asset_id, img)
    return index


def _index_renders(
    render_aggregate: Dict[str, Any],
    final_by_asset_id: Dict[str, str],
    partial_by_asset_id: Dict[str, Dict[str, Any]],
    *,
    promote_partial_review: bool,
) -> Dict[str, Dict[str, Any]]:
    """按 (wall_id, candidate_id) 索引渲染结果与可交付图 URL。"""
    index: Dict[str, Dict[str, Any]] = {}
    for out in render_aggregate.get("outputs", []) or []:
        asset_id = str(out.get("asset_id", ""))
        key = f"{out.get('wall_id')}::{out.get('candidate_id')}"

        final_url = final_by_asset_id.get(asset_id)
        final_source = "final_hd" if final_url else None
        partial_img = partial_by_asset_id.get(asset_id)
        promoted_partial = False

        if not final_url and promote_partial_review and partial_img and partial_img.get("url"):
            final_url = partial_img.get("url")
            final_source = "partial_review_hardpaste_preview"
            promoted_partial = True

        index[key] = {
            "asset_id": asset_id,
            "final_image_url": final_url,
            "final_image_source": final_source,
            "exit_code": out.get("exit_code"),
            "production_usable": bool(out.get("production_usable")),
            "deliverable": bool(final_url),
            "promoted_partial_review": promoted_partial,
            "partial_review_image": partial_img,
            "qa": out.get("selected_render_size_qa") or out.get("primary_render_size_qa"),
        }
    return index


def _plan_to_candidate(plan: Dict[str, Any], rank: int, render_info: Dict[str, Any]) -> Dict[str, Any]:
    install = plan.get("install_reference", {}) or {}
    final_url = render_info.get("final_image_url")
    return {
        "rank": rank,
        "wall_id": plan.get("wall_id"),
        "score": plan.get("score"),
        "risk_level": plan.get("risk_level"),
        "reason_tags": plan.get("reason_tags", []),
        "install": {
            "center_height_cm": install.get("center_height_cm"),
            "height_finalized": bool(install.get("metric_height_finalized")),
            "height_rule_label": install.get("height_rule_label"),
            "bottom_edge_cm": install.get("bottom_edge_cm"),
            "top_edge_cm": install.get("top_edge_cm"),
            "nail_height_cm": install.get("nail_height_cm"),
            "bottom_gap_to_furniture_cm": install.get("bottom_gap_to_furniture_cm"),
            "center_px": install.get("center_px_original"),
            "quad_px": install.get("projected_quad_px_original"),
        },
        "composition_context": plan.get("composition_context"),
        "cleanup_info": plan.get("cleanup_info"),
        "structural_clearance": plan.get("structural_clearance"),
        "geometry_confidence": plan.get("geometry_confidence"),
        "soft_risk_info": plan.get("soft_risk_info"),
        # 有 final_image_url 就视作本轮 worker 可交付；若是 partial 提升，会通过下方字段保留来源。
        "needs_review": not bool(final_url),
        "final_image_url": final_url,
        "final_image_source": render_info.get("final_image_source"),
        "render_exit_code": render_info.get("exit_code"),
        "production_usable": bool(render_info.get("production_usable")),
        "promoted_partial_review": bool(render_info.get("promoted_partial_review")),
        "partial_review_image": render_info.get("partial_review_image"),
        "qa": render_info.get("qa"),
    }


def build(
    client_result: Dict[str, Any],
    render_aggregate: Dict[str, Any],
    r2_result: Dict[str, Any],
    job: Dict[str, Any],
) -> Dict[str, Any]:
    final_by_asset_id = r2_result.get("final_by_asset_id", {}) or {}
    partial_images = r2_result.get("partial_review_images", []) or []
    partial_by_asset_id = _index_partials(partial_images)
    promote_partial_review = _truthy_env("MOLINK_PROMOTE_PARTIAL_REVIEW_AS_FINAL", "0")
    render_index = _index_renders(
        render_aggregate,
        final_by_asset_id,
        partial_by_asset_id,
        promote_partial_review=promote_partial_review,
    )

    plans: List[Dict[str, Any]] = []
    best = client_result.get("best_plan")
    if best:
        plans.append(best)
    plans.extend(client_result.get("alternative_plans", []) or [])

    candidates: List[Dict[str, Any]] = []
    for i, plan in enumerate(plans):
        key = f"{plan.get('wall_id')}::{plan.get('candidate_id')}"
        candidates.append(_plan_to_candidate(plan, i + 1, render_index.get(key, {})))

    # 仅保留拿到终图/兜底图的候选靠前（无图候选仍保留，供人工/调试）
    candidates.sort(key=lambda c: (c.get("final_image_url") is None, c.get("rank", 99)))

    exit_code = client_result.get("exit_code", "")
    usable = int(render_aggregate.get("production_usable_output_count", 0) or 0)
    final_url_count = sum(1 for c in candidates if c.get("final_image_url"))
    promoted_partial_count = sum(1 for c in candidates if c.get("promoted_partial_review"))
    status = _status_from_exit(
        exit_code,
        usable,
        bool(partial_images),
        deliverable_count=final_url_count,
        promoted_partial_count=promoted_partial_count,
    )

    return {
        "job_id": job.get("job_id"),
        "order_id": job.get("order_id"),
        "status": status,
        "exit_code": exit_code,
        "candidates": candidates,
        "not_recommended": client_result.get("not_recommended", []),
        "partial_review_images": partial_images,
        "error": None,
        "metrics": {
            "render_provider": (job.get("render") or {}).get("provider", "bfl"),
            "render_elapsed_s": render_aggregate.get("elapsed_s"),
            "production_usable_output_count": usable,
            "final_url_count": final_url_count,
            "partial_review_image_count": len(partial_images),
            "promoted_partial_review_count": promoted_partial_count,
            "promote_partial_review_as_final": promote_partial_review,
        },
    }


def build_failed_payload(job: Dict[str, Any], client_result: Dict[str, Any]) -> Dict[str, Any]:
    exit_code = client_result.get("exit_code", "FAILED_UNKNOWN")
    status = "no_safe_wall" if exit_code == "PARTIAL_ALL_REJECTED_WITH_REASONS" else "failed"
    return {
        "job_id": job.get("job_id"),
        "order_id": job.get("order_id"),
        "status": status,
        "exit_code": exit_code,
        "candidates": [],
        "not_recommended": client_result.get("not_recommended", []),
        "partial_review_images": [],
        "error": None,
        "metrics": {"render_provider": (job.get("render") or {}).get("provider", "bfl")},
    }
