"""worker/render_api.py —— 把离线脚本 run_bfl_hanging_flux_pipeline.run_pipeline
封装为 worker 内可调用函数，复用既有 QA 与 final_hd/partial_review 语义。

通过 bfl.parse_args([]) 取脚本自身的 argparse 默认值，
而非手工镜像——脚本新增参数时无需同步修改此处。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import run_bfl_hanging_flux_pipeline as bfl  # noqa: E402


def normalize_render_provider(value: str | None) -> str:
    key = str(value or os.getenv("HANGING_RENDER_PROVIDER") or "apiyi_gpt_image2_vip").strip().lower()
    aliases = {
        "apiyi": "apiyi_gpt_image2_vip",
        "apiyi_gpt_image2_vip": "apiyi_gpt_image2_vip",
        "gpt_image2_vip": "apiyi_gpt_image2_vip",
        "gpt-image-2-vip": "apiyi_gpt_image2_vip",
        "bfl": "bfl",
        "bfl_official": "bfl",
        "bfl_flux2": "bfl",
        "mock": "mock",
        "dry_run": "mock",
        "noop": "mock",
    }
    normalized = aliases.get(key)
    if normalized:
        return normalized
    print(f"[worker/render_api] unknown render provider={key!r}; fallback to apiyi_gpt_image2_vip", flush=True)
    return "apiyi_gpt_image2_vip"


def _call_bfl_parse_args(argv):
    """兼容 bfl.parse_args(argv) 和 bfl.parse_args() 两种签名。"""
    import inspect
    import sys

    argv = list(argv or [])

    try:
        sig = inspect.signature(bfl.parse_args)
        can_take_argv = len(sig.parameters) >= 1
    except Exception:
        can_take_argv = False

    if can_take_argv:
        return bfl.parse_args(argv)

    old_argv = sys.argv[:]
    try:
        sys.argv = [getattr(bfl, "__file__", "run_bfl_hanging_flux_pipeline.py")] + argv
        return bfl.parse_args()
    finally:
        sys.argv = old_argv


def _default_args():
    """获取脚本 argparse 默认值；用占位 required 参数避免读取真实 sys.argv。"""
    import argparse

    try:
        return _call_bfl_parse_args([
            "--hanging-zip", "__dummy_hanging__.zip",
            "--output-dir", "__dummy_output__",
        ])
    except (SystemExit, Exception) as exc:
        print(f"[worker/render_api] parse_args fallback: {type(exc).__name__}: {exc}", flush=True)

    # 极端兜底：保持与 run_bfl_hanging_flux_pipeline.py 当前默认参数一致
    return argparse.Namespace(
        hanging_zip="",
        original_room=None,
        artwork_reference=None,
        output_dir="",
        provider="apiyi_gpt_image2_vip",
        erase_dilate_pixels=10,
        seed=None,
        safety_tolerance=None,
        limit=None,
        dual_ab_compare=False,
        dual_ab_compare_limit=1,
        qa_core_erode_px=1,
        qa_allowed_frame_shadow_px=16,
        qa_forbidden_outer_guard_px=8,
        qa_changed_pixel_delta=30.0,
        qa_min_core_masked_ssim=0.78,
        qa_max_core_color_delta=45.0,
        qa_max_forbidden_outer_guard_changed_ratio=0.65,
        hardpaste_supersample_scale=4,
        size_locked_transition_feather_px=2,
        continue_on_error=False,
        finalize_policy="qa_advisory",
        guarantee_final_hd=True,
        render_prompt_upsampling=None,
        render_mask_scope=os.getenv("MOLINK_RENDER_MASK_SCOPE", "edit_region"),
    )
def run_pipeline(result_zip: str, output_dir: str, provider: str = "apiyi_gpt_image2_vip", limit: int | None = None) -> Dict[str, Any]:
    args = _default_args()
    # 调用者覆盖：必要字段 + worker 层有意与脚本默认不同的字段
    args.hanging_zip = str(result_zip)
    args.output_dir = str(output_dir)
    args.provider = normalize_render_provider(provider)
    # worker 层容错：单张失败继续处理其余张，而非终止整个批次。
    # QA 仅作为质量诊断，不再把有效 Flux 输出挡在 final_hd 外。
    args.continue_on_error = True
    args.finalize_policy = "qa_advisory"
    args.guarantee_final_hd = True
    args.qa_core_erode_px = int(os.getenv("MOLINK_FLUX_QA_CORE_ERODE_PX", "1"))
    args.qa_allowed_frame_shadow_px = int(os.getenv("MOLINK_FLUX_QA_ALLOWED_FRAME_SHADOW_PX", "16"))
    args.qa_forbidden_outer_guard_px = int(os.getenv("MOLINK_FLUX_QA_FORBIDDEN_OUTER_GUARD_PX", "8"))
    args.qa_changed_pixel_delta = float(os.getenv("MOLINK_FLUX_QA_CHANGED_PIXEL_DELTA", "30"))
    args.qa_min_core_masked_ssim = float(os.getenv("MOLINK_FLUX_QA_MIN_CORE_MASKED_SSIM", "0.78"))
    args.qa_max_core_color_delta = float(os.getenv("MOLINK_FLUX_QA_MAX_CORE_COLOR_DELTA", "45"))
    args.qa_max_forbidden_outer_guard_changed_ratio = float(os.getenv("MOLINK_FLUX_QA_MAX_FORBIDDEN_OUTER_GUARD_CHANGED_RATIO", "0.65"))
    if limit is not None:
        args.limit = limit
    return bfl.run_pipeline(args)
