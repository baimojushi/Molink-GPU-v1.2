#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hanging_mvp.hardpaste import load_rgba, save_rgb, warp_artwork_mount_to_geometry, warp_artwork_to_quad
from hanging_mvp.render_qa import evaluate_mask_anchored_render_qa
from image_editing.pipeline import cleanup_objects, render_hanging
from image_editing.provider_registry import build_image_edit_provider


MOCK_PROVIDER_WARNING = (
    "🔴 WARNING: provider=mock is a local no-op dry run. "
    "Final images are copied/intermediate inputs and must not be interpreted as BFL success or failure."
)


def _is_mock_provider(name: str | None) -> bool:
    return str(name or "").strip().lower() in {"mock", "dry_run", "noop"}


def _red(text: str) -> str:
    return f"\033[31m{text}\033[0m"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in str(value)).strip("_") or "item"


def _extract_zip(zip_path: Path, work_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(work_dir)
    return work_dir


def _load_manifest(extracted: Path) -> dict[str, Any]:
    manifest = extracted / "flux_inputs" / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"missing flux_inputs/manifest.json in hanging result: {manifest}")
    data = _read_json(manifest)
    if not isinstance(data.get("items"), list):
        raise ValueError("flux_inputs/manifest.json must contain an items list")
    return data


def _resolve_path(extracted: Path, rel_or_abs: str | None) -> Path | None:
    if not rel_or_abs:
        return None
    p = Path(rel_or_abs)
    return p if p.is_absolute() else extracted / p


def _require_existing(path: Path | None, *, label: str, asset_id: str) -> Path:
    if path is None or not path.exists():
        raise FileNotFoundError(f"manifest item {asset_id} references missing {label}: {path}")
    return path


def _copy_inputs_for_trace(item: dict[str, Any], manifest: dict[str, Any], extracted: Path, item_dir: Path) -> dict[str, str]:
    copied: dict[str, str] = {}
    keys = (
        "original_room_path",
        "artwork_path",
        "hard_paste_path",
        "remove_mask_path",
        "art_region_mask_path",
        "front_region_mask_path",
        "frame_side_mask_path",
        "mount_mask_path",
        "flux2_edit_mask_path",
        "mount_debug_path",
        "candidate_path",
        "render_prompt_path",
    )
    for key in keys:
        rel = item.get(key) or manifest.get(key)
        if not rel:
            continue
        src = _resolve_path(extracted, str(rel))
        if src is None or not src.exists():
            raise FileNotFoundError(f"manifest item {item.get('asset_id')} references missing file: {rel}")
        dst = item_dir / "inputs" / Path(str(rel)).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        copied[key] = str(dst)
    return copied


def _quad_from_item(item: dict[str, Any], extracted: Path) -> list[list[float]] | None:
    quad = item.get("target_quad_original") or item.get("polygon_px")
    if quad:
        return [[float(p[0]), float(p[1])] for p in quad[:4]]
    candidate_path = _resolve_path(extracted, item.get("candidate_path"))
    if candidate_path and candidate_path.exists():
        cand = _read_json(candidate_path)
        projection = cand.get("projection", {}) if isinstance(cand.get("projection"), dict) else {}
        quad = projection.get("projected_quad_px_original")
        if quad:
            return [[float(p[0]), float(p[1])] for p in quad[:4]]
    return None


def _is_strict_v7_item(item: dict[str, Any], manifest: dict[str, Any]) -> bool:
    erase_order = str(item.get("erase_order") or manifest.get("erase_order") or "").strip()
    if erase_order == "original_room_then_hardpaste_after_erase":
        return True
    try:
        if int(manifest.get("version", 0) or 0) >= 2:
            return True
    except Exception:
        pass
    if item.get("original_room_path") or item.get("artwork_path") or item.get("target_quad_original"):
        return True
    return False


def _hardpaste_after_erase(
    clean_room: Path,
    artwork: Path,
    quad: list[list[float]],
    out_path: Path,
    *,
    mount_policy: dict[str, Any] | None = None,
    supersample_scale: int = 4,
) -> tuple[Path, dict[str, Any]]:
    room_rgb = np.array(Image.open(clean_room).convert("RGB"))
    art_rgba = load_rgba(artwork)
    policy = mount_policy if isinstance(mount_policy, dict) else {}
    if policy.get("precompose_frame_sides"):
        mounted = warp_artwork_mount_to_geometry(
            room_rgb,
            art_rgba,
            front_quad_original=policy.get("front_quad_original") or quad,
            side_faces_original=policy.get("visible_side_faces_original") or [],
            edge_sample_px=int(policy.get("edge_sample_px", 8)),
            side_shading_strength=float(policy.get("side_shading_strength", 0.18)),
            supersample_scale=int(supersample_scale),
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_rgb(str(out_path), mounted["image"])
        return out_path, {**(mounted.get("debug") or {}), "policy": policy}
    pasted, _art_region = warp_artwork_to_quad(room_rgb, art_rgba, quad, supersample_scale=int(supersample_scale))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_rgb(str(out_path), pasted)
    return out_path, {"mode": "flat_hardpaste", "policy": policy}


def _run_mask_anchored_qa(args: argparse.Namespace, *, hard_paste: Path, render: Path, art_mask: Path) -> dict[str, Any]:
    return evaluate_mask_anchored_render_qa(
        hard_paste_path=hard_paste,
        render_path=render,
        art_region_mask_path=art_mask,
        core_erode_px=int(args.qa_core_erode_px),
        allowed_frame_shadow_px=int(args.qa_allowed_frame_shadow_px),
        forbidden_outer_guard_px=int(args.qa_forbidden_outer_guard_px),
        changed_pixel_delta=float(args.qa_changed_pixel_delta),
        min_core_masked_ssim=float(args.qa_min_core_masked_ssim),
        max_core_color_delta=float(args.qa_max_core_color_delta),
        max_forbidden_outer_guard_changed_ratio=float(args.qa_max_forbidden_outer_guard_changed_ratio),
    )


def _is_nonempty_image_file(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        p = Path(path)
        return p.exists() and p.is_file() and p.stat().st_size > 0
    except Exception:
        return False


def _env_truthy(name: str, default: str = "1") -> bool:
    value = os.getenv(name, default)
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def _qa_hard_block_reasons(qa: dict[str, Any] | None, args: argparse.Namespace) -> list[str]:
    if not isinstance(qa, dict) or qa.get("passed"):
        return []
    reasons: list[str] = []
    try:
        if float(qa.get("core_masked_ssim", 1.0)) < float(args.qa_hard_min_core_masked_ssim):
            reasons.append("hard_core_masked_ssim_too_low")
    except Exception:
        pass
    try:
        if float(qa.get("core_color_delta", 0.0)) > float(args.qa_hard_max_core_color_delta):
            reasons.append("hard_core_color_delta_too_high")
    except Exception:
        pass
    try:
        if float(qa.get("forbidden_outer_guard_changed_ratio", 0.0)) > float(args.qa_hard_max_forbidden_outer_guard_changed_ratio):
            reasons.append("hard_forbidden_outer_guard_changed_ratio_too_high")
    except Exception:
        pass
    return reasons


def _qa_quality_score(qa: dict[str, Any] | None) -> float:
    if not isinstance(qa, dict):
        return -1.0
    if qa.get("passed"):
        return 100.0
    score = 0.0
    try:
        score += float(qa.get("core_masked_ssim", 0.0)) * 40.0
    except Exception:
        pass
    try:
        score -= min(float(qa.get("core_color_delta", 0.0)), 120.0) / 3.0
    except Exception:
        pass
    try:
        score -= min(float(qa.get("forbidden_outer_guard_changed_ratio", 0.0)), 1.0) * 30.0
    except Exception:
        pass
    return score


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    provider_name = str(args.provider or "bfl").strip().lower()
    provider = build_image_edit_provider(provider_name)
    using_mock_provider = _is_mock_provider(provider_name)
    if using_mock_provider:
        print(_red(MOCK_PROVIDER_WARNING), file=sys.stderr)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_dir = output_dir / "final_hd"
    final_dir.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="hanging_flux_") as td:
        extracted = _extract_zip(Path(args.hanging_zip).resolve(), Path(td))
        manifest = _load_manifest(extracted)
        items = manifest.get("items", [])
        if args.limit is not None:
            items = items[: args.limit]
        aggregate: dict[str, Any] = {
            "version": 2,
            "source_zip": str(Path(args.hanging_zip).resolve()),
            "provider": provider_name,
            "provider_warning": MOCK_PROVIDER_WARNING if using_mock_provider else None,
            "started_at_epoch": time.time(),
            "manifest_scope": manifest.get("scope"),
            "manifest_version": manifest.get("version"),
            "input_item_count": len(items),
            "erase_order": "original_room_then_hardpaste_after_erase",
            "flux2_mask_required": True,
            "flux2_mask_sent": True,
            "outputs": [],
            "errors": [],
            "warnings": ([{
                "provider": provider_name,
                "warning_label": "🔴 MOCK PROVIDER / NO-OP RESULT",
                "message": MOCK_PROVIDER_WARNING,
                "production_usable": False,
            }] if using_mock_provider else []),
            "dual_ab_compare": {
                "enabled": bool(args.dual_ab_compare),
                "requested_limit": int(args.dual_ab_compare_limit),
                "executed_count": 0,
                "note": "Compare safe dual_erased_room_plus_target against a single-input baseline before expanding dual coverage.",
            },
            "render_size_qa": {
                "method": "mask_anchored_core_ssim_and_forbidden_outer_guard",
                "bbox_redetection_used_for_decision": False,
                "decision_mode": str(getattr(args, "finalize_policy", "qa_advisory")),
                "guarantee_final_hd": bool(getattr(args, "guarantee_final_hd", True)),
            },
        }
        dual_ab_executed = 0
        for idx, item in enumerate(items):
            asset_id = _safe_name(item.get("asset_id") or f"item_{idx:03d}")
            item_dir = output_dir / asset_id
            item_dir.mkdir(parents=True, exist_ok=True)
            try:
                copied = _copy_inputs_for_trace(item, manifest, extracted, item_dir)
                hard_paste_legacy = _require_existing(_resolve_path(extracted, item.get("hard_paste_path")), label="hard_paste_path", asset_id=asset_id)
                remove_mask = _require_existing(_resolve_path(extracted, item.get("remove_mask_path")), label="remove_mask_path", asset_id=asset_id)
                art_region_mask = _require_existing(_resolve_path(extracted, item.get("art_region_mask_path")), label="art_region_mask_path", asset_id=asset_id)
                front_region_candidate = _resolve_path(extracted, item.get("front_region_mask_path"))
                if front_region_candidate is not None and front_region_candidate.exists():
                    qa_art_mask = front_region_candidate
                    qa_art_mask_source = "front_region_mask_path"
                else:
                    qa_art_mask = art_region_mask
                    qa_art_mask_source = "art_region_mask_path_legacy_fallback"

                flux2_edit_mask_candidate = _resolve_path(extracted, item.get("flux2_edit_mask_path"))
                if flux2_edit_mask_candidate is not None and flux2_edit_mask_candidate.exists():
                    flux2_edit_mask = flux2_edit_mask_candidate
                    flux2_edit_mask_source = "flux2_edit_mask_path"
                else:
                    # Backwards-compatible fallback for older manifests.  New
                    # service exports must include flux2_edit_mask_path.
                    flux2_edit_mask = qa_art_mask
                    flux2_edit_mask_source = f"missing_flux2_edit_mask_path_fallback_to_{qa_art_mask_source}"
                    aggregate["warnings"].append({
                        "asset_id": asset_id,
                        "mode": "missing_flux2_edit_mask_path",
                        "message": "Manifest did not include flux2_edit_mask_path; falling back to QA art mask so FLUX.2 is still masked.",
                    })
                prompt = item.get("render_prompt")
                if not prompt and item.get("render_prompt_path"):
                    prompt = _require_existing(_resolve_path(extracted, item["render_prompt_path"]), label="render_prompt_path", asset_id=asset_id).read_text(encoding="utf-8")
                if not prompt:
                    raise ValueError(f"manifest item {asset_id} has no render prompt")

                original_room = Path(args.original_room).resolve() if args.original_room else _resolve_path(extracted, item.get("original_room_path") or manifest.get("original_room_path"))
                artwork_path = _resolve_path(extracted, item.get("artwork_path") or manifest.get("artwork_path"))
                if artwork_path is None and args.artwork_reference:
                    artwork_path = Path(args.artwork_reference).resolve()
                quad = _quad_from_item(item, extracted)

                legacy_fallback_reasons: list[str] = []
                if original_room is None:
                    legacy_fallback_reasons.append("missing_original_room_path")
                elif not original_room.exists():
                    legacy_fallback_reasons.append(f"original_room_not_found:{original_room}")
                if artwork_path is None:
                    legacy_fallback_reasons.append("missing_artwork_path")
                elif not artwork_path.exists():
                    legacy_fallback_reasons.append(f"artwork_not_found:{artwork_path}")
                if not quad:
                    legacy_fallback_reasons.append("missing_target_quad_original")

                strict_v7_item = _is_strict_v7_item(item, manifest)
                if strict_v7_item and legacy_fallback_reasons:
                    raise FileNotFoundError(
                        f"strict v7 manifest item {asset_id} cannot run erase-before-hardpaste; "
                        f"missing/invalid inputs={legacy_fallback_reasons}. "
                        "Refusing legacy hardpaste->erase fallback because it makes erase see the newly pasted artwork."
                    )

                sequence_mode = "v7_original_erase_hardpaste_flux2"
                mount_policy = item.get("artwork_mount_policy") if isinstance(item.get("artwork_mount_policy"), dict) else {}
                if not legacy_fallback_reasons:
                    erase_input = original_room
                    erase_result = cleanup_objects(
                        provider,
                        erase_input,
                        remove_mask,
                        output_dir=item_dir,
                        output_name="01_erased.png",
                        metadata={
                            "asset_id": asset_id,
                            "candidate_id": item.get("candidate_id"),
                            "wall_id": item.get("wall_id"),
                            "dilate_pixels": args.erase_dilate_pixels,
                            "erase_input_semantics": "original_room",
                        },
                    )
                    hard_paste_after_erase, mount_trace = _hardpaste_after_erase(
                        erase_result.image,
                        artwork_path,
                        quad,
                        item_dir / "02_hard_paste_after_erase.png",
                        mount_policy=mount_policy,
                        supersample_scale=int(args.hardpaste_supersample_scale),
                    )
                else:
                    # Backwards compatibility for older manifests. This path is
                    # intentionally marked legacy because erase sees the pasted art.
                    sequence_mode = "legacy_hardpaste_erase_flux2"
                    warning = (
                        f"WARNING legacy_hardpaste_erase_flux2 fallback for asset_id={asset_id}; "
                        f"reasons={legacy_fallback_reasons}; v7 manifests should provide "
                        "original_room_path, artwork_path and target_quad_original. "
                        "Output quality may degrade because erase sees already-pasted artwork."
                    )
                    print(warning, file=sys.stderr)
                    aggregate["warnings"].append({
                        "asset_id": asset_id,
                        "sequence_mode": sequence_mode,
                        "reasons": list(legacy_fallback_reasons),
                        "message": warning,
                    })
                    erase_input = hard_paste_legacy
                    erase_result = cleanup_objects(
                        provider,
                        erase_input,
                        remove_mask,
                        output_dir=item_dir,
                        output_name="01_erased.png",
                        metadata={
                            "asset_id": asset_id,
                            "candidate_id": item.get("candidate_id"),
                            "wall_id": item.get("wall_id"),
                            "dilate_pixels": args.erase_dilate_pixels,
                            "erase_input_semantics": "legacy_hard_paste",
                        },
                    )
                    hard_paste_after_erase = erase_result.image
                    mount_trace = {"mode": "legacy_hardpaste_erase_flux2", "policy": mount_policy}

                requested_flux2_input_mode = str(item.get("flux2_input_mode") or "single_after_erase_hardpaste")
                flux2_input_mode = requested_flux2_input_mode
                render_primary = hard_paste_after_erase
                render_refs: list[Path] = []
                if requested_flux2_input_mode in {"dual_erased_room_plus_target", "dual_original_plus_target"} and sequence_mode.startswith("v7"):
                    flux2_input_mode = "dual_erased_room_plus_target"
                    render_refs = [erase_result.image]
                    render_input_semantics = {
                        "input_image": "cleaned_hard_paste_target_authoritative_canvas",
                        "input_image_2": "erased_room_texture_lighting_reference",
                        "order_policy": "input_image_is_authoritative_target; input_image_2_is_clean_reference_only; raw_original_reference_prohibited",
                    }
                    if requested_flux2_input_mode == "dual_original_plus_target":
                        warning = f"WARNING downgraded legacy raw-original dual mode to clean-reference dual for asset_id={asset_id}"
                        print(warning, file=sys.stderr)
                        aggregate["warnings"].append({
                            "asset_id": asset_id,
                            "mode": "raw_original_dual_downgraded",
                            "message": warning,
                        })
                else:
                    flux2_input_mode = "single_after_erase_hardpaste"
                    render_input_semantics = {
                        "input_image": "cleaned_hard_paste_target_authoritative",
                        "raw_original_reference_prohibited": True,
                    }

                render_result = render_hanging(
                    provider,
                    render_primary,
                    prompt,
                    art_mask=flux2_edit_mask,
                    references=render_refs,
                    output_dir=item_dir,
                    output_name="03_flux2_render.png" if sequence_mode.startswith("v7") else "02_flux2_render.png",
                    metadata={
                        "asset_id": asset_id,
                        "candidate_id": item.get("candidate_id"),
                        "wall_id": item.get("wall_id"),
                        "seed": args.seed,
                        "safety_tolerance": args.safety_tolerance,
                        "sequence_mode": sequence_mode,
                        "flux2_input_mode": flux2_input_mode,
                        "render_input_semantics": render_input_semantics,
                        "art_region_mask_usage": "front_region_mask_for_core_qa; flux2_edit_mask_sent_to_flux2",
                        "flux2_edit_mask_path": str(flux2_edit_mask),
                        "flux2_edit_mask_source": flux2_edit_mask_source,
                    },
                )
                primary_render_size_qa = _run_mask_anchored_qa(
                    args,
                    hard_paste=hard_paste_after_erase,
                    render=render_result.image,
                    art_mask=qa_art_mask,
                )

                dual_ab_compare_trace: dict[str, Any] | None = None
                if (
                    args.dual_ab_compare
                    and flux2_input_mode == "dual_erased_room_plus_target"
                    and sequence_mode.startswith("v7")
                    and dual_ab_executed < int(args.dual_ab_compare_limit)
                ):
                    try:
                        baseline_prompt = (
                            f"{prompt} A/B comparison baseline: use this single input image as the authoritative target; "
                            "do not use any original-room reference image for this baseline render."
                        )
                        baseline_result = render_hanging(
                            provider,
                            hard_paste_after_erase,
                            baseline_prompt,
                            art_mask=flux2_edit_mask,
                            references=[],
                            output_dir=item_dir,
                            output_name="03b_flux2_render_single_baseline.png",
                            metadata={
                                "asset_id": asset_id,
                                "candidate_id": item.get("candidate_id"),
                                "wall_id": item.get("wall_id"),
                                "seed": args.seed,
                                "safety_tolerance": args.safety_tolerance,
                                "sequence_mode": sequence_mode,
                                "flux2_input_mode": "single_after_erase_hardpaste_ab_baseline",
                                "render_input_semantics": {
                                    "input_image": "cleaned_hard_paste_target_authoritative",
                                    "ab_compare_role": "single_input_baseline_for_dual_mode_validation",
                                },
                                "art_region_mask_usage": "front_region_mask_for_core_qa; flux2_edit_mask_sent_to_flux2",
                                "flux2_edit_mask_path": str(flux2_edit_mask),
                                "flux2_edit_mask_source": flux2_edit_mask_source,
                                "ab_compare": True,
                            },
                        )
                        baseline_final_path = output_dir / "final_hd_ab_compare" / f"{asset_id}__single_baseline.png"
                        baseline_final_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(baseline_result.image, baseline_final_path)
                        dual_ab_executed += 1
                        aggregate["dual_ab_compare"]["executed_count"] = dual_ab_executed
                        dual_ab_compare_trace = {
                            "enabled": True,
                            "purpose": "validate whether safe dual_erased_room_plus_target improves or degrades the same candidate against a single-input baseline",
                            "single_baseline_path": str(baseline_result.image),
                            "single_baseline_final_path": str(baseline_final_path),
                            "single_baseline_trace": baseline_result.trace,
                            "dual_primary_path": str(render_result.image),
                        }
                    except Exception as ab_exc:
                        warning = f"WARNING dual_ab_compare baseline failed for asset_id={asset_id}: {ab_exc}"
                        print(warning, file=sys.stderr)
                        aggregate["warnings"].append({
                            "asset_id": asset_id,
                            "mode": "dual_ab_compare",
                            "message": warning,
                        })
                        dual_ab_compare_trace = {
                            "enabled": True,
                            "error": str(ab_exc),
                            "dual_primary_path": str(render_result.image),
                        }

                selected_render_result = render_result
                selected_render_size_qa = primary_render_size_qa
                render_fallback_trace: dict[str, Any] | None = None
                if not bool(primary_render_size_qa.get("passed")) and flux2_input_mode == "dual_erased_room_plus_target":
                    fallback_prompt = (
                        f"{prompt} SAFE FALLBACK: use this single cleaned hard-paste input as the only authoritative canvas. "
                        "Do not use any reference image. Preserve the exact painting-core pixel footprint."
                    )
                    fallback_result = render_hanging(
                        provider,
                        hard_paste_after_erase,
                        fallback_prompt,
                        art_mask=flux2_edit_mask,
                        references=[],
                        output_dir=item_dir,
                        output_name="03c_flux2_render_single_fallback.png",
                        metadata={
                            "asset_id": asset_id,
                            "candidate_id": item.get("candidate_id"),
                            "wall_id": item.get("wall_id"),
                            "seed": args.seed,
                            "safety_tolerance": args.safety_tolerance,
                            "sequence_mode": sequence_mode,
                            "flux2_input_mode": "single_after_erase_hardpaste_size_qa_fallback",
                            "art_region_mask_usage": "front_region_mask_for_core_qa; flux2_edit_mask_sent_to_flux2",
                            "flux2_edit_mask_path": str(flux2_edit_mask),
                            "flux2_edit_mask_source": flux2_edit_mask_source,
                            "size_qa_fallback": True,
                        },
                    )
                    fallback_size_qa = _run_mask_anchored_qa(
                        args,
                        hard_paste=hard_paste_after_erase,
                        render=fallback_result.image,
                        art_mask=qa_art_mask,
                    )
                    render_fallback_trace = {
                        "trigger": "primary_render_size_qa_failed",
                        "mode": "single_after_erase_hardpaste",
                        "render_path": str(fallback_result.image),
                        "render_trace": fallback_result.trace,
                        "render_size_qa": fallback_size_qa,
                    }
                    if bool(fallback_size_qa.get("passed")):
                        selected_render_result = fallback_result
                        selected_render_size_qa = fallback_size_qa
                    elif _qa_quality_score(fallback_size_qa) > _qa_quality_score(selected_render_size_qa):
                        selected_render_result = fallback_result
                        selected_render_size_qa = fallback_size_qa
                        render_fallback_trace["selected_despite_qa_failure"] = True

                hardpaste_preview_path: Path | None = None
                final_path: Path | None = None
                qa_passed = bool(selected_render_size_qa.get("passed"))
                qa_hard_block_reasons = _qa_hard_block_reasons(selected_render_size_qa, args)
                qa_hard_blocked = bool(qa_hard_block_reasons)
                finalize_policy = str(getattr(args, "finalize_policy", "qa_advisory") or "qa_advisory").strip().lower()
                guarantee_final_hd = bool(getattr(args, "guarantee_final_hd", True))
                selected_flux_file_ok = _is_nonempty_image_file(selected_render_result.image)
                selected_trace = dict(selected_render_result.trace or {})
                selected_mask_sent = bool(selected_trace.get("mask_sent_to_flux2"))
                allow_qa_advisory_final = finalize_policy in {"qa_advisory", "lax", "always_final"}
                should_finalize_flux = (
                    selected_mask_sent
                    and not qa_hard_blocked
                    and (qa_passed or (guarantee_final_hd and allow_qa_advisory_final and selected_flux_file_ok))
                )

                finalization_trace: dict[str, Any] = {
                    "policy": finalize_policy,
                    "raw_flux_passed": qa_passed,
                    "qa_hard_blocked": qa_hard_blocked,
                    "qa_hard_block_reasons": qa_hard_block_reasons,
                    "qa_hard_thresholds": {
                        "min_core_masked_ssim": float(args.qa_hard_min_core_masked_ssim),
                        "max_core_color_delta": float(args.qa_hard_max_core_color_delta),
                        "max_forbidden_outer_guard_changed_ratio": float(args.qa_hard_max_forbidden_outer_guard_changed_ratio),
                    },
                    "guarantee_final_hd": guarantee_final_hd,
                    "selected_flux_file_ok": selected_flux_file_ok,
                    "selected_flux2_mask_sent": selected_mask_sent,
                    "never_silent_hardpaste_success": True,
                    "hardpaste_delivery_allowed": False,
                }
                if should_finalize_flux:
                    if qa_passed:
                        item_exit_code = "OK_SELECTED_SINGLE_FALLBACK" if render_fallback_trace and selected_render_result is not render_result else "OK_SELECTED"
                    else:
                        item_exit_code = "OK_SELECTED_QA_ADVISORY_ACCEPTED"
                        aggregate["warnings"].append({
                            "asset_id": asset_id,
                            "mode": "flux_qa_advisory_accepted",
                            "message": (
                                f"FLUX render QA did not pass for {asset_id}, but a masked non-empty Flux render was produced; "
                                "QA is advisory and the render was promoted to final_hd. Severe artwork-identity QA remains a hard gate."
                            ),
                            "qa": selected_render_size_qa,
                        })
                    final_path = final_dir / f"{asset_id}.png"
                    shutil.copyfile(selected_render_result.image, final_path)
                    production_usable = not using_mock_provider
                else:
                    partial_dir = output_dir / "partial_review"
                    partial_dir.mkdir(parents=True, exist_ok=True)
                    preview_suffix = "frame_side_hardpaste_preview" if bool(mount_policy.get("precompose_frame_sides")) else "hardpaste_preview"
                    hardpaste_preview_path = partial_dir / f"{asset_id}__{preview_suffix}.png"
                    shutil.copyfile(hard_paste_after_erase, hardpaste_preview_path)
                    item_exit_code = "PARTIAL_HARDPASTE_REVIEW"
                    production_usable = False
                    aggregate["warnings"].append({
                        "asset_id": asset_id,
                        "mode": "flux_unavailable_or_strict_gate",
                        "message": f"No deliverable masked FLUX image was available for {asset_id}; hardpaste preview saved for manual review",
                        "qa": selected_render_size_qa,
                        "qa_hard_block_reasons": qa_hard_block_reasons,
                        "selected_flux2_mask_sent": selected_mask_sent,
                    })
                finalization_trace.update({
                    "final_path": str(final_path) if final_path is not None else None,
                    "production_usable": bool(production_usable),
                })

                item_trace = {
                    "asset_id": asset_id,
                    "candidate_id": item.get("candidate_id"),
                    "wall_id": item.get("wall_id"),
                    "score": item.get("score"),
                    "risk_level": item.get("risk_level"),
                    "sequence_mode": sequence_mode,
                    "legacy_fallback_reasons": list(legacy_fallback_reasons),
                    "erase_input": str(erase_input),
                    "hard_paste_after_erase": str(hard_paste_after_erase),
                    "mount_trace": mount_trace,
                    "requested_flux2_input_mode": requested_flux2_input_mode,
                    "flux2_input_mode": flux2_input_mode,
                    "render_input_semantics": render_input_semantics,
                    "flux2_mask_sent": selected_mask_sent,
                    "flux2_edit_mask_path": str(flux2_edit_mask),
                    "flux2_edit_mask_source": flux2_edit_mask_source,
                    "art_region_mask_usage": "front_region_mask_for_core_qa; flux2_edit_mask sent_to_flux2",
                    "qa_art_mask_path": str(qa_art_mask),
                    "qa_art_mask_source": qa_art_mask_source,
                    "art_region_mask_path": str(art_region_mask),
                    "render_size_qa_status": selected_render_size_qa.get("status"),
                    "primary_render_size_qa": primary_render_size_qa,
                    "selected_render_size_qa": selected_render_size_qa,
                    "post_flux_finalization": finalization_trace,
                    "render_fallback": render_fallback_trace,
                    "exit_code": item_exit_code,
                    "production_usable": bool(production_usable),
                    "input_files": copied,
                    "erase_trace": erase_result.trace,
                    "render_trace": render_result.trace,
                    "selected_render_trace": selected_render_result.trace,
                    "selected_render_path": str(selected_render_result.image),
                    "mock_provider_warning": MOCK_PROVIDER_WARNING if using_mock_provider else None,
                    "final_path": str(final_path) if final_path is not None else None,
                    "hardpaste_preview_path": str(hardpaste_preview_path) if hardpaste_preview_path is not None else None,
                    "exit_review": item.get("exit_review"),
                    "manifest_selection": item.get("manifest_selection"),
                    "dual_ab_compare": dual_ab_compare_trace,
                }
                (item_dir / "edit_trace.json").write_text(json.dumps(item_trace, ensure_ascii=False, indent=2), encoding="utf-8")
                aggregate["outputs"].append(item_trace)
            except Exception as exc:
                error = {"asset_id": asset_id, "error": str(exc)}
                aggregate["errors"].append(error)
                if not args.continue_on_error:
                    raise
        if using_mock_provider:
            print(_red(MOCK_PROVIDER_WARNING), file=sys.stderr)
        aggregate["finished_at_epoch"] = time.time()
        aggregate["elapsed_s"] = round(aggregate["finished_at_epoch"] - aggregate["started_at_epoch"], 3)
        aggregate["exit_code_counts"] = {}
        for output in aggregate["outputs"]:
            code = str(output.get("exit_code") or "UNKNOWN")
            aggregate["exit_code_counts"][code] = int(aggregate["exit_code_counts"].get(code, 0)) + 1
        aggregate["production_usable_output_count"] = sum(1 for output in aggregate["outputs"] if output.get("production_usable"))
        (output_dir / "edit_trace.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
        return aggregate


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official BFL processing for approved hanging assets.")
    parser.add_argument("--hanging-zip", required=True, help="Zip returned by /analyze_hanging_mvp")
    parser.add_argument("--original-room", default=None, help="Optional original room image override for legacy manifests")
    parser.add_argument("--artwork-reference", default=None, help="Original artwork image override for legacy manifests")
    parser.add_argument("--output-dir", required=True, help="Directory for final_hd images and traces")
    parser.add_argument("--provider", default="bfl", choices=["bfl", "mock"], help="Use bfl for official API, mock for local dry run")
    parser.add_argument("--erase-dilate-pixels", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--safety-tolerance", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Process only first N approved items after manifest selection")
    parser.add_argument("--dual-ab-compare", action="store_true", help="For safe dual_erased_room_plus_target items, also render a single-input baseline for validation.")
    parser.add_argument("--dual-ab-compare-limit", type=int, default=1, help="Maximum safe dual-mode items to render with the extra single-input A/B baseline.")
    parser.add_argument("--qa-core-erode-px", type=int, default=int(os.getenv("MOLINK_FLUX_QA_CORE_ERODE_PX", "1")))
    parser.add_argument("--qa-allowed-frame-shadow-px", type=int, default=int(os.getenv("MOLINK_FLUX_QA_ALLOWED_FRAME_SHADOW_PX", "16")))
    parser.add_argument("--qa-forbidden-outer-guard-px", type=int, default=int(os.getenv("MOLINK_FLUX_QA_FORBIDDEN_OUTER_GUARD_PX", "8")))
    parser.add_argument("--qa-changed-pixel-delta", type=float, default=float(os.getenv("MOLINK_FLUX_QA_CHANGED_PIXEL_DELTA", "30")))
    parser.add_argument("--qa-min-core-masked-ssim", type=float, default=float(os.getenv("MOLINK_FLUX_QA_MIN_CORE_MASKED_SSIM", "0.78")))
    parser.add_argument("--qa-max-core-color-delta", type=float, default=float(os.getenv("MOLINK_FLUX_QA_MAX_CORE_COLOR_DELTA", "45")))
    parser.add_argument("--qa-max-forbidden-outer-guard-changed-ratio", type=float, default=float(os.getenv("MOLINK_FLUX_QA_MAX_FORBIDDEN_OUTER_GUARD_CHANGED_RATIO", "0.65")))
    parser.add_argument("--qa-hard-min-core-masked-ssim", type=float, default=float(os.getenv("MOLINK_FLUX_QA_HARD_MIN_CORE_MASKED_SSIM", "0.45")), help="Hard delivery gate: below this artwork-core SSIM, a Flux render cannot enter final_hd even under qa_advisory.")
    parser.add_argument("--qa-hard-max-core-color-delta", type=float, default=float(os.getenv("MOLINK_FLUX_QA_HARD_MAX_CORE_COLOR_DELTA", "90")), help="Hard delivery gate: above this artwork-core color delta, a Flux render cannot enter final_hd even under qa_advisory.")
    parser.add_argument("--qa-hard-max-forbidden-outer-guard-changed-ratio", type=float, default=float(os.getenv("MOLINK_FLUX_QA_HARD_MAX_FORBIDDEN_OUTER_GUARD_CHANGED_RATIO", "0.85")), help="Hard delivery gate for severe outer-region change around the target artwork.")
    parser.add_argument("--finalize-policy", default=os.getenv("MOLINK_FLUX_FINALIZE_POLICY", "qa_advisory"), choices=["qa_gate_only", "qa_advisory", "lax", "always_final"], help="qa_advisory keeps QA metrics but allows non-empty Flux renders into final_hd.")
    parser.add_argument("--guarantee-final-hd", dest="guarantee_final_hd", action="store_true", default=_env_truthy("MOLINK_FLUX_GUARANTEE_FINAL_HD", "1"), help="Promote a non-empty Flux render to final_hd even when advisory QA fails.")
    parser.add_argument("--no-guarantee-final-hd", dest="guarantee_final_hd", action="store_false")
    parser.add_argument("--hardpaste-supersample-scale", type=int, default=4, help="Supersampling scale for hard-paste after erase; improves small artwork anti-aliasing without rejecting small projections.")
    parser.add_argument("--size-locked-transition-feather-px", type=int, default=2, help="Feather width for size-locked composite edge band.")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    trace = run_pipeline(parse_args())
    print(json.dumps({"output_count": len(trace["outputs"]), "error_count": len(trace["errors"]), "trace": "edit_trace.json"}, ensure_ascii=False, indent=2))
