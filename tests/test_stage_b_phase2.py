import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from hanging_mvp.flux_export import build_stage_b_prompt
from semantic_mvp.vlm_client import parse_styling_analysis_with_vlm
from worker import render_api, result_builder


WALLPAPER = {"enabled": True, "tone_rgb": [58, 62, 71]}


def _make_stage_a_fixture(tmp_path: Path, protect_mode: str = "valid") -> tuple[Path, Path, bytes]:
    output_dir = tmp_path / "render"
    final_dir = output_dir / "final_hd"
    final_dir.mkdir(parents=True)
    source = np.zeros((80, 120, 3), dtype=np.uint8)
    source[:, :] = [220, 215, 205]
    final_path = final_dir / "wall_a__candidate_1.png"
    Image.fromarray(source).save(final_path)
    original_bytes = final_path.read_bytes()

    mask = np.zeros((80, 120), dtype=np.uint8)
    if protect_mode != "empty":
        mask[25:55, 45:75] = 255
    mask_path = tmp_path / "mount_mask.png"
    Image.fromarray(mask).save(mask_path)
    decor = np.zeros((80, 120), dtype=np.uint8)
    decor[45:, :] = 255
    decor_path = tmp_path / "decor_mask.png"
    Image.fromarray(decor).save(decor_path)
    wall = np.zeros((80, 120), dtype=np.uint8)
    wall[:58, :] = 255
    wall_path = tmp_path / "wall_mask.png"
    Image.fromarray(wall).save(wall_path)
    artwork_path = tmp_path / "artwork.png"
    Image.fromarray(np.full((40, 30, 3), [72, 110, 164], dtype=np.uint8)).save(artwork_path)
    item = {
        "asset_id": "wall_a__candidate_1",
        "wall_id": "wall_a",
        "candidate_id": "candidate_1",
    }
    if protect_mode != "missing_field":
        item["stage_b_protect_zone_mask_path"] = "flux_inputs/items/wall_a__candidate_1/stage_b_protect_zone_mask.png"
    item["stage_b_decor_zone_mask_path"] = "flux_inputs/items/wall_a__candidate_1/stage_b_decor_zone_mask.png"
    item["stage_b_wall_zone_mask_path"] = "flux_inputs/items/wall_a__candidate_1/stage_b_wall_zone_mask.png"
    item["styling_zone_source"] = "sam_support_union"
    manifest = {"artwork_path": "flux_inputs/artwork.png", "items": [item]}
    archive_path = tmp_path / "hanging_result.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("flux_inputs/manifest.json", json.dumps(manifest))
        archive.write(artwork_path, "flux_inputs/artwork.png")
        archive.write(decor_path, "flux_inputs/items/wall_a__candidate_1/stage_b_decor_zone_mask.png")
        archive.write(wall_path, "flux_inputs/items/wall_a__candidate_1/stage_b_wall_zone_mask.png")
        if protect_mode != "missing_file" and protect_mode != "missing_field":
            archive.write(mask_path, "flux_inputs/items/wall_a__candidate_1/stage_b_protect_zone_mask.png")
    return archive_path, output_dir, original_bytes


def test_stage_b_prompt_has_four_independent_combinations():
    none = build_stage_b_prompt(False, None)
    soft = build_stage_b_prompt(True, None)
    wall = build_stage_b_prompt(False, WALLPAPER)
    both = build_stage_b_prompt(True, WALLPAPER)

    assert none == ""
    assert "small accessories" in soft and "Recolor only" not in soft
    assert "Recolor only" in wall and "small accessories" not in wall
    assert "small accessories" in both and "Recolor only" in both
    assert "harmonious" in both


def test_algorithmic_styling_analysis_honors_fixed_wall_tone(tmp_path):
    artwork = tmp_path / "artwork.png"
    Image.fromarray(np.full((48, 64, 3), [180, 62, 94], dtype=np.uint8)).save(artwork)
    analysis = parse_styling_analysis_with_vlm(
        artwork.read_bytes(),
        "mock",
        soft_furnishing_requested=True,
        wallpaper_recolor=WALLPAPER,
    )
    assert analysis["status"] == "fallback_algorithmic"
    assert analysis["suggested_wall_tone_rgb"] == WALLPAPER["tone_rgb"]
    assert len(analysis["accent_colors_abstracted_rgb"]) == 2
    prompt = build_stage_b_prompt(True, WALLPAPER, analysis, {"has_frame": True})
    assert "abstracted accent RGB values" in prompt
    assert "preserve the complete visible physical frame" in prompt


def test_stage_b_pipeline_calls_zero_or_once_per_asset_for_all_combinations(tmp_path):
    combinations = [
        (False, None, 0, "not_requested"),
        (True, None, 1, "succeeded"),
        (False, WALLPAPER, 1, "succeeded"),
        (True, WALLPAPER, 1, "succeeded"),
    ]
    for index, (soft, wallpaper, calls, status) in enumerate(combinations):
        case_dir = tmp_path / str(index)
        case_dir.mkdir()
        archive, output_dir, _ = _make_stage_a_fixture(case_dir)
        aggregate = render_api.run_stage_b_pipeline(
            str(archive),
            str(output_dir),
            provider="mock",
            soft_furnishing_requested=soft,
            wallpaper_recolor=wallpaper,
        )
        assert aggregate["call_count"] == calls
        assert aggregate["outputs"][0]["styling_status"] == status
        assert (output_dir / "pre_styling_hd" / "wall_a__candidate_1.png").exists()


def test_stage_b_failure_keeps_stage_a_and_marks_fallback(tmp_path, monkeypatch):
    archive, output_dir, original_bytes = _make_stage_a_fixture(tmp_path)

    class FailingProvider:
        def edit(self, request):
            raise RuntimeError("synthetic stage-b failure")

    monkeypatch.setattr(render_api, "build_image_edit_provider", lambda _name: FailingProvider())
    aggregate = render_api.run_stage_b_pipeline(
        str(archive),
        str(output_dir),
        provider="mock",
        soft_furnishing_requested=True,
    )
    final_path = output_dir / "final_hd" / "wall_a__candidate_1.png"
    assert aggregate["call_count"] == 1
    assert aggregate["outputs"][0]["styling_status"] == "failed_fallback_to_plain"
    assert final_path.read_bytes() == original_bytes


def test_stage_b_content_qa_rejects_whole_decor_repaint(tmp_path, monkeypatch):
    archive, output_dir, original_bytes = _make_stage_a_fixture(tmp_path)

    class WholeImageRewriteProvider:
        def edit(self, request):
            out = request.output_dir / request.output_name
            with Image.open(request.image) as source:
                Image.fromarray(np.full((source.height, source.width, 3), [220, 20, 30], dtype=np.uint8)).save(out)
            return SimpleNamespace(image=out, trace={"synthetic": "whole_image_rewrite"})

    monkeypatch.setattr(render_api, "build_image_edit_provider", lambda _name: WholeImageRewriteProvider())
    aggregate = render_api.run_stage_b_pipeline(
        str(archive),
        str(output_dir),
        provider="mock",
        soft_furnishing_requested=True,
    )
    final_path = output_dir / "final_hd" / "wall_a__candidate_1.png"
    assert aggregate["outputs"][0]["styling_status"] == "qa_rejected_fallback_to_plain"
    assert aggregate["outputs"][0]["qa"]["protect_changed_pixel_count"] == 0
    assert "decor_changed_ratio_above_threshold" in aggregate["outputs"][0]["qa"]["failures"]
    assert final_path.read_bytes() == original_bytes


def test_stage_b_content_qa_rejects_random_noise_inside_large_edit_mask(tmp_path, monkeypatch):
    archive, output_dir, original_bytes = _make_stage_a_fixture(tmp_path)

    class RandomNoiseProvider:
        def edit(self, request):
            out = request.output_dir / request.output_name
            with Image.open(request.image) as source:
                noise = np.random.default_rng(7).integers(0, 256, (source.height, source.width, 3), dtype=np.uint8)
                Image.fromarray(noise).save(out)
            return SimpleNamespace(image=out, trace={"synthetic": "random_noise"})

    monkeypatch.setattr(render_api, "build_image_edit_provider", lambda _name: RandomNoiseProvider())
    aggregate = render_api.run_stage_b_pipeline(
        str(archive),
        str(output_dir),
        provider="mock",
        soft_furnishing_requested=True,
        wallpaper_recolor=WALLPAPER,
    )
    final_path = output_dir / "final_hd" / "wall_a__candidate_1.png"
    qa = aggregate["outputs"][0]["qa"]
    assert aggregate["outputs"][0]["styling_status"] == "qa_rejected_fallback_to_plain"
    assert qa["passed"] is False
    assert qa["content_quality"]["decor"]["masked_ssim"] < 0.60
    assert final_path.read_bytes() == original_bytes


def test_stage_b_content_qa_accepts_small_local_decor_edit(tmp_path, monkeypatch):
    archive, output_dir, _ = _make_stage_a_fixture(tmp_path)

    class SmallAccessoryProvider:
        def edit(self, request):
            out = request.output_dir / request.output_name
            with Image.open(request.image) as source:
                styled = np.asarray(source.convert("RGB")).copy()
            styled[65:71, 10:20] = [72, 108, 84]
            Image.fromarray(styled).save(out)
            return SimpleNamespace(image=out, trace={"synthetic": "small_accessory"})

    monkeypatch.setattr(render_api, "build_image_edit_provider", lambda _name: SmallAccessoryProvider())
    aggregate = render_api.run_stage_b_pipeline(
        str(archive),
        str(output_dir),
        provider="mock",
        soft_furnishing_requested=True,
    )
    qa = aggregate["outputs"][0]["qa"]
    assert aggregate["outputs"][0]["styling_status"] == "succeeded"
    assert qa["passed"] is True
    assert 0.0 < qa["content_quality"]["decor"]["changed_ratio"] < 0.08


def test_stage_b_wall_qa_accepts_recolor_that_preserves_luma_gradients(tmp_path, monkeypatch):
    archive, output_dir, _ = _make_stage_a_fixture(tmp_path)
    final_path = output_dir / "final_hd" / "wall_a__candidate_1.png"
    yy, xx = np.mgrid[0:80, 0:120]
    gradient = np.stack([
        150 + xx // 4,
        155 + yy // 3,
        165 + (xx + yy) // 8,
    ], axis=2).astype(np.uint8)
    Image.fromarray(gradient).save(final_path)

    class GradientPreservingRecolorProvider:
        def edit(self, request):
            out = request.output_dir / request.output_name
            with Image.open(request.image) as source:
                base = np.asarray(source.convert("RGB"), dtype=np.float32)
            styled = np.clip(base * 0.72 + np.array([12.0, 8.0, 18.0]), 0, 255).astype(np.uint8)
            Image.fromarray(styled).save(out)
            return SimpleNamespace(image=out, trace={"synthetic": "gradient_preserving_recolor"})

    monkeypatch.setattr(render_api, "build_image_edit_provider", lambda _name: GradientPreservingRecolorProvider())
    aggregate = render_api.run_stage_b_pipeline(
        str(archive),
        str(output_dir),
        provider="mock",
        wallpaper_recolor=WALLPAPER,
    )
    qa = aggregate["outputs"][0]["qa"]
    assert aggregate["outputs"][0]["styling_status"] == "succeeded"
    assert qa["passed"] is True
    assert qa["content_quality"]["wall"]["gradient_correlation"] > 0.80


def test_stage_b_qa_rejects_wrong_provider_size_and_keeps_stage_a(tmp_path, monkeypatch):
    archive, output_dir, original_bytes = _make_stage_a_fixture(tmp_path)

    class WrongSizeProvider:
        def edit(self, request):
            out = request.output_dir / request.output_name
            Image.fromarray(np.full((20, 20, 3), 90, dtype=np.uint8)).save(out)
            return SimpleNamespace(image=out, trace={"synthetic": "wrong_size"})

    monkeypatch.setattr(render_api, "build_image_edit_provider", lambda _name: WrongSizeProvider())
    aggregate = render_api.run_stage_b_pipeline(
        str(archive),
        str(output_dir),
        provider="mock",
        soft_furnishing_requested=True,
    )
    final_path = output_dir / "final_hd" / "wall_a__candidate_1.png"
    assert aggregate["call_count"] == 1
    assert aggregate["outputs"][0]["styling_status"] == "qa_rejected_fallback_to_plain"
    assert aggregate["outputs"][0]["qa"]["reason"] == "provider_output_size_mismatch"
    assert final_path.read_bytes() == original_bytes


def test_stage_b_missing_invalid_or_empty_protect_mask_fails_closed(tmp_path):
    for mode in ("missing_field", "missing_file", "empty"):
        case_dir = tmp_path / mode
        case_dir.mkdir()
        archive, output_dir, original_bytes = _make_stage_a_fixture(case_dir, protect_mode=mode)
        aggregate = render_api.run_stage_b_pipeline(
            str(archive),
            str(output_dir),
            provider="mock",
            soft_furnishing_requested=True,
        )
        final_path = output_dir / "final_hd" / "wall_a__candidate_1.png"
        assert aggregate["call_count"] == 0
        assert aggregate["outputs"][0]["styling_status"] == "failed_fallback_to_plain"
        assert final_path.read_bytes() == original_bytes
        assert "mask" in aggregate["outputs"][0]["error"].lower()


def test_result_payload_exposes_pre_styling_url_and_status():
    client_result = {
        "exit_code": "SUCCESS",
        "best_plan": {
            "wall_id": "wall_a",
            "candidate_id": "candidate_1",
            "install_reference": {},
        },
        "alternative_plans": [],
    }
    render_aggregate = {
        "production_usable_output_count": 1,
        "outputs": [{
            "asset_id": "wall_a__candidate_1",
            "wall_id": "wall_a",
            "candidate_id": "candidate_1",
            "production_usable": True,
        }],
        "stage_b": {
            "requested": True,
            "call_count": 1,
            "outputs": [{
                "asset_id": "wall_a__candidate_1",
                "styling_status": "succeeded",
                "styling_zone_source": "sam_support_union",
                "qa": {"passed": True},
            }],
        },
    }
    r2_result = {
        "final_by_asset_id": {"wall_a__candidate_1": "https://example/final.png"},
        "pre_styling_by_asset_id": {"wall_a__candidate_1": "https://example/stage-a.png"},
        "partial_review_images": [],
    }
    payload = result_builder.build(client_result, render_aggregate, r2_result, {"job_id": "j1", "order_id": 1})
    candidate = payload["candidates"][0]
    assert candidate["final_image_url"].endswith("final.png")
    assert candidate["pre_styling_image_url"].endswith("stage-a.png")
    assert candidate["styling_status"] == "succeeded"
    assert candidate["styling_zone_source"] == "sam_support_union"
    assert candidate["styling_qa"]["passed"] is True
    assert payload["metrics"]["stage_b_call_count"] == 1
