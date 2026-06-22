from __future__ import annotations

import inspect
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from image_editing import pipeline
from image_editing.providers.base import ImageEditProvider


def test_provider_protocol_has_only_erase_and_edit_capabilities():
    members = {name for name, value in inspect.getmembers(ImageEditProvider) if not name.startswith("_")}
    assert "erase" in members
    assert "edit" in members
    assert "fill" not in members
    assert not hasattr(pipeline, "repair_wall")


def test_flux_pipeline_processes_every_manifest_item_with_mock_provider(tmp_path: Path):
    img = Image.new("RGB", (32, 24), (220, 220, 220))
    mask = Image.new("L", (32, 24), 0)
    art = Image.new("RGB", (8, 8), (255, 0, 0))
    bundle = tmp_path / "hanging.zip"
    with zipfile.ZipFile(bundle, "w") as z:
        items = []
        for idx in range(2):
            prefix = f"flux_inputs/items/wall_a__candidate_{idx}"
            hard = tmp_path / f"hard_{idx}.png"
            rm = tmp_path / f"remove_{idx}.png"
            am = tmp_path / f"art_{idx}.png"
            img.save(hard)
            mask.save(rm)
            mask.save(am)
            z.write(hard, f"{prefix}/hard_paste.png")
            z.write(rm, f"{prefix}/remove_mask.png")
            z.write(am, f"{prefix}/art_region_mask.png")
            z.writestr(f"{prefix}/render_prompt.txt", "render the hard-pasted artwork realistically")
            z.writestr(f"{prefix}/candidate.json", json.dumps({"candidate_id": f"candidate_{idx}"}))
            items.append({
                "asset_id": f"wall_a__candidate_{idx}",
                "candidate_id": f"candidate_{idx}",
                "wall_id": "wall_a",
                "hard_paste_path": f"{prefix}/hard_paste.png",
                "remove_mask_path": f"{prefix}/remove_mask.png",
                "art_region_mask_path": f"{prefix}/art_region_mask.png",
                "render_prompt_path": f"{prefix}/render_prompt.txt",
                "render_prompt": "render the hard-pasted artwork realistically",
                "exit_review": {"approved_for_flux_render": True},
            })
        z.writestr("flux_inputs/manifest.json", json.dumps({"version": 1, "items": items}))
    art_path = tmp_path / "artwork.png"
    art.save(art_path)
    out = tmp_path / "out"
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_bfl_hanging_flux_pipeline.py"
    subprocess.run(
        [sys.executable, str(script), "--hanging-zip", str(bundle), "--artwork-reference", str(art_path), "--output-dir", str(out), "--provider", "mock"],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    trace = json.loads((out / "edit_trace.json").read_text(encoding="utf-8"))
    assert len(trace["outputs"]) == 2
    assert (out / "final_hd" / "wall_a__candidate_0.png").exists()
    assert (out / "final_hd" / "wall_a__candidate_1.png").exists()


def test_v7_flux_pipeline_erases_original_then_hardpastes_and_dual_inputs(tmp_path: Path):
    original = Image.new("RGB", (64, 48), (200, 200, 200))
    mask = Image.new("L", (64, 48), 0)
    art_region = Image.new("L", (64, 48), 0)
    artwork = Image.new("RGBA", (16, 12), (255, 0, 0, 255))
    bundle = tmp_path / "hanging_v7.zip"
    with zipfile.ZipFile(bundle, "w") as z:
        prefix = "flux_inputs/items/wall_a__candidate_0"
        original_path = tmp_path / "original_room.png"
        artwork_path = tmp_path / "artwork.png"
        hard = tmp_path / "hard.png"
        rm = tmp_path / "remove.png"
        am = tmp_path / "art_mask.png"
        fm = tmp_path / "front_mask.png"
        original.save(original_path)
        artwork.save(artwork_path)
        original.save(hard)
        mask.save(rm)
        art_region.save(am)
        art_region.save(fm)
        z.write(original_path, "flux_inputs/original_room.png")
        z.write(artwork_path, "flux_inputs/artwork.png")
        z.write(hard, f"{prefix}/hard_paste.png")
        z.write(rm, f"{prefix}/remove_mask.png")
        z.write(am, f"{prefix}/art_region_mask.png")
        z.write(fm, f"{prefix}/front_region_mask.png")
        z.writestr(f"{prefix}/render_prompt.txt", "keep bbox fixed")
        z.writestr(f"{prefix}/candidate.json", json.dumps({"candidate_id": "candidate_0"}))
        item = {
            "asset_id": "wall_a__candidate_0",
            "candidate_id": "candidate_0",
            "wall_id": "wall_a",
            "original_room_path": "flux_inputs/original_room.png",
            "artwork_path": "flux_inputs/artwork.png",
            "hard_paste_path": f"{prefix}/hard_paste.png",
            "remove_mask_path": f"{prefix}/remove_mask.png",
            "art_region_mask_path": f"{prefix}/art_region_mask.png",
            "front_region_mask_path": f"{prefix}/front_region_mask.png",
            "render_prompt_path": f"{prefix}/render_prompt.txt",
            "render_prompt": "keep bbox fixed",
            "target_quad_original": [[10, 10], [30, 10], [30, 25], [10, 25]],
            "flux2_input_mode": "dual_original_plus_target",
            "exit_review": {"approved_for_flux_render": True},
        }
        z.writestr("flux_inputs/manifest.json", json.dumps({
            "version": 2,
            "original_room_path": "flux_inputs/original_room.png",
            "artwork_path": "flux_inputs/artwork.png",
            "items": [item],
        }))
    out = tmp_path / "out"
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_bfl_hanging_flux_pipeline.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--hanging-zip",
            str(bundle),
            "--output-dir",
            str(out),
            "--provider",
            "mock",
            "--dual-ab-compare",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    item_trace = json.loads((out / "wall_a__candidate_0" / "edit_trace.json").read_text(encoding="utf-8"))
    aggregate = json.loads((out / "edit_trace.json").read_text(encoding="utf-8"))
    assert item_trace["sequence_mode"] == "v7_original_erase_hardpaste_flux2"
    assert item_trace["legacy_fallback_reasons"] == []
    assert item_trace["requested_flux2_input_mode"] == "dual_original_plus_target"
    assert item_trace["flux2_input_mode"] == "dual_erased_room_plus_target"
    assert item_trace["flux2_mask_sent"] is False
    assert item_trace["render_trace"]["reference_count"] == 1
    assert item_trace["render_input_semantics"]["input_image"] == "cleaned_hard_paste_target_authoritative_canvas"
    assert item_trace["render_input_semantics"]["input_image_2"] == "erased_room_texture_lighting_reference"
    assert "raw_original_reference_prohibited" in item_trace["render_input_semantics"]["order_policy"]
    assert item_trace["render_trace"]["warning_label"] == "🔴 MOCK PROVIDER / NO-OP RESULT"
    assert "provider=mock" in aggregate["provider_warning"]
    assert item_trace["dual_ab_compare"]["enabled"] is True
    assert item_trace["dual_ab_compare"]["single_baseline_trace"]["reference_count"] == 0
    assert aggregate["dual_ab_compare"]["executed_count"] == 1
    assert item_trace["qa_art_mask_source"] == "front_region_mask_path"
    assert item_trace["render_size_qa_status"] in {"passed", "skipped_empty_art_region_mask"}
    assert item_trace["exit_code"] == "OK_SELECTED"
    assert item_trace["production_usable"] is False
    assert (out / "wall_a__candidate_0" / "01_erased.png").exists()
    assert (out / "wall_a__candidate_0" / "02_hard_paste_after_erase.png").exists()
    assert (out / "wall_a__candidate_0" / "03_flux2_render.png").exists()
    assert (out / "wall_a__candidate_0" / "03b_flux2_render_single_baseline.png").exists()
    assert (out / "final_hd_ab_compare" / "wall_a__candidate_0__single_baseline.png").exists()



def test_legacy_fallback_prints_warning_to_stderr(tmp_path: Path):
    img = Image.new("RGB", (32, 24), (220, 220, 220))
    mask = Image.new("L", (32, 24), 0)
    bundle = tmp_path / "legacy_hanging.zip"
    with zipfile.ZipFile(bundle, "w") as z:
        prefix = "flux_inputs/items/legacy_item"
        hard = tmp_path / "hard.png"
        rm = tmp_path / "remove.png"
        am = tmp_path / "art.png"
        img.save(hard)
        mask.save(rm)
        mask.save(am)
        z.write(hard, f"{prefix}/hard_paste.png")
        z.write(rm, f"{prefix}/remove_mask.png")
        z.write(am, f"{prefix}/art_region_mask.png")
        z.writestr(f"{prefix}/render_prompt.txt", "legacy render")
        items = [{
            "asset_id": "legacy_item",
            "candidate_id": "candidate_legacy",
            "wall_id": "wall_a",
            "hard_paste_path": f"{prefix}/hard_paste.png",
            "remove_mask_path": f"{prefix}/remove_mask.png",
            "art_region_mask_path": f"{prefix}/art_region_mask.png",
            "render_prompt_path": f"{prefix}/render_prompt.txt",
            "render_prompt": "legacy render",
        }]
        z.writestr("flux_inputs/manifest.json", json.dumps({"version": 1, "items": items}))
    out = tmp_path / "out"
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_bfl_hanging_flux_pipeline.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--hanging-zip", str(bundle), "--output-dir", str(out), "--provider", "mock"],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert "WARNING legacy_hardpaste_erase_flux2 fallback" in proc.stderr
    trace = json.loads((out / "edit_trace.json").read_text(encoding="utf-8"))
    assert any(w.get("sequence_mode") == "legacy_hardpaste_erase_flux2" for w in trace["warnings"])
    assert any(w.get("warning_label") == "🔴 MOCK PROVIDER / NO-OP RESULT" for w in trace["warnings"])
    item_trace = json.loads((out / "legacy_item" / "edit_trace.json").read_text(encoding="utf-8"))
    assert item_trace["sequence_mode"] == "legacy_hardpaste_erase_flux2"
    assert "missing_original_room_path" in item_trace["legacy_fallback_reasons"]


def test_bfl_partial_preview_filename_marks_frame_side_mode(tmp_path: Path):
    original = Image.new("RGB", (64, 48), (200, 200, 200))
    front_mask = Image.new("L", (64, 48), 0)
    for x in range(12, 30):
        for y in range(12, 28):
            front_mask.putpixel((x, y), 255)
    art_region = front_mask.copy()
    remove_mask = Image.new("L", (64, 48), 0)
    artwork = Image.new("RGBA", (16, 12), (255, 0, 0, 255))
    bundle = tmp_path / "hanging_v7_partial.zip"
    with zipfile.ZipFile(bundle, "w") as z:
        prefix = "flux_inputs/items/wall_a__candidate_0"
        original_path = tmp_path / "original_room.png"
        artwork_path = tmp_path / "artwork.png"
        hard = tmp_path / "hard.png"
        rm = tmp_path / "remove.png"
        am = tmp_path / "art_mask.png"
        fm = tmp_path / "front_mask.png"
        original.save(original_path)
        artwork.save(artwork_path)
        original.save(hard)
        remove_mask.save(rm)
        art_region.save(am)
        front_mask.save(fm)
        z.write(original_path, "flux_inputs/original_room.png")
        z.write(artwork_path, "flux_inputs/artwork.png")
        z.write(hard, f"{prefix}/hard_paste.png")
        z.write(rm, f"{prefix}/remove_mask.png")
        z.write(am, f"{prefix}/art_region_mask.png")
        z.write(fm, f"{prefix}/front_region_mask.png")
        z.writestr(f"{prefix}/render_prompt.txt", "keep bbox fixed")
        z.writestr(f"{prefix}/candidate.json", json.dumps({"candidate_id": "candidate_0"}))
        item = {
            "asset_id": "wall_a__candidate_0",
            "candidate_id": "candidate_0",
            "wall_id": "wall_a",
            "original_room_path": "flux_inputs/original_room.png",
            "artwork_path": "flux_inputs/artwork.png",
            "hard_paste_path": f"{prefix}/hard_paste.png",
            "remove_mask_path": f"{prefix}/remove_mask.png",
            "art_region_mask_path": f"{prefix}/art_region_mask.png",
            "front_region_mask_path": f"{prefix}/front_region_mask.png",
            "render_prompt_path": f"{prefix}/render_prompt.txt",
            "render_prompt": "keep bbox fixed",
            "target_quad_original": [[12, 12], [30, 12], [30, 28], [12, 28]],
            "artwork_mount_policy": {
                "precompose_frame_sides": True,
                "front_quad_original": [[12, 12], [30, 12], [30, 28], [12, 28]],
                "visible_side_faces_original": [],
            },
            "exit_review": {"approved_for_flux_render": True},
        }
        z.writestr("flux_inputs/manifest.json", json.dumps({
            "version": 2,
            "original_room_path": "flux_inputs/original_room.png",
            "artwork_path": "flux_inputs/artwork.png",
            "items": [item],
        }))
    out = tmp_path / "out"
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_bfl_hanging_flux_pipeline.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--hanging-zip",
            str(bundle),
            "--output-dir",
            str(out),
            "--provider",
            "mock",
            "--qa-min-core-masked-ssim",
            "1.01",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    item_trace = json.loads((out / "wall_a__candidate_0" / "edit_trace.json").read_text(encoding="utf-8"))
    assert item_trace["exit_code"] == "PARTIAL_HARDPASTE_REVIEW"
    assert item_trace["qa_art_mask_source"] == "front_region_mask_path"
    preview_path = Path(item_trace["hardpaste_preview_path"])
    assert preview_path.name.endswith("__frame_side_hardpaste_preview.png")
    assert preview_path.exists()


def test_strict_v7_manifest_refuses_legacy_fallback_when_artwork_missing(tmp_path: Path):
    original = Image.new("RGB", (64, 48), (200, 200, 200))
    mask = Image.new("L", (64, 48), 0)
    bundle = tmp_path / "hanging_v7_missing_artwork.zip"
    with zipfile.ZipFile(bundle, "w") as z:
        prefix = "flux_inputs/items/wall_a__candidate_0"
        original_path = tmp_path / "original_room.png"
        hard = tmp_path / "hard.png"
        rm = tmp_path / "remove.png"
        am = tmp_path / "art_mask.png"
        original.save(original_path)
        original.save(hard)
        mask.save(rm)
        mask.save(am)
        z.write(original_path, "flux_inputs/original_room.png")
        z.write(hard, f"{prefix}/hard_paste.png")
        z.write(rm, f"{prefix}/remove_mask.png")
        z.write(am, f"{prefix}/art_region_mask.png")
        z.writestr(f"{prefix}/render_prompt.txt", "strict v7 render")
        item = {
            "asset_id": "wall_a__candidate_0",
            "candidate_id": "candidate_0",
            "wall_id": "wall_a",
            "original_room_path": "flux_inputs/original_room.png",
            "artwork_path": "flux_inputs/artwork.png",
            "hard_paste_path": f"{prefix}/hard_paste.png",
            "remove_mask_path": f"{prefix}/remove_mask.png",
            "art_region_mask_path": f"{prefix}/art_region_mask.png",
            "render_prompt_path": f"{prefix}/render_prompt.txt",
            "render_prompt": "strict v7 render",
            "target_quad_original": [[10, 10], [30, 10], [30, 25], [10, 25]],
            "erase_order": "original_room_then_hardpaste_after_erase",
        }
        z.writestr("flux_inputs/manifest.json", json.dumps({
            "version": 2,
            "erase_order": "original_room_then_hardpaste_after_erase",
            "original_room_path": "flux_inputs/original_room.png",
            "artwork_path": "flux_inputs/artwork.png",
            "items": [item],
        }))
    out = tmp_path / "out"
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_bfl_hanging_flux_pipeline.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--hanging-zip", str(bundle), "--output-dir", str(out), "--provider", "mock"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "references missing file" in proc.stderr or "strict v7 manifest" in proc.stderr
    assert "legacy_hardpaste_erase_flux2 fallback" not in proc.stderr


def test_bfl_client_from_env_enforces_poll_safety_floor(monkeypatch):
    from image_editing.providers.bfl import BflClient

    monkeypatch.setenv("BFL_TIMEOUT_S", "0.5")
    monkeypatch.setenv("BFL_POLL_INTERVAL_S", "0.01")
    client = BflClient.from_env()
    assert client.timeout_s == 10.0
    assert client.poll_interval_s == 0.25
