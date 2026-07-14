import json
import shutil
import zipfile

import numpy as np
from PIL import Image

from worker import pipeline


def test_primary_wall_reuses_persisted_stage_a_without_render_call(tmp_path, monkeypatch):
    source = tmp_path / "persisted-stage-a.png"
    pixels = np.zeros((32, 48, 3), dtype=np.uint8)
    pixels[:, :] = [91, 113, 137]
    Image.fromarray(pixels).save(source)

    manifest = {
        "items": [
            {"asset_id": "wall_main__candidate_1", "wall_id": "wall_main", "candidate_id": "candidate_1"},
            {"asset_id": "wall_other__candidate_2", "wall_id": "wall_other", "candidate_id": "candidate_2"},
        ]
    }
    archive_path = tmp_path / "hanging_result.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("flux_inputs/manifest.json", json.dumps(manifest))

    downloads = []

    def fake_download(url, destination):
        downloads.append(url)
        shutil.copyfile(source, destination)
        return destination

    monkeypatch.setattr(pipeline, "_download", fake_download)
    output_dir = tmp_path / "flux_out"
    aggregate = pipeline.prepare_pre_styling_render_aggregate(
        archive_path,
        output_dir,
        stage_a_base_by_wall_id={"wall_main": "https://example.test/main-stage-a.png"},
        target_wall_ids=["wall_main"],
        provider="apiyi_gpt_image2_vip",
    )

    assert downloads == ["https://example.test/main-stage-a.png"]
    assert aggregate["stage_a_render_call_count"] == 0
    assert aggregate["production_usable_output_count"] == 1
    assert aggregate["outputs"][0]["asset_id"] == "wall_main__candidate_1"
    rendered = np.asarray(Image.open(output_dir / "final_hd" / "wall_main__candidate_1.png").convert("RGB"))
    assert np.array_equal(rendered, pixels)


def test_primary_wall_reuse_fails_closed_when_base_does_not_match_manifest(tmp_path):
    archive_path = tmp_path / "hanging_result.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("flux_inputs/manifest.json", json.dumps({
            "items": [{"asset_id": "wall_a__candidate_1", "wall_id": "wall_a", "candidate_id": "candidate_1"}]
        }))

    try:
        pipeline.prepare_pre_styling_render_aggregate(
            archive_path,
            tmp_path / "out",
            stage_a_base_by_wall_id={"wall_missing": "https://example.test/missing.png"},
            target_wall_ids=["wall_a"],
        )
    except RuntimeError as exc:
        assert "did not match" in str(exc)
    else:
        raise AssertionError("missing Stage A base mapping must fail closed")
