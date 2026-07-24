from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image

from image_editing.providers.apiyi_gpt_image2 import ApiyiGptImage2Client
from image_editing.providers.base import ImageEditRequest


def test_apiyi_explicit_full_space_mode_never_uses_local_crop(tmp_path: Path, monkeypatch):
    image_path = tmp_path / "room.png"
    Image.fromarray(np.full((72, 96, 3), [180, 175, 170], dtype=np.uint8)).save(image_path)
    captured = {}

    def fake_post(self, *, data, files):
        captured["prompt"] = data["prompt"]
        file_list = list(files)
        captured["upload_names"] = [item[1][0] for item in file_list]
        handle = file_list[0][1][1]
        handle.seek(0)
        with Image.open(handle) as sent:
            captured["sent_size"] = list(sent.size)
            buf = io.BytesIO()
            sent.convert("RGB").save(buf, format="PNG")
        return {"data": [{"b64_json": base64.b64encode(buf.getvalue()).decode("ascii")}]}

    monkeypatch.setattr(ApiyiGptImage2Client, "_post_images_edit", fake_post)
    client = ApiyiGptImage2Client(
        api_key="test",
        compress_inputs=False,
        include_references=False,
        allow_unmasked_fallback=False,
        resize_to_input_size=True,
    )
    result = client.edit(
        ImageEditRequest(
            image=image_path,
            mask=None,
            prompt="Integrate the mounted artwork.",
            output_dir=tmp_path,
            output_name="rendered.png",
            metadata={
                "full_space_edit_mode": "render_artwork_anchor",
                "front_polygon_px": [[30, 20], [50, 20], [50, 42], [30, 42]],
                "front_bbox_px": [30, 20, 50, 42],
            },
        )
    )

    assert result.trace["output_scope"] == "full_space_artwork_anchor"
    assert result.trace["full_space_artwork_anchor"] is True
    assert result.trace["mask_enforced_locally"] is False
    assert result.trace["mask_adapter"]["local_crop_disabled"] is True
    assert result.trace["target_image_sent_to_api"] == str(image_path)
    assert captured["sent_size"] == [96, 72]
    assert "FULL-SPACE PROTECTED ARTWORK ANCHOR" in captured["prompt"]
    assert "不是生图编辑区域" in captured["prompt"]


def test_apiyi_styling_scope_is_full_frame_and_not_mask_bounded(tmp_path: Path, monkeypatch):
    image_path = tmp_path / "room.png"
    Image.fromarray(np.full((60, 80, 3), [150, 145, 140], dtype=np.uint8)).save(image_path)
    captured = {}

    def fake_post(self, *, data, files):
        captured["prompt"] = data["prompt"]
        handle = list(files)[0][1][1]
        handle.seek(0)
        with Image.open(handle) as sent:
            buf = io.BytesIO()
            sent.convert("RGB").save(buf, format="PNG")
        return {"data": [{"b64_json": base64.b64encode(buf.getvalue()).decode("ascii")}]}

    monkeypatch.setattr(ApiyiGptImage2Client, "_post_images_edit", fake_post)
    client = ApiyiGptImage2Client(
        api_key="test",
        compress_inputs=False,
        include_references=False,
        allow_unmasked_fallback=False,
    )
    result = client.edit(
        ImageEditRequest(
            image=image_path,
            prompt="Add coordinated cushions and a small plant.",
            output_dir=tmp_path,
            output_name="styled.png",
            metadata={
                "full_space_edit_mode": "styling_artwork_protected",
                "front_polygon_px": [[25, 15], [45, 15], [45, 35], [25, 35]],
            },
        )
    )
    assert result.trace["output_scope"] == "full_space_styling_artwork_protected"
    assert result.trace["full_space_styling_artwork_protected"] is True
    assert result.trace["mask_enforced_locally"] is False
    assert "不提供用于限制创作范围的编辑 mask" in captured["prompt"]
    assert "挂画占位是唯一受保护区域" in captured["prompt"]
