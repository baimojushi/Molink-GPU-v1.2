---
name: erase-output-size-normalization
description: BFL erase-v1 API may return a different image size than the input — always resize back to original dimensions and add a defensive guard in the pipeline to refuse mismatched sizes before hardpaste
source: auto-skill
extracted_at: '2026-06-25T23:10:00.000Z'
---

# BFL Erase Output Size Normalization

## Problem

The BFL erase-v1 API (`POST /v1/flux-tools/erase-v1`) may return an erased image at a different resolution than the input image. This breaks the downstream hanging pipeline because:

- Hard-paste uses original-image-space quad coordinates
- Masks and artifact positions are computed in the original coordinate space
- If the erase canvas is resized, all downstream operations project to the wrong pixels

## Fix in BflImageEditProvider.erase()

Capture the original input size before any downscale, then resize the provider output back after download:

```python
with Image.open(image) as src_img:
    original_size = src_img.size

# ... existing downscale + BFL API call ...

self.client.download_sample(result, out_path)

resize_back_trace: dict[str, Any] = {"applied": False}
with Image.open(out_path) as erased_img:
    if erased_img.size != original_size:
        raw_out_path = out_path.with_name(f"{out_path.stem}.provider_raw{out_path.suffix}")
        shutil.move(str(out_path), str(raw_out_path))
        erased_img.convert("RGB").resize(original_size, Image.Resampling.LANCZOS).save(out_path)
        resize_back_trace = {
            "applied": True,
            "provider_output_path": str(raw_out_path),
            "provider_output_size": list(erased_img.size),
            "restored_output_size": list(original_size),
            "reason": "keep_cleanup_output_in_original_coordinate_space",
        }

trace = {
    # ... existing trace fields ...
    "resize_back_to_input_size": resize_back_trace,
}
```

## Defensive Guard in Pipeline

Add a size-mismatch guard in `run_bfl_hanging_flux_pipeline.py` before `_hardpaste_after_erase`:

```python
# V7 path (erase from original_room):
with Image.open(erase_result.image) as erased_img, Image.open(original_room) as orig_img:
    if erased_img.size != orig_img.size:
        raise RuntimeError(
            f"cleanup output size {erased_img.size} differs from original_room size {orig_img.size}; "
            "refusing to hardpaste original-space quad onto resized erase canvas"
        )

# Legacy path (erase from hard_paste):
with Image.open(erase_result.image) as erased_img, Image.open(erase_input) as inp_img:
    if erased_img.size != inp_img.size:
        raise RuntimeError(
            f"cleanup output size {erased_img.size} differs from erase_input size {inp_img.size}; "
            "refusing to hardpaste onto resized erase canvas"
        )
```

## Files Modified

- `image_editing/providers/bfl.py` — `BflImageEditProvider.erase()`
- `scripts/run_bfl_hanging_flux_pipeline.py` — both v7 and legacy hardpaste paths
