---
name: bfl-erase-image-downscale
description: BFL erase-v1 API returns 422 when original_room exceeds ~2048px longest dimension — downscale image+mask before erase, scale dilate_pixels proportionally
source: auto-skill
extracted_at: '2026-06-25T11:44:47.577Z'
---

## Problem

BFL erase-v1 API (`/v1/flux-tools/erase-v1`) returns **422 Unprocessable Entity** when the input image (original_room) exceeds approximately 2048px on the longest dimension. A 3648x2736 room image base64-encoded to ~10MB triggers this failure.

**Diagnosis pattern:**
- Compare failed vs successful job input sizes: `room=(3648, 2736) 2.1MB` vs `room=(1664, 768) 0.4MB`
- Check `05_render/render_aggregate.json` for `"errors": [{"error": "BFL API request failed after 3 attempts: 422 ..."}]`
- Check `_request_json()` in `bfl.py` — without HTTP body capture, the actual 422 reason is lost

## Fix 1 — HTTP error body capture in BflClient

In `image_editing/providers/bfl.py`, `BflClient._request_json()`, add before `resp.raise_for_status()`:

```python
if resp.status_code >= 400:
    raise RuntimeError(f"BFL {method} {url} -> {resp.status_code}: {resp.text[:1000]}")
```

## Fix 2 — Downscale before BFL erase

In `BflImageEditProvider.erase()`, after `_ensure_same_image_size(image, mask)` and before building the payload, insert downscale logic (see code block below).

Add env var to `/etc/molink/gpu.env` (worker config, NOT project `.env`):

```
BFL_ERASE_MAX_LONG_EDGE=2048
```

Set to `0` to disable downscale.

## Where to apply

- `image_editing/providers/bfl.py`: `BflClient._request_json()` — HTTP error body capture
- `image_editing/providers/bfl.py`: `BflImageEditProvider.erase()` — downscale before base64 encoding
- `/etc/molink/gpu.env`: add `BFL_ERASE_MAX_LONG_EDGE=2048` (requires worker restart)
