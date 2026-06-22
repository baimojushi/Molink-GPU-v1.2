---
name: artwork-vlm-integration
description: Artwork VLM frame analysis requires both pipeline form fields and VLM client image size/MIME handling
source: auto-skill
extracted_at: '2026-06-14T05:22:39.514Z'
---

# Artwork VLM Integration

Artwork frame analysis (`artwork_analysis` in semantic output) requires cooperation between pipeline script and VLM client. Both layers must be correct.

## Pipeline requirements

`run_v1_full_pipeline.sh` must pass two form fields to **both** the parallel seed call and the serial semantic call:

```bash
ARTWORK_VLM_ENABLE="${ARTWORK_VLM_ENABLE:-1}"  # default enabled

# Parallel seed call:
-F "artwork_image=@${ARTWORK_IMAGE}" \
-F "artwork_vlm_enable=${ARTWORK_VLM_ENABLE}"

# Serial semantic call (non-parallel branch):
-F "artwork_image=@${ARTWORK_IMAGE}" \
-F "artwork_vlm_enable=${ARTWORK_VLM_ENABLE}"
```

The alignment call (step 2, consumes `semantic_seed_json`) intentionally skips VLM/SAM and does NOT need these fields.

## VLM client requirements

`semantic_mvp/vlm_client.py` must handle two failure modes that produce dashscope 400 errors:

### 1. MIME auto-detection

Do not hardcode `image/jpeg`. Artwork images may be PNG, WebP, etc. Use `_detect_mime()` to inspect magic bytes and set the correct MIME in the data URL.

### 2. Image size limiting

Dashscope rejects payloads that are too large. A 39MB PNG produces ~52MB base64, which triggers 400 Bad Request.

`_resize_image()` must cap the maximum side to `VLM_MAX_IMAGE_SIDE` (default 2048px) before encoding:
- If both dimensions ≤ max_side, pass through unchanged
- Otherwise resize preserving aspect ratio, save as JPEG quality 85%

## Verification

After pipeline run, check:

```bash
jq '.artwork_analysis' semantics_seed.json
jq '.artwork_analysis' semantics.json
```

**Expected (working):**
```json
{"enabled": true, "status": "ok", "source": "openai_compatible", "has_frame": false, "confidence": 0.95, ...}
```

**Before fix (skipped):**
```json
{"enabled": false, "status": "not_requested", "reason": "artwork_image_not_supplied_or_artwork_vlm_disabled"}
```

**Before fix (VLM error):**
```json
{"enabled": true, "status": "error", "reason": "artwork_frame_vlm_failed: 400 Client Error: Bad Request for url: ..."}
```

If `status=uncertain`, VLM ran but model was unsure — distinct from "not_requested" or "error".
