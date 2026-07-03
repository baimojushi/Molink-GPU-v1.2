---
name: artwork-vlm-not-invoked
description: parse_artwork_frame_with_vlm exists but is never called in production — artwork surface style and frame thickness default to rules config, not VLM analysis of actual artwork image
source: auto-skill
extracted_at: '2026-07-04T01:30:00.000Z'
---

## Problem

`semantic_mvp/vlm_client.parse_artwork_frame_with_vlm()` exists and can classify artwork images by surface style:
- `refined_paper_or_delicate` (fine art, paper, 国画)
- `rough_or_canvas` (canvas, oil painting, acrylic, fabric)
- `chinese_hanging_scroll` (立轴, 国画立轴)

This classification drives `side_thickness_policy` (`none`/`normal`/`double`) which controls frame side face depth in render.

**However, the function is never called in production code.** It is only invoked in test files (`tests/test_artwork_mount_policy.py`).

## Current flow (broken)

1. `hanging_mvp/service.py:analyze_hanging_mvp()` receives `artwork_image: UploadFile`
2. `artwork_analysis = semantics.get("artwork_analysis")` — but semantic service never outputs this field
3. `_augment_artwork_analysis_with_pix()` fills `visual_complexity` from pixel HF energy only
4. Result: `artwork_surface_style="uncertain"`, `side_thickness_policy="normal"` (default), `frame_depth_m=0.025` (default × 1.0)

## Expected flow

Hanging endpoint should call:
```python
artwork_analysis = parse_artwork_frame_with_vlm(artwork_raw, vlm_mode, art_w, art_h)
```

This would populate `artwork_surface_style`, `side_thickness_policy`, `has_frame`, `confidence`, `frame_width_ratio`, and `dominant_edge_colors_rgb` from actual VLM analysis of the artwork image.

## Fix location

In `hanging_mvp/service.py`, around line 1037, after loading `artwork_raw`:
```python
# Current (broken):
artwork_analysis = semantics.get("artwork_analysis") if isinstance(semantics.get("artwork_analysis"), dict) else {}

# Should be:
artwork_analysis = parse_artwork_frame_with_vlm(artwork_raw, vlm_mode, art_w, art_h)
# Then augment with pixel-based complexity:
artwork_analysis = _augment_artwork_analysis_with_pix(artwork_analysis, art_rgba)
```

## Impact

Without this fix:
- All artwork gets the same default frame thickness regardless of actual surface type
- Canvas/oil paintings don't get the `double` side thickness they need for realistic rendering
- Chinese scroll paintings incorrectly render frame side faces
- Edge color matching for frame compositing is not available
