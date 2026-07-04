---
name: artwork-vlm-not-invoked
description: parse_artwork_frame_with_vlm exists but was never called in production — now fixed via _analyze_artwork_image_with_vlm() in hanging_mvp/service.py; artwork surface style and frame thickness are derived from VLM analysis of actual artwork image
source: auto-skill
extracted_at: '2026-07-04T01:30:00.000Z'
updated_at: '2026-07-04T02:45:00.000Z'
---

## Problem (resolved)

`semantic_mvp/vlm_client.parse_artwork_frame_with_vlm()` exists and classifies artwork images by surface style:
- `refined_paper_or_delicate` (fine art, paper, 国画)
- `rough_or_canvas` (canvas, oil painting, acrylic, fabric)
- `chinese_hanging_scroll` (立轴, 国画立轴)

This classification drives `side_thickness_policy` (`none`/`normal`/`double`) which controls frame side face depth in render.

**Was never called in production code** — only in test files. Fixed by two patches:

## Applied fix

### Patch 1: semantic_mvp + worker pipeline (molink-gpu-artwork-vlm-fix.patch)
- `worker/pipeline.py`: semantic request now includes `artwork_image` and `artwork_vlm_enable`
- `worker/main.py`: passes artwork path to semantic call
- `semantic_mvp/service.py`: accepts `artwork_image` / `artwork_vlm_enable`, calls `parse_artwork_frame_with_vlm()`, writes result to `artwork_analysis`

### Patch 2: hanging-level VLM fallback (molink-gpu-artwork-vlm-frame-fix.patch)
- `hanging_mvp/service.py`: adds `_analyze_artwork_image_with_vlm()` function
  - Checks `ARTWORK_FRAME_VLM_ENABLED` env var (default=1)
  - Calls `parse_artwork_frame_with_vlm(artwork_raw, mode, art_w, art_h)`
  - Falls back to seed analysis on VLM failure (fail-closed, never blocks hanging)
- Replaces `artwork_analysis = semantics.get("artwork_analysis")` with:
  ```python
  seed_artwork_analysis = semantics.get("artwork_analysis") if isinstance(semantics.get("artwork_analysis"), dict) else {}
  artwork_analysis = _analyze_artwork_image_with_vlm(artwork_raw, art_rgba, seed_artwork_analysis)
  artwork_analysis = _augment_artwork_analysis_with_pix(artwork_analysis, art_rgba)
  ```

## VLM output fields now populated

- `has_frame`, `confidence`, `frame_style`
- `artwork_surface_style` (refined_paper_or_delicate / rough_or_canvas / chinese_hanging_scroll / uncertain)
- `side_thickness_policy` (none / normal / double)
- `frame_width_ratio` (left/right/top/bottom)
- `dominant_edge_colors_rgb`
- `mood_register`, `visual_complexity`
- `source` (artwork_image_vlm / semantic_vlm / rules_bool)

## Env vars required

```bash
ARTWORK_FRAME_VLM_ENABLED=1       # default, enables VLM analysis
ARTWORK_FRAME_VLM_MODE=auto       # or openai_compatible
VLM_PROVIDER=openai_compatible
VLM_API_URL=...
VLM_API_KEY=...
VLM_MODEL=...
ARTWORK_FRAME_MIN_CONFIDENCE=0.72  # threshold for should_precompose_frame_sides
```

## Impact (after fix)

- Canvas/oil paintings → `rough_or_canvas` → `side_thickness_policy: double` → 2x frame depth
- Chinese scroll paintings → `chinese_hanging_scroll` → `side_thickness_policy: none` → no side faces
- Fine art/paper → `refined_paper_or_delicate` → `side_thickness_policy: normal`
- VLM unavailable → fallback to seed/rules analysis, hanging continues uninterrupted
