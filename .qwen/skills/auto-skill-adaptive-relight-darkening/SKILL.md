---
name: adaptive-relight-darkening-compensation
description: When render darkens artwork more than surrounding environment, art_compensation_delta must brighten it back to maintain brightness consistency (project)
source: auto-skill
extracted_at: '2026-06-27T00:15:00.000Z'
---

## Problem

Adaptive relight only compensates for underexposed artwork (`under_score > 0`) or when environment gets brighter (`ambient_delta > 0`). When render darkens the artwork more than the surrounding wall (`render_object_delta < ambient_delta`), no compensation was applied, causing artwork to appear unnaturally dark compared to its surroundings.

## Rule

In `_apply_adaptive_relight()` (run_bfl_hanging_flux_pipeline.py), add darkening compensation:

```python
# When render darkens artwork more than environment, compensate
if render_object_delta < ambient_delta - 3.0:
    darkening_gap = ambient_delta - render_object_delta
    art_compensation = max(art_compensation, min(darkening_gap * 0.6, 20.0))
```

## Why

- `ambient_delta` = median luma change in wall ring around artwork
- `render_object_delta` = median luma change in artwork region
- When `render_object_delta < ambient_delta`, artwork was darkened MORE than surroundings
- The 3.0 threshold avoids noise-triggering
- 60% compensation ratio preserves some of render's intentional darkening
- 20.0 cap prevents over-compensation

## How to apply

When modifying adaptive relight logic, always check both brightening AND darkening consistency between artwork and environment. The artwork's brightness change should be explainable by either:
1. Underexposed original (dark capture)
2. Environment getting brighter/darker synchronously
3. Render making coherent lighting changes
