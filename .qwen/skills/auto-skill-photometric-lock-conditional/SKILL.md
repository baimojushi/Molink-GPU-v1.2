---
name: photometric-lock-conditional-trigger
description: photometric_lock must only run when render QA fails — skip it when QA passes to preserve render quality (project)
source: auto-skill
extracted_at: '2026-06-27T00:15:00.000Z'
---

## Problem

`MOLINK_FINAL_PHOTOMETRIC_LOCK=1` unconditionally applies photometric lock after every render, even when the render output is already correct. This can degrade good renders by forcing unnecessary brightness/color adjustments.

## Rule

Run QA **before** photometric_lock. Only apply photometric_lock when:
- QA fails (`qa_passed == False`), OR
- `MOLINK_FORCE_PHOTOMETRIC_LOCK=1` is explicitly set

## Implementation

```python
# In run_bfl_hanging_flux_pipeline.py:
# 1. Run QA first
primary_render_size_qa = _run_mask_anchored_qa(args, hard_paste, render_result.image, qa_art_mask)
qa_passed = primary_render_size_qa.get("qa_passed", False)

# 2. Only apply photometric_lock if QA failed
if _env_truthy("MOLINK_FINAL_PHOTOMETRIC_LOCK", "1") and (not qa_passed or _env_truthy("MOLINK_FORCE_PHOTOMETRIC_LOCK", "0")):
    # ... apply photometric_lock
```

## Why

When render QA passes (SSIM good, no forbidden outer guard violation, luma/chroma within bounds), the render output is already faithful to the hard-paste target. Applying photometric_lock would only risk degrading it.

## How to apply

When modifying the render pipeline, always run QA before photometric_lock and gate the lock on QA failure.
