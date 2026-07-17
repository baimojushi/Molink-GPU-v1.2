---
name: render-qa-diagnosis
description: Diagnose render QA failures — collect diagnostic bundle, parse result_payload.json QA section, trace SSIM/color delta failures through hardpaste→flux pipeline
source: auto-skill
extracted_at: '2026-07-17T02:56:13.555Z'
---

## When to use
Pipeline returns `status=render_review`, `exit_code=OK_SELECTED`, `candidates=N`, `final_urls=0` — candidates exist but no production-ready images.

## Root cause pattern
Render QA uses `mask_anchored_adaptive_relight_and_forbidden_outer_guard` method with these hard thresholds:
- `core_masked_ssim ≥ 0.78` — structural similarity of artwork region after render
- `core_color_delta ≤ 45.0` — color difference in artwork region
- `forbidden_outer_guard_changed_ratio ≤ 0.65` — unchanged pixel ratio outside artwork

If both `core_masked_ssim` and `core_color_delta` fail, `render_exit_code=PARTIAL_HARDPASTE_REVIEW`, `production_usable=false`. A partial preview image is generated from `frame_side_hardpaste_preview` but marked `needs_review=true`.

## Diagnostic workflow

### 1. Locate run directory
```bash
ls /root/autodl-fs/hanging_runs/ | grep <job_short_id>
```

### 2. Parse result_payload.json
```bash
python3 -c "
import json
with open('99_result/result_payload.json') as f:
    d = json.load(f)
c = d['candidates'][0]
qa = c.get('qa', {})
print('render_exit_code:', c.get('render_exit_code'))
print('production_usable:', c.get('production_usable'))
print('core_masked_ssim:', qa.get('core_masked_ssim'), '≥ 0.78?')
print('core_color_delta:', qa.get('core_color_delta'), '≤ 45.0?')
print('outer_guard_changed:', qa.get('forbidden_outer_guard_changed_ratio'), '≤ 0.65?')
print('failures:', qa.get('failures'))
print('partial_review:', c.get('partial_review_image', {}).get('reason'))
"
```

### 3. Package diagnostic artifacts
```bash
mkdir -p /root/diag_<job_short_id>
cp -r <run_dir>/05_render /root/diag_<job_short_id>/
cp <run_dir>/04_hanging/client_result.json /root/diag_<job_short_id>/
cp <run_dir>/99_result/result_payload.json /root/diag_<job_short_id>/
grep '<job_short_id>' /opt/molink/hanging/logs/worker.log > /root/diag_<job_short_id>/worker.log
tar czf /root/diag_<job_short_id>.tar.gz -C /root diag_<job_short_id>
rm -rf /root/diag_<job_short_id>
```

### 4. Key files for QA analysis
| File | Purpose |
|------|---------|
| `05_render/.../inputs/original_room.png` | Source room image |
| `05_render/.../inputs/hard_paste.png` | Hardpaste result before QA |
| `05_render/.../inputs/art_region_mask.png` | Artwork region mask used for SSIM |
| `05_render/.../02_hard_paste_after_erase.png` | Post-erase hardpaste |
| `05_render/.../03_flux2_render.png` | Final render output |
| `05_render/.../partial_review/*.png` | QA-failed fallback preview |
| `05_render/.../edit_trace.json` | Render provider trace |

## Worker progress bug
`worker/main.py:send_result()` treats `render_review` as a success for `result_stage` (goes to `r2_upload`) but sends `stage=failed` progress message because status is not `succeeded` or `succeeded_partial`. Business servers interpret `stage=failed` as pipeline failure. This is a cosmetic bug — the pipeline actually completed with a partial result.

## Fix patterns
- If SSIM too low: check hardpaste alignment, mask precision, artwork warping
- If color delta too high: check color space conversion, luminance matching, wall tone application
- If both fail: likely render provider returned significantly different pixels than hardpaste input
