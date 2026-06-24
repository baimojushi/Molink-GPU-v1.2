---
name: render-provider-tracing
description: Trace render provider (apiyi_gpt_image2_vip vs bfl) through the full pipeline from backend job to final render, to diagnose why the wrong model is used
source: auto-skill
extracted_at: '2026-06-24T14:57:26.677Z'
---

# Render Provider Tracing

## Problem

Render task uses an unexpected model (e.g. BFL FLUX.2 instead of APIYI gpt-image-2-vip), but the local codebase has `apiyi_gpt_image2_vip` as the correct default everywhere.

## Full Provider Chain

```
Backend → WebSocket job payload → worker/main.py → render_api.py → run_bfl_hanging_flux_pipeline.py → provider dispatch
```

### Step 1 — Check the actual run output

```bash
# Read edit_trace.json from the run directory
python3 -c "
import json
with open('<run_dir>/05_render/flux_out/edit_trace.json') as f:
    trace = json.load(f)
print('provider:', trace.get('provider'))
print('render_provider:', trace.get('render_provider'))

# Per-item trace
with open('<run_dir>/05_render/flux_out/<item>/edit_trace.json') as f:
    item = json.load(f)
print('render_trace.provider:', item.get('render_trace', {}).get('provider'))
print('render_trace.endpoint:', item.get('render_trace', {}).get('endpoint'))
print('selected_apiyi_gpt_image2_vip:', item.get('selected_apiyi_gpt_image2_vip'))
"
```

### Step 2 — Check the manifest

```bash
python3 -c "
import json, zipfile
with zipfile.ZipFile('<run_dir>/04_hanging/hanging_result.zip') as z:
    m = json.loads(z.read('flux_inputs/manifest.json'))
    print('render_provider:', m.get('render_provider'))
"
```

**Note:** `render_provider` in the manifest is **metadata only** (set by `hanging_mvp/service.py` as `"bfl_flux2_edit"`). It does **not** control the actual provider used.

### Step 3 — Check the render aggregate

```bash
python3 -c "
import json
with open('<run_dir>/05_render/render_aggregate.json') as f:
    agg = json.load(f)
print('provider:', agg.get('provider'))
"
```

If this shows `bfl`, the root cause is confirmed: `args.provider` was `"bfl"` when `run_pipeline()` was called.

### Step 4 — Trace back to worker/main.py

The provider flows through:

```
worker/main.py L251:
  render_provider = (job.get("render") or {}).get("provider", "apiyi_gpt_image2_vip")
  → render_api.run_pipeline(provider=render_provider)

worker/render_api.py L83:
  def run_pipeline(..., provider="apiyi_gpt_image2_vip"):
    args.provider = provider
    → bfl.run_pipeline(args)  # calls run_bfl_hanging_flux_pipeline.run_pipeline

scripts/run_bfl_hanging_flux_pipeline.py L449:
  provider_name = str(args.provider or "apiyi_gpt_image2_vip").strip().lower()
  provider = build_image_edit_provider(provider_name)
```

### Step 5 — Root cause

If the local codebase is correct (all defaults are `apiyi_gpt_image2_vip`), the issue is in the **backend job payload**:

```python
job["render"]["provider"]  # This field is set to "bfl" by the backend, overriding the local default
```

The local default `"apiyi_gpt_image2_vip"` only applies when `job["render"]["provider"]` is absent.

## Fix

- **Backend side:** Set `job["render"]["provider"]` to `"apiyi_gpt_image2_vip"`, or omit it to use the local default.
- **Not the fix:** Changing local defaults, env vars, or `.env` files — the worker explicitly overrides `args.provider` from the job payload, so local defaults are bypassed.

## Key files

| File | Role |
|---|---|
| `worker/main.py` L251 | Extracts `job["render"]["provider"]` with `"apiyi_gpt_image2_vip"` fallback |
| `worker/render_api.py` L83 | Passes provider to pipeline script |
| `scripts/run_bfl_hanging_flux_pipeline.py` L449 | Resolves provider name, builds provider instance |
| `hanging_mvp/service.py` L2357/2499 | Sets `render_provider: bfl_flux2_edit` in manifest (metadata only, not controlling) |
| `image_editing/provider_registry.py` | Dispatches to actual provider implementation |
