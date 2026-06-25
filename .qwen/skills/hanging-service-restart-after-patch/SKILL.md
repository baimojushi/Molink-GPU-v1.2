---
name: hanging-service-restart-after-patch
description: Hanging service must restart after code patch deployment — stale process runs old mask collection logic
source: auto-skill
extracted_at: '2026-06-08T00:30:00.000Z'
---

# Hanging Service Restart After Patch Deployment

## Problem

After deploying code patches (e.g., `right_wall_erase_scope_patch`), the hanging service continues running old bytecode. Stale processes cause:

- Existing content (paintings, framed art on walls) detected by semantic but **not** collected into `existing_masks`
- `existing_content_count: 0` in the BFL erase trace
- BFL erase only removes foreground objects (people), leaving existing artwork uncleared
- Hard-paste overlaps with existing artwork because remove_mask was empty for existing_content

## Root Cause

### Old code (exact match — fails with suffix-style VLM labels):
```python
existing_masks = [o for o in objects.values() if o["label"] in existing_content_labels]
```

VLM returns `painting_01`, `framed_art_01`, `window_right` — but `existing_content_classes` defines `painting`, `framed_art`, `window`. Exact match fails → zero masks.

### New code (role-based + canonicalization):
```python
existing_masks = [
    o for o in objects.values()
    if str(o.get("role", "")).strip().lower() == "existing_content"
    or label_matches(o.get("label"), existing_content_labels)
    or label_matches(o.get("raw_label"), existing_content_labels)
]
```

Uses `role == "existing_content"` as primary recall signal, plus `label_matches` with suffix stripping for label-based matching.

## Diagnosis Procedure

1. **Check process start time vs patch deployment time:**
   ```bash
   ps -p $(pgrep -f 'uvicorn.*6010') -o pid,lstart,etimes=
   git log --format="%ci %s" -3 -- hanging_mvp/service.py hanging_mvp/masks.py
   ```
   If process started **before** patch deployment → stale process.

2. **Verify existing_masks collection in flight:**
   ```bash
   python3 -c "
   import json
   from hanging_mvp.masks import load_semantic_masks, label_matches, canonical_semantic_label
   semantics = json.load(open('path/to/semantics.json'))
   h, w = semantics['processed_size']
   objects = load_semantic_masks(semantics, (h, w))
   existing = [o for o in objects.values() if o.get('role') == 'existing_content']
   print(f'existing_masks: {len(existing)}')
   "
   ```

3. **Check BFL trace for existing_content_count:**
   ```bash
   python3 -c "
   import json
   trace = json.load(open('flux_result_bfl/edit_trace.json'))
   for out in trace['outputs']:
       print(f'{out[\"asset_id\"]}: existing_content_count={out.get(\"erase_input\", {}).get(\"existing_content_count\", \"N/A\")}')
   "
   ```

## Fix

Restart hanging service after any code patch deployment:

```bash
bash /opt/metric3d_api/scripts/manage_all_mvp_apis.sh restart hanging
```

Or restart all services:
```bash
bash /opt/metric3d_api/scripts/manage_all_mvp_apis.sh restart
```

## Worker restart after patch

If the patch modifies `image_editing/providers/*.py`, `scripts/run_bfl_hanging_flux_pipeline.py`, or any code executed by the GPU worker process, the worker must also be restarted:

```bash
pkill -f 'worker.main|worker/start_worker.sh' || true
: > /opt/molink/hanging/logs/worker.log
cd /opt/molink/hanging
bash /opt/molink/hanging/worker/start_worker.sh
```

The worker runs as a persistent background process and never reloads code automatically. Deploying a patch without restarting the worker means the next job still executes the old bytecode.

## Prevention Rule

After deploying any patch that modifies `hanging_mvp/*.py`, `semantic_mvp/*.py`, or `scripts/*.py`:

1. Restart affected service(s) via `manage_all_mvp_apis.sh restart`
2. Verify `/health` returns 200
3. Run a test invocation to confirm new behavior
