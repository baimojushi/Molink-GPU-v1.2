---
name: furniture-support-requires-sam-mask
description: Furniture support association (table, cabinet, counter, etc.) requires SAM-generated mask with 3D points — semantic seed mode without metric3d point cloud cannot produce masks, causing fallback to floor_center_152 rule
source: auto-skill
extracted_at: '2026-07-16T11:29:14.992Z'
---

## Problem

Hanging pipeline places artwork at 1.52m (floor_center_152 rule) even when furniture (table, cabinet, counter) is visibly present below the wall. The furniture is detected by VLM but not used as a support surface.

## Root cause chain

```
1. Worker pipeline calls semantic service WITHOUT metric3d point cloud
   ↓
2. Semantic runs VLM 2D detection → detects "table" label ✓
   ↓
3. But SAM segmentation is skipped (no xyz input) → no mask field
   ↓
4. hanging_mvp/support_context.py:_estimate_axis_support_context()
   ↓
5. _object_points_near_axis(obj, xyz, ...) needs obj["mask"]
   ↓
6. mask is None → returns empty point array
   ↓
7. valid_points_near_axis: 0 → fails min_valid_points threshold (200)
   ↓
8. No furniture support qualifies → falls back to floor rule → 1.52m
```

## Key code logic

`support_context.py` line ~267:
```python
labels_by_priority = ["sofa", "counter", "counter_fixture", "cabinet", "sideboard", "table", "floor"]
```

`_object_points_near_axis` (line ~240) extracts points from `obj.get("mask")`. Without mask, it returns empty.

`small_shelf` is NOT in `labels_by_priority` — even with mask, it would not be used as support.

## Diagnosis

Check `hanging_plan.json`:
```bash
cat <run_dir>/04_hanging/extracted/hanging_plan.json | python3 -c "
import json, sys
plan = json.load(sys.stdin)
sc = plan['selected_candidate']['support_context']
print('label:', sc.get('label'))           # 'floor' means no furniture support
print('source:', sc.get('source'))         # 'trusted_metric_floor_fallback'
print('rule_id:', sc.get('rule_id'))       # 'floor_center_152'
"
```

Check `semantics.json` for mask presence:
```bash
cat <run_dir>/03_semantic/semantics.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
for obj in data.get('objects', []):
    label = obj.get('label')
    if label in ['table', 'cabinet', 'counter', 'sofa']:
        print(f'{label}: mask={obj.get(\"mask\") is not None}, num_points={obj.get(\"num_points\")}')
"
```

## Resolution

To enable furniture support association, the pipeline MUST provide metric3d point cloud to semantic service so SAM can generate masks. This requires:
1. Worker pipeline calls metric3d first (not just semantic seed)
2. Metric3d point cloud (xyz) is passed to semantic service
3. Semantic runs SAM segmentation → generates mask
4. Hanging service uses mask to extract 3D points for support association

This is a **pipeline architecture requirement**, not a code bug. The hanging code correctly handles the case when masks are unavailable by falling back to the floor rule.

## When to apply

- Diagnosing why artwork is placed at 1.52m when furniture is visible
- Investigating why `furniture_supports` count is 0 despite VLM detecting table/cabinet
- Explaining the difference between semantic seed mode (2D only) and full semantic mode (2D + 3D via SAM)
