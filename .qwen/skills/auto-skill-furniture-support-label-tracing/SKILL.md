---
name: furniture-support-label-tracing
description: Trace how semantic furniture labels map to hanging pipeline support_context labels — geometry_support is remapped to nearest semantic furniture label (table/cabinet/etc) for user-facing output
source: auto-skill
extracted_at: '2026-07-17T04:50:01.394Z'
updated_at: '2026-07-17'
---

## Problem

When diagnosing why a hanging candidate places artwork at a certain height or on a certain support, the `support_context.label` in the candidate JSON may show a semantic furniture label (e.g. `table`) even though the height came from geometry inference, or it may show `geometry_support` if no semantic furniture was detected nearby.

## Architecture

The hanging pipeline has two paths to determine support:

1. **Semantic support**: VLM detects furniture (e.g. `table_05`, label=`table`, role=`support`) in `03_semantic/semantics.json`
2. **Geometry probe**: `support_context.py` runs `_infer_wall_adjacent_geometry_support()` which infers support from wall-adjacent point cloud height bands, producing `label=geometry_support`, `object_id=geometry_support_at_axis`

### Geometry → Semantic label remapping (V6+)

When `geometry_support` is selected (no semantic furniture passed wall association), `_estimate_axis_support_context()` **remaps** the label to the nearest semantic furniture probe by comparing `support_top_abs_m_p85` heights. If any semantic furniture exists (even if rejected), its label is used instead of `geometry_support`.

```python
# In _estimate_axis_support_context():
if selected_label == "geometry_support":
    # Find nearest semantic furniture probe by height proximity
    nearest_sem_label = ...  # closest support_top_abs_m_p85
    if nearest_sem_label is not None:
        selected_label = nearest_sem_label  # e.g. "table"
        selected_probe = {**selected_probe, "semantic_proxy_for": nearest_sem_label}
```

This ensures LLM output shows user-friendly labels like "边柜/茶几上方" instead of the raw `geometry_support` string.

### Diagnostic markers

- `support_context.label` → the user-facing label (remapped semantic or original)
- `support_context.semantic_proxy_for` → present when label was remapped from geometry_support
- `support_context.source` → `"geometry_wall_adjacent_support_probe"` when height came from geometry inference
- `support_context.object_id` → `"geometry_support_at_axis"` when geometry was the source

## Diagnostic paths

### 1. Check semantic output for detected furniture

```bash
python3 -c "
import json
with open('<run_dir>/03_semantic/semantics.json') as f:
    d = json.load(f)
for obj in d.get('objects', []):
    if obj.get('role') == 'support':
        print(f\"id={obj.get('object_id')}, label={obj.get('label')}, raw_label={obj.get('raw_label')}\")
"
```

### 2. Check candidate support_context for remapping

```bash
python3 -c "
import json
with open('<run_dir>/05_render/flux_out/<wall_cand>/inputs/candidate.json') as f:
    d = json.load(f)
sc = d.get('support_context', {})
print('label:', sc.get('label'))
print('semantic_proxy_for:', sc.get('semantic_proxy_for'))
print('object_id:', sc.get('object_id'))
print('source:', sc.get('source'))
print('height_mode:', sc.get('height_mode'))
print('rule_id:', sc.get('rule_id'))
"
```

If `label=table` but `semantic_proxy_for=table` and `source=geometry_wall_adjacent_support_probe`, the label was remapped from geometry inference.

### 3. Check ranked candidates for support_label

```bash
python3 -c "
import json
with open('<run_dir>/04_hanging/extracted/ranked_candidates.json') as f:
    d = json.load(f)
for i, c in enumerate(d):
    print(f\"Candidate {i}: wall={c.get('wall_id')}, score={c.get('score')}\")
"
```

## Key insight

`geometry_support` is a geometry-derived fallback that is **remapped** to the nearest semantic furniture label before reaching user-facing output. The `semantic_proxy_for` field in `support_context` indicates when this remapping occurred. Height computation still uses the geometry probe's `support_top_abs_m` value regardless of the remapped label.

### Height fix (V6+): semantic proxy bypasses floor_default

When `semantic_proxy_for` is present, `_resolve_furniture_height()` disables the `furniture_never_below_floor_default` constraint. This prevents the 1.52m floor default from overriding the geometry-inferred height. Without this fix, remapped labels (table/cabinet/etc) would incorrectly use `max(preferred_center_height, 1.52)` instead of the actual geometry clearance height.

```python
# In _resolve_furniture_height():
never_below_floor_default = (
    bool(rule_value(rules, ["support_surface_constraints", "furniture_never_below_floor_default"], True))
    and not is_semantic_proxy  # <-- skip floor_default when height is from geometry
)
```

When V5+ furniture clearance rules are active, remapped labels (table, cabinet, counter) still apply their per-label `min_bottom_gap_to_support_m` values from `rules_v1.json` — but the floor_default ceiling is lifted for semantic proxies.
