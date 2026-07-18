---
name: geometry-support-label-remap
description: Remap geometry_support to nearest semantic furniture label (including bar) and skip floor_default height override for semantic proxies
source: auto-skill
extracted_at: '2026-07-17T11:32:40.895Z'
---

## Problem

When `_infer_wall_adjacent_geometry_support()` produces a `geometry_support` probe, the LLM receives `above_furniture_label_zh = "geometry_support"` — a raw internal label with no Chinese mapping in `LABEL_ZH`. Users see "geometry_support" instead of "边柜/茶几上方".

Additionally, when `geometry_support` is remapped to a semantic label (e.g., `table`), `_resolve_furniture_height()` applies `furniture_never_below_floor_default=True`, which overrides the geometry-inferred height with the 1.52m floor default.

## Fix location

`hanging_mvp/support_context.py` — `_estimate_axis_support_context()` and `_resolve_furniture_height()`.

## Label remap logic

After `selected_label` is determined as `geometry_support`, scan all probes (including rejected semantic furniture) for the nearest semantic label by `support_top_abs_m_p85` distance:

```python
if selected_label == "geometry_support":
    geo_top = selected_probe.get("support_top_abs_m_p85")
    nearest_sem_label = None
    nearest_sem_dist = float("inf")
    for p in probes:
        p_label = p.get("label", "")
        if p_label in {"sofa", "table", "cabinet", "counter", "counter_fixture", "sideboard", "bar"}:
            p_top = p.get("support_top_abs_m_p85")
            dist = abs(float(p_top) - float(geo_top)) if p_top is not None and geo_top is not None else 0.0
            if dist < nearest_sem_dist:
                nearest_sem_dist = dist
                nearest_sem_label = p_label
    if nearest_sem_label is not None:
        selected_label = nearest_sem_label
        selected_probe = {**selected_probe, "semantic_proxy_for": nearest_sem_label}
```

**Key**: remap regardless of height difference — VLM confirmed furniture exists, wall association just failed.

## Height fix

When `semantic_proxy_for` is present, skip `furniture_never_below_floor_default` in `_resolve_furniture_height()`:

```python
def _resolve_furniture_height(..., is_semantic_proxy: bool = False):
    never_below_floor_default = (
        bool(rule_value(rules, ["support_surface_constraints", "furniture_never_below_floor_default"], True))
        and not is_semantic_proxy
    )
```

Call site:
```python
is_proxy = isinstance(selected_probe, dict) and "semantic_proxy_for" in selected_probe
furniture_height = _resolve_furniture_height(
    ..., is_semantic_proxy=is_proxy,
)
```

## Return value propagation

Add `semantic_proxy_for` to the return dict:
```python
**({"semantic_proxy_for": selected_probe["semantic_proxy_for"]}
  if isinstance(selected_probe, dict) and "semantic_proxy_for" in selected_probe else {}),
```

## Supported semantic furniture labels

The remap scans for these labels: `sofa`, `table`, `cabinet`, `counter`, `counter_fixture`, `sideboard`, `bar`.

Height resolution handles: `table`, `cabinet`, `counter`, `bar` (via `_resolve_furniture_height`). Other labels fall back to `cabinet` defaults.

## Result

- LLM receives `above_furniture_label_zh = "边柜/茶几上方"` (from `LABEL_ZH["table"]`) instead of `"geometry_support"`
- Height comes from geometry probe clearance, not 1.52m floor default
- `source` remains `geometry_wall_adjacent_support_probe` for debugging
