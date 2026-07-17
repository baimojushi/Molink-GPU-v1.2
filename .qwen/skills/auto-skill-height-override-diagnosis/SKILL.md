---
name: height-override-diagnosis
description: Diagnose why hanging pipeline places artwork at wrong height — trace through center_t_override, height_contract, and metric_rule authority
source: auto-skill
extracted_at: '2026-07-16T04:24:34.309Z'
---

## Symptom

Artwork placed at wrong height (e.g. 2.15m instead of expected 1.35m for `table_center_135` rule).

## Root cause mechanism

The hanging pipeline has two competing height sources:
1. **Metric rule** (authoritative): `table_center_135`, `floor_center_152`, etc. — fixed height above floor
2. **center_t_override** (semantic geometry): proposed by `semantic_clear_subregion_max_empty_rect_center` or similar axis sources

`resolve_candidate_center_t()` in `height_fit_contract.py` is supposed to enforce metric rule authority — when `height_mode == "metric_rule"`, it must reject any `center_t_m_override` from axis_item.

## Diagnosis workflow

### Step 1: Extract height data from hanging_plan.json

```bash
# Find the order's hanging_plan.json
find /root/hanging_runs /root/autodl-fs/hanging_runs -name "hanging_plan.json" -path "*<order_id>*" | head -1

# Extract key fields
cat <path>/hanging_plan.json | python3 -c "
import json, sys
plan = json.load(sys.stdin)
sc = plan['selected_candidate']['support_context']
hc = plan['selected_candidate'].get('height_contract', {})
print('rule_id:', sc.get('rule_id'))
print('height_mode:', sc.get('height_mode'))
print('floor_height_abs_m:', sc['floor_height_abs_m'])
print('center_height_above_floor_m:', sc['center_height_above_floor_m'])
print('center_t_override_m:', sc.get('center_t_override_m'))
print('center_t_override_source:', sc.get('center_t_override_source'))
print()
print('=== HEIGHT CONTRACT ===')
print('override_applied:', hc.get('center_t_override_applied'))
print('override_ignored_reason:', hc.get('center_t_override_ignored_reason'))
print('resolved_center_height_above_floor_m:', hc.get('resolved_center_height_above_floor_m'))
"
```

### Step 2: Interpret results

| Pattern | Meaning | Fix |
|---------|---------|-----|
| `center_t_override_m` is set AND `height_contract` is None | `resolve_candidate_center_t()` was never called — early return path in service.py | Check service.py for early continue/before the height_contract call |
| `center_t_override_applied: true` with `height_mode: metric_rule` | Bug in `resolve_candidate_center_t()` — override should be rejected | Fix height_fit_contract.py logic |
| `center_t_override_applied: false` but height still wrong | Override was correctly rejected, but rule_center_height_abs_m itself is wrong | Check `_estimate_axis_support_context()` rule calculation |
| `center_t_override_m` not in axis_item but present in support_context | Something else is setting it — check support_context.py or client_result.py | Search codebase for assignments |

### Step 3: Verify code is actually deployed

```bash
# Check if __pycache__ is stale
find /opt/molink/hanging/hanging_mvp -name "*.pyc" -newer /opt/molink/hanging/hanging_mvp/height_fit_contract.py

# Clear cache and restart
find /opt/molink/hanging/hanging_mvp -name "*.pyc" -delete
find /opt/molink/hanging/hanging_mvp -name "__pycache__" -exec rm -rf {} + 2>/dev/null
pkill -f "hanging_mvp.service"
# Restart hanging service
```

### Step 4: Check order timestamp vs deploy timestamp

```bash
# Order run time (from hanging_runs directory name)
ls -la /root/autodl-fs/hanging_runs/ | grep <order_id>

# Deploy time (from backup directory)
ls -la /root/autodl-tmp/qwen_metric3d/artifacts/tmp/deploy_backups/
```

If order ran before code deploy, the old buggy code was active.

## Key fields explained

- `floor_height_abs_m`: Signed height of floor relative to camera (negative = floor below camera)
- `center_height_abs_m`: Absolute height of artwork center in camera coordinate frame
- `center_height_above_floor_m`: `center_height_abs_m - floor_height_abs_m` — what user sees as "离地高度"
- `center_t_override_m`: Wall-local t-coordinate proposed by semantic geometry (not metric rule)
- `height_contract`: Result of `resolve_candidate_center_t()` — shows whether override was applied or rejected

## Common pitfall

If `height_contract` field is missing/None in hanging_plan.json, the code never reached `resolve_candidate_center_t()`. This means either:
1. An early `continue`/`return` in service.py candidate loop
2. Python was running cached `.pyc` from before the fix was deployed
3. Service was not restarted after code deployment

Always clear `__pycache__` and restart the hanging service after deploying height-related patches.
