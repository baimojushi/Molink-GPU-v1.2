---
name: height-contract-v3-diagnosis
description: Diagnose hanging pipeline candidate rejection under HEIGHT-FIX-V3 — semantic_clear_rect is diagnostic-only, only metric_height_rule_consistency is a hard gate (project)
source: auto-skill
extracted_at: '2026-07-16T10:52:51.536Z'
---

## HEIGHT-FIX-V3 Behavior Change

**V2 behavior (deprecated):** `semantic_clear_rect_vertical_fit` was a hard gate — if the metric rule height fell outside the semantic clear rectangle's t-range, the candidate was rejected with `metric_height_outside_semantic_clear_rectangle`.

**V3 behavior (current):** `semantic_clear_rect_vertical_fit` is **diagnostic only** — it never rejects candidates. The only height-related hard gate is `metric_height_rule_consistent`.

## Candidate Rejection Gates Under V3

### Height-related gates (from `height_contract_gate_reasons`)

| Reason | Triggered when | Severity |
|--------|---------------|----------|
| `metric_height_rule_deviation_exceeds_threshold` | `metric_height_rule_consistent` is False | Hard reject |
| `metric_height_outside_semantic_clear_rectangle` | **NEVER** (removed in V3) | N/A |

### Non-height gates (still active)

- `insufficient_safe_wall_support` — wall_inside_ratio below threshold
- `artwork_edge_support_below_hard_reject_threshold` — min edge support too low
- `artwork_corner_support_below_hard_reject_threshold` — min corner support too low
- `blocking_existing_content_overlap` — artwork overlaps furniture/objects

## Diagnosis Workflow

### Step 1: Check rejection reasons

```bash
cat <hanging_plan.json> | python3 -c "
import json, sys
plan = json.load(sys.stdin)
rejected = plan.get('debug', {}).get('rejected_candidates', [])
for rc in rejected[:3]:
    print(f\"candidate: {rc.get('candidate_id')}\")
    print(f\"  wall: {rc.get('wall_id')}\")
    print(f\"  reject_reasons: {rc.get('reject_reasons')}\")
    print()
"
```

### Step 2: Interpret results

| Pattern | Meaning |
|---------|---------|
| `reject_reasons: ['metric_height_outside_semantic_clear_rectangle']` | **V2 code still running** — service not restarted after V3 deploy |
| `reject_reasons: ['metric_height_rule_deviation_exceeds_threshold']` | Height rule is inconsistent — check `height_contract.metric_height_rule_consistent` |
| `reject_reasons: ['insufficient_safe_wall_support', ...]` | Geometry rejection, not height-related |
| `reject_reasons: []` or `None` | Bug — reasons not being populated |

### Step 3: Verify V3 is deployed

```bash
# Check build ID in service.py
grep HANGING_PIPELINE_BUILD_ID /opt/molink/hanging/hanging_mvp/service.py
# Should show: "2026-07-16-metric-height-exact-footprint-v3"

# Check height_fit_contract.py policy
grep "policy" /opt/molink/hanging/hanging_mvp/height_fit_contract.py
# Should show: "diagnostic_only_exact_projected_footprint_is_authoritative"
```

### Step 4: Check near-miss candidates

If all candidates rejected but there are `near_miss_candidates`, these are close to passing and may need manual review:

```bash
cat <hanging_plan.json> | python3 -c "
import json, sys
plan = json.load(sys.stdin)
for nm in plan.get('near_miss_candidates', []):
    print(f\"candidate: {nm.get('candidate_id')}\")
    print(f\"  near_miss_reason: {nm.get('near_miss_reason')}\")
    print(f\"  reject_reasons: {nm.get('reject_reasons')}\")
    print(f\"  min_edge_support: {nm.get('min_edge_support_ratio')}\")
    print(f\"  min_corner_support: {nm.get('min_corner_support_ratio')}\")
"
```

## Key Invariants

1. **`semantic_clear_rect_vertical_fit.passed` is always True in V3** — it's diagnostic, not a gate
2. **`semantic_clear_rect_vertical_fit.enforced` is always False in V3**
3. **Only `metric_height_rule_consistent` matters for height rejection**
4. **If `metric_height_outside_semantic_clear_rectangle` appears in reject_reasons, V2 code is still running**

## How to apply

When diagnosing why a candidate was rejected:
1. Check `reject_reasons` first — if it contains height-related reasons, verify V3 is deployed
2. If only geometric reasons (edge/corner/wall support), this is not a height contract issue
3. Near-miss candidates with perfect support ratios (1.0) but no rejection reasons may indicate a different gate was triggered
