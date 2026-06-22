---
name: mock-provider-safety-guards
description: Prevent misinterpretation of mock/no-op results as real BFL API outputs
source: auto-skill
extracted_at: '2026-06-04T08:30:00.000Z'
---

# Mock Provider Safety Guards

## Problem

Mock provider produces output by copying input images (erase copies original, flux2 copies hard-paste). Without explicit warnings, these no-op results can be mistaken for real BFL API outputs, leading to false conclusions about pipeline health.

## Shell script guard (`run_full_hanging_bfl_test_opt.sh`)

- Default `PROVIDER=bfl` — real API is the production default
- `PROVIDER=mock|dry_run|noop` requires explicit `ALLOW_MOCK=1`
- Without `ALLOW_MOCK=1`, script exits with code 64 and red stderr error
- Mock runs produce red `🔴 WARNING` messages on stderr and in `MOCK_PROVIDER_WARNING.txt`

```bash
# Allowed dry-run
ALLOW_MOCK=1 PROVIDER=mock bash scripts/run_full_hanging_bfl_test_opt.sh

# Blocked by default
PROVIDER=mock bash scripts/run_full_hanging_bfl_test_opt.sh
# → [ERR] Refusing provider=mock. Set ALLOW_MOCK=1 only for explicit dry-run.
# → exit 64
```

## Python pipeline guard (`run_bfl_hanging_flux_pipeline.py`)

When `--provider mock` is used:
- Red warning printed to stderr at startup and before final summary
- `MOCK_PROVIDER_WARNING` constant defines the warning text

## Trace-level markers

Mock provider injects into every trace:

| Field | Value | Purpose |
|---|---|---|
| `warning` | `"🔴 WARNING: provider=mock is a local no-op dry run..."` | Human-readable warning |
| `warning_label` | `"🔴 MOCK PROVIDER / NO-OP RESULT"` | Machine-searchable label |
| `noop_result` | `true` | Explicit flag |
| `production_usable` | `false` | Quality gate |

Aggregate `edit_trace.json` also includes:
- `warnings[]` array with mock warning objects when provider is mock
- `provider_warning` field set to `MOCK_PROVIDER_WARNING` text

## BFL provider (`bfl.py`)

`MockProvider._trace()` method adds all warning fields uniformly for both erase and flux2_edit capabilities.

## When to check

- After any pipeline run, verify `edit_trace.json` does NOT contain `noop_result: true`
- In automated validation, assert `trace["provider"] != "mock"` for production runs
- Red text in stderr during shell runs indicates mock/no-op — not a BFL failure
