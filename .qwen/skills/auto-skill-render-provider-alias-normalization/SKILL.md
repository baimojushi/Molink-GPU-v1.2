---
name: render-provider-alias-normalization
description: Normalize render provider names across pipeline layers (CLI script, worker API, worker main, result builder) using a canonical name + alias map
source: auto-skill
extracted_at: '2026-06-25T01:20:00Z'
---

## Problem

Provider names arrive from multiple sources with inconsistent naming:
- CLI flags: `--provider apiyi_gpt_image2_vip`
- Job payloads: `render.provider`, `render.render_provider`, `render_provider`, `hanging_provider`
- Environment: `HANGING_RENDER_PROVIDER`
- User shorthand: `apiyi`, `gpt-image-2-vip`, `gpt_image2_vip`

Each layer previously resolved the name independently, causing mismatched telemetry and fallback behavior.

## Solution pattern

### 1. Canonical name + alias map

Define one canonical name per provider and map all known aliases to it:

```python
DEFAULT_RENDER_PROVIDER = "apiyi_gpt_image2_vip"
RENDER_PROVIDER_ALIASES = {
    "apiyi": "apiyi_gpt_image2_vip",
    "apiyi_gpt_image2_vip": "apiyi_gpt_image2_vip",
    "gpt_image2_vip": "apiyi_gpt_image2_vip",
    "gpt-image-2-vip": "apiyi_gpt_image2_vip",
    "bfl": "bfl",
    "bfl_official": "bfl",
    "bfl_flux2": "bfl",
    "mock": "mock",
    "dry_run": "mock",
    "noop": "mock",
}
```

### 2. normalize_render_provider function

Single normalization function used at every layer:

```python
def normalize_render_provider(value: str | None) -> str:
    key = str(value or os.getenv("HANGING_RENDER_PROVIDER") or DEFAULT_RENDER_PROVIDER).strip().lower()
    normalized = RENDER_PROVIDER_ALIASES.get(key)
    if normalized:
        return normalized
    print(f"[layer] unknown render provider={key!r}; fallback to {DEFAULT_RENDER_PROVIDER}", flush=True)
    return DEFAULT_RENDER_PROVIDER
```

### 3. resolve_render_provider at job boundary

At the worker job entry point, scan all possible provider fields in priority order:

```python
def resolve_render_provider(job: dict) -> tuple[str, str]:
    render = job.get("render") if isinstance(job.get("render"), dict) else {}
    candidates = [
        ("render.provider", render.get("provider")),
        ("render.render_provider", render.get("render_provider")),
        ("render_provider", job.get("render_provider")),
        ("hanging_provider", job.get("hanging_provider")),
        ("provider", job.get("provider")),
        ("env.HANGING_RENDER_PROVIDER", os.getenv("HANGING_RENDER_PROVIDER")),
    ]
    for source, value in candidates:
        if value is not None and str(value).strip():
            return normalize_render_provider(value), source
    return DEFAULT_RENDER_PROVIDER, "default"
```

### 4. Distinguish requested vs actual in telemetry

- `requested_render_provider`: what the caller asked for (pre-normalization)
- `actual_render_provider`: what was actually used (post-normalization)
- `render_provider_source`: which field provided the value

This allows diagnosing cases where an alias was provided but a different canonical provider was used.

### 5. Apply at every layer

| Layer | File | What it does |
|-------|------|-------------|
| CLI script | `scripts/run_bfl_hanging_flux_pipeline.py` | `_normalize_render_provider_name()` + `--provider` expanded choices |
| Worker API | `worker/render_api.py` | `normalize_render_provider()` called before passing to script |
| Worker main | `worker/main.py` | `resolve_render_provider()` at job entry; `normalize_render_provider()` constants |
| Result builder | `worker/result_builder.py` | `_job_requested_render_provider()` + `_actual_render_provider()` in metrics |

## Key invariant

After normalization, provider comparisons should use exact string equality (`==`), not set membership (`in {...}`), because the normalized value is guaranteed to be canonical:

```python
# Before (fragile — easy to miss an alias):
using_apiyi = provider_name in {"apiyi_gpt_image2_vip", "gpt_image2_vip", "gpt-image-2-vip", "apiyi"}

# After (safe — canonical name only):
using_apiyi = provider_name == "apiyi_gpt_image2_vip"
```
