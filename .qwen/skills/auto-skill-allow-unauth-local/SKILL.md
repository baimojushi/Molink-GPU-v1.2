---
name: allow-unauth-local
description: Persist ALLOW_UNAUTH_LOCAL=1 via .env.local for local dev; change to 0 before building the AutoDL private image
source: auto-skill
extracted_at: '2026-06-22T00:00:00Z'
---

## Problem

All three MVP services (metric3d, semantic, hanging) require `API_KEY` by default. Without it, every POST returns HTTP 500:

```
API_KEY is not configured; set ALLOW_UNAUTH_LOCAL=1 for local development
```

## Solution

Place `ALLOW_UNAUTH_LOCAL` in `.env.local` so `manage_all_mvp_apis.sh` sources it for every service start.

### File: `.env.local`

```bash
# Local development overrides — set before making the private image.
# IMPORTANT: change to 0 before building the AutoDL private image.
export ALLOW_UNAUTH_LOCAL=1
```

### File: `manage_all_mvp_apis.sh`

Add sourcing after `mkdir -p "$LOG_DIR" "$PID_DIR"`:

```bash
[ -f "$ROOT/.env.local" ] && source "$ROOT/.env.local"
```

This propagates to all three services (metric3d on 6006, semantic on 6011, hanging on 6010) because they are all launched via the manage script, which sets environment before spawning nohup uvicorn processes.

## Key points

- Do NOT set `ALLOW_UNAUTH_LOCAL=1` in individual start scripts (start_giant_auto.sh, start_semantic_v1_sam.sh, start_hanging_v1.sh) — centralize in `.env.local` sourced by `manage_all_mvp_apis.sh`.
- `.env.local` should be gitignored (it is tracked in `.gitignore` via `.env*` patterns already).
- **Before building the AutoDL private image**: change `ALLOW_UNAUTH_LOCAL=1` to `ALLOW_UNAUTH_LOCAL=0` in `.env.local`, or delete the file entirely.
