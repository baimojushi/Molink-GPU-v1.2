---
name: metric3d-restart-requires-allow-unauth-local
description: metric3d service restart must use start_giant_auto.sh script to set ALLOW_UNAUTH_LOCAL=1 — manual uvicorn launch without this env var causes 500 API_KEY not configured errors
source: auto-skill
extracted_at: '2026-07-16T16:33:26.862Z'
---

# metric3d Restart Requires ALLOW_UNAUTH_LOCAL=1

## Symptom

After restarting metric3d service, worker pipeline fails with:

```
RuntimeError: Metric3D HTTP 500: {"detail":"API_KEY is not configured; set ALLOW_UNAUTH_LOCAL=1 for local development"}
```

## Root Cause

metric3d service's `require_auth` middleware checks:
1. `API_KEY` environment variable is set
2. If not set, `ALLOW_UNAUTH_LOCAL=1` must be set for localhost requests

When metric3d is restarted manually using:
```bash
python -m uvicorn app:app --host 0.0.0.0 --port 6006
```

The environment variables are not loaded from `.env` file, causing `API_KEY` to be unset and `ALLOW_UNAUTH_LOCAL` to be missing.

## Correct Restart Procedure

**Always use the official startup script:**

```bash
cd /opt/molink/hanging/services/metric3d_api
bash start_giant_auto.sh
```

This script:
1. Sources conda environment
2. Sets all required env vars (MODEL_NAME, MAX_SIDE, FOV_CANDIDATES, etc.)
3. **Sets `ALLOW_UNAUTH_LOCAL=1`** (added 2026-07-17 to fix 500 errors)
4. Starts uvicorn with proper environment

**Important**: The `start_giant_auto.sh` script was modified on 2026-07-17 to include `export ALLOW_UNAUTH_LOCAL=1`. If the script is ever reverted, this line must be re-added after the `FOV_CANDIDATES` export and before the `python -m uvicorn` line.

## Manual Restart (if script unavailable)

```bash
cd /opt/molink/hanging/services/metric3d_api
export ALLOW_UNAUTH_LOCAL=1
set -a && source .env && set +a
/root/miniconda3/envs/metric3d/bin/python -m uvicorn app:app --host 0.0.0.0 --port 6006
```

## Verification

After restart, verify:
```bash
curl -s http://localhost:6006/health | python3 -m json.tool
```

Should return `{"status": "ok"}`, not 500 error.

## Related Services

Same pattern applies to:
- **semantic service**: Uses `ALLOW_UNAUTH_LOCAL=1` from `start_semantic_v1_sam.sh`
- **hanging service**: Uses `ALLOW_UNAUTH_LOCAL=1` from `start_hanging_v1.sh`

All three services have identical `require_auth` logic that rejects unauthenticated localhost requests unless `ALLOW_UNAUTH_LOCAL=1` is set.
