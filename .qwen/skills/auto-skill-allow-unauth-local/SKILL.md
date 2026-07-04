---
name: allow-unauth-local
description: All GPU services (semantic, hanging, worker) require ALLOW_UNAUTH_LOCAL=1 and API_KEY set in their startup scripts for local dev — each service has its own require_auth with identical logic
source: auto-skill
extracted_at: '2026-07-04T07:33:32.875Z'
---

## Problem

All GPU services (semantic on 6011, hanging on 6010) have `require_auth` that checks `API_KEY` at module load time. Without `ALLOW_UNAUTH_LOCAL=1`, every POST without Authorization header returns HTTP 500 (or 401 if API_KEY is set):

```
API_KEY is not configured; set ALLOW_UNAUTH_LOCAL=1 for local development
```

Worker does not send Authorization header to semantic or hanging endpoints.

## Root cause

Each service has its own `require_auth` function with this logic:
```python
def require_auth(a):
    if not API_KEY:
        if ALLOW_UNAUTH_LOCAL:
            return
        raise HTTPException(500, "API_KEY is not configured...")
    if a != f"Bearer {API_KEY}":
        raise HTTPException(401, "Unauthorized")
```

If `API_KEY` is set (from VLM_API_KEY) but `ALLOW_UNAUTH_LOCAL` is not, requests without auth header get 401. **The fix is to check `ALLOW_UNAUTH_LOCAL` first, regardless of API_KEY:**

```python
def require_auth(a):
    if ALLOW_UNAUTH_LOCAL:
        return
    if not API_KEY:
        raise HTTPException(500, "API_KEY is not configured...")
    if a != f"Bearer {API_KEY}":
        raise HTTPException(401, "Unauthorized")
```

## Solution

Each service startup script must source `.env` and export both variables:

### `scripts/start_semantic_v1_sam.sh`
### `scripts/start_hanging_v1.sh`

Add after other env sourcing:
```bash
# Local dev: allow unauthenticated requests and set API_KEY from VLM_API_KEY
[ -f .env ] && source .env
export ALLOW_UNAUTH_LOCAL="${ALLOW_UNAUTH_LOCAL:-1}"
if [ -z "${API_KEY:-}" ] && [ -n "${VLM_API_KEY:-}" ]; then
    export API_KEY="$VLM_API_KEY"
fi
```

## Key points

- **semantic_mvp/service.py**: `require_auth` fixed to check ALLOW_UNAUTH_LOCAL first
- **hanging_mvp/service.py**: Same require_auth pattern, same fix needed
- **Worker** does not send Authorization header — it relies on ALLOW_UNAUTH_LOCAL
- After any service restart, verify with `curl http://localhost:PORT/health` then test POST without auth header
- If a service returns 500/401 on POST, check its startup script for missing ALLOW_UNAUTH_LOCAL/API_KEY exports
