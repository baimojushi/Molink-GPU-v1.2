---
name: worker-service-auth-header
description: Worker pipeline HTTP calls to metric3d/semantic/hanging must carry Authorization: Bearer {API_KEY} header — services with API_KEY set reject unauthenticated requests with 401 even on localhost
source: auto-skill
extracted_at: '2026-07-12T15:29:37.284Z'
---

## Problem

Worker calls to `http://127.0.0.1:6010/analyze_hanging_mvp` (and similarly metric3d at :6006, semantic at :6011) return **401 Unauthorized** even though:

- `ALLOW_UNAUTH_LOCAL=1` is set in gpu.env
- The service process has the env var loaded
- Requests come from localhost (127.0.0.1)

## Root cause

In `hanging_mvp/service.py` (and similarly in other services):

```python
def require_auth(a):
    if not API_KEY:
        if ALLOW_UNAUTH_LOCAL:
            return
        raise HTTPException(500, "API_KEY is not configured; set ALLOW_UNAUTH_LOCAL=1 for local development")
    if a != f"Bearer {API_KEY}":
        raise HTTPException(401, "Unauthorized")
```

When `API_KEY` is set (which it always is in production), the `ALLOW_UNAUTH_LOCAL` bypass is **never reached** — the code falls through to the `Bearer {API_KEY}` check. If the caller sends no Authorization header, `a` is `None` and the 401 is raised.

## Fix

`worker/pipeline.py` must send the Authorization header on all localhost service calls:

```python
API_KEY = os.getenv("API_KEY", "")

def _auth_headers() -> dict:
    return {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}

# Then on every httpx.post to metric3d/semantic/hanging:
httpx.post(url, ..., headers=_auth_headers())
```

The `API_KEY` is loaded from the same gpu.env that the services read, so the token matches.

## Affected calls

All three `call_*` functions in `worker/pipeline.py`:
- `call_metric3d()` → `httpx.post(..., headers=_auth_headers())`
- `call_semantic()` → `httpx.post(..., headers=_auth_headers())`
- `call_hanging()` → `httpx.post(..., headers=_auth_headers())`

## After applying fix

Restart the worker:
```bash
pkill -f "worker.main" && bash /opt/molink/hanging/worker/start_worker.sh &
```
