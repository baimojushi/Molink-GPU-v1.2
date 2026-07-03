---
name: semantic-sam-lazy-load-crash
description: Semantic service startup and crash issues — auth 401/500, OMP_NUM_THREADS, SAM subprocess isolation, require_auth logic order
source: auto-skill
extracted_at: '2026-06-07T12:31:54.042Z'
updated_at: '2026-07-04T01:03:00Z'
---

## Symptom

Semantic service (`:6011`) starts successfully, `/health` returns 200. On POST requests to `/parse_semantics_mvp`, returns 401 Unauthorized or 500 Internal Server Error, or the uvicorn process exits silently.

## Issue 3: require_auth logic order causing 401 on all requests (FIXED 2026-07-04)

**Symptom**: All POST requests return `{"detail":"Unauthorized"}` (401) even though `ALLOW_UNAUTH_LOCAL=1` is set and `API_KEY` has a value.

**Root cause**: `require_auth()` in `semantic_mvp/service.py` checked `API_KEY` emptiness BEFORE checking `ALLOW_UNAUTH_LOCAL`:

```python
# BROKEN: ALLOW_UNAUTH_LOCAL only effective when API_KEY is empty
def require_auth(a):
    if not API_KEY:
        if ALLOW_UNAUTH_LOCAL:
            return
        raise HTTPException(500, "...")
    if a != f"Bearer {API_KEY}":
        raise HTTPException(401, "Unauthorized")  # ← hits this when API_KEY set but no auth header
```

Worker calls semantic without `Authorization` header. When `API_KEY` is set (from `VLM_API_KEY`), `require_auth` requires `Authorization: Bearer <key>` — which the worker never sends.

**Fix applied** in `semantic_mvp/service.py`:
```python
# FIXED: ALLOW_UNAUTH_LOCAL checked first, bypasses all auth
def require_auth(a):
    if ALLOW_UNAUTH_LOCAL:
        return
    if not API_KEY:
        raise HTTPException(500, "API_KEY is not configured; set ALLOW_UNAUTH_LOCAL=1 for local development")
    if a != f"Bearer {API_KEY}":
        raise HTTPException(401, "Unauthorized")
```

**Fix applied** in `scripts/start_semantic_v1_sam.sh`:
```bash
# Ensure ALLOW_UNAUTH_LOCAL and API_KEY are set for local dev
[ -f .env ] && source .env
export ALLOW_UNAUTH_LOCAL="${ALLOW_UNAUTH_LOCAL:-1}"
if [ -z "${API_KEY:-}" ] && [ -n "${VLM_API_KEY:-}" ]; then
    export API_KEY="$VLM_API_KEY"
fi
```

## Issue 1: OMP_NUM_THREADS=0 (FIXED 2026-06-07)

`OMP_NUM_THREADS=0` was set in the shell environment, causing OpenMP/PyTorch undefined behavior during SAM CUDA inference. This was the primary crash cause.

**Fix applied** in `scripts/start_semantic_v1_sam.sh`:
```bash
export OMP_NUM_THREADS="4"
```

### Issue 2: VLM + SAM combination crash (FIXED 2026-06-07 via subprocess isolation)

VLM + SAM combination caused the process to silently crash on first request. Root cause was a GPU/CUDA context conflict when both run in the same uvicorn worker.

**FIXED**: SAM refinement is now isolated in a short-lived subprocess. If it crashes, only the child dies — the main uvicorn worker survives and returns HTTP 200 with a bbox fallback.

**Implementation** (`semantic_mvp/sam_subprocess_runner.py`):
```python
# In service.py — replace inline call:
from .sam_subprocess_runner import refine_objects_with_sam_subprocess

# Instead of:
#   objects, sam_debug = refine_objects_with_sam(image_np, objects, ...)
# Use:
objects, sam_debug = refine_objects_with_sam_subprocess(
    image_np, objects,
    processed_size=(original_w, original_h),
    original_size=(original_w, original_h),
    enabled=sam_enabled,
    refine_labels_csv=sam_refine_labels,
    max_side=sam_max_side,
    timeout_seconds=int(os.getenv("SAM_SUBPROCESS_TIMEOUT_S", "120")),
)
```

**Subprocess runner pattern:**
1. Serialize image via `numpy.tobytes()` + shape for IPC
2. Use `multiprocessing.get_context("spawn")` for clean GPU context (not fork)
3. Load SAM predictor inside the child (not in parent)
4. Put result on a `multiprocessing.Queue`
5. Parent checks exit code: non-zero → fallback to bbox, return 200
6. Timeout: configurable via `SAM_SUBPROCESS_TIMEOUT_S` env var

**Verified behavior matrix after fix:**

| vlm_mode | sam_enable | HTTP | Process |
|---|---|---|---|
| mock | 1 | ✅ 200 | ✅ alive |
| auto | 0 | ✅ 200 | ✅ alive |
| auto | 1 | ✅ 200 | ✅ alive (SAM in subprocess) |

## Previous workarounds (now superseded)

### Workaround 1: Pre-load SAM at startup (STILL APPLIED)

SAM pre-load at startup still runs via `_startup_sam_warmup()` for fast first request. This is still useful even with subprocess isolation.

### Workaround 2: Avoid VLM + SAM on same request (NO LONGER NEEDED)

Subprocess isolation makes VLM + SAM safe to use together.
