---
name: hanging-service-restart-after-patch
description: All MVP services (semantic, hanging, metric3d) and worker must restart after code patch deployment — each runs as a separate persistent process; patching one module requires restarting only its owning service, not the worker that calls it via HTTP
source: auto-skill
extracted_at: '2026-06-08T00:30:00.000Z'
---

# Service Restart After Patch Deployment

## Problem

After deploying code patches, services continue running old bytecode. Each service runs as a separate persistent process:

- **semantic** (port 6011) — `semantic_mvp/service.py`
- **hanging** (port 6010) — `hanging_mvp/service.py`
- **metric3d** (port 6006) — metric3d API
- **worker** — `worker/main.py` (calls semantic/hanging via HTTP)

**Critical:** The worker calls semantic/hanging services **via HTTP**, not via direct imports. Patching `semantic_mvp/*.py` requires restarting the **semantic service**, NOT the worker. Restarting only the worker after a semantic patch is a no-op.

## Architecture

```
worker (HTTP client) ──POST──> semantic service (port 6011, uvicorn)
                            └──> hanging service (port 6010, uvicorn)
                            └──> metric3d service (port 6006, uvicorn)
```

The worker is a GPU-bound orchestrator. Semantic/hanging/metric3d are independent HTTP API servers. A patch to `semantic_mvp/` affects the semantic service process only.

## Diagnosis Procedure

1. **Check process start time vs patch deployment time:**
   ```bash
   bash /opt/molink/hanging/manage_all_mvp_apis.sh status
   ```
   Compare process start times with patch `git apply` time. If process started **before** patch → stale process.

2. **Check which service owns the patched module:**
   | Patched module | Service to restart |
   |---|---|
   | `semantic_mvp/*.py` | semantic (port 6011) |
   | `hanging_mvp/*.py` | hanging (port 6010) |
   | `worker/*.py` | worker |
   | `metric3d_ctl.sh` or metric3d code | metric3d (port 6006) |

3. **Verify patch activation in output:**
   Check the next job's `semantics.json` for new fields the patch introduced. Their absence means old code is still running.

## Fix

### Restart specific service
```bash
bash /opt/molink/hanging/manage_all_mvp_apis.sh restart semantic
bash /opt/molink/hanging/manage_all_mvp_apis.sh restart hanging
bash /opt/molink/hanging/manage_all_mvp_apis.sh restart metric3d
```

### Restart worker only
```bash
pkill -f 'worker.main|worker/start_worker.sh' || true
: > /opt/molink/hanging/logs/worker.log
cd /opt/molink/hanging
bash /opt/molink/hanging/worker/start_worker.sh
```

### Restart all services
```bash
bash /opt/molink/hanging/manage_all_mvp_apis.sh restart
```

## Stale .pyc cache can block patch deployment

Python `.pyc` bytecode cache files in `__pycache__/` directories may be older than patched `.py` source files. Under certain configurations, processes may load stale `.pyc` even when `.py` has been modified.

**Symptom:** Patch applied via `git apply`, service restarted, but output shows pre-patch behavior.

**Fix — always clear cache before restarting after a patch:**
```bash
# 1. Clear ALL __pycache__ directories in the project
find /opt/molink/hanging -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# 2. Restart affected service(s)
bash /opt/molink/hanging/manage_all_mvp_apis.sh restart semantic  # or hanging/worker/all
```

## Prevention Rule

After deploying any patch that modifies `hanging_mvp/*.py`, `semantic_mvp/*.py`, `worker/*.py`, or metric3d code:

1. Identify which service owns the patched module (see table above)
2. Clear `__pycache__` directories in affected modules
3. Restart **only the affected service(s)** — restarting the worker alone does NOT reload semantic/hanging code
4. Verify `/health` returns 200 for each service
5. Run a test invocation to confirm new behavior
