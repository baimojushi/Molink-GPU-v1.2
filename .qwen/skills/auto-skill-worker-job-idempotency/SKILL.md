---
name: worker-job-idempotency
description: Worker must guard against duplicate job_id execution — use _executed_jobs set to reject re-dispatch after reconnect or restart
source: auto-skill
extracted_at: '2026-06-30T06:13:29.596Z'
---

## Problem

Worker WebSocket reconnection or manual restart causes the business server to re-dispatch a job already completed or in-flight. Without idempotency protection, the same `job_id` executes twice, producing duplicate artifact packages for a single order.

## Trigger scenarios

- Worker disconnects during long-running job → business server re-dispatches on reconnect
- Worker restarts after patch deployment → receives stale dispatch from business server queue
- Business server does not track job completion state accurately

## Fix pattern

In `worker/main.py`:

1. **Track executed jobs**: Add `_executed_jobs: set[str]` to `Worker.__init__`
2. **Check on dispatch**: In `run_job()` entry, reject if `job_id` already in `_executed_jobs`, send `job_skipped` message
3. **Record on completion**: Use `finally` block in `run_job()` to add `job_id` to `_executed_jobs` — covers all exit paths (success, failure, exception)

## Key code pattern

```python
def __init__(self) -> None:
    self._executed_jobs: set[str] = set()

async def run_job(self, ws, job) -> None:
    job_id = job.get("job_id")
    if job_id and job_id in self._executed_jobs:
        await ws.send(json.dumps({"type": "job_skipped", "job_id": job_id, "reason": "duplicate"}))
        return
    # ... execute pipeline ...
    finally:
        if job_id:
            self._executed_jobs.add(job_id)
```

## Limitations

This is process-memory-level protection. The `_executed_jobs` set clears on Worker process restart. For cross-restart protection, persist job IDs to a disk manifest and reload on startup.

## How to apply

When investigating duplicate artifact packages or orders producing multiple results, check if Worker was restarted or reconnected during the order lifecycle. The `_executed_jobs` guard prevents re-execution within a single Worker process lifetime.
