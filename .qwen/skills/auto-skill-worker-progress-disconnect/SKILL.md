---
name: worker-progress-disconnect
description: Worker must not treat progress WebSocket send failures as pipeline failures — progress drops are acceptable, only pipeline errors should terminate jobs (project)
source: auto-skill
extracted_at: '2026-06-26T21:50:00.000Z'
---

## Problem

When `progress(ws, ...)` raises `ConnectionClosedError` during pipeline execution, the worker's outer `except` block catches it and marks the job as `FAILED_WORKER_EXCEPTION`. This is incorrect because:

1. `progress` is just a status message to the business server — not the pipeline itself
2. Metric3d/semantic/hanging are independent HTTP processes that continue running even if the WebSocket drops
3. The business server receives the failure, transitions the order to `hanging_failed`, and ignores subsequent retransmitted results
4. Worker reconnects but does not wait for the actual pipeline to finish

**Observed in job_3cf946b5:**
- Metric3d completed, worker sent `progress(semantic, 35%)`
- WebSocket dropped during `ws.send` (idle timeout during metric3d processing)
- Worker immediately marked job as failed, sent `FAILED_WORKER_EXCEPTION`
- Business server turned order to `hanging_failed` → manual review
- Pipeline likely continued in background but result was never consumed

## Fix applied to worker/main.py

### 1. `progress()` method — retry and drop, never raise

```python
async def progress(self, ws, job, stage, state, pct, message, ...):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            await ws.send(json.dumps(payload, ensure_ascii=False))
            return
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            print(f"[progress] failed to send after {max_retries} attempts: {e}")
            print(f"[progress] progress message dropped, pipeline continues")
            return
```

### 2. `on_job_complete()` — idle message failure is non-fatal

```python
async def on_job_complete(self, ws) -> None:
    ...
    try:
        await ws.send(json.dumps({"type": "worker_idle", ...}))
    except Exception as e:
        print(f"[on_job_complete] idle message failed: {e}, countdown starts anyway")
    self.idle_since = time.time()
    self._shutdown_task = asyncio.create_task(self._idle_countdown(ws))
```

### 3. `send_result()` — already has reconnect-and-retransmit logic

The `send_result` method (deployed earlier) creates a fresh WebSocket connection, re-registers, and retransmits the result when the original connection dies.

## Correct behavior after fix

| Scenario | Behavior |
|---|---|
| `progress` send fails | Retry 3x, then drop silently. Pipeline continues. |
| Pipeline step fails (metric3d/semantic/hanging/render) | Outer except catches → build_failed_payload → send_result |
| `send_result` send fails | Create fresh ws, re-register, retransmit result |
| Pipeline completes but ws is dead | send_result reconnect path delivers result |

## Business server coordination

Business server must also:
1. Accept delayed `result` messages and override `hanging_failed` status
2. Not re-dispatch jobs that already have R2 files uploaded
3. Implement WebSocket keepalive/ping to prevent idle timeouts
