---
name: websocket-send-result-timeout
description: Worker send_result() fails with ConnectionClosedError after long-running render jobs — business server WebSocket idle timeout; job completes locally but result delivery fails
source: auto-skill
extracted_at: '2026-06-26T00:30:00.000Z'
---

# WebSocket send_result Timeout

## Problem

After a job completes all pipeline stages (metric3d → semantic → hanging → render → r2_upload), the worker's `send_result()` call fails with:

```
websockets.exceptions.ConnectionClosedError: no close frame received or sent
[worker-debug] send_result ... status=failed exit_code=FAILED_WORKER_EXCEPTION
```

The job actually succeeded locally (diagnostics zip created, all stages passed), but the WebSocket connection to the business server was closed during the long-running render phase.

## Root Cause

The business server has a WebSocket idle timeout. Render jobs (especially full-space textured wall rendering) take long enough that the server silently closes the connection. The worker only discovers this when it tries to `send_result()` at the end.

## Affected Jobs

Observed on:
- `job_1bfacbe3-c592-4da4-bdf6-ffea6072d226` (2026-06-25 23:29)
- `job_dd30a2f9-7178-44fd-8b70-441be7d29648` (2026-06-26 00:21)

Both had `05_render` directory present, `exit_code=OK_SELECTED` in `client_result.json`, but `FAILED_WORKER_EXCEPTION` in worker log.

## How to Diagnose

1. Check worker log: `grep <job_id> /opt/molink/hanging/logs/worker.log`
2. Look for `send_result ... status=failed ... error=no close frame received or sent`
3. Verify run directory has all stages: `ls <run_dir>/` should show `05_render`
4. Check `client_result.json`: `exit_code` is likely `OK_SELECTED`, not a pipeline failure

## Workaround

The business server automatically retries the job after WebSocket reconnection. The second attempt usually succeeds because:
- The job is re-queued by the business server
- The worker reconnects with a fresh WebSocket
- Total wall-clock time is shorter (no initial connection setup delay)

## Code Fix (deployed 2026-06-26)

`worker/main.py` — `Worker.send_result()` now includes reconnect-and-retransmit logic:

1. Try to send result via current `ws`
2. If `ConnectionClosedError`: create fresh WebSocket → re-register → retransmit result + progress via new connection
3. If retransmit also fails: log error, result saved locally only

This ensures job results are delivered even if the original connection died during pipeline execution. The fix is in commit `f772156`.

## Recommended Server-Side Fix

On the business server side:
- Increase WebSocket idle timeout to accommodate render duration (typically 3-8 minutes)
- Add WebSocket keepalive/ping frames during long-running stages
- Or: switch to HTTP callback for `send_result` instead of WebSocket
