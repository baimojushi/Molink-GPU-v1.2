---
name: worker-progress-on-failure
description: Worker send_result() must emit "failed" progress when status is not succeeded, not unconditional "done"
source: auto-skill
extracted_at: '2026-06-21T14:31:28.463Z'
---

## Problem
Worker `send_result()` was unconditionally sending `progress("done", 100, "完成")` after every result, including failures. Clients see "完成" even when the pipeline crashed, masking errors.

## Pattern
In `send_result()`, check payload status before emitting progress:

```python
if payload.get("status") in {"succeeded", "succeeded_partial"}:
    await self.progress(ws, job, "done", "done", 100, "完成")
else:
    await self.progress(ws, job, "failed", "failed", 92, "本次自动生成遇到异常，团队会继续为您处理…")
```

## Why
- `succeeded_partial` is still a usable delivery (some images generated)
- `failed` or any other status means the client should not expect deliverable output
- The 92% progress + failure message tells the frontend to show an error state, not a completion state

## When to apply
- Any worker/async task that sends progress updates and can succeed or fail
- Especially important when the client UX depends on distinguishing completion from failure
