---
name: http-error-body-capture
description: Replace generic raise_for_status() with explicit status+body capture to expose root cause on HTTP 4xx/5xx
source: auto-skill
extracted_at: '2026-06-21T14:31:28.463Z'
---

## Problem
`response.raise_for_status()` throws `HTTPError` with only status code and reason phrase. The actual error body (e.g.,鉴权失败、CUDA OOM、模型异常、图片格式错误) is lost, making diagnosis impossible from logs alone.

## Pattern
Replace:

```python
r.raise_for_status()
```

With:

```python
if r.status_code >= 400:
    raise RuntimeError(f"ServiceName HTTP {r.status_code}: {r.text[:1000]}")
```

## Why
- The response body from internal services (metric3d, semantic, BFL, etc.) contains the actual error: authentication failure, CUDA out-of-memory, invalid input image, model load error, etc.
- Truncating to 1000 chars avoids bloating logs while preserving useful context.
- `RuntimeError` is sufficient; the stack trace plus body text points directly to the fault.

## When to apply
- Any `httpx.post()` / `httpx.get()` / `requests.post()` / `requests.get()` call to internal microservices or third-party APIs where the response body carries diagnostic information.
- Especially critical for GPU services (CUDA errors, OOM kills, weight loading failures) and file-processing endpoints (image format/size validation errors).

## Applied fixes
- `image_editing/providers/bfl.py`: `BflClient._request_json()` — replaced `raise_for_status()` with explicit status+body capture for 4xx/5xx responses.
