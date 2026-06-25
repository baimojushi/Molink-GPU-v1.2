---
name: bfl-download-retry
description: BFL CDN sample.png downloads require retry protection — transient SSL EOF errors will abort the erase→hardpaste→render pipeline without it
source: auto-skill
extracted_at: '2026-06-25T02:57:00.000Z'
---

## Problem

`BflClient.download_sample()` performs a single `requests.get()` to download the BFL CDN `sample.png` URL returned after erase/edit polling completes. This call has no retry protection.

BFL CDN occasionally returns transient SSL EOF errors:

```
SSLError(SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1017)'))
```

This is a network-layer transient failure, not a task failure. Without retry, the entire hanging pipeline aborts at the erase stage, producing `status=failed` even though `exit_code=OK_SELECTED` (candidates were selected successfully).

## Fix

Wrap `download_sample` in the same `max_retries` exponential-backoff loop used by `_request_json`:

```python
def download_sample(self, result: Mapping[str, Any], output_path: Path) -> None:
    sample_url = ...  # extract from result
    if not sample_url:
        raise RuntimeError(...)
    last_error: Exception | None = None
    for attempt in range(self.max_retries):
        try:
            resp = requests.get(str(sample_url), timeout=self.timeout_s)
            resp.raise_for_status()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(resp.content)
            return
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= self.max_retries - 1:
                break
            time.sleep(2**attempt)
    raise RuntimeError(f"BFL sample download failed after {self.max_retries} attempts: {last_error}")
```

## Key rule

**Any HTTP download of BFL CDN result assets (sample.png, output images, etc.) must be wrapped in retry logic.** The BFL CDN is not guaranteed to serve large image payloads on the first attempt. A single-request download is a bug waiting to manifest as a full pipeline failure.

## Diagnosis checklist

When a hanging job shows `status=failed` with `exit_code=OK_SELECTED` and `error` contains `SSLError` or `UNEXPECTED_EOF_WHILE_READING`:
1. Check `05_render/flux_out/edit_trace.json` for the download error in `errors[]`
2. Verify `download_sample` or any direct `requests.get()` of CDN URLs lacks retry
3. Add retry loop matching `_request_json` pattern
