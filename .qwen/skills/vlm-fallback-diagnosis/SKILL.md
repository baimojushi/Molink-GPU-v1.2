---
name: vlm-fallback-diagnosis
description: Diagnose why VLM semantic parsing falls back to mock responses instead of calling the real API
source: auto-skill
extracted_at: '2026-06-07T03:45:00.000Z'
---

## Symptom

`/parse_semantics_mvp` returns mock wall/floor objects with `"notes": "mock coarse wall"` or a `"warning": "vlm_failed_fallback_to_mock"` instead of real VLM detections.

## Root causes

### 1. Startup script does not source `.env.vlm_sam`

The most common cause. The file `.env.vlm_sam` may contain valid credentials, but if the startup script doesn't `source` it, the env vars are never exported.

**Fix:** Add this line to the startup script before launching uvicorn:

```bash
[ -f /opt/metric3d_api/.env.vlm_sam ] && source /opt/metric3d_api/.env.vlm_sam
```

Verify with: `ps aux | grep semantic` to find the running process, then check its env: `cat /proc/<pid>/environ | tr '\0' '\n' | grep VLM`

### 2. `.env.vlm_sam` missing or empty fields

Check the file exists and has non-empty values:

```bash
cat /opt/metric3d_api/.env.vlm_sam
```

Required fields: `VLM_PROVIDER`, `VLM_API_URL`, `VLM_API_KEY`, `VLM_MODEL`. All three must be non-empty.

### 3. Fallback logic in `vlm_client.py`

The `parse_with_vlm()` function in `semantic_mvp/vlm_client.py` follows this decision tree:

```python
def parse_with_vlm(..., mode: str, ...):
    if mode == "mock":
        return mock_objects(...)           # explicit mock
    if mode == "openai_compatible":
        return call_openai_compatible(...) # explicit API call
    # auto mode:
    if os.getenv("VLM_PROVIDER") == "openai_compatible":
        try:
            return call_openai_compatible(...)
        except Exception as e:
            return {**mock_objects(...), "warning": f"vlm_failed_fallback_to_mock: {e}"}
    return {**mock_objects(...), "warning": "no_vlm_provider_configured_fallback_to_mock"}
```

The `no_vlm_provider_configured_fallback_to_mock` warning means `VLM_PROVIDER` is unset or not `"openai_compatible"`.

### 4. VLM timeout (120s boundary)

`qwen3-vl-32b-thinking` calls the DashScope API with a default 120-second timeout. If the model response takes longer than 120 seconds (common on GPU-heavy instances), the request times out and falls back to mock:

```
"vlm_failed_fallback_to_mock: HTTPSConnectionPool(host='dashscope.aliyuncs.com', port=443): Read timed out. (read timeout=120)"
```

**Fix options:**
- **Reduce model depth**: Switch to `qwen3-vl-plus` in `.env.vlm_sam` (20-40s per call, stays under timeout)
- **Increase timeout**: Add `VLM_TIMEOUT_S="180"` or higher to `.env.vlm_sam`
- **Split roles**: Use `qwen3-vl-flash` for BFL exit review (`.env.bfl`), `qwen3-vl-plus` for semantic service (`.env.vlm_sam`)

### 5. DashScope free tier exhausted (403 Forbidden)

When the configured model's free tier quota is depleted, DashScope returns HTTP 403:

```json
{"error": {"message": "The free tier of the model has been exhausted...", "type": "AllocationQuota.FreeTierOnly", "code": "AllocationQuota.FreeTierOnly"}}
```

The semantic service does NOT catch this — it raises `HTTPError: 403 Client Error: Forbidden` and returns HTTP 500. No fallback to mock.

**Diagnosis:**

```bash
source /opt/metric3d_api/.env.vlm_sam
curl -s -X POST "$VLM_API_URL" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VLM_API_KEY" \
  -d '{"model":"'"$VLM_MODEL"'","messages":[{"role":"user","content":[{"type":"text","text":"OK"}]}],"max_tokens":5}'
```

If 403, test `qwen3-vl-plus` under the same key — it often still works:

```bash
# same curl but with "model":"qwen3-vl-plus"
```

**Fix:** Change `VLM_MODEL` in `.env.vlm_sam` to `qwen3-vl-plus`, then restart semantic service:

```bash
sed -i 's/VLM_MODEL=.*/VLM_MODEL="qwen3-vl-plus"/' /opt/metric3d_api/.env.vlm_sam
bash /opt/metric3d_api/scripts/manage_all_mvp_apis.sh restart
```

### 6. `.env.bfl` VLM fields empty

The `.env.bfl` file must have `VLM_API_URL`, `VLM_API_KEY`, and `VLM_MODEL` set for the BFL hard-paste exit review to function. If empty, high-risk candidates are silently blocked. The VLM credentials are separate from `.env.vlm_sam` — both files must be configured independently.

## Quick diagnosis checklist

```bash
# 1. File exists?
ls -la /opt/metric3d_api/.env.vlm_sam

# 2. Fields populated?
grep -E 'VLM_PROVIDER|VLM_API_KEY|VLM_API_URL|VLM_MODEL' /opt/metric3d_api/.env.vlm_sam

# 3. Startup script sources it?
grep 'vlm_sam' /opt/metric3d_api/scripts/start_semantic_v1_sam.sh

# 4. Running process has the env?
PID=$(pgrep -f 'uvicorn semantic_mvp')
cat /proc/$PID/environ 2>/dev/null | tr '\0' '\n' | grep VLM

# 5. Test VLM API directly (free-tier 403 check)
source /opt/metric3d_api/.env.vlm_sam
curl -s -X POST "$VLM_API_URL" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VLM_API_KEY" \
  -d '{"model":"'"$VLM_MODEL"'","messages":[{"role":"user","content":[{"type":"text","text":"OK"}]}],"max_tokens":5}'
# If 403 "free tier exhausted": change VLM_MODEL to qwen3-vl-plus, restart semantic

# 6. Test semantic service with real VLM
curl -s -X POST http://127.0.0.1:6011/parse_semantics_mvp \
  -F "room_image=@/path/to/test.jpg" -F "vlm_mode=openai_compatible" -F 'return_debug_assets=false' | python3 -m json.tool
```
