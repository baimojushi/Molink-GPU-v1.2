---
name: worker-api-key-env
description: Worker processes load API keys from /etc/molink/gpu.env, not from project .env — provider API keys must be added to gpu.env and require worker restart to take effect
source: auto-skill
extracted_at: '2026-06-25T01:45:00Z'
---

## Problem

Provider API keys (e.g. `APIYI_API_KEY`) configured in project `.env` are invisible to the worker process. Render calls fail with:

```
"APIYI_API_KEY, YI_API_KEY or GPT_IMAGE2_API_KEY is required for APIYI GPT Image provider"
```

## Why

`worker/start_worker.sh` sources these system files for environment variables:

```bash
set -a
[ -f /etc/profile.d/molink-gpu.sh ] && source /etc/profile.d/molink-gpu.sh
[ -f /etc/molink/gpu.env ] && source /etc/molink/gpu.env
set +a
```

The project `.env` is **not** sourced by the worker. It is only useful for interactive CLI sessions where you run `source .env` manually.

## Solution

1. Append the provider API key and related env vars to `/etc/molink/gpu.env`:
   ```bash
   echo 'APIYI_API_KEY="sk-..."' >> /etc/molink/gpu.env
   echo 'APIYI_API_BASE="https://api.apiyi.com/v1"' >> /etc/molink/gpu.env
   # ... other provider-specific vars
   ```

2. Restart the worker to pick up the new environment:
   ```bash
   pkill -f "worker.main" && bash /opt/molink/hanging/worker/start_worker.sh &
   ```

3. Verify in the worker log or `runtime_context.json` that the provider is recognized.

## Affected providers

Any provider that requires an API key and is called from the worker process:
- APIYI (`APIYI_API_KEY`, `YI_API_KEY`, `GPT_IMAGE2_API_KEY`)
- BFL (`BFL_API_KEY` — already in gpu.env)
- VLM (`VLM_API_KEY` — already in gpu.env)
- Any future provider additions
