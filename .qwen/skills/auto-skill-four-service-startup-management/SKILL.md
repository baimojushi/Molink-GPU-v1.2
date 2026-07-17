---
name: four-service-startup-management
description: Start, stop, restart, and diagnose the four MVP services (metric3d, semantic, hanging, worker) using verified unified scripts — never use manual uvicorn commands or bash -c
source: auto-skill
extracted_at: '2026-07-17T01:10:00.000Z'
---

# Four-Service Startup Management

## Verified commands (use only these)

All scripts are under `/opt/molink/hanging/`.

### Unified management (all 4 services)

```bash
# Start all services (metric3d, semantic, hanging, worker)
bash /opt/molink/hanging/scripts/start_all_mvp_apis.sh start

# Restart all services
bash /opt/molink/hanging/scripts/start_all_mvp_apis.sh restart

# Check status
bash /opt/molink/hanging/scripts/start_all_mvp_apis.sh status

# View logs
bash /opt/molink/hanging/scripts/start_all_mvp_apis.sh logs
```

### Worker only

```bash
# Restart worker
bash /opt/molink/hanging/scripts/restart_worker.sh

# Manual worker restart (if script unavailable)
pkill -f 'worker.main' || true
sleep 2
: > /opt/molink/hanging/logs/worker.log
cd /opt/molink/hanging && bash /opt/molink/hanging/worker/start_worker.sh &
```

### Individual services (only when unified script fails)

```bash
# metric3d (port 6006)
bash /opt/molink/hanging/services/metric3d_api/start_giant_auto.sh

# semantic (port 6011)
bash /opt/molink/hanging/scripts/start_semantic_v1_sam.sh

# hanging (port 6010)
bash /opt/molink/hanging/scripts/start_hanging_v1.sh
```

## Critical: ALLOW_UNAUTH_LOCAL=1

All three API services (metric3d, semantic, hanging) require `ALLOW_UNAUTH_LOCAL=1` environment variable to accept local worker requests. The verified startup scripts set this automatically:

- `start_giant_auto.sh` → exports `ALLOW_UNAUTH_LOCAL=1`
- `start_semantic_v1_sam.sh` → exports `ALLOW_UNAUTH_LOCAL=1`
- `start_hanging_v1.sh` → sources `.env` which sets `ALLOW_UNAUTH_LOCAL=1`

**Never start services with manual `uvicorn` commands** — they will miss this env var and return 500 errors:
```
API_KEY is not configured; set ALLOW_UNAUTH_LOCAL=1 for local development
```

## Common failure modes

### 1. Port already in use on restart

Symptom: `ERROR: [Errno 98] error while attempting to bind on address ('0.0.0.0', 6006): address already in use`

Fix: `start_all_mvp_apis.sh restart` handles this automatically (kills old process first). If using individual scripts, `pkill -9 -f "<service_name>"` first.

### 2. Service crashes and restarts without env var

Symptom: Service returns 500 with `API_KEY is not configured` even though startup script was used.

Root cause: A monitoring script or manual restart started the service without the proper startup script.

Fix: Kill all instances, then use `start_all_mvp_apis.sh restart`.

### 3. Worker exits after job failure

This is normal behavior. Worker exits with non-zero code on `FAILED_WORKER_EXCEPTION`. Restart with `restart_worker.sh`.

## Verification

After starting/restarting, always verify:

```bash
curl -s http://localhost:6006/health  # metric3d
curl -s http://localhost:6011/health  # semantic
curl -s http://localhost:6010/health  # hanging
tail -5 /opt/molink/hanging/logs/worker.log  # worker registration
```

All should return `{"status":"ok",...}`. Worker log should show `已注册 pro-xxxxx`.

## Service ports

| Service    | Port | Protocol     |
|------------|------|--------------|
| metric3d   | 6006 | HTTP REST    |
| semantic   | 6011 | HTTP REST    |
| hanging    | 6010 | HTTP REST    |
| worker     | none | WebSocket to wss://www.molink.art |
