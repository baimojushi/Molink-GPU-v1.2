---
name: metric3d-zombie-process-diagnosis
description: Diagnose and recover metric3d service when port is listening and /health returns 200, but /analyze_geometry calls timeout or return 400
source: auto-skill
extracted_at: '2026-06-07T09:00:00.000Z'
---

## Symptom

`manage_all_mvp_apis.sh start` reports `[skip:metric3d] port 6006 already listening`, `/health` returns HTTP 200, but `/analyze_geometry` calls:
- Timeout (curl hangs, no response body)
- Return HTTP 400 with empty or minimal body
- No entries in application logs (requests never reach handler)

## Root cause

Uvicorn process is alive and serving the synchronous `/health` endpoint, but the model inference worker thread is stuck from a prior failed request (often OOM-killed mid-inference). The process becomes a zombie: accepts connections but cannot execute `/analyze_geometry`.

This is distinct from:
- **Port occupied, health DOWN**: process not yet ready → wait (covered in four-service-startup-management)
- **Port not listening**: process crashed → check logs and restart (covered in four-service-startup-management)

## Diagnosis procedure

```bash
# 1. Health endpoint works (always responds fast)
curl -s --connect-timeout 5 http://127.0.0.1:6006/health
# → Returns 200, {"status":"ok",...}

# 2. Actual endpoint hangs or 400s
curl -s --connect-timeout 30 -X POST http://127.0.0.1:6006/analyze_geometry \
  -F "file=@/path/to/room.jpg" -o /dev/null -w "HTTP:%{http_code}\n" 2>&1
# → Hangs, returns 400, or times out

# 3. Confirm uvicorn process is alive but no inference is running
ps aux | grep uvicorn | grep 6006 | grep -v grep
# → Process exists with normal CPU/memory (no active inference)

# 4. Check for stale OOM in dmesg (if available)
dmesg 2>/dev/null | tail -50 | grep -i -E "oom|kill"
```

## Recovery

```bash
# 1. Kill ALL holders of port 6006 (including zombie uvicorn)
lsof -ti:6006 2>/dev/null | xargs -r kill -9

# 2. Start fresh
cd /opt/metric3d_api/services/metric3d_api
bash start_giant_auto.sh &>/root/autodl-tmp/tmp/metric3d_restart.log &

# 3. Wait for model load (15-30s for vit_giant2)
for i in $(seq 1 20); do
  resp=$(curl -s --connect-timeout 3 http://127.0.0.1:6006/health 2>&1)
  [ -n "$resp" ] && echo "READY: $resp" && break
  echo "waiting... $i"
  sleep 3
done

# 4. Verify actual endpoint works (not just /health)
curl -s --connect-timeout 30 -X POST http://127.0.0.1:6006/analyze_geometry \
  -F "file=@/path/to/test.jpg" -o /dev/null -w "HTTP:%{http_code} TIME:%{time_total}\n"
# → Expected: HTTP:200 TIME:~30s
```

## Important notes

- **Root cause (missing weight → OOM → zombie) permanently fixed 2026-06-08.** `get_model()` in `app.py` now loads weights directly from `/root/autodl-tmp/models/` via `torch.load()`, bypassing `load_state_dict_from_url()`. Weight-related zombies should no longer occur. This diagnostic still applies to other zombie causes (GPU OOM, memory pressure, deadlocks).
- **`manage_all_mvp_apis.sh start` now DOES detect this state** (fixed 2026-06-07). `_start_one()` calls `_health_ok` after detecting port occupancy. If port is open but HTTP health fails, `_force_stop_port` kills the zombie and falls through to restart.
- The `restart` subcommand of `manage_all_mvp_apis.sh` does stop and restart, so `manage_all_mvp_apis.sh restart` would recover from this. But it's slower than the targeted fix above.
- **`start_giant_auto.sh` has `set -e`**: if uvicorn fails to bind (`address already in use`), the script exits immediately. Log files may not be created because the process dies before writing.
- When starting manually, ensure `cd` to the correct `services/metric3d_api/` directory before running `start_giant_auto.sh`, otherwise Python imports may fail silently.
- If background shell reports exit but uvicorn is still running, it's because `&` backgrounding in a subshell doesn't keep the parent alive. The uvicorn child process persists independently.
- After recovery, verify with a real `/analyze_geometry` call — `/health` alone is insufficient to confirm the service can handle inference requests.
- An `OMP_NUM_THREADS` warning may appear in conda environment output (`libgomp: Invalid value for environment variable OMP_NUM_THREADS`) — this is benign and does not affect functionality.

## Prevention

**Implemented 2026-06-07.** `manage_all_mvp_apis.sh start` now calls `_health_ok` after detecting port occupancy. If port is open but HTTP health fails, it runs `_force_stop_port` (pidfile kill + pkill by port pattern + 10s port release wait) then falls through to restart. The manager no longer blindly skips on port-open alone.
