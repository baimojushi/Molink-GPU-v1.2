---
name: metric3d-port-conflict-on-restart
description: metric3d restart fails with 'address already in use' if old process not killed first — always pkill before restarting
source: auto-skill
extracted_at: '2026-07-17T00:52:00.000Z'
---

# metric3d Port Conflict on Restart

## Symptom

After metric3d crashes or is killed, restart fails with:

```
ERROR: [Errno 98] error while attempting to bind on address ('0.0.0.0', 6006): address already in use
```

## Root Cause

The old metric3d process may still be holding port 6006 even if it's not responding to health checks. Starting a new process fails immediately.

## Correct Restart Procedure

**Step 1: Kill old process**
```bash
pkill -9 -f "metric3d_api" || true
sleep 2
```

**Step 2: Verify port is free**
```bash
ps aux | grep "metric3d_api" | grep -v grep  # Should be empty
```

**Step 3: Start with correct script**
```bash
cd /opt/molink/hanging/services/metric3d_api
bash start_giant_auto.sh
```

**Step 4: Wait and verify**
```bash
sleep 10
curl -s --connect-timeout 3 http://localhost:6006/health | python3 -c "import sys,json; print('metric3d:', json.load(sys.stdin)['status'])"
```

## Common Mistake

Running `start_giant_auto.sh` without killing the old process first. The script will fail silently (exit code 1) and the service remains broken.

## Verification Checklist

After restart, confirm:
- [ ] `ps aux | grep metric3d_api` shows a new PID
- [ ] `curl http://localhost:6006/health` returns `{"status": "ok"}`
- [ ] Worker pipeline can call metric3d without 500 errors
