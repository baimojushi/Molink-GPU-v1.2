---
name: hanging-service-restart-verification
description: After deploying code patches, verify hanging service actually restarted — pkill may fail silently on pts/0 foreground processes; always check PID and start time changed
source: auto-skill
extracted_at: '2026-07-17T00:22:00.000Z'
---

## Problem

After deploying new hanging code and running `pkill -f "hanging_mvp.service"`, the old process (PID 1303, started at 11:37) remained alive despite the kill signal. Subsequent orders continued using old code, causing height calculation bugs to persist.

## Root cause

The hanging process was launched from a pts/0 terminal with `nohup ... &`. The process became a foreground process group member on that terminal, making it resistant to `pkill -f` signals from other terminals.

## Correct restart procedure

### Method 1: Use the startup script (recommended)

```bash
pkill -9 -f "hanging_mvp.service"
sleep 2
cd /opt/molink/hanging && bash /opt/molink/hanging/scripts/start_hanging_v1.sh
```

### Method 2: Manual restart

```bash
# Force kill with signal 9
pkill -9 -f "hanging_mvp.service"
sleep 2

# Clear Python cache
find /opt/molink/hanging/hanging_mvp -name "*.pyc" -delete
find /opt/molink/hanging/hanging_mvp -name "__pycache__" -exec rm -rf {} +

# Start with correct environment
cd /opt/molink/hanging && bash /opt/molink/hanging/scripts/start_hanging_v1.sh
```

## Verification checklist

After restart, ALWAYS verify:

```bash
# 1. Check PID changed
ps aux | grep "hanging_mvp.service" | grep -v grep

# 2. Check start time is recent (not from before deploy)
ps -p <PID> -o pid,lstart,etime

# 3. Check health endpoint
curl -s http://localhost:6010/health

# 4. Check environment variables loaded
cat /proc/<PID>/environ | tr '\0' '\n' | grep -E "ALLOW_UNAUTH_LOCAL|API_KEY"
```

## Common pitfalls

- `pkill -f` (without -9) may only send SIGTERM, which uvicorn may ignore
- Process started with `nohup ... &` on pts/0 may survive pkill from other terminals
- Python `__pycache__` may serve old bytecode even after code files are updated
- Manual `nohup python -m uvicorn ...` without the startup script misses `ALLOW_UNAUTH_LOCAL` and `API_KEY` env vars, causing 500 errors

## How to apply

After any hanging code patch deployment:
1. Deploy code
2. Clear `__pycache__`
3. Force kill with `pkill -9`
4. Start with `start_hanging_v1.sh` (not bare nohup)
5. Verify PID, start time, health, and env vars
