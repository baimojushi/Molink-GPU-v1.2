---
name: hanging-restart-verification
description: After deploying code patches, verify hanging service actually restarted — pkill may fail silently on pts/0 foreground processes; always check PID and start time changed
source: auto-skill
extracted_at: '2026-07-16T04:53:29.509Z'
---

## Problem

Deploying new code to `/opt/molink/hanging` and running `pkill -f "hanging_mvp.service"` followed by a restart command **does not guarantee** the old process is terminated. The hanging service runs in a pts/0 terminal session as a foreground process group member, which can resist SIGTERM and even SIGKILL in some AutoDL container configurations.

**Observed behavior (2026-07-16):**
- Deployed HEIGHT-FIX-V2 zip at ~11:57
- Ran `pkill -f "hanging_mvp.service"` and restart command
- New orders (12:06, 12:40) still showed height override bug
- Root cause: hanging process PID 1303 (started 11:37) was still running old code
- `pkill -9` eventually killed it; new PID 5936 started at 12:52 with new code
- **Second issue:** Manual restart without proper startup script caused 500 errors (job_2013416c) due to missing `ALLOW_UNAUTH_LOCAL` and `API_KEY` environment variables

## Verification procedure

After any code deployment that modifies `hanging_mvp/` modules:

1. **Record pre-deployment PID and start time:**
   ```bash
   ps aux | grep "hanging_mvp.service" | grep -v grep
   # Note: PID and start time (e.g., PID 1303, started 11:37)
   ```

2. **Deploy code and clear Python cache:**
   ```bash
   find /opt/molink/hanging/hanging_mvp -name "*.pyc" -delete
   find /opt/molink/hanging/hanging_mvp -name "__pycache__" -exec rm -rf {} + 2>/dev/null
   ```

3. **Force kill old process:**
   ```bash
   pkill -9 -f "hanging_mvp.service" || true
   sleep 2
   ```

4. **Verify old process is dead:**
   ```bash
   ps aux | grep "hanging_mvp.service" | grep -v grep
   # Should return empty — if PID 1303 still appears, kill failed
   ```

5. **Start new service using the proper startup script:**
   ```bash
   bash /opt/molink/hanging/scripts/start_hanging_v1.sh &
   ```
   **WARNING:** Do NOT manually run `nohup python -m uvicorn ...` without sourcing the startup script. The script sets critical environment variables (`ALLOW_UNAUTH_LOCAL=1`, `API_KEY` from `.env`). Missing these causes 500 Internal Server Error on every request.

6. **Verify new PID, start time, AND environment variables:**
   ```bash
   sleep 8
   NEW_PID=$(pgrep -f "hanging_mvp.service" | head -1)
   ps aux | grep "$NEW_PID" | grep -v grep
   # Confirm: NEW PID (different from pre-deployment), start time = now
   
   # CRITICAL: Verify environment variables
   cat /proc/$NEW_PID/environ 2>/dev/null | tr '\0' '\n' | grep -E "ALLOW_UNAUTH_LOCAL|API_KEY"
   # Must show: ALLOW_UNAUTH_LOCAL=1 and API_KEY=<value>
   ```

7. **Health check:**
   ```bash
   curl -s --connect-timeout 3 http://localhost:6010/health | python3 -m json.tool
   ```

## Key invariants

**PID must change after restart.** If `ps aux` shows the same PID and start time as before the deploy, the service is still running old code regardless of what the restart script reported.

**Environment variables must be present.** If `ALLOW_UNAUTH_LOCAL` or `API_KEY` are missing from the process environment, all requests will fail with 500 Internal Server Error. Always use `scripts/start_hanging_v1.sh` to start the service, never manually invoke uvicorn without the env setup.

## How to apply

When deploying any patch that modifies hanging_mvp/*.py files, always:
1. Verify the hanging service PID changed after restart
2. Verify environment variables are loaded correctly
3. Do not trust "restart completed" messages — only trust `ps aux` output showing a new PID with a recent start time and `cat /proc/PID/environ` showing correct env vars
