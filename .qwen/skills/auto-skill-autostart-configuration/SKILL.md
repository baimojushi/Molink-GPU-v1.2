---
name: autostart-configuration
description: Direct-process autostart for all 4 services (metric3d, semantic, hanging, worker) on AutoDL container boot — no supervisord
source: auto-skill
extracted_at: '2026-07-15T09:05:00.000Z'
---

# Autostart Configuration

## Architecture

AutoDL container uses `/init/` boot mechanism with its own `/bin/supervisord` managing system services (autopanel, jupyterlab, proxy, sshd). This supervisord config at `/init/supervisor/supervisor.ini` is **encrypted and read-only** — molink services cannot be added.

**Previous approach (BROKEN)**: A second supervisord instance for molink services conflicted with AutoDL's built-in supervisord, causing port contention, startup failure, and missing worker process. Server crashed on reboot.

**Current approach**: `molink-autostart.sh` launches all 4 services directly as background subprocesses with proper conda env activation, env var sourcing, and PID file tracking. No supervisord dependency.

## Management script

| File | Purpose |
|------|---------|
| `/opt/molink/hanging/ops/molink-autostart.sh` | Start/stop/restart/status wrapper (also installed at `/etc/init.d/molink-autostart`) |
| `/etc/rc.local` | Boot entry point — calls `bash /opt/molink/hanging/ops/molink-autostart.sh start` |

## PID file tracking

Each service writes its PID to `/tmp/molink-{metric3d,semantic,hanging,worker}.pid` for status/stop management.

## Service launch order

All 4 services start concurrently as background processes:

| Program | Port | Env source | Notes |
|---------|------|-----------|-------|
| metric3d | 6006 | PYTHONPATH + model paths | Starts in `/opt/molink/hanging/services/metric3d_api` |
| semantic | 6011 | `.env.vlm_sam` + `.env` + ALLOW_UNAUTH_LOCAL | SAM model config via env vars |
| hanging | 6010 | `.env` + ALLOW_UNAUTH_LOCAL | |
| worker | WebSocket | `/etc/molink/gpu.env` + `/etc/profile.d/molink-gpu.sh` | Must start AFTER APIs are healthy |

## Management commands

```bash
# Start all molink services
bash /opt/molink/hanging/ops/molink-autostart.sh start

# Stop all molink services (kills tracked PIDs)
bash /opt/molink/hanging/ops/molink-autostart.sh stop

# Restart all molink services
bash /opt/molink/hanging/ops/molink-autostart.sh restart

# Check status (reads PID files, verifies processes alive)
bash /opt/molink/hanging/ops/molink-autostart.sh status

# View logs
tail -f /var/log/molink/metric3d.log
tail -f /var/log/molink/semantic.log
tail -f /var/log/molink/hanging.log
tail -f /var/log/molink/worker.log
```

## Boot autostart

`/etc/rc.local` already contains:
```bash
#!/bin/bash
# Molink GPU services autostart
bash /opt/molink/hanging/ops/molink-autostart.sh start
```

## Important constraints

- **Do not** use `manage_all_mvp_apis.sh` while autostart services are running — pidfile conflicts will occur
- **Do not** start services manually while autostart is active — duplicate port binding will fail
- Worker loads API keys from `/etc/molink/gpu.env`, not from project `.env` files
- `ALLOW_UNAUTH_LOCAL=1` must be set for semantic and hanging to accept local requests
- Health check polls ports 6006/6011/6010 for up to 60s after launch

## Troubleshooting

1. **Service not running**: Check `/var/log/molink/{service}.log` for startup errors (missing env vars, model paths)
2. **Port already in use**: AutoDL's built-in supervisor may still be managing a conflicting process — check `ps aux | grep uvicorn`
3. **Worker not connecting**: Verify `/etc/molink/gpu.env` exists and contains WORKER_SECRET, API_KEY, VLM_API_KEY
4. **Autostart didn't run after reboot**: Check `dmesg | grep rc.local` or boot logs for execution errors
