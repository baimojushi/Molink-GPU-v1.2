---
name: three-service-startup-management
description: Start, stop, restart, and diagnose the three MVP services (metric3d, semantic, hanging) using unified scripts
source: auto-skill
extracted_at: '2026-06-08T01:30:00.000Z'
---

# Three-Service Startup Management

## Service inventory

| Service | Port | Health endpoint | Startup script | Kind |
|---|---|---|---|---|
| metric3d | 6006 | `/health` | `start_giant_auto.sh` (via `metric3d_ctl.sh`) | api |
| semantic | 6011 | `/health` | `scripts/start_semantic_v1_sam.sh` | api |
| hanging | 6010 | `/health` | `scripts/start_hanging_v1.sh` | api |

Config defined in `scripts/manage_all_mvp_apis.sh` as pipe-delimited `SERVICES` array entries.

**Note:** `auto_update` daemon was removed 2026-06-08. It performed `git pull + checkout` which reset local code patches. All three services are now the permanent set.

## One-command start

```bash
cd /opt/metric3d_api
bash scripts/start_all_mvp_apis.sh
```

Starts all three services in order (metric3d → semantic → hanging), waits for port readiness, and tails recent logs (80 lines per service). Exit with `Ctrl-C`; services remain running in background.

## Management commands

```bash
cd /opt/metric3d_api
bash scripts/manage_all_mvp_apis.sh status    # show UP/DOWN for each service
bash scripts/manage_all_mvp_apis.sh start      # start stopped services (idempotent)
bash scripts/manage_all_mvp_apis.sh stop       # stop all services
bash scripts/manage_all_mvp_apis.sh restart    # stop then start
bash scripts/manage_all_mvp_apis.sh logs       # tail -F all three log files
bash scripts/manage_all_mvp_apis.sh start hanging   # start single service
```

Default `MVP_SERVICES=metric3d,semantic,hanging`. No `--with-auto-update` or `--start-auto-update` flags exist.

## Log files

All logs under `/opt/metric3d_api/logs/`:
- `metric3d.log`
- `semantic.log`
- `hanging.log`

`MVP_START_LOG_LINES` env var controls recent log lines on start (default: 80).
`MVP_PRINT_START_LOG_TAIL=1` enables `_print_recent_log()` output on start.

## Startup troubleshooting

1. **Port occupied, health DOWN**: Service process exists but not yet ready. Wait 10-20s, then `status`.
2. **Port not listening**: Service failed to start. Check `<service>.log` for traceback.
3. **Hanging health returns 500**: Dependency issue (metric3d or semantic DOWN). Start dependencies first.
4. **Services in partial state after restart**: `stop` clears pidfiles, `start` checks pidfiles → use `restart` for clean cycle.

## Zombie process auto-recovery (implemented 2026-06-07)

`manage_all_mvp_apis.sh start` now detects and recovers from zombie processes automatically:

1. `_health_ok` checks that port open → HTTP 2xx on `/health`
2. If port is open but health fails, `_force_stop_port` kills the stale process (pidfile + pkill fallback) and waits up to 10s for port release
3. Then falls through to normal restart logic

Before this fix, `_start_one` only checked `_port_open` (TCP connect), skipping restart if the port accepted connections — even when the process was a zombie serving stale `/health` 200s but returning 400/timeouts on real endpoints.

## Post-deployment port alignment verification

After deploying a patch that touches service code or config, verify that the manager's `SERVICE_SPECS` port definitions match the actual `--port` values in startup scripts:

```bash
# 1. Check manager spec definitions
grep '^\s*"' scripts/manage_all_mvp_apis.sh

# 2. Check actual port in each startup script
grep -n 'PORT\|--port' scripts/start_hanging_v1.sh scripts/start_semantic_v1_sam.sh \
  services/metric3d_api/start_giant_auto.sh services/metric3d_api/start_giant.sh \
  start_hanging_mvp.sh start_semantic_mvp.sh

# 3. Verify port connectivity
python3 -c "
import socket
for port in [6006, 6010, 6011]:
    s = socket.socket(); s.settimeout(1)
    try:
        s.connect(('127.0.0.1', port)); print(f'Port {port}: OPEN')
    except:
        print(f'Port {port}: CLOSED')
    finally:
        s.close()
"

# 4. Health endpoint check
curl -s http://127.0.0.1:6006/health
curl -s http://127.0.0.1:6010/health
curl -s http://127.0.0.1:6011/health
```

Expected alignment:
| Service | Manager Spec | Startup Script Default | Actual |
|---------|-------------|----------------------|--------|
| metric3d | 6006 | start_giant_auto.sh: `--port 6006` | 6006 |
| hanging | 6010 | start_hanging_v1.sh: `--port $HANGING_MVP_PORT` (default 6010) | 6010 |
| semantic | 6011 | start_semantic_v1_sam.sh: `--port $SEMANTIC_MVP_PORT` (default 6011) | 6011 |

## Symlinks

Project root contains symlinks for convenience:
- `start_all_mvp_apis.sh` → `scripts/start_all_mvp_apis.sh`
- `manage_all_mvp_apis.sh` → `scripts/manage_all_mvp_apis.sh`

## CLI entrypoint

```bash
# Interactive or fully automated
python3 scripts/hanging_mvp_cli.py run \
  --room /root/autodl-tmp/pictures/qwen/room-exif.jpg \
  --artwork /root/autodl-tmp/pictures/qwen/artwork51.2×67.8cm.png \
  --art-width-cm 51.2 --art-height-cm 67.8 \
  --output-dir /root/hanging_runs --skip-bfl
```

CLI auto-starts all three services before running the pipeline. No `--start-auto-update` flag exists.
