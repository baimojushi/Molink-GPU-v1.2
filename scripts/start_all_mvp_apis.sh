#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/opt/molink/hanging}"
ACTION="${1:-start}"
MANAGER="$ROOT/scripts/manage_all_mvp_apis.sh"
if [ ! -f "$MANAGER" ]; then
  MANAGER="$ROOT/manage_all_mvp_apis.sh"
fi

case "$ACTION" in
  start|restart|reload)
    export MVP_PRINT_START_LOG_TAIL="${MVP_PRINT_START_LOG_TAIL:-1}"
    export MVP_START_LOG_LINES="${MVP_START_LOG_LINES:-80}"
    export MVP_LOG_LINES="${MVP_LOG_LINES:-120}"
    bash "$MANAGER" "$ACTION"
    bash "$MANAGER" status
    echo "[logs] streaming four-service logs for diagnostics. Press Ctrl-C to detach; services keep running."
    exec bash "$MANAGER" logs
    ;;
  status|logs|tail|stop|suspend|hangup)
    exec bash "$MANAGER" "$ACTION"
    ;;
  *)
    echo "Usage: $0 {start|restart|stop|status|logs}"
    exit 1
    ;;
esac
