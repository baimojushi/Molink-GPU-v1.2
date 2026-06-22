#!/usr/bin/env bash
set +e

ROOT="${ROOT:-/opt/molink/hanging}"
LOG_DIR="$ROOT/logs"
RUN_DIR="$ROOT/run"
mkdir -p "$LOG_DIR" "$RUN_DIR"

case "$-" in
  *i*) ;;
  *) return 0 2>/dev/null || exit 0 ;;
esac

if [ "${ARTWORK_HANGING_BASHRC_AUTOSTART_RAN:-0}" = "1" ]; then
  return 0 2>/dev/null || exit 0
fi
export ARTWORK_HANGING_BASHRC_AUTOSTART_RAN=1

if [ -f "$RUN_DIR/disable_bashrc_autostart" ]; then
  echo "[artwork-autostart] disabled by $RUN_DIR/disable_bashrc_autostart"
  return 0 2>/dev/null || exit 0
fi

{
  echo "===== $(date '+%Y-%m-%d %H:%M:%S') bashrc autostart ====="

  if [ -x "$ROOT/manage_all_mvp_apis.sh" ]; then
    MVP_PRINT_START_LOG_TAIL=0 bash "$ROOT/manage_all_mvp_apis.sh" start || true
  elif [ -x "$ROOT/scripts/manage_all_mvp_apis.sh" ]; then
    MVP_PRINT_START_LOG_TAIL=0 bash "$ROOT/scripts/manage_all_mvp_apis.sh" start || true
  else
    echo "[warn] manage_all_mvp_apis.sh not found"
  fi
} >> "$LOG_DIR/bashrc_autostart.log" 2>&1 &

return 0 2>/dev/null || exit 0
