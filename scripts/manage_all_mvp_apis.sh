#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/opt/molink/hanging}"
ACTION="${1:-status}"
LOG_DIR="$ROOT/logs"
PID_DIR="$ROOT/run"
mkdir -p "$LOG_DIR" "$PID_DIR"

# Load production environment first, then local overrides
[ -f /etc/molink/gpu.env ] && source /etc/molink/gpu.env
[ -f "$ROOT/.env.local" ] && source "$ROOT/.env.local"

# name|port|health|fallback-start-scripts|kind
SERVICE_SPECS=(
  "metric3d|6006|/health|start_giant_auto.sh,start_giant.sh,start.sh,services/metric3d_api/start_giant_auto.sh,services/metric3d_api/start_giant.sh,services/metric3d_api/start.sh,metric3d_ctl.sh|api"
  "semantic|6011|/health|scripts/start_semantic_v1_sam.sh,start_semantic_mvp.sh|api"
  "hanging|6010|/health|scripts/start_hanging_v1.sh,start_hanging_mvp.sh|api"
)
MVP_SERVICES="${MVP_SERVICES:-metric3d,semantic,hanging}"

_service_enabled() {
  local name="$1"
  local csv=",$MVP_SERVICES,"
  [ "${csv#*,${name},}" != "$csv" ]
}

_have_systemctl() {
  command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files >/dev/null 2>&1
}

_unit_for_name() {
  case "$1" in
    metric3d) echo "metric3d-api.service" ;;
    semantic) echo "semantic-mvp.service" ;;
    hanging) echo "hanging-mvp.service" ;;
    *) echo "" ;;
  esac
}

_have_unit() {
  local unit="$1"
  [ -n "$unit" ] || return 1
  systemctl list-unit-files 2>/dev/null | awk '{print $1}' | grep -qx "$unit"
}

_port_open() {
  python3 - "$1" <<'PY'
import socket, sys
s = socket.socket()
s.settimeout(0.35)
try:
    s.connect(("127.0.0.1", int(sys.argv[1])))
    print("1")
except Exception:
    print("0")
finally:
    s.close()
PY
}

_http_health() {
  python3 - "$1" "$2" <<'PY'
import http.client, os, sys
port = int(sys.argv[1])
path = sys.argv[2]
try:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=float(os.getenv("MVP_HEALTH_TIMEOUT", "5.0")))
    conn.request("GET", path)
    resp = conn.getresponse()
    body = resp.read(180).decode("utf-8", "ignore").replace("\n", " ")
    print(f"HTTP {resp.status} {body}")
except Exception as e:
    print(f"DOWN {type(e).__name__}: {e}")
PY
}

_find_start_script() {
  local csv="$1"
  IFS=',' read -ra scripts <<< "$csv"
  for s in "${scripts[@]}"; do
    if [ -f "$ROOT/$s" ]; then
      echo "$ROOT/$s"
      return 0
    fi
  done
  return 1
}

_print_recent_log() {
  local name="$1"
  local log="$LOG_DIR/${name}.log"
  local lines="${MVP_START_LOG_LINES:-80}"
  [ -f "$log" ] || return 0
  echo "----- recent log: $log -----"
  tail -n "$lines" "$log" || true
  echo "----- end log: $log -----"
}

_start_bg_plain() {
  local name="$1"
  local port="$2"
  local script="$3"
  local log="$LOG_DIR/${name}.log"
  local pidfile="$PID_DIR/${name}.pid"

  echo "[start:$name] $script -> $log"
  (
    cd "$ROOT"
    nohup bash "$script" > "$log" 2>&1 &
    echo $! > "$pidfile"
  )

  for _ in $(seq 1 25); do
    if [ "$(_port_open "$port")" = "1" ]; then
      echo "[ok:$name] port $port is listening"
      if [ "${MVP_PRINT_START_LOG_TAIL:-0}" = "1" ]; then
        _print_recent_log "$name"
      fi
      return 0
    fi
    sleep 1
  done

  echo "[warn:$name] port $port not listening after 25s; check $log"
  _print_recent_log "$name"
}

_start_metric3d_ctl() {
  local script="$1"
  local port="6006"
  local log="$LOG_DIR/metric3d.log"
  local pidfile="$PID_DIR/metric3d.pid"

  echo "[start:metric3d] using metric3d_ctl.sh -> $log"
  (
    cd "$ROOT"
    nohup bash "$script" start > "$log" 2>&1 &
    echo $! > "$pidfile"
  )

  for _ in $(seq 1 25); do
    if [ "$(_port_open "$port")" = "1" ]; then
      echo "[ok:metric3d] port $port is listening"
      if [ "${MVP_PRINT_START_LOG_TAIL:-0}" = "1" ]; then
        _print_recent_log "metric3d"
      fi
      return 0
    fi
    sleep 1
  done

  echo "[warn:metric3d] metric3d_ctl.sh start did not open port 6006 after 25s"
  echo "                 trying: bash metric3d_ctl.sh restart"
  (
    cd "$ROOT"
    nohup bash "$script" restart >> "$log" 2>&1 &
    echo $! > "$pidfile"
  )

  for _ in $(seq 1 25); do
    if [ "$(_port_open "$port")" = "1" ]; then
      echo "[ok:metric3d] port $port is listening"
      if [ "${MVP_PRINT_START_LOG_TAIL:-0}" = "1" ]; then
        _print_recent_log "metric3d"
      fi
      return 0
    fi
    sleep 1
  done

  echo "[warn:metric3d] port 6006 still not listening; check $log"
  _print_recent_log "metric3d"
}

_daemon_pidfile_for_name() {
  echo "$PID_DIR/${1}.pid"
}

_start_daemon() {
  local name="$1"
  local script="$2"
  local log="$LOG_DIR/${name}.log"
  local pidfile
  pidfile="$(_daemon_pidfile_for_name "$name")"

  if [ -f "$pidfile" ]; then
    local pid
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      echo "[skip:$name] daemon already running pid=$pid"
      return 0
    fi
  fi

  echo "[start:$name] $script -> $log"
  (
    cd "$ROOT"
    bash "$script" main
  ) 2>&1 | tee -a "$log"

  if [ -f "$pidfile" ]; then
    echo "[ok:$name] pid=$(cat "$pidfile" 2>/dev/null || true)"
  else
    echo "[warn:$name] pidfile not found after start: $pidfile"
  fi
  if [ "${MVP_PRINT_START_LOG_TAIL:-0}" = "1" ]; then
    _print_recent_log "$name"
  fi
}

_health_ok() {
  local port="$1" health="$2"
  local resp
  resp=$(_http_health "$port" "$health")
  [[ "$resp" == HTTP\ 2* ]]
}

_force_stop_port() {
  local name="$1" port="$2"
  echo "[zombie:$name] port $port open but unhealthy; force-stopping"
  # Try pidfile first
  local pidfile="$PID_DIR/${name}.pid"
  if [ -f "$pidfile" ]; then
    local pid
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      sleep 1
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
  fi
  # Fallback: pkill by uvicorn pattern
  case "$name" in
    metric3d)  pkill -f 'uvicorn.*6006' 2>/dev/null || true ;;
    semantic)  pkill -f 'uvicorn.*6011' 2>/dev/null || true ;;
    hanging)   pkill -f 'uvicorn.*6010' 2>/dev/null || true ;;
  esac
  # Wait for port to release
  for _ in $(seq 1 10); do
    if [ "$(_port_open "$port")" != "1" ]; then
      echo "[ok:$name] port $port released"
      return 0
    fi
    sleep 1
  done
  echo "[warn:$name] port $port still in use after force-stop"
}

_start_one() {
  local spec="$1"
  IFS='|' read -r name port health scripts kind <<< "$spec"

  if [ "$kind" = "daemon" ]; then
    local daemon_script
    if daemon_script="$(_find_start_script "$scripts")"; then
      _start_daemon "$name" "$daemon_script"
    else
      echo "[warn:$name] no fallback start script found"
      echo "             expected one of: $scripts"
    fi
    return 0
  fi

  if [ "$(_port_open "$port")" = "1" ]; then
    # Port open — but verify HTTP health to catch zombie processes
    if _health_ok "$port" "$health"; then
      echo "[skip:$name] port $port healthy"
    else
      _force_stop_port "$name" "$port"
      # Fall through to restart logic below
    fi
    if [ "${MVP_PRINT_START_LOG_TAIL:-0}" = "1" ]; then
      _print_recent_log "$name"
    fi
    if [ "$(_port_open "$port")" = "1" ] && _health_ok "$port" "$health"; then
      return 0
    fi
  fi

  if _have_systemctl; then
    local unit
    unit="$(_unit_for_name "$name")"
    if _have_unit "$unit"; then
      echo "[systemd:start:$name] $unit"
      systemctl start "$unit" || true
      sleep 1
      if [ "$(_port_open "$port")" = "1" ]; then
        echo "[ok:$name] port $port is listening"
        return 0
      fi
    fi
  fi

  local script
  if script="$(_find_start_script "$scripts")"; then
    if [ "$name" = "metric3d" ] && [ "$(basename "$script")" = "metric3d_ctl.sh" ]; then
      _start_metric3d_ctl "$script"
    else
      _start_bg_plain "$name" "$port" "$script"
    fi
  else
    echo "[warn:$name] no fallback start script found for port $port"
    echo "             expected one of: $scripts"
  fi
}

_stop_one() {
  local spec="$1"
  IFS='|' read -r name port health scripts kind <<< "$spec"
  local pidfile="$PID_DIR/${name}.pid"
  if [ "$kind" = "daemon" ]; then
    pidfile="$(_daemon_pidfile_for_name "$name")"
  fi

  if _have_systemctl; then
    local unit
    unit="$(_unit_for_name "$name")"
    if _have_unit "$unit"; then
      echo "[systemd:stop:$name] $unit"
      systemctl stop "$unit" || true
    fi
  fi

  if [ "$name" = "metric3d" ] && [ -f "$ROOT/metric3d_ctl.sh" ]; then
    echo "[stop:metric3d] bash metric3d_ctl.sh stop"
    (cd "$ROOT" && bash "$ROOT/metric3d_ctl.sh" stop >> "$LOG_DIR/metric3d.log" 2>&1) || true
  fi

  if [ -f "$pidfile" ]; then
    local pid
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      echo "[stop:$name] pid $pid"
      kill "$pid" 2>/dev/null || true
      sleep 1
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
  fi

  case "$name" in
    semantic)
      pkill -f 'uvicorn semantic_mvp.service:app' || true
      pkill -f 'semantic_mvp.service:app' || true
      ;;
    hanging)
      pkill -f 'uvicorn hanging_mvp.service:app' || true
      pkill -f 'hanging_mvp.service:app' || true
      ;;
    metric3d)
      pkill -f 'uvicorn.*metric3d' || true
      pkill -f 'metric3d.*uvicorn' || true
      pkill -f 'metric3d_api.*app' || true
      ;;
  esac
}

_status_one() {
  local spec="$1"
  IFS='|' read -r name port health scripts kind <<< "$spec"
  if [ "$kind" = "daemon" ]; then
    local pidfile pid
    pidfile="$(_daemon_pidfile_for_name "$name")"
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      echo "[UP]   $name daemon pid=$pid"
    else
      echo "[DOWN] $name daemon"
    fi
    [ -f "$LOG_DIR/${name}.log" ] && echo "       log: $LOG_DIR/${name}.log"
    return 0
  fi

  if [ "$(_port_open "$port")" = "1" ]; then
    echo "[UP]   $name port=$port health=$(_http_health "$port" "$health")"
  else
    echo "[DOWN] $name port=$port health=$(_http_health "$port" "$health")"
  fi
  [ -f "$LOG_DIR/${name}.log" ] && echo "       log: $LOG_DIR/${name}.log"
  [ -f "$PID_DIR/${name}.pid" ] && echo "       pidfile: $(cat "$PID_DIR/${name}.pid" 2>/dev/null || true)"
}

do_start() {
  # 确保 cron 守护进程运行（Metric3D 看门狗依赖）
  if ! pgrep -x cron >/dev/null 2>&1; then
    if command -v cron >/dev/null 2>&1; then
      /usr/sbin/cron
      echo "[start:cron] cron daemon started"
    fi
  else
    echo "[skip:cron] already running"
  fi
  for spec in "${SERVICE_SPECS[@]}"; do
    IFS='|' read -r name _port _health _scripts _kind <<< "$spec"
    if _service_enabled "$name"; then
      _start_one "$spec"
    else
      echo "[skip:$name] disabled by MVP_SERVICES=$MVP_SERVICES"
    fi
  done
  # 反隧道 worker：三段服务启动后拉起（默认开启，置 START_WORKER=0 关闭）
  if [ "${START_WORKER:-1}" = "1" ]; then
    local repo_root worker_log
    repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    worker_log="$LOG_DIR/worker.log"
    if pgrep -f 'worker.main' >/dev/null 2>&1; then
      echo "[start:worker] already running"
    else
      echo "[start:worker] python3 -m worker.main -> $worker_log"
      ( cd "$repo_root" && nohup bash worker/start_worker.sh > "$worker_log" 2>&1 & )
    fi
  else
    echo "[skip:worker] disabled by START_WORKER=0"
  fi
}

do_stop() {
  if [ "${START_WORKER:-1}" = "1" ]; then
    if pgrep -f 'worker.main' >/dev/null 2>&1; then
      echo "[stop:worker] terminating worker.main"
      pkill -f 'worker.main' 2>/dev/null || true
    fi
  fi
  for (( idx=${#SERVICE_SPECS[@]}-1 ; idx>=0 ; idx-- )); do
    IFS='|' read -r name _port _health _scripts _kind <<< "${SERVICE_SPECS[$idx]}"
    if _service_enabled "$name"; then
      _stop_one "${SERVICE_SPECS[$idx]}"
    else
      echo "[skip:$name] disabled by MVP_SERVICES=$MVP_SERVICES"
    fi
  done
}

do_restart() {
  do_stop
  sleep 1
  do_start
}

do_status() {
  echo "== MVP API status =="
  echo "service filter: $MVP_SERVICES"
  for spec in "${SERVICE_SPECS[@]}"; do
    IFS='|' read -r name _port _health _scripts _kind <<< "$spec"
    if _service_enabled "$name"; then
      _status_one "$spec"
    else
      echo "[SKIP] $name disabled by MVP_SERVICES"
    fi
  done
  echo
  echo "== Matching processes =="
  ps -ef | grep -E 'metric3d|semantic_mvp\.service:app|hanging_mvp\.service:app' | grep -v grep || true
}

do_logs() {
  local lines="${MVP_LOG_LINES:-120}"
  touch "$LOG_DIR/metric3d.log" "$LOG_DIR/semantic.log" "$LOG_DIR/hanging.log" "$LOG_DIR/worker.log"
  echo "[logs] tail -F four-service logs (metric3d/semantic/hanging/worker). Press Ctrl-C to stop watching; services keep running."
  tail -n "$lines" -F \
    "$LOG_DIR/metric3d.log" \
    "$LOG_DIR/semantic.log" \
    "$LOG_DIR/hanging.log" \
    "$LOG_DIR/worker.log"
}

case "$ACTION" in
  start) do_start ;;
  stop|suspend|hangup) do_stop ;;
  restart|reload) do_restart ;;
  status) do_status ;;
  logs|tail) do_logs ;;
  *)
    echo "Usage: $0 {start|stop|suspend|restart|status|logs}"
    exit 1
    ;;
esac
