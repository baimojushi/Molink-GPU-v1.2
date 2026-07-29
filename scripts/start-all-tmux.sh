#!/usr/bin/env bash
# MoLink GPU — unified tmux service launcher
# Starts all 4 services in persistent tmux sessions.
# Usage: bash start-all-tmux.sh [restart|status]
set -euo pipefail

ACTION="${1:-start}"
ROOT="${ROOT:-/opt/molink/app}"

# Ensure profile.d for conda activation
export BASH_ENV="$HOME/miniconda3/etc/profile.d/conda.sh"
[ -f /opt/miniconda3/etc/profile.d/conda.sh ] && export BASH_ENV=/opt/miniconda3/etc/profile.d/conda.sh

_kill_tmux() {
  tmux kill-session -t molink-metric3d 2>/dev/null || true
  tmux kill-session -t molink-semantic 2>/dev/null || true
  tmux kill-session -t molink-hanging 2>/dev/null || true
  tmux kill-session -t molink-worker 2>/dev/null || true
}

_start_metric3d() {
  tmux new-session -d -s molink-metric3d \
    "cd $ROOT/scripts && bash $ROOT/start_giant_auto.sh 2>&1"
}

_start_semantic() {
  tmux new-session -d -s molink-semantic \
    "cd $ROOT/scripts && bash $ROOT/scripts/start_semantic_v1_sam.sh 2>&1"
}

_start_hanging() {
  tmux new-session -d -s molink-hanging \
    "cd $ROOT/scripts && bash $ROOT/scripts/start_hanging_v1.sh 2>&1"
}

_start_worker() {
  tmux new-session -d -s molink-worker \
    "cd $ROOT && bash $ROOT/worker/start_worker.sh 2>&1"
}

_check() {
  local name="$1" port="$2"
  local pid=""
  pid=$(tmux list-panes -t "molink-$name" -F '#{pane_pid}' 2>/dev/null | head -1 || true)
  if [ -z "$pid" ]; then
    pid=$(pgrep -f "uvicorn.*$port" | head -1)
  fi
  if [ -n "$pid" ] && curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    echo "  [UP] $name:$port pid=$pid"
  else
    echo "  [DOWN] $name:$port"
  fi
}

case "$ACTION" in
  restart)
    echo "[stop] killing tmux sessions..."
    _kill_tmux
    sleep 2
    ;&
  start)
    echo "[start] Metric3D (cold start ~90s)..."
    _start_metric3d
    echo "[start] Semantic..."
    _start_semantic
    echo "[start] Hanging..."
    _start_hanging
    sleep 5
    echo "[start] Worker..."
    _start_worker
    echo
    echo "=== status ==="
    _check metric3d 6006
    _check semantic 6011
    _check hanging 6010
    sleep 3
    worker_pid=$(pgrep -f 'worker.main' | head -1 || true)
    if [ -n "$worker_pid" ]; then
      echo "  [UP] worker pid=$worker_pid"
    else
      echo "  [DOWN] worker"
    fi
    ;;
  status)
    echo "=== MoLink tmux sessions ==="
    _check metric3d 6006
    _check semantic 6011
    _check hanging 6010
    worker_pid=$(pgrep -f 'worker.main' | head -1 || true)
    if [ -n "$worker_pid" ]; then
      conn="no WS"
      ss -tnp 2>/dev/null | grep -q "$worker_pid" 2>/dev/null && conn="ESTAB" || true
      echo "  [UP] worker pid=$worker_pid $conn"
    else
      echo "  [DOWN] worker"
    fi
    ;;
  stop)
    echo "[stop] killing all MoLink tmux sessions..."
    _kill_tmux
    pkill -f 'uvicorn.*:6006\|uvicorn.*:6011\|uvicorn.*:6010\|worker.main' 2>/dev/null || true
    echo "  done"
    ;;
  *)
    echo "Usage: $0 {start|restart|status|stop}"
    exit 1
    ;;
esac
