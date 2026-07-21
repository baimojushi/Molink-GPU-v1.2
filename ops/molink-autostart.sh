#!/bin/bash
# Molink GPU services autostart — direct process launch (no supervisord)
# Install: cp /opt/molink/hanging/ops/molink-autostart.sh /etc/init.d/molink-autostart && chmod +x /etc/init.d/molink-autostart && update-rc.d molink-autostart defaults
# Or via /etc/rc.local: bash /opt/molink/hanging/ops/molink-autostart.sh start

PIDFILE="/tmp/molink-autostart.pid"

cleanup() {
  echo "[molink] stopping all services..."
  for label in metric3d semantic hanging worker; do
    [ -f "/tmp/molink-${label}.pid" ] && kill "$(cat /tmp/molink-${label}.pid)" 2>/dev/null && rm -f "/tmp/molink-${label}.pid"
  done
  rm -f "$PIDFILE"
}

start() {
  echo "[molink] starting services..."
  mkdir -p /var/log/molink /tmp/molink-jobs

  # metric3d
  (
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate metric3d
    cd /opt/molink/hanging/services/metric3d_api
    export PYTHONPATH="/opt/molink/hanging"
    exec python -m uvicorn app:app --host 0.0.0.0 --port 6006 >> /var/log/molink/metric3d.log 2>&1
  ) &
  echo $! > /tmp/molink-metric3d.pid

  # semantic
  (
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate metric3d
    cd /opt/molink/hanging
    [ -f /opt/metric3d_api/.env.vlm_sam ] && source /opt/metric3d_api/.env.vlm_sam
    [ -f .env ] && source .env
    export ALLOW_UNAUTH_LOCAL="${ALLOW_UNAUTH_LOCAL:-1}"
    [ -z "${API_KEY:-}" ] && [ -n "${VLM_API_KEY:-}" ] && export API_KEY="$VLM_API_KEY"
    export OMP_NUM_THREADS="4"
    export SEMANTIC_MVP_PORT="${SEMANTIC_MVP_PORT:-6011}"
    export SAM_ENABLE="${SAM_ENABLE:-1}"
    export SAM_BACKEND="${SAM_BACKEND:-sam1}"
    export SAM_MODEL_TYPE="${SAM_MODEL_TYPE:-vit_b}"
    export SAM_CHECKPOINT="${SAM_CHECKPOINT:-/opt/molink-cache/sam/sam_vit_b_01ec64.pth}"
    export SAM_MAX_IMAGE_SIDE="${SAM_MAX_IMAGE_SIDE:-1536}"
    exec python -m uvicorn semantic_mvp.service:app --host 0.0.0.0 --port "$SEMANTIC_MVP_PORT" >> /var/log/molink/semantic.log 2>&1
  ) &
  echo $! > /tmp/molink-semantic.pid

  # hanging
  (
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate metric3d
    cd /opt/molink/hanging
    [ -f .env.bfl ] && source .env.bfl
    [ -f .env ] && source .env
    export ALLOW_UNAUTH_LOCAL="${ALLOW_UNAUTH_LOCAL:-1}"
    [ -z "${API_KEY:-}" ] && [ -n "${VLM_API_KEY:-}" ] && export API_KEY="$VLM_API_KEY"
    export HANGING_MVP_PORT="${HANGING_MVP_PORT:-6010}"
    exec python -m uvicorn hanging_mvp.service:app --host 0.0.0.0 --port "$HANGING_MVP_PORT" >> /var/log/molink/hanging.log 2>&1
  ) &
  echo $! > /tmp/molink-hanging.pid

  # worker
  (
    source /root/miniconda3/etc/profile.d/conda.sh && conda activate metric3d
    cd /opt/molink/hanging
    set -a
    [ -f /etc/profile.d/molink-gpu.sh ] && source /etc/profile.d/molink-gpu.sh
    [ -f /etc/molink/gpu.env ] && source /etc/molink/gpu.env
    set +a
    export AUTODL_INSTANCE_ID="${MANUAL_INSTANCE_ID:-${AUTODL_INSTANCE_ID:-pro-7818bc0d0a9d}}"
    export INSTANCE_ID="$AUTODL_INSTANCE_ID"
    export MOLINK_WS_URL="${MOLINK_WS_URL:-wss://www.molink.art/api/gpu/worker-connect}"
    export GPU_MODEL="${GPU_MODEL:-TITAN Xp}"
    export GPU_COUNT="${GPU_COUNT:-1}"
    export PYTHONUNBUFFERED=1
    export IDLE_TIMEOUT_S="${IDLE_TIMEOUT_S:-0}"
    exec python -u -m worker.main >> /var/log/molink/worker.log 2>&1
  ) &
  echo $! > /tmp/molink-worker.pid

  echo $! > "$PIDFILE"

  # wait for health
  sleep 5
  echo "[molink] checking services..."
  for port in 6006 6011 6010; do
    for i in $(seq 1 30); do
      if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
        echo "[molink] port $port ok"
        break
      fi
      sleep 2
    done
  done
}

case "$1" in
  start)
    start
    ;;
  stop)
    cleanup
    ;;
  restart)
    cleanup
    sleep 2
    start
    ;;
  status)
    for label in metric3d semantic hanging worker; do
      pidfile="/tmp/molink-${label}.pid"
      if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
          echo "$label: running (pid $pid)"
        else
          echo "$label: dead (stale pidfile)"
        fi
      else
        echo "$label: not started"
      fi
    done
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status}"
    exit 1
    ;;
esac
