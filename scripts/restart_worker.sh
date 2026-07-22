#!/usr/bin/env bash
# Comprehensive all-GPU-service restart: hanging (6010), semantic (6011), worker
# Usage: bash scripts/restart_worker.sh [--no-wait] [service...]
#   --no-wait    skip health-check polling
#   service(s)   one or more of: hanging semantic worker (default: all three)
set -euo pipefail

APP="/opt/molink/hanging"
ENV_BFL="$APP/.env.bfl"
ENV_LOCAL="$APP/.env"
ENV_VLM_SAM="/opt/metric3d_api/.env.vlm_sam"
ENV_GPU="/etc/molink/gpu.env"
ENV_PROFILE="/etc/profile.d/molink-gpu.sh"
CONDA_INIT="/root/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="metric3d"
HANGING_PORT="${HANGING_MVP_PORT:-6010}"
SEMANTIC_PORT="${SEMANTIC_MVP_PORT:-6011}"
LOG_DIR="$APP/logs"

NO_WAIT=false
SERVICES=()
for arg in "$@"; do
    case "$arg" in
        --no-wait) NO_WAIT=true ;;
        hanging|semantic|worker) SERVICES+=("$arg") ;;
        *) echo "Unknown option/arg: $arg"; exit 1 ;;
    esac
done
[ ${#SERVICES[@]} -eq 0 ] && SERVICES=(hanging semantic worker)

_conda() {
    # shellcheck disable=SC1090
    source "$CONDA_INIT" && conda activate "$CONDA_ENV"
}

_wait_health() {
    local port="$1" label="$2"
    local max_attempts=30
    echo "[$label] waiting for health on port $port..."
    for i in $(seq 1 "$max_attempts"); do
        if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
            echo "[$label] health ok"
            return 0
        fi
        sleep 2
    done
    echo "[$label] WARNING: health check timeout after ${max_attempts}s"
    return 1
}

_stop_service() {
    local pattern="$1" label="$2"
    echo "[$label] stopping..."
    pkill -f "$pattern" 2>/dev/null || true
    sleep 1
}

_start_hanging() {
    echo "[hanging] starting..."
    mkdir -p "$LOG_DIR"
    (
        _conda
        cd "$APP"
        [ -f "$ENV_BFL" ] && source "$ENV_BFL"
        [ -f "$ENV_LOCAL" ] && source "$ENV_LOCAL"
        export ALLOW_UNAUTH_LOCAL="${ALLOW_UNAUTH_LOCAL:-1}"
        [ -z "${API_KEY:-}" ] && [ -n "${VLM_API_KEY:-}" ] && export API_KEY="$VLM_API_KEY"
        export HANGING_MVP_PORT="$HANGING_PORT"
        exec python -m uvicorn hanging_mvp.service:app \
            --host 0.0.0.0 --port "$HANGING_PORT" \
            >> "$LOG_DIR/hanging.log" 2>&1
    ) &
    HANGING_PID=$!
    echo "[hanging] PID $HANGING_PID"
    echo "$HANGING_PID" > /tmp/molink-hanging.pid
}

_start_semantic() {
    echo "[semantic] starting..."
    mkdir -p "$LOG_DIR"
    (
        _conda
        cd "$APP"
        [ -f "$ENV_VLM_SAM" ] && source "$ENV_VLM_SAM"
        [ -f "$ENV_LOCAL" ] && source "$ENV_LOCAL"
        export ALLOW_UNAUTH_LOCAL="${ALLOW_UNAUTH_LOCAL:-1}"
        [ -z "${API_KEY:-}" ] && [ -n "${VLM_API_KEY:-}" ] && export API_KEY="$VLM_API_KEY"
        export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
        export SEMANTIC_MVP_PORT="$SEMANTIC_PORT"
        export SAM_ENABLE="${SAM_ENABLE:-1}"
        export SAM_BACKEND="${SAM_BACKEND:-sam1}"
        export SAM_MODEL_TYPE="${SAM_MODEL_TYPE:-vit_b}"
        export SAM_CHECKPOINT="${SAM_CHECKPOINT:-/opt/molink-cache/sam/sam_vit_b_01ec64.pth}"
        export SAM_MAX_IMAGE_SIDE="${SAM_MAX_IMAGE_SIDE:-1536}"
        exec python -m uvicorn semantic_mvp.service:app \
            --host 0.0.0.0 --port "$SEMANTIC_PORT" \
            >> "$LOG_DIR/semantic.log" 2>&1
    ) &
    SEMANTIC_PID=$!
    echo "[semantic] PID $SEMANTIC_PID"
    echo "$SEMANTIC_PID" > /tmp/molink-semantic.pid
}

_start_worker() {
    echo "[worker] starting..."
    mkdir -p "$LOG_DIR"
    (
        _conda
        cd "$APP"
        set -a
        [ -f "$ENV_PROFILE" ] && source "$ENV_PROFILE"
        [ -f "$ENV_GPU" ] && source "$ENV_GPU"
        set +a
        export AUTODL_INSTANCE_ID="${MANUAL_INSTANCE_ID:-${AUTODL_INSTANCE_ID:-pro-7818bc0d0a9d}}"
        export INSTANCE_ID="$AUTODL_INSTANCE_ID"
        export MOLINK_WS_URL="${MOLINK_WS_URL:-wss://www.molink.art/api/gpu/worker-connect}"
        export GPU_MODEL="${GPU_MODEL:-TITAN Xp}"
        export GPU_COUNT="${GPU_COUNT:-1}"
        export PYTHONUNBUFFERED=1
        export IDLE_TIMEOUT_S="${IDLE_TIMEOUT_S:-0}"
        exec python -u -m worker.main >> "$LOG_DIR/worker.log" 2>&1
    ) &
    WORKER_PID=$!
    echo "[worker] PID $WORKER_PID"
    echo "$WORKER_PID" > /tmp/molink-worker.pid
}

# ── main ──────────────────────────────────────────────────────────────────

for svc in "${SERVICES[@]}"; do
    case "$svc" in
        hanging)  _stop_service 'hanging_mvp.service:app' "hanging" ;;
        semantic) _stop_service 'semantic_mvp.service:app' "semantic" ;;
        worker)   _stop_service 'worker.main|worker/start_worker.sh' "worker" ;;
    esac
done
sleep 1

NEED_HEALTH_HANGING=false
NEED_HEALTH_SEMANTIC=false

for svc in "${SERVICES[@]}"; do
    case "$svc" in
        hanging)  _start_hanging;  NEED_HEALTH_HANGING=true ;;
        semantic) _start_semantic; NEED_HEALTH_SEMANTIC=true ;;
        worker)   _start_worker ;;
    esac
done

if [ "$NO_WAIT" = false ]; then
    sleep 3
    $NEED_HEALTH_HANGING && _wait_health "$HANGING_PORT" "hanging" || true
    $NEED_HEALTH_SEMANTIC && _wait_health "$SEMANTIC_PORT" "semantic" || true
    # worker has no health endpoint; just check process
    if [[ " ${SERVICES[*]} " =~ " worker " ]]; then
        WPID=$(pgrep -f 'worker.main' | head -1) || true
        if [ -n "$WPID" ]; then
            echo "[worker] running (PID $WPID)"
        else
            echo "[worker] WARNING: worker PID not found"
        fi
    fi
fi

echo "── restart summary ──"
for svc in "${SERVICES[@]}"; do
    case "$svc" in
        hanging)  WPID=$(pgrep -f 'hanging_mvp.service:app' | head -1) || true ;;
        semantic) WPID=$(pgrep -f 'semantic_mvp.service:app' | head -1) || true ;;
        worker)   WPID=$(pgrep -f 'worker.main' | head -1) || true ;;
    esac
    [ -n "${WPID:-}" ] && echo "  $svc: running (PID $WPID)" || echo "  $svc: NOT RUNNING"
done
