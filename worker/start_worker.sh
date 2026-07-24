#!/usr/bin/env bash
set -euo pipefail

APP="${MOLINK_APP:-/opt/molink/hanging}"
cd "$APP"

# 关键：set -a 会让 source 进来的变量自动 export 给 python 子进程
set -a
[ -f /etc/profile.d/molink-gpu.sh ] && source /etc/profile.d/molink-gpu.sh
[ -f /etc/molink/gpu.env ] && source /etc/molink/gpu.env
set +a

# 强制使用 Pro ID 复现
export AUTODL_INSTANCE_ID="${MANUAL_INSTANCE_ID:-${AUTODL_INSTANCE_ID:-pro-7818bc0d0a9d}}"
export INSTANCE_ID="$AUTODL_INSTANCE_ID"
export MOLINK_WS_URL="${MOLINK_WS_URL:-wss://www.molink.art/api/gpu/worker-connect}"
export GPU_MODEL="${GPU_MODEL:-TITAN Xp}"
export GPU_COUNT="${GPU_COUNT:-1}"
export PYTHONUNBUFFERED=1
# Disable idle auto-shutdown for dev environment (set to 0 = never shutdown)
export IDLE_TIMEOUT_S="${IDLE_TIMEOUT_S:-0}"
export MOLINK_SKIP_ARTWORK_PROTECTION="${MOLINK_SKIP_ARTWORK_PROTECTION:-1}"

mkdir -p "$APP/logs"

python - <<'PY'
import os, hashlib

def fp(v):
    v = str(v or "")
    return hashlib.sha256(v.encode()).hexdigest()[:12] if v else "(empty)"

print("[start_worker.sh] env check "
      f"instance_id={os.getenv('AUTODL_INSTANCE_ID')} "
      f"ws_url={os.getenv('MOLINK_WS_URL')} "
      f"secret_len={len(os.getenv('WORKER_SECRET') or '')} "
      f"secret_fp={fp(os.getenv('WORKER_SECRET'))}",
      flush=True)

if not os.getenv("WORKER_SECRET"):
    raise SystemExit("[start_worker.sh] WORKER_SECRET is empty before launching worker.main")
PY

exec python -u -m worker.main 2>&1 | tee -a "$APP/logs/worker.log"
