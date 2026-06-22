#!/usr/bin/env bash
set -Eeuo pipefail

mkdir -p /var/log/molink /tmp/molink-jobs
exec >> /var/log/molink/boot.log 2>&1

echo "========== MoLink AutoDL Boot =========="
date
hostname
whoami

source /etc/profile.d/molink-gpu.sh
[ -f /etc/molink/gpu.env ] && set -a && source /etc/molink/gpu.env && set +a

mkdir -p "$MOLINK_CACHE" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME"
mkdir -p /var/log/molink /tmp/molink-jobs

cd "$MOLINK_APP"

echo "[boot] disk usage before:"
df -h
du -hxd1 /opt 2>/dev/null || true

if [ ! -f "$MOLINK_CACHE/.prepared" ]; then
  echo "[boot] first prepare: download/cache models"
  bash ops/download_models.sh

  echo "[boot] optional warmup"
  timeout 600 bash scripts/manage_all_mvp_apis.sh restart || true
  sleep 20

  curl -fsS http://127.0.0.1:6006/health || true
  curl -fsS http://127.0.0.1:6011/health || true
  curl -fsS http://127.0.0.1:6010/health || true

  bash scripts/manage_all_mvp_apis.sh stop || true

  date > "$MOLINK_CACHE/.prepared"
  echo "[boot] prepare finished"
fi

echo "[boot] start production"
exec bash ops/start_prod.sh
