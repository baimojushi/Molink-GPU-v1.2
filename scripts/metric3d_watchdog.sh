#!/usr/bin/env bash
# Metric3D 探活看门狗 — 每分钟由 crontab 调用
# 检查 :6006/health，失联时重启
set -euo pipefail

HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:6006/health}"
LOG_FILE="${LOG_FILE:-/opt/molink/hanging/logs/metric3d_watchdog.log}"
RESTART_SCRIPT="${RESTART_SCRIPT:-/opt/molink/hanging/start_giant_auto.sh}"

# 健康检查
if curl -sf --max-time 5 "$HEALTH_URL" >/dev/null 2>&1; then
    # 每分钟写入心跳，证明 crontab 生效
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ok" >> "$LOG_FILE"
    exit 0
fi

# 探活失败 — 写日志、重启
echo "[$(date '+%Y-%m-%d %H:%M:%S')] metric3d 失联，执行重启..." >> "$LOG_FILE"
nohup bash "$RESTART_SCRIPT" >> "$LOG_FILE" 2>&1 &
echo "[$(date '+%Y-%m-%d %H:%M:%S')] metric3d 重启命令已发出 (PID $!)" >> "$LOG_FILE"
