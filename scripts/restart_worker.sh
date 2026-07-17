#!/usr/bin/env bash
# Restart worker process — safe for production
set -euo pipefail

echo "[restart_worker] stopping old worker..."
pkill -f 'worker.main|worker/start_worker.sh' || true
sleep 2

echo "[restart_worker] clearing worker.log..."
: > /opt/molink/hanging/logs/worker.log

echo "[restart_worker] starting new worker..."
cd /opt/molink/hanging
bash /opt/molink/hanging/worker/start_worker.sh &

sleep 3
WORKER_PID=$(pgrep -f 'worker.main' | head -1)
if [ -n "$WORKER_PID" ]; then
    echo "[restart_worker] worker started (PID $WORKER_PID)"
else
    echo "[restart_worker] WARNING: worker PID not found"
    exit 1
fi
