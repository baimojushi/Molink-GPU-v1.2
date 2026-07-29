#!/usr/bin/env bash
set -euo pipefail

# Kill any existing sessions
tmux kill-session -t molink-hanging 2>/dev/null || true
tmux kill-session -t molink-worker 2>/dev/null || true

# Start Hanging in tmux
cd /opt/molink/hanging
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate metric3d
tmux new-session -d -s molink-hanging "cd /opt/molink/hanging && source /opt/miniconda3/etc/profile.d/conda.sh && conda activate metric3d && python -m uvicorn hanging_mvp.service:app --host 0.0.0.0 --port 6010"

# Wait for hanging
sleep 4

# Start Worker in tmux
tmux new-session -d -s molink-worker "bash /opt/molink/app/worker/start_worker.sh"

echo "Hanging: $(curl -sf http://127.0.0.1:6010/health && echo UP || echo DOWN)"
sleep 2
tmux capture-pane -t molink-worker -p | tail -5
echo "Worker PID: $(pgrep -f 'worker.main' | head -1)"
