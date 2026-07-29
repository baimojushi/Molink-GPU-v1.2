#!/usr/bin/env bash
set -euo pipefail
tmux new-session -d -s molink-worker "bash /opt/molink/app/worker/start_worker.sh" 2>&1 || tmux new-session -d -s molink-worker "bash /opt/molink/app/worker/start_worker.sh"
sleep 3
tmux capture-pane -t molink-worker -p 2>/dev/null | tail -10 | head -10
echo "---"
echo "Worker tmux session 'molink-worker' started."
echo "To view logs: tmux capture-pane -t molink-worker -p"
echo "To kill: tmux kill-session -t molink-worker"
