#!/bin/bash
set -e
# MoLink tmux startup
set -a; source /etc/molink/gpu.env; set +a
APP_DIR=/opt/molink/app
SESSION=molink-gpu
CONDA_BIN=/opt/miniconda3/envs/metric3d/bin

tmux kill-session -t $SESSION 2>/dev/null || true
tmux new-session -d -s $SESSION -n metric3d
tmux new-window -t $SESSION -n semantic
tmux new-window -t $SESSION -n hanging
tmux new-window -t $SESSION -n worker
tmux new-window -t $SESSION -n monitor
# Metric3D on GPU-A
tmux send-keys -t $SESSION:metric3d \x27export CUDA_VISIBLE_DEVICES=$METRIC3D_GPU_UUID\x27 Enter
tmux send-keys -t $SESSION:metric3d \x27cd $APP_DIR test
abcd efg hij
tmux send-keys one two three
tmux send-keys -t SESSION:metric3d 