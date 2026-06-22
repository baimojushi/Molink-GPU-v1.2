#!/usr/bin/env bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh || source ~/miniconda3/etc/profile.d/conda.sh
conda activate metric3d
cd "$ROOT"
export MODEL_NAME=${MODEL_NAME:-metric3d_vit_giant2}
export METRIC3D_REPO=${METRIC3D_REPO:-/opt/Metric3D}
export TORCH_HUB_DIR=/opt/molink-cache/torch/hub
export MAX_SIDE=${MAX_SIDE:-1536}
export FOV_CANDIDATES=${FOV_CANDIDATES:-35,45,55,60,70,80,90}
export PLANE_RESIDUAL_THRESH_M=${PLANE_RESIDUAL_THRESH_M:-0.035}
python -m uvicorn app:app --host 0.0.0.0 --port 6006
