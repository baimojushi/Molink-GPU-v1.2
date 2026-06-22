#!/usr/bin/env bash
set -e

source /root/miniconda3/etc/profile.d/conda.sh || source ~/miniconda3/etc/profile.d/conda.sh
conda activate metric3d

cd "$ROOT"

export MODEL_NAME=${MODEL_NAME:-metric3d_vit_small}
export METRIC3D_REPO=${METRIC3D_REPO:-/opt/Metric3D}
export TORCH_HUB_DIR=/opt/molink-cache/torch/hub
export MAX_PIXELS=${MAX_PIXELS:-2359296}

uvicorn app:app --host 0.0.0.0 --port 6006
