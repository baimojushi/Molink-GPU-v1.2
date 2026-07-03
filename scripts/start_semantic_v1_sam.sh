#!/usr/bin/env bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-metric3d}"
cd "${ROOT:-/opt/molink/hanging}"
[ -f /opt/metric3d_api/.env.vlm_sam ] && source /opt/metric3d_api/.env.vlm_sam
# Local dev: allow unauthenticated requests and set API_KEY from VLM_API_KEY
[ -f .env ] && source .env
export ALLOW_UNAUTH_LOCAL="${ALLOW_UNAUTH_LOCAL:-1}"
if [ -z "${API_KEY:-}" ] && [ -n "${VLM_API_KEY:-}" ]; then
    export API_KEY="$VLM_API_KEY"
fi
# Fix: OMP_NUM_THREADS=0 causes OpenMP/PyTorch undefined behavior in SAM inference
export OMP_NUM_THREADS="4"
export SEMANTIC_MVP_PORT="${SEMANTIC_MVP_PORT:-6011}"
export SAM_ENABLE="${SAM_ENABLE:-1}"
export SAM_BACKEND="${SAM_BACKEND:-sam1}"
export SAM_MODEL_TYPE="${SAM_MODEL_TYPE:-vit_b}"
export SAM_CHECKPOINT="${SAM_CHECKPOINT:-/opt/molink-cache/sam/sam_vit_b_01ec64.pth}"
export SAM_MAX_IMAGE_SIDE="${SAM_MAX_IMAGE_SIDE:-1536}"
python -m uvicorn semantic_mvp.service:app --host 0.0.0.0 --port "$SEMANTIC_MVP_PORT"
