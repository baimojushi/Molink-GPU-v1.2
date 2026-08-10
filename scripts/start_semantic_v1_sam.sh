#!/usr/bin/env bash
set -euo pipefail

# Load production environment and export VLM variables
if [ -f /etc/molink/gpu.env ]; then
    source /etc/molink/gpu.env
    export VLM_PROVIDER VLM_API_URL VLM_API_KEY VLM_MODEL
    export SPATIAL_VLM_API_URL SPATIAL_VLM_API_KEY SPATIAL_VLM_MODEL SPATIAL_VLM_TIMEOUT_S
    export API_KEY ALLOW_UNAUTH_LOCAL
    export R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_PUBLIC_BASE_URL
    export APIYI_API_KEY APIYI_API_BASE APIYI_GPT_IMAGE_MODEL APIYI_GPT_IMAGE_TIMEOUT_S
    export BFL_API_KEY HANGING_RENDER_PROVIDER
    export NO_PROXY HTTP_PROXY HTTPS_PROXY
    export OMP_NUM_THREADS
fi

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
