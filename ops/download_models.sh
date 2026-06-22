#!/usr/bin/env bash
set -Eeuo pipefail

source /etc/profile.d/molink-gpu.sh
[ -f /etc/molink/gpu.env ] && set -a && source /etc/molink/gpu.env && set +a

mkdir -p "$METRIC3D_MODEL_DIR" /opt/molink-cache/sam "$TORCH_HOME/hub/checkpoints"

echo "[models] cache root: $MOLINK_CACHE"

METRIC3D_GIANT="$METRIC3D_MODEL_DIR/metric_depth_vit_giant2_800k.pth"
if [ ! -s "$METRIC3D_GIANT" ]; then
  echo "[models] downloading Metric3D giant checkpoint..."
  wget -c -O "$METRIC3D_GIANT" "https://hf-mirror.com/JUGGHM/Metric3D/resolve/main/metric_depth_vit_giant2_800k.pth" || \
  wget -c -O "$METRIC3D_GIANT" "https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_giant2_800k.pth"
fi

ln -sfn "$METRIC3D_GIANT" "$TORCH_HOME/hub/checkpoints/metric_depth_vit_giant2_800k.pth"

SAM_CKPT="${SAM_CHECKPOINT:-/opt/molink-cache/sam/sam_vit_b_01ec64.pth}"
if [ "${SAM_ENABLE:-1}" = "1" ] && [ ! -s "$SAM_CKPT" ]; then
  echo "[models] SAM checkpoint missing: $SAM_CKPT"
  echo "[models] If automatic URL is unavailable, place sam_vit_b_01ec64.pth here manually."
fi

echo "[models] current model files:"
find /opt/molink-cache -maxdepth 3 -type f -size +10M -printf "%p %s bytes\n" | sort || true

echo "[models] done"
