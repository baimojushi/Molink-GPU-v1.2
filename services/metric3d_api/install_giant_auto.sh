#!/usr/bin/env bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh || source ~/miniconda3/etc/profile.d/conda.sh
conda activate metric3d
cd /opt/Metric3D
grep -n "metric3d_vit_giant2" /opt/Metric3D/hubconf.py || { git -C /opt/Metric3D pull || true; grep -n "metric3d_vit_giant2" /opt/Metric3D/hubconf.py; }
python -m pip install -U fastapi "uvicorn[standard]" python-multipart pillow opencv-python-headless numpy requests scipy exifread piexif
rm -f /opt/molink-cache/torch/hub/checkpoints/metric_depth_vit_small_800k.pth
mkdir -p /opt/molink-cache/torch/hub/checkpoints
cd /opt/molink-cache/torch/hub/checkpoints
if [ ! -f metric_depth_vit_giant2_800k.pth ]; then
  wget -c -O metric_depth_vit_giant2_800k.pth https://hf-mirror.com/JUGGHM/Metric3D/resolve/main/metric_depth_vit_giant2_800k.pth || \
  wget -c -O metric_depth_vit_giant2_800k.pth https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_giant2_800k.pth
fi
ls -lh /opt/molink-cache/torch/hub/checkpoints/metric_depth_vit_giant2_800k.pth
cat > /opt/molink/hanging/warmup_giant.py <<'PY'
import os, torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
model=torch.hub.load(os.getenv("METRIC3D_REPO","/opt/Metric3D"),os.getenv("MODEL_NAME","metric3d_vit_giant2"),pretrain=True,source="local")
model.cuda().eval()
print("metric3d_vit_giant2 loaded on", next(model.parameters()).device)
PY
export MODEL_NAME=metric3d_vit_giant2
python /opt/molink/hanging/warmup_giant.py
