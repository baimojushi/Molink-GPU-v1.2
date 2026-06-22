#!/usr/bin/env bash
set -Eeuo pipefail

source /etc/profile.d/molink-gpu.sh
[ -f /etc/molink/gpu.env ] && set -a && source /etc/molink/gpu.env && set +a

cd "$MOLINK_APP"

echo "========== disk =========="
df -h
du -hxd1 /opt 2>/dev/null || true
du -hxd1 /opt/molink-cache 2>/dev/null || true

echo "========== python =========="
which python || true
python -V || true
python - <<'PY'
import os, sys
print("python", sys.version)
for k in ["MOLINK_APP","MOLINK_CACHE","HF_HOME","TORCH_HOME","METRIC3D_CKPT_PATH","SAM_CHECKPOINT"]:
    print(k, os.getenv(k))
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch import failed:", repr(e))
PY

echo "========== source hardcoded path audit =========="
grep -RInE "/root/autodl-tmp|/root/.cache|/opt/models|/root/metric3d|metric_depth_.*pth|sam_vit_.*pth" "$MOLINK_APP" --include="*.py" --include="*.sh" --include="*.env*" --exclude-dir=.git 2>/dev/null | grep -v ".qwen/hooks" || true

echo "========== secret audit =========="
echo "Do not print secrets. Checking only placeholder names."
grep -RInE "BFL_API_KEY=|R2_SECRET_ACCESS_KEY=|WORKER_SECRET=|AUTODL_TOKEN=" "$MOLINK_APP" --include="*.py" --include="*.sh" --include="*.env*" --exclude-dir=.git 2>/dev/null | grep -v ".qwen" || true

echo "========== model files =========="
find "$MOLINK_CACHE" -maxdepth 4 -type f -size +10M -printf "%p %s bytes\n" | sort || true

echo "========== service scripts =========="
test -x scripts/manage_all_mvp_apis.sh
test -x ops/download_models.sh
test -x ops/start_prod.sh
test -x ops/autodl_boot.sh

echo "preflight done"
