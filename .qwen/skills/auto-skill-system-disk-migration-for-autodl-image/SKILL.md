---
name: system-disk-migration-for-autodl-image
description: Migrate GPU service code from data disk to system disk with model cache separation, for creating a 30GB AutoDL private image
source: auto-skill
extracted_at: '2026-06-19T02:58:00.000Z'
---

# System Disk Migration for AutoDL Private Image

When preparing a 30GB system-disk image on AutoDL, large model weights must NOT be included. They are downloaded on first boot of the Pro instance.

## Fixed directory structure

```bash
MOLINK_ROOT=/opt/molink/hanging        # source code
MOLINK_CACHE=/opt/molink-cache         # model cache (excluded from image)
MOLINK_LOG=/var/log/molink             # production logs
MOLINK_ETC=/etc/molink                 # secrets (600 root:root)
MOLINK_JOBS=/tmp/molink-jobs           # job scratch space
```

## Step 1 — Rsync source code

Find source directory (check `/root/metric3d_api`, `/opt/metric3d_api`, `/root/autodl-tmp/metric3d_api`), then:

```bash
rsync -a --delete \
  --exclude ".git" --exclude "__pycache__" --exclude "*.pyc" \
  --exclude "logs" --exclude "run" --exclude "outputs" \
  --exclude "artifacts" --exclude ".cache" --exclude "*.zip" \
  --exclude ".pytest_cache" \
  "$SOURCE_DIR"/ /opt/molink/hanging/

# Replace original with compatibility symlink
rm -rf /opt/metric3d_api    # must stop services first
ln -sfn /opt/molink/hanging /opt/metric3d_api
```

## Step 2 — Create `/etc/profile.d/molink-gpu.sh`

Must set all cache paths to `/opt/molink-cache`:

```bash
export MOLINK_HOME=/opt/molink
export MOLINK_APP=/opt/molink/hanging
export MOLINK_CACHE=/opt/molink-cache
export HF_HOME=/opt/molink-cache/hf
export TRANSFORMERS_CACHE=/opt/molink-cache/hf/transformers
export HUGGINGFACE_HUB_CACHE=/opt/molink-cache/hf/hub
export TORCH_HOME=/opt/molink-cache/torch
export TORCH_HUB_DIR=/opt/molink-cache/torch/hub
export XDG_CACHE_HOME=/opt/molink-cache/xdg
export METRIC3D_MODEL_DIR=/opt/molink-cache/metric3d
export METRIC3D_CKPT_PATH=/opt/molink-cache/metric3d/metric_depth_vit_giant2_800k.pth
export SAM_CHECKPOINT=/opt/molink-cache/sam/sam_vit_b_01ec64.pth
export ROOT=/opt/molink/hanging
export PYTHONPATH=/opt/molink/hanging:${PYTHONPATH:-}
export PATH="/root/miniconda3/envs/metric3d/bin:${PATH}"
```

The conda Python path line is critical — without it `python` will not be found.

## Step 3 — Create `/etc/molink/gpu.env`

Secrets file, permissions 600 root:root. Must contain:
- `MOLINK_WS_URL`, `WORKER_SECRET` — worker WebSocket connection
- `BFL_API_KEY` — BFL rendering API
- `R2_*` — Cloudflare R2 storage keys
- `VLM_API_KEY`, `VLM_API_URL`, `VLM_MODEL` — VLM semantic parsing
- `AUTODL_TOKEN`, `AUTODL_INSTANCE_ID` — AutoDL instance management
- Service URLs, model paths, SAM config

**Critical:** GPU_MODEL must be quoted if it contains spaces: `GPU_MODEL="TITAN Xp"` not `GPU_MODEL=TITAN Xp`.

## Step 4 — Audit and fix hardcoded paths

Run the audit:
```bash
grep -RInE '/root/autodl-tmp|/root/.cache|/opt/models|/root/metric3d' \
  /opt/molink/hanging --include="*.py" --include="*.sh" --include="*.env*" \
  --exclude-dir=.git | grep -v ".qwen/hooks"
```

Fix patterns:
- `/root/autodl-tmp/models/*.pth` → `/opt/molink-cache/metric3d/*.pth`
- `/opt/models/sam/*.pth` → `/opt/molink-cache/sam/*.pth`
- `TORCH_HUB_DIR=/root/autodl-tmp/...` → `TORCH_HUB_DIR=/opt/molink-cache/torch/hub`
- `ROOT="${ROOT:-/opt/metric3d_api}"` → `ROOT="${ROOT:-/opt/molink/hanging}"`
- `cd /opt/metric3d_api` → `cd "$ROOT"`
- Output paths `/root/metric3d_*.zip` → `/tmp/molink-jobs/metric3d_*.zip`

## Step 5 — Create ops scripts

Four scripts in `/opt/molink/hanging/ops/`:
1. **download_models.sh** — downloads Metric3D/SAM checkpoints to cache (first boot only)
2. **start_prod.sh** — starts MVP APIs, waits for health, execs worker
3. **autodl_boot.sh** — idempotent entrypoint: sources env, creates dirs, calls download_models on first boot (guarded by `.prepared` timestamp), then execs start_prod.sh
4. **preflight_gpu.sh** — diagnostics: disk, python, torch/cuda, path audit, secret audit

## Step 6 — Install dependencies

```bash
source /etc/profile.d/molink-gpu.sh
set -a; source /etc/molink/gpu.env; set +a
cd /opt/molink/hanging
pip install -r requirements.txt -r requirements-sam.txt -r worker/requirements-worker.txt
```

SAM git install may fail on first attempt; check if `segment_anything` is already installed before retrying.

## Step 7 — Health check

```bash
bash scripts/manage_all_mvp_apis.sh restart
sleep 20
curl -fsS http://127.0.0.1:6006/health
curl -fsS http://127.0.0.1:6011/health
curl -fsS http://127.0.0.1:6010/health
```

## Step 8 — Clean for image

```bash
cd /opt/molink/hanging
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
rm -rf logs run outputs artifacts .pytest_cache

# Remove model weights from cache (will be re-downloaded on first boot)
rm -f /opt/molink-cache/metric3d/*.pth
rm -f /opt/molink-cache/sam/*.pth
rm -f /opt/molink-cache/torch/hub/checkpoints/*.pth
rm -f /opt/molink-cache/.prepared

pip cache purge
conda clean -a -y
apt-get clean
```

## Boot command for AutoDL Pro start_command

```bash
bash -lc 'source /etc/profile.d/molink-gpu.sh; set -a; source /etc/molink/gpu.env; set +a; exec /opt/molink/hanging/ops/autodl_boot.sh'
```

**Never put API keys in start_command** — they must be in `/etc/molink/gpu.env` baked into the private image.

## Symlink strategy for pre-image testing (models on data disk)

When testing on the **current instance** before creating the 30GB image, model weights (5.2GB Metric3D + 358MB SAM) would exceed the system disk limit. Use symlinks instead:

```bash
# Remove real dirs in molink-cache, link to data disk locations
rm -rf /opt/molink-cache/metric3d /opt/molink-cache/sam
ln -sfn /root/autodl-tmp/models /opt/molink-cache/metric3d
ln -sfn /opt/models/sam /opt/molink-cache/sam

# Torch hub checkpoint symlink
ln -sfn /root/autodl-tmp/models/metric_depth_vit_giant2_800k.pth \
  /opt/molink-cache/torch/hub/checkpoints/metric_depth_vit_giant2_800k.pth
```

This keeps `/opt/molink-cache` at ~24KB on system disk while models live on data disk. **Before creating the image**, delete these symlinks and recreate the empty dirs — the Pro instance will download models on first boot.

## Service startup workaround

`manage_all_mvp_apis.sh restart` can timeout when stdout is captured. Start services directly:

```bash
source /etc/profile.d/molink-gpu.sh
set -a; source /etc/molink/gpu.env; set +a
cd /opt/molink/hanging
mkdir -p logs run
export OMP_NUM_THREADS=4
nohup python -m uvicorn app:app --host 0.0.0.0 --port 6006 > logs/metric3d.log 2>&1 &
nohup python -m uvicorn semantic_mvp.service:app --host 0.0.0.0 --port 6011 > logs/semantic.log 2>&1 &
nohup python -m uvicorn hanging_mvp.service:app --host 0.0.0.0 --port 6010 > logs/hanging.log 2>&1 &
```

## Known issues

- `GPU_MODEL=TITAN Xp` (unquoted) causes `set -a; source gpu.env` to fail with "Xp: command not found"
- `segment-anything` pip install from git may fail due to network; check pre-installation first
- `manage_all_mvp_apis.sh` can timeout when run in a shell that captures all output; use `nohup ... &` for each service
- Long hex tokens (keys, JWTs) must be copied via `grep+sed` — never visually transcribed (see `auto-skill-hex-token-copy-safety`)
