# MoLink GPU Server - System Disk Migration Report

Generated: 2026-06-19

## 1. Source Directory

```
/opt/molink/hanging    <- real source (system disk)
/opt/metric3d_api      -> /opt/molink/hanging  (compatibility symlink)
```

## 2. System Disk Usage

```
Filesystem      Size  Used Avail Use% Mounted on
overlay          30G   23G  7.4G  76% /
```

## 3. /opt Directory Usage

```
1.4G    /opt/nvidia
358M    /opt/models          (legacy - SAM ckpt, can be removed)
355M    /opt/Metric3D        (legacy - torch hub repo, can be removed)
11M     /opt/metric3d_api_backups
1.1M    /opt/molink          (hanging source code)
480K    /opt/metric3d_api_deploy
212K    /opt/metric3d_api_patch_backups
0       /opt/molink-cache    (empty - models downloaded on first boot)
```

## 4. /opt/molink-cache Usage

```
0 bytes (all model directories created but empty)
  - metric3d/        (Metric3D checkpoints, downloaded on first boot)
  - sam/             (SAM checkpoints, downloaded on first boot)
  - hf/              (HuggingFace cache)
  - torch/           (PyTorch hub cache)
  - xdg/             (XDG cache)
  - checkpoints/     (torch hub checkpoints dir)
```

## 5. Path-Modified Files

All references to `/root/autodl-tmp`, `/root/.cache`, `/opt/models/sam/`, `/opt/metric3d_api` (as default) replaced with:
- Checkpoints: `/opt/molink-cache/metric3d/` and `/opt/molink-cache/sam/`
- Torch hub: `/opt/molink-cache/torch/hub/checkpoints/`
- ROOT default: `/opt/molink/hanging`

Key files modified:
- `services/metric3d_api/app.py` - `_MODEL_TYPE_CKPT_MAP` and `CKPT_PATH` default
- `app.py` (symlink to above)
- `semantic_mvp/sam_refiner.py` - SAM_CHECKPOINT default
- `scripts/start_semantic_v1_sam.sh` - SAM_CHECKPOINT default
- `.env.vlm_sam` - SAM_CHECKPOINT
- `scripts/install_v1_full_sam.sh` - SAM_DIR default
- All `scripts/*.sh` - ROOT default fallback
- All `services/metric3d_api/start*.sh` - cd target and TORCH_HUB_DIR
- `services/metric3d_api/install_giant*.sh` - torch cache paths
- `services/metric3d_api/metric3d_cli.py` and `metric3d_ctl.sh` - output paths
- `worker/start_worker.sh` - WORKER_DEPS_DIR default

## 6. Remaining Hardcoded Model Paths

ALL CLEAN (excluding `.qwen/hooks/` which are Qwen-specific, not production).

The remaining `metric_depth_*.pth` and `sam_vit_*.pth` grep matches are:
- Correct `/opt/molink-cache/` paths in app.py, sam_refiner.py
- Just filenames (without directory prefix) in install scripts' case statements and wget commands

## 7. /etc/molink/gpu.env

```
-rw------- 1 root root 1576 Jun 19 02:48 /etc/molink/gpu.env
Permissions: 600 root:root  ✓
```

Contains: BFL_API_KEY, R2 keys, VLM_API_KEY, service URLs, model paths, SAM config.

## 8. Three-Service Health Check Results

All three services started successfully from `/opt/molink/hanging`:

| Service | Port | Health | Response |
|---------|------|--------|----------|
| metric3d | 6006 | OK | `{"status":"ok","model":"metric3d_vit_giant2","device":"cuda"}` |
| semantic | 6011 | OK | `{"status":"ok","sam":{"checkpoint":"/opt/molink-cache/sam/sam_vit_b_01ec64.pth","device":"cuda"}}` |
| hanging | 6010 | OK | `{"status":"ok","api":"hanging-art-mvp-geometry-api"}` |

## 9. Worker Connection Status

**NOT tested** — two variables remain as placeholders in `/etc/molink/gpu.env`:

| Variable | Current Value | Required For |
|----------|--------------|--------------|
| `WORKER_SECRET` | `__REPLACE_ME__` | WebSocket authentication with molink.art |
| `AUTODL_TOKEN` | `__REPLACE_ME__` | AutoDL instance shutdown on idle |

All other variables (BFL_API_KEY, R2 keys, VLM_API_KEY, service URLs) have real values populated from existing `.env` files.

## 10. Recommended AutoDL Pro Boot Command

```bash
bash -lc 'source /etc/profile.d/molink-gpu.sh; set -a; source /etc/molink/gpu.env; set +a; exec /opt/molink/hanging/ops/autodl_boot.sh'
```

This command:
- Loads cache/model path environment
- Sources secrets from `/etc/molink/gpu.env` (not passed in start_command)
- Executes the idempotent boot script which:
  1. Downloads models on first boot (skipped if `.prepared` exists)
  2. Starts all three MVP services
  3. Waits for health checks
  4. Starts the worker

## Directory Structure

```
/opt/molink/hanging/          <- source code
  ops/
    autodl_boot.sh            <- idempotent boot entrypoint
    start_prod.sh             <- production service+worker start
    download_models.sh        <- first-boot model downloader
    preflight_gpu.sh          <- pre-flight diagnostic

/etc/molink/
  gpu.env                     <- secrets (600 root:root)

/etc/profile.d/
  molink-gpu.sh               <- environment for all shells

/opt/molink-cache/            <- model cache (empty, populated on first boot)
  metric3d/
  sam/
  hf/
  torch/hub/checkpoints/
  xdg/
```
