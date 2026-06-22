---
name: system-disk-cleanup
description: Identify and clean invalid storage on system disk for Metric3D API project
source: auto-skill
extracted_at: '2026-06-20T10:40:00.000Z'
---

## System Disk Cleanup Procedure

### Step 1 — Audit disk usage

Run `df -h /` to get system disk total and usage percentage.

Run `du -sh /opt/* 2>/dev/null | sort -h` to find top consumers on system disk.

Run `du -sh /root/autodl-tmp/* 2>/dev/null | sort -h` to break down data disk.

### Step 2 — System disk `/opt/` layout

| Path | Type | Action |
|---|---|---|
| `/opt/Metric3D` | 官方源码 | 保留 |
| `/opt/models` | SAM 权重（系统盘副本） | 保留（运行时软链接目标） |
| `/opt/molink/hanging` | 项目主代码库 | 保留 |
| `/opt/metric3d_api` | → `/opt/molink/hanging` 软链接 | 保留（兼容旧路径） |
| `/opt/molink-cache` | 框架缓存 | 清空 `xdg/`、`torch/` 子目录 |
| `/opt/nvidia` | NVIDIA 容器组件 | 不可删 |
| `/opt/metric3d_api_backups` | 历史代码备份 | 可删 |
| `/opt/metric3d_api_patch_backups` | 历史补丁备份 | 可删 |
| `/opt/metric3d_api_deploy` | 已部署旧包产物 | 可删 |

### Step 3 — Data disk `/root/autodl-tmp/` stale targets

| Path | Type | Action |
|---|---|---|
| `artwork_hanging_bfl_patch_fixed/` | 旧补丁解压残留 | 可删 |
| `tmp/semantic_*.log` | 调试日志残片 | 可删 |
| `tmp/test_*` dirs & zips | 旧测试产物 | 可删 |
| `tmp/qwen/*` (debug zips, jsons) | Qwen 调试产物 | 可删 |
| `tmp/qwen/*.zip` | 旧源码包 | 可删 |
| `qwen_metric3d/npm-cache/` | npm 缓存 | 可清空 |

### Step 4 — Identify invalid storage

**Torch/HF cache in wrong location**: Model weights cached in `artifacts/cache/torch/hub/checkpoints/` should be in `/root/autodl-tmp/models/`. Move, don't delete, if the model is needed.

**Scattered backups**: Files in `artifacts/backups/` or `data_disk_tmp/` that duplicate runs in `hanging_runs/` — delete the duplicate, keep the authoritative copy in `hanging_runs/`.

**Pictures in wrong dirs**: `artifacts/pictures/`, `artifacts/images/` should consolidate to `data_disk_pictures` (symlink at `/root/pictures/`).

**Stale temp files**: `artifacts/tmp/pytest-of-*`, `artifacts/tmp/node-compile-cache`, `artifacts/tmp/tmp*` — safe to delete.

**Old hanging_runs**: Runs older than 3 days (`find /root/hanging_runs/ -maxdepth 1 -type d -mtime +3`) — safe to delete if not needed for debugging.

**Scattered files in /root**: Images (`.png`, `.jpg`) and old `.zip` files in `/root` root — move to `/root/pictures/qwen/` or delete.

**npm cache**: `/root/autodl-tmp/qwen_metric3d/npm-cache/` — safe to clear.

### Step 5 — Execute cleanup

Delete stale system-disk backups and deploy artifacts:
```
rm -rf /opt/metric3d_api_backups /opt/metric3d_api_patch_backups /opt/metric3d_api_deploy
```

Clean framework runtime caches:
```
rm -rf /opt/molink-cache/xdg /opt/molink-cache/torch
```

Delete data-disk stale artifacts:
```
rm -rf /root/autodl-tmp/artwork_hanging_bfl_patch_fixed
rm -rf /root/autodl-tmp/tmp/semantic_*.log /root/autodl-tmp/tmp/test_*
rm -rf /root/autodl-tmp/tmp/qwen/*.zip /root/autodl-tmp/tmp/qwen/*.json /root/autodl-tmp/tmp/qwen/*.txt
```

Move model weights:
```
mkdir -p /root/autodl-tmp/models
mv <wrong_path>/<model_file>.pth /root/autodl-tmp/models/
rm -rf <wrong_path>/torch
```

Consolidate pictures:
```
mkdir -p /root/pictures/qwen
cp -r <artifacts_pictures_path> /root/pictures/qwen/
rm -rf <artifacts_pictures_path>
```

Clean old hanging_runs:
```
find /root/hanging_runs/ -maxdepth 1 -type d -mtime +3 -exec rm -rf {} +
```

Clean scattered files in /root:
```
mv /root/*.png /root/*.jpg /root/pictures/qwen/
rm -f /root/*.zip /root/Untitled\ Folder
```

### Step 6 — Verify

Run `du -sh /opt/* 2>/dev/null | sort -h` — stale dirs should be gone.

Run `du -sh /root/autodl-tmp/* 2>/dev/null | sort -h` — tmp should be minimal.

Run `du -sh /root/autodl-tmp/tmp/qwen/ 2>/dev/null` — should be empty or near-zero.

### Key paths to remember

- Project root: `/opt/molink/hanging` (symlinked as `/opt/metric3d_api`)
- System disk: `overlay` mounted on `/` (30G)
- Data disk root: `/root/autodl-tmp/`
- Model weights: `/root/autodl-tmp/models/`
- Pictures: `/root/pictures/` (symlink from `data_disk_pictures`)
- Hanging runs: `/root/hanging_runs/` (symlink from `data_disk_outputs`)
- Qwen runtime: `/root/autodl-tmp/qwen_metric3d/`
- Framework cache: `/opt/molink-cache/` (softlinks for metric3d→data_disk models, sam→/opt/models)
