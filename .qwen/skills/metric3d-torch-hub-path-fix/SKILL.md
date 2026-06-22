---
name: metric3d-torch-hub-path-fix
description: Metric3D model weights must be loaded from data disk path, not via torch.hub.load_state_dict_from_url — the latter attempts network download when cache is empty, causing OOM kill and zombie process
source: auto-skill
extracted_at: '2026-06-07T09:00:00.000Z'
---

## Problem

`torch.hub.load_state_dict_from_url()` inside `hubconf.py` hardcodes weight cache to `/root/.cache/torch/hub/checkpoints/`. When this cache is empty (system disk cleanup, reboot), it attempts to download 5.2GB from HuggingFace over network → fails (no network) or triggers OOM → process killed → zombie port → HTTP 400.

## Permanent fix (preferred, implemented 2026-06-08)

`get_model()` in `services/metric3d_api/app.py` loads weights directly from disk:

```python
def get_model():
    global _MODEL
    if _MODEL is None:
        with _LOCK:
            if _MODEL is None:
                m = torch.hub.load(METRIC3D_REPO, MODEL_NAME, pretrain=False, source="local")
                ckpt = CKPT_PATH  # env var, default: /root/autodl-tmp/models/...
                if os.path.isfile(ckpt):
                    state = torch.load(ckpt, map_location="cpu")
                    if "model_state_dict" in state:
                        state = state["model_state_dict"]
                    m.load_state_dict(state, strict=False)
                else:
                    m = torch.hub.load(METRIC3D_REPO, MODEL_NAME, pretrain=True, source="local")
                m.to(DEVICE).eval()
                _MODEL = m
    return _MODEL
```

Key properties:
- `pretrain=False` skips `load_state_dict_from_url()` entirely
- Weight path from `CKPT_PATH` env var (default: `/root/autodl-tmp/models/metric_depth_vit_giant2_800k.pth`)
- Falls back to `pretrain=True` only if data disk file is missing
- Double-checked locking for thread safety

### Weight path mapping

`_MODEL_TYPE_CKPT_MAP` in `app.py` provides default data disk paths for all model variants:

| MODEL_NAME | Default CKPT_PATH |
|---|---|
| metric3d_convnext_tiny | /root/autodl-tmp/models/convtiny_hourglass_v1.pth |
| metric3d_convnext_large | /root/autodl-tmp/models/convlarge_hourglass_0.3_150_step750k_v1.1.pth |
| metric3d_vit_small | /root/autodl-tmp/models/metric_depth_vit_small_800k.pth |
| metric3d_vit_large | /root/autodl-tmp/models/metric_depth_vit_large_800k.pth |
| metric3d_vit_giant2 | /root/autodl-tmp/models/metric_depth_vit_giant2_800k.pth |

Weights are managed by `services/metric3d_api/install_giant_auto.sh` which downloads/copies them to `/root/autodl-tmp/models/`.

## Legacy workaround (obsolete after 2026-06-08 fix)

Before the direct-load fix, two workarounds were used:

### Workaround 1: Cache copy to /root/.cache/torch/hub/checkpoints/

```bash
mkdir -p /root/.cache/torch/hub/checkpoints
cp /root/autodl-tmp/models/metric_depth_vit_giant2_800k.pth \
   /root/.cache/torch/hub/checkpoints/metric_depth_vit_giant2_800k.pth
```

This makes `load_state_dict_from_url()` find the local file and skip network download.

### Workaround 2: TORCH_HUB_DIR export in startup scripts

```bash
export TORCH_HUB_DIR=/root/autodl-tmp/qwen_metric3d/artifacts/cache/torch/hub
```

**Why this alone was insufficient:** `TORCH_HUB_DIR` only affects `torch.hub.get_dir()` which controls repo code lookup. It does NOT change `load_state_dict_from_url()` checkpoint cache path — that still defaults to `/root/.cache/torch/hub/checkpoints/`.

## Why weight cache "disappears"

Two independent cache paths cause inconsistency:

| Path | Managed by | Persistence |
|------|-----------|-------------|
| `/root/.cache/torch/hub/checkpoints/` | `load_state_dict_from_url()` | System disk — wiped on cleanup/reboot |
| `/root/autodl-tmp/models/` | `install_giant_auto.sh` | Data disk — persistent |

The permanent fix reads directly from the data disk path, eliminating the dependency on the ephemeral system cache.

## Diagnostic commands

```bash
# Verify weight exists at data disk path
ls -lh /root/autodl-tmp/models/metric_depth_vit_giant2_800k.pth

# Check current torch cache (may be empty)
ls -la /root/.cache/torch/hub/checkpoints/

# Check if metric3d process is zombie (health OK but 400 on inference)
curl -s http://127.0.0.1:6006/health && echo ""
curl -s --max-time 60 -X POST http://127.0.0.1:6006/analyze_geometry \
  -F "file=@/path/to/test.jpg" -o /dev/null -w "HTTP:%{http_code}\n"
```
