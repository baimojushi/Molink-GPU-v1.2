---
name: sam-model-switching
description: Switch SAM backend model (vit_b, vit_l, mobile_sam) — env var config, checkpoint download, and service restart procedure
source: auto-skill
extracted_at: '2026-07-14T12:53:59.492Z'
---

# SAM Model Switching

## When to use

User wants to change the SAM model used for semantic mask refinement — either switching between SAM vit_b/vit_l/vit_h, or trying MobileSAM/FastSAM alternatives.

## SAM backend options

| Backend | SAM_BACKEND | SAM_MODEL_TYPE | Checkpoint size | Notes |
|---------|-------------|----------------|-----------------|-------|
| SAM vit_b | `sam1` | `vit_b` | ~375MB | Default, already installed |
| SAM vit_l | `sam1` | `vit_l` | ~1.2GB | Higher accuracy, more VRAM |
| SAM vit_h | `sam1` | `vit_h` | ~2.4GB | Highest accuracy, most VRAM |
| MobileSAM | `mobile_sam` | `vit_t` | ~39MB | Lightweight, requires pip install |

## Env vars in `/etc/molink/gpu.env`

```bash
SAM_ENABLE=1
SAM_BACKEND=sam1           # or mobile_sam
SAM_MODEL_TYPE=vit_b       # or vit_l, vit_h, vit_t
SAM_CHECKPOINT=/opt/molink-cache/sam/sam_vit_b_01ec64.pth
SAM_MAX_IMAGE_SIDE=1536
```

## Checkpoint download

Checkpoints are at:
- SAM vit_b/l/h: `/opt/molink-cache/sam/sam_vit_{b,l,h}_*.pth` (Facebook Research)
- MobileSAM: `/opt/molink-cache/sam/mobile_sam.pt` (ChaoningZhang/MobileSAM)

**Network-unreachable fallback:** GitHub and HuggingFace are often blocked on AutoDL machines. If external downloads fail:
1. Check if checkpoint already exists: `ls -la /opt/molink-cache/sam/`
2. If vit_b checkpoint exists (375MB), use it instead of downloading MobileSAM
3. If no checkpoint exists, user must manually download and transfer via SCP

## Service restart after model change

Semantic service must restart to pick up new SAM config:

```bash
# Kill existing semantic
pkill -f 'semantic_mvp' 2>/dev/null || true
sleep 2

# Start with metric3d conda env Python (conda activate fails in background shells)
cd /opt/molink/hanging && nohup env \
  SAM_ENABLE=1 \
  SAM_BACKEND=sam1 \
  SAM_MODEL_TYPE=vit_b \
  SAM_CHECKPOINT=/opt/molink-cache/sam/sam_vit_b_01ec64.pth \
  SAM_MAX_IMAGE_SIDE=1536 \
  ALLOW_UNAUTH_LOCAL=1 \
  OMP_NUM_THREADS=4 \
  SEMANTIC_MVP_PORT=6011 \
  /root/miniconda3/envs/metric3d/bin/python -m uvicorn \
  semantic_mvp.service:app --host 0.0.0.0 --port 6011 \
  > logs/semantic_new.log 2>&1 &

# Wait and verify
sleep 10
ps aux | grep 'semantic_mvp' | grep -v grep
curl -s http://127.0.0.1:6011/health | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d.get('sam',{}), indent=2))"
```

**Important:** `nohup` returns a wrapper PID, the actual uvicorn PID is different. Use `ps aux | grep semantic_mvp` to find the real PID for verification.

## Verify SAM loaded correctly

Health endpoint should show:
```json
{
  "enabled_by_env": true,
  "backend": "sam1",
  "model_type": "vit_b",
  "checkpoint": "/opt/molink-cache/sam/sam_vit_b_01ec64.pth",
  "device": "cuda",  // or "cpu" if no GPU
  "available": true,
  "loaded": false,   // false = lazy load, true = pre-warmed
  "error": ""
}
```

## Troubleshooting

### `SAM_MAX_IMAGE_SIDE` missing
- Default is 1536 in the code, but explicit is better
- Add to gpu.env: `echo 'SAM_MAX_IMAGE_SIDE=1536' >> /etc/molink/gpu.env`

### MobileSAM install fails
- `pip install mobile-sam` may not be on PyPI mirrors
- Use: `pip install git+https://github.com/ChaoningZhang/MobileSAM.git`
- Must install into metric3d conda env: `/root/miniconda3/envs/metric3d/bin/python -m pip install ...`

### MobileSAM checkpoint download fails
- External network (GitHub/HuggingFace) often unreachable on AutoDL
- Fallback to existing SAM vit_b checkpoint if available
- Model key for MobileSAM is `vit_t`, not `tiny_vit`

### SAM not enabled after restart
- Check process env vars: `cat /proc/<PID>/environ | tr '\0' '\n' | grep '^SAM_'`
- The `nohup env ...` pattern ensures vars are passed; `export` in parent shell may not propagate to detached child
