---
name: external-storage-relocation
description: Relocate large service output (hanging_runs) from system disk `/` to AutoFS `/root/autodl-fs/` using symlink + path persistence in scripts
source: auto-skill
extracted_at: '2026-06-17T21:08:33.277Z'
---

## External Storage Relocation Procedure

Use this skill when the system disk overlay (`/`, 30G) is running low (80%+) due to accumulated service output directories. The primary target for relocation is `/root/hanging_runs/`.

### Disk layout — DO NOT confuse these

| Mount | Size | Purpose | Action |
|---|---|---|---|
| `/` (overlay) | 30G | System disk — **this is what fills up** | Move hanging_runs here off |
| `/root/autodl-tmp` | 50G data disk (`/dev/md0`) | Model weights, artifacts, cache, tmp | **Do NOT move** — this is storage |
| `/root/autodl-fs` | 200G AutoFS | External backup/archive mount | Move large outputs here |

**Critical rule**: `/root/autodl-tmp` is the data disk, not the system disk. Model weights (`/root/autodl-tmp/models/`) and artifacts (`/root/autodl-tmp/qwen_metric3d/artifacts/`) should stay there — they belong on the data disk by design.

### Target: `/root/hanging_runs/`

This is the BFL pipeline output directory on the system disk. Typically 3-5GB after a few runs.

### Step 1 — Move to AutoFS, create symlink

```bash
# Move (mv across filesystems = copy + delete source)
mv /root/hanging_runs /root/autodl-fs/hanging_runs

# If mv times out (large dir), use rsync then delete partial
rsync -av /root/hanging_runs/ /root/autodl-fs/hanging_runs/
rm -rf /root/hanging_runs

# Symlink for backward compatibility
ln -s /root/autodl-fs/hanging_runs /root/hanging_runs
```

### Step 2 — Persist new path in scripts

Update all hardcoded `/root/hanging_runs/` references to use `/root/autodl-fs/hanging_runs/` as the new default, with an env var override:

**`scripts/run_full_hanging_bfl_test_opt.sh`:**
```bash
HANGING_RUNS_ROOT="${HANGING_RUNS_ROOT:-/root/autodl-fs/hanging_runs}"
OUT_ROOT="${OUT_ROOT:-${HANGING_RUNS_ROOT}/${RUN_ID}}"
```

**`scripts/run_full_hanging_pipeline.sh`:**
```bash
HANGING_RUNS_ROOT="${HANGING_RUNS_ROOT:-/root/autodl-fs/hanging_runs}"
OUT_DIR="${HANGING_RUNS_ROOT}/room_full_$(date +%Y%m%d_%H%M%S)"
```

**`scripts/hanging_mvp_cli.py`:**
```python
def _default_output_dir() -> Path:
    hanging_runs_root = os.environ.get("HANGING_RUNS_ROOT", "/root/autodl-fs/hanging_runs")
    return Path(hanging_runs_root) / time.strftime("room_full_%Y%m%d_%H%M%S")
```

### Step 3 — Verify

```bash
ls -la /root/hanging_runs       # should show symlink
ls /root/hanging_runs/ | head   # should list run directories
df -h /                          # system disk usage should drop
bash /opt/metric3d_api/scripts/manage_all_mvp_apis.sh restart
```

### Gotchas

- **mv across filesystems**: `mv` between `/` and `/root/autodl-fs` performs a copy-then-delete, which can timeout on large dirs. If interrupted, `rsync` the remainder and delete the partial source manually.
- **`--remove-source-files` on rsync**: Dangerous with AutoFS (network mount). If rsync times out mid-transfer, `--remove-source-files` leaves partial data. Use plain `rsync -av`, verify sizes, then manually `rm -rf` the source.
- **Don't migrate data disk content**: `/root/autodl-tmp/` models and artifacts are on the 50G data disk — they do not belong on the system disk and do not need relocation.
