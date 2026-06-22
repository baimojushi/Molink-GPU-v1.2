---
name: metric3d-hanging-artifact-structure
description: Understanding Metric3D zip contents, hanging pipeline output filtering, and artifact location conventions
source: auto-skill
extracted_at: '2026-06-04T13:33:08.267Z'
---

# Metric3D Hanging Artifact Structure

## Metric3D zip contents

The `/analyze_geometry` endpoint returns a zip containing:

| File | Type | Purpose |
|---|---|---|
| `meta.json` | JSON | Intrinsics, scale, quality metadata |
| `depth_m.npy` | NPY | Depth map in meters (float32) |
| `depth_mm_uint16.png` | PNG | Depth in millimeters (uint16) |
| `depth_vis.png` | PNG | Depth visualization |
| `normal_xyz.npy` | NPY | Normal vectors (float32) |
| `normal_rgb.png` | PNG | Normal visualization (RGB) |
| `points_xyz_m.npy` | NPY | 3D point cloud (only if `include_xyz=true`) |

Source: `services/metric3d_api/app.py` lines 278-285.

## Hanging service input filtering

`hanging_mvp/metric3d_io.py` `load_metric3d_zip()` extracts **only 3 files**:
- `meta.json`
- `depth_m.npy`
- `normal_xyz.npy`

The PNG visualization images (`depth_vis.png`, `normal_rgb.png`, `depth_mm_uint16.png`, `normal_rgb.png`) are **discarded** during consumption. They are not passed through to the hanging result zip.

## Hanging result zip structure

`hanging_v1_result.zip` from `/analyze_hanging_mvp` contains:

```
hanging_plan.json
ranked_candidates.json
passed_walls.json
client_result.json
flux_inputs/
  manifest.json
  original_room.png
  artwork.png
  items/
    <asset_id>/
      hard_paste.png
      art_region_mask.png
      remove_mask.png
      debug_overlay.png
      candidate.json
      render_prompt.txt
debug_assets/
  semantic_wall_debug.png
wall_assets/
  <wall_id>/
    wall_debug_overlay.png
    normal_consistency_mask.png
```

**Note:** No metric3d depth/normal/bbox images are included. The hanging service consumes the numpy arrays for geometry computation but does not preserve the visualization images.

## Artifact locations

| Artifact | Location |
|---|---|
| Code | `/opt/metric3d_api/` |
| Run outputs | `/root/hanging_runs/<run_id>/` |
| Backups | `/root/autodl-tmp/qwen_metric3d/artifacts/backups/qwen/` |
| Temporary work | `data_disk_tmp/qwen/` (symlink → `/root/autodl-tmp/qwen_metric3d/tmp/`) |
| Logs | `data_disk_logs/qwen/` |
| Pictures | `data_disk_pictures/qwen/` |

## Backup unpack workflow

To inspect a backup zip:

```bash
# Find backup
find /root/autodl-tmp -name "<backup_name>.zip" -type f

# Unpack to temp directory
mkdir -p data_disk_tmp/qwen/
unzip -o /root/autodl-tmp/qwen_metric3d/artifacts/backups/qwen/<backup_name>.zip -d data_disk_tmp/qwen/
```

## Metric3D zip preservation

If you need metric3d visualization images (depth_vis.png, normal_rgb.png), you must:

1. **Copy before passing to hanging:** Save `room_metric3d_raw.zip` (downloaded by `run_full_hanging_bfl_test_opt.sh`) before calling `/analyze_hanging_mvp`
2. **Extract manually:** `unzip -l room_metric3d_raw.zip` to list contents, then extract specific PNGs
3. **Re-run metric3d:** Call `/analyze_geometry` again with the same room image and `include_candidates=true`

The test script downloads metric3d zip to `$OUT_ROOT/room_metric3d_raw.zip` (line 64-71). If this file is missing after a run, it was either:
- Not downloaded (metric3d service unavailable)
- Deleted after use
- Not included in the backup archive

## Metric3D backfill workflow

When a hanging run directory is missing metric3d output, re-run inference and copy the result into place:

```bash
# 1. Extract original room image from hanging result
mkdir -p /root/autodl-tmp/tmp/qwen/metric3d_temp
cd /root/autodl-tmp/tmp/qwen/metric3d_temp
python3 -c "import zipfile; z=zipfile.ZipFile('/root/hanging_runs/<RUN_ID>/hanging_v1_result.zip'); z.extract('flux_inputs/original_room.png')"

# 2. Verify checkpoint exists (blocks inference if missing)
find / -name "metric_depth_vit_giant2_800k.pth" 2>/dev/null
# If not in torch cache, copy from artifacts:
cp /root/autodl-tmp/qwen_metric3d/artifacts/cache/torch/hub/checkpoints/metric_depth_vit_giant2_800k.pth /root/.cache/torch/hub/checkpoints/

# 3. Start metric3d service (if not running)
cd /opt/metric3d_api/services/metric3d_api
bash metric3d_ctl.sh start

# 4. Run inference
python3 metric3d_cli.py --url http://127.0.0.1:6006 infer \
  --image /root/autodl-tmp/tmp/qwen/metric3d_temp/flux_inputs/original_room.png \
  --include-candidates \
  --output /root/autodl-tmp/tmp/qwen/metric3d_temp/metric3d_result.zip

# 5. Copy result to hanging run directory
cp /root/autodl-tmp/tmp/qwen/metric3d_temp/metric3d_result.zip /root/hanging_runs/<RUN_ID>/metric3d_result.zip
```

**Output zip contents:** `meta.json`, `depth_m.npy`, `depth_mm_uint16.png`, `depth_vis.png`, `normal_xyz.npy`, `normal_rgb.png` (25MB for typical room image).

## Metric3D checkpoint location and TORCH_HOME

**Authoritative path:** `/root/autodl-tmp/qwen_metric3d/artifacts/cache/torch/hub/checkpoints/metric_depth_vit_giant2_800k.pth`

The metric3d service process has `TORCH_HOME=/root/autodl-tmp/qwen_metric3d/artifacts/cache/torch` set in its environment. This overrides the default `~/.cache/torch/`, so `torch.hub.load()` and `torch.hub.load_state_dict_from_url()` look for checkpoints under the artifacts/cache path.

**Diagnostic commands:**

```bash
# Verify running service has correct TORCH_HOME
cat /proc/$(cat /opt/metric3d_api/run/metric3d.pid)/environ | tr '\0' '\n' | grep TORCH

# Check checkpoint exists at correct location
ls -lh /root/autodl-tmp/qwen_metric3d/artifacts/cache/torch/hub/checkpoints/metric_depth_vit_giant2_800k.pth

# Full filesystem search (takes ~30s)
find / -name "metric_depth_vit_giant2_800k.pth" 2>/dev/null
```

**If inference returns 500 with URLError/Network unreachable:**

1. Check `tail /opt/metric3d_api/logs/metric3d.log` — if traceback shows `torch.hub.load_state_dict_from_url` failing with `URLError: [Errno 99]` or `[Errno 101]`, the checkpoint is missing from the TORCH_HOME path.
2. The file may exist at `/root/.cache/torch/hub/checkpoints/` (old default) — this is the WRONG location for the running service.
3. **Fix:** Move (don't symlink) the file to the correct path:
   ```bash
   mkdir -p /root/autodl-tmp/qwen_metric3d/artifacts/cache/torch/hub/checkpoints
   mv /root/.cache/torch/hub/checkpoints/metric_depth_vit_giant2_800k.pth \
      /root/autodl-tmp/qwen_metric3d/artifacts/cache/torch/hub/checkpoints/metric_depth_vit_giant2_800k.pth
   ```
4. Restart metric3d: `bash scripts/manage_all_mvp_apis.sh restart metric3d`
5. Verify: `curl -fS -X POST http://127.0.0.1:6006/analyze_geometry -F "file=@/path/to/room.jpg" -F 'camera_json={}' -o /tmp/test.zip`

**Why move, not symlink:** Symlinks add maintenance overhead and can break across filesystem reorganizations. Direct placement is simpler and more reliable.

**Checkpoint size:** ~5.2GB (5510989187 bytes).

**Old memory entries may say** "copy FROM artifacts TO ~/.cache/torch" — this was incorrect. The service reads FROM the artifacts path because of its `TORCH_HOME` env var.

## Bbox backfill workflow

When `metric3d_result.zip` lacks bbox data (metric3d service does not produce bbox — it is a hanging service product), inject it from `hanging_v1_result.zip`:

```python
import zipfile, json, shutil, tempfile, os

hanging_path = '/root/hanging_runs/<RUN_ID>/hanging_v1_result.zip'
metric3d_path = '/root/hanging_runs/<RUN_ID>/metric3d_result.zip'

# 1. Extract bbox from hanging candidate.json files
hanging = zipfile.ZipFile(hanging_path)
bbox_data = {}
for name in hanging.namelist():
    if name.startswith('flux_inputs/items/') and name.endswith('candidate.json'):
        cand = json.loads(hanging.read(name))
        bfl = cand.get('bfl_v7_export', {})
        if bfl:
            cand_id = name.split('/')[-2]  # e.g. auto_wall_08__cand_005
            bbox_data[cand_id] = {
                'bbox_px': bfl.get('bbox_px'),           # [x0, y0, x1, y1]
                'polygon_px': bfl.get('polygon_px'),      # [[x,y], ...] 4 corners
                'projected_quad_px_original': cand.get('projection', {}).get('projected_quad_px_original'),
                'center_px_original': cand.get('projection', {}).get('center_px_original'),
            }

# 2. Build bbox.json with image_size and intrinsics from metric3d
metric3d = zipfile.ZipFile(metric3d_path)
meta = json.loads(metric3d.read('meta.json'))
bbox_export = {
    'image_size': meta['image']['original_size'],   # [1280, 1707]
    'processed_size': meta['image']['processed_size'],
    'intrinsics': meta.get('selected_intrinsics'),
    'candidates': bbox_data
}

# 3. Inject bbox.json into metric3d zip
with tempfile.TemporaryDirectory() as tmpdir:
    metric3d.extractall(tmpdir)
    with open(os.path.join(tmpdir, 'bbox.json'), 'w') as f:
        json.dump(bbox_export, f, indent=2, ensure_ascii=False)
    with zipfile.ZipFile(metric3d_path, 'w', zipfile.ZIP_DEFLATED) as out:
        for root, dirs, files in os.walk(tmpdir):
            for fname in files:
                full = os.path.join(root, fname)
                arcname = os.path.relpath(full, tmpdir)
                out.write(full, arcname)
```

**Result:** `metric3d_result.zip` now contains 7 files: `meta.json`, `depth_m.npy`, `depth_mm_uint16.png`, `depth_vis.png`, `normal_xyz.npy`, `normal_rgb.png`, `bbox.json`.

**Source of bbox:** `hanging_mvp/service.py` line ~1479 — `cand["bfl_v7_export"]` is populated with `bbox_px`, `polygon_px`, and other fields during the hanging pipeline. These are embedded in each `candidate.json` under `flux_inputs/items/<asset_id>/`.

## Key distinction

- **Old `room-exif` runs:** Included `room_metric3d.zip` in run directory → contains full metric3d output with PNGs
- **New `room-raw_bfl` runs:** May or may not include metric3d zip in run directory; backup archives may exclude it; backfill using the workflow above
- **Bbox is never in metric3d output:** metric3d service produces geometry only (depth, normals, intrinsics). Bbox/polygon data originates from the hanging service's wall detection and projection pipeline. Must be extracted from `hanging_v1_result.zip` candidate.json files and injected separately.
