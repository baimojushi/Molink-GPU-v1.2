---
name: pipeline-e2e-sim-run
description: End-to-end pipeline simulation using hanging_mvp_cli.py — auth handling, skip-bfl local testing, output summary, and zero-candidate diagnosis
source: auto-skill
extracted_at: '2026-07-14T14:00:09.825Z'
---

# Pipeline End-to-End Simulation Run

## Auth requirement

When services have `API_KEY` set (from `VLM_API_KEY` in `/etc/molink/gpu.env`), the pipeline script MUST pass `API_KEY` in the environment. Without it, curl calls return HTTP 401.

```bash
API_KEY="sk-bdf4a9c050d2479385cd795518cb99bd" python3 scripts/hanging_mvp_cli.py run \
  --no-start-services --skip-bfl \
  --room <room_path> --artwork <artwork_path> \
  --art-width-cm <w> --art-height-cm <h> \
  --output-dir <output_dir>
```

The `run_v1_full_pipeline.sh` script reads `API_KEY` from env and adds `-H "Authorization: Bearer $API_KEY"` to all curl calls.

## Standard local test command

```bash
cd /opt/molink/hanging
API_KEY="$(grep '^API_KEY=' /etc/molink/gpu.env | cut -d= -f2)" \
python3 scripts/hanging_mvp_cli.py run \
  --no-start-services \
  --skip-bfl \
  --room /root/autodl-tmp/pictures/qwen/room-raw.jpg \
  --artwork /root/autodl-tmp/pictures/qwen/artwork51.2×67.8cm.png \
  --art-width-cm 51.2 \
  --art-height-cm 67.8 \
  --output-dir /root/autodl-tmp/qwen_metric3d/artifacts/outputs/pipeline_test_$(date +%Y%m%d_%H%M%S) \
  --scene-profile gallery \
  --vlm-mode auto \
  --sam-enable
```

Key flags:
- `--no-start-services` — reuse already-running services (ports 6006, 6011, 6010)
- `--skip-bfl` — stop after local hanging pipeline, no external API calls
- `--output-dir` — write to data disk artifacts, never project root

## Output verification

After pipeline completes, check these files in `<output_dir>/result/`:

| File | Purpose |
|---|---|
| `client_result.json` | Top-level status, exit_code, rejection reasons |
| `hanging_plan.json` | Full plan with selected_candidate, passed_walls, rejected_walls |
| `ranked_candidates.json` | Approved candidates (empty if all rejected) |
| `rejected_walls.json` | Per-wall rejection details with floor estimation debug |
| `flux_inputs/manifest.json` | Flux pipeline config, semantic inventory, NMS settings |
| `rejected_wall_assets/<wall_id>/reject_summary.json` | Per-wall rejection debug |
| `rejected_wall_assets/<wall_id>/debug_overlay.png` | Visual debug overlay |

## Zero-candidate diagnosis

When `exit_code` is `PARTIAL_ALL_REJECTED_WITH_REASONS` and `ranked_candidates.json` is empty:

1. Read `rejected_walls.json` — find the `reason` field for each wall.
2. Common rejection reasons:
   - `trusted_wall_floor_intersection_required` — floor plane estimation failed (check `floor_ransac_inlier_ratio`, threshold typically ~0.3)
   - `no_safe_axis_and_insufficient_solid_component` — wall lacks a stable hanging axis
   - `geometry_gate` — candidate failed hard geometric constraints
3. Check `floor_estimation.attempts[]` in rejected_walls.json — each attempt shows `source`, `reject_reasons`, and metrics.
4. Floor estimation fallback chain: `wall_floor_plane_intersection` → `floor_mask_projected_to_current_wall_plane` → `solid_wall_mask_bottom_percentile` → `visible_wall_bottom_percentile_fallback`.

## Service health pre-check

```bash
for port in 6006 6011 6010; do
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$port/health")
  echo "port $port: $code"
done
```

All three must return 200 before running the pipeline.

## File storage compliance

- Output goes to `/root/autodl-tmp/qwen_metric3d/artifacts/outputs/` (data disk)
- Never write pipeline results to `/opt/molink/hanging/` (project root)
- Test images sourced from `/root/autodl-tmp/pictures/qwen/`
