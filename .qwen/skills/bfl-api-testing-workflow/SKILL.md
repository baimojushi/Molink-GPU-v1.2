---
name: bfl-api-testing-workflow
description: Standard workflow for testing BFL erase/flux2 API using hanging v1 results
source: auto-skill
extracted_at: '2026-06-06T17:31:14.532Z'
---

# BFL API Testing Workflow

## Prerequisites

- BFL API key configured in `.env.bfl` (file must use `export` prefix on each variable)
- A `hanging_v1_result.zip` with `flux_inputs/manifest.json`
- Artwork reference image available

## One-command full pipeline (recommended)

```bash
cd /opt/metric3d_api
bash scripts/run_full_hanging_pipeline.sh ROOM_IMAGE ARTWORK_IMAGE [OUT_DIR] [FLAGS...]
```

This script chains: start 4 services → v1 pipeline → BFL real API. BFL env vars are auto-exported via `set -a; source .env.bfl; set +a`.

Flags: `--skip-bfl` (skip BFL stage), `--bfl-limit N` (limit candidates), `--continue-on-error`.

## Manual test steps

### 1. Source and export BFL environment

```bash
cd /opt/metric3d_api
set -a; source .env.bfl; set +a
```

**Critical:** Use `set -a` (or explicit `export`) before sourcing. `source .env.bfl` alone only sets variables in the current shell — Python subprocesses cannot read them via `os.getenv()`, causing silent `provider=None` and zero API steps.

**Verify:** `echo $BFL_API_KEY` should show your key. If empty, the `.env.bfl` file is missing or lacks `export` prefixes.

### 2. Run with real BFL API (limited candidates)

```bash
HANGING_ZIP=/root/hanging_runs/<run>/hanging_v1_result.zip
BFL_DIR=/root/hanging_runs/<run>/flux_result_bfl

python3 scripts/run_bfl_hanging_flux_pipeline.py \
  --hanging-zip "$HANGING_ZIP" \
  --output-dir "$BFL_DIR" \
  --provider bfl \
  --limit 1 \
  --continue-on-error \
  --dual-ab-compare \
  --dual-ab-compare-limit 1
```

`--limit 1` restricts to one candidate. `--dual-ab-compare` runs a single-input baseline alongside dual-input for the same candidate.

### 4. Verify BFL artifacts

```bash
find "$BFL_DIR" -maxdepth 4 -type f | sort
```

Expected:
- `edit_trace.json` with `provider: bfl`
- `final_hd/<asset_id>.png` — dual render result
- `final_hd_ab_compare/<asset_id>__single_baseline.png` — single baseline
- Per-asset directory with `01_erased.png`, `02_hard_paste_after_erase.png`, `03_flux2_render.png`
- `inputs/` with `original_room.png`, `artwork.png`, `hard_paste.png`, `remove_mask.png`, `art_region_mask.png`

### 5. Key trace fields to verify

| Field | Expected Value | Meaning |
|---|---|---|
| `sequence_mode` | `v7_original_erase_hardpaste_flux2` | Correct v7 order |
| `flux2_mask_sent` | `false` | Mask not sent to Flux2 (N1 fix) |
| `art_region_mask_usage` | `qa_only_not_sent_to_flux2` | QA-only semantics |
| `flux2_input_mode` | `dual_original_plus_target` or `single` | Render mode |
| `render_input_semantics.input_image` | `cleaned_hard_paste_target_authoritative_canvas` | Image 1 = authoritative canvas |
| `render_input_semantics.input_image_2` | `original_room_texture_lighting_reference` | Image 2 = reference only |
| `erase_provider` | `bfl_erase_v1` | Erase API used |
| `render_provider` | `bfl_flux2_edit` | Render API used |
| `provider_warning` | `null` (or warning text if mock) | Mock safety guard |

## Dual input order contract

Flux2 dual mode uses explicit order:
- **Image 1** = `hard_paste_after_erase` (authoritative canvas for artwork placement, size, crop, aspect ratio)
- **Image 2** = `original_room` (texture/lighting/perspective reference only)
- `order_policy`: `"input_image_is_authoritative_target; input_image_2_is_reference_only"`

Prompt includes `STRICT SIZE LOCK` and `Do not change the painting core size` rules to prevent Flux2 from scaling or cropping the artwork.

## Dual mode activation

`--dual-ab-compare` only produces dual renders when manifest items have `flux2_input_mode` set to `dual_erased_room_plus_target` or `dual_original_plus_target`. Otherwise the pipeline falls back to `single_after_erase_hardpaste` and `executed_count` in the `dual_ab_compare` section will be 0.

To force dual mode on all candidates, the manifest's `flux2_input_mode` field must be set before running the pipeline, or use `--dual-ab-compare-limit` with candidates that already have dual mode requested.

## QA size validation and output placement

The pipeline runs mask-anchored QA after each Flux2 render. Key thresholds:
- `min_core_masked_ssim` (default ~0.85): SSIM within the artwork core region
- `max_forbidden_outer_guard_changed_ratio` (default ~0.05): fraction of pixels changed outside the guard region

If QA fails:
- Outputs go to `partial_review/` not `final_hd/`
- `core_restored_preview.png` shows render with core pixels restored from hard-paste
- `hardpaste_preview.png` shows the hard-paste before Flux2 render
- A single-input fallback render may be attempted (`03c_flux2_render_single_fallback.png`)

**Typical failure modes:**
- `core_masked_ssim < 0.5`: Flux2 significantly altered the artwork core (wrong colors, texture changes)
- `forbidden_outer_guard_changed_ratio > 0.1`: Flux2 modified wall/room texture beyond acceptable tolerance

## Trace capture gap

Per-asset `edit_trace.json` files may show `"steps": []` even when BFL API calls succeed and output files exist. This is a known trace capture limitation in `run_bfl_hanging_flux_pipeline.py` — the step details from individual provider results are not aggregated into the per-asset trace. Verify success by checking that `01_erased.png`, `02_hard_paste_after_erase.png`, and `03_flux2_render.png` exist in the output directory.

## A/B comparison review

Compare dual vs single render for same candidate:
```
final_hd/<asset_id>.png                    # dual input
final_hd_ab_compare/<asset_id>__single_baseline.png  # single input
```

Decision rule:
- Keep dual mode only if artwork position/scale preserved as well as single baseline while improving wall texture consistency
- Otherwise revert non-person cleanup cases to single-input mode
