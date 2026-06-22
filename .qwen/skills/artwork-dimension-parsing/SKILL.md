---
name: artwork-dimension-parsing
description: Parse artwork physical dimensions from filenames and enforce as required inputs in the hanging pipeline
source: auto-skill
extracted_at: '2026-06-06T10:38:22.322Z'
---

## Goal

Automatically extract artwork physical dimensions (width × height) from filenames like `artwork51.2×67.8cm.png` and inject them as required inputs into the hanging MVP pipeline. No hardcoded defaults — dimensions must come from upstream or the pipeline errors.

## Parser script

`scripts/parse_artwork_dimensions.py` extracts dimensions from filenames.

### Supported formats

- Separators: `×` (multiplication sign), `x`, `X`, `*`, `·`
- Units: `cm` (default if absent), `mm`, `m`
- Output: metres (cm → /100, mm → /1000, m → direct)
- Error: unknown unit suffix causes exit 1

### Usage

```bash
ARTWORK_FILE="/path/to/artwork51.2×67.8cm.png"
eval "$(python3 scripts/parse_artwork_dimensions.py "$ARTWORK_FILE")"
# Sets: ARTWORK_WIDTH_M=0.512000  ARTWORK_HEIGHT_M=0.678000
```

## Pipeline integration

The test script `scripts/run_full_hanging_bfl_test_opt.sh` enforces dimension requirements:

```bash
# Mandatory check — exit 1 if not provided
if [ -z "${ARTWORK_WIDTH_M:-}" ] || [ -z "${ARTWORK_HEIGHT_M:-}" ]; then
  echo "[ERR] ARTWORK_WIDTH_M and ARTWORK_HEIGHT_M environment variables are required"
  exit 1
fi
```

### Full call pattern

```bash
ARTWORK_FILE="/path/to/artwork51.2×67.8cm.png"
eval "$(python3 scripts/parse_artwork_dimensions.py "$ARTWORK_FILE")"
ARTWORK="$ARTWORK_FILE" \
  ROOM="/path/to/room.jpg" \
  ARTWORK_WIDTH_M="$ARTWORK_WIDTH_M" \
  ARTWORK_HEIGHT_M="$ARTWORK_HEIGHT_M" \
  bash scripts/run_full_hanging_bfl_test_opt.sh
```

## Service-side validation

`hanging_mvp/service.py` validates dimensions at runtime:

```python
_artwork_cfg = rules.get("artwork", {})
if not _artwork_cfg.get("physical_width_m") or not _artwork_cfg.get("physical_height_m"):
    err = {"status": "failed", "exit_code": "FAILED_MISSING_ARTWORK_DIMENSIONS",
           "error": "artwork.physical_width_m and artwork.physical_height_m are required inputs"}
    return _zip_response(...)
```

## Configuration defaults removed

- `configs/rules_v1.json`: `physical_width_m` and `physical_height_m` set to `null`
- `hanging_mvp/rules_mvp.json`: same

Previously these had fallback values (0.512/0.678 in JSON, 0.80/0.60 in code). All fallbacks removed to enforce explicit input.

## Key files

| File | Role |
|------|------|
| `scripts/parse_artwork_dimensions.py` | Filename → metres parser |
| `scripts/run_full_hanging_bfl_test_opt.sh` | Test pipeline with dimension guard |
| `hanging_mvp/service.py` | Runtime validation (FAILED_MISSING_ARTWORK_DIMENSIONS) |
| `configs/rules_v1.json` | Null placeholder (no default) |
| `hanging_mvp/rules_mvp.json` | Null placeholder (no default) |
