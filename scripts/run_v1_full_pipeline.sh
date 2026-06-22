#!/usr/bin/env bash
set -euo pipefail

ROOM_IMAGE="${1:?usage: run_v1_full_pipeline.sh ROOM_IMAGE ARTWORK_IMAGE OUT_DIR [RULES_JSON]}"
ARTWORK_IMAGE="${2:?usage: run_v1_full_pipeline.sh ROOM_IMAGE ARTWORK_IMAGE OUT_DIR [RULES_JSON]}"
OUT_DIR="${3:?usage: run_v1_full_pipeline.sh ROOM_IMAGE ARTWORK_IMAGE OUT_DIR [RULES_JSON]}"
RULES_JSON="${4:-configs/rules_v1.json}"
METRIC_URL="${METRIC_URL:-http://127.0.0.1:6006}"
SEMANTIC_URL="${SEMANTIC_URL:-http://127.0.0.1:6011}"
HANGING_URL="${HANGING_URL:-http://127.0.0.1:6010}"
SCENE_PROFILE="${SCENE_PROFILE:-gallery}"
VLM_MODE="${VLM_MODE:-auto}"
SAM_ENABLE_FORM="${SAM_ENABLE_FORM:-1}"
ARTWORK_VLM_ENABLE="${ARTWORK_VLM_ENABLE:-1}"
CAMERA_JSON="${CAMERA_JSON-}"
if [ -z "$CAMERA_JSON" ]; then CAMERA_JSON="{}"; fi
INCLUDE_XYZ="${INCLUDE_XYZ:-false}"
INCLUDE_CANDIDATES="${INCLUDE_CANDIDATES:-true}"
SEMANTIC_PARALLEL="${SEMANTIC_PARALLEL:-true}"
API_KEY="${API_KEY:-}"

AUTH_ARGS=()
if [ -n "$API_KEY" ]; then
  AUTH_ARGS=(-H "Authorization: Bearer $API_KEY")
fi

# Per-call rules override: inject artwork dimensions if env vars are set
RULES_FIELD="${RULES_JSON}"
if [ -n "${ARTWORK_WIDTH_M:-}" ] && [ -n "${ARTWORK_HEIGHT_M:-}" ]; then
  RULES_TMP="$OUT_DIR/rules_call.json"
  python3 - "$RULES_JSON" "$RULES_TMP" "$ARTWORK_WIDTH_M" "$ARTWORK_HEIGHT_M" <<'PY'
import json, sys
rules_file, out, w, h = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
with open(rules_file, "r", encoding="utf-8") as f:
    rules = json.load(f)
rules.setdefault("artwork", {})["physical_width_m"] = float(w)
rules.setdefault("artwork", {})["physical_height_m"] = float(h)
with open(out, "w", encoding="utf-8") as f:
    json.dump(rules, f, ensure_ascii=False, indent=2)
PY
  RULES_FIELD="$RULES_TMP"
fi

mkdir -p "$OUT_DIR"

if [[ "$SEMANTIC_PARALLEL" =~ ^(1|true|yes|on)$ ]]; then
  echo "[step 1: parallel, calls 1-2/4] Metric3D /analyze_geometry + Semantic seed /parse_semantics_mvp"
  METRIC_LOG="$OUT_DIR/metric3d.log"
  SEMANTIC_SEED_LOG="$OUT_DIR/semantic_seed.log"
  curl -fsS -X POST "$METRIC_URL/analyze_geometry" \
    "${AUTH_ARGS[@]}" \
    -F "file=@${ROOM_IMAGE}" \
    -F "camera_json=${CAMERA_JSON}" \
    -F "include_xyz=${INCLUDE_XYZ}" \
    -F "include_candidates=${INCLUDE_CANDIDATES}" \
    -o "$OUT_DIR/room_metric3d.zip" >"$METRIC_LOG" 2>&1 &
  METRIC_PID=$!

  curl -fsS -X POST "$SEMANTIC_URL/parse_semantics_mvp" \
    "${AUTH_ARGS[@]}" \
    -F "room_image=@${ROOM_IMAGE}" \
    -F "artwork_image=@${ARTWORK_IMAGE}" \
    -F "artwork_vlm_enable=${ARTWORK_VLM_ENABLE}" \
    -F "scene_profile=${SCENE_PROFILE}" \
    -F "vlm_mode=${VLM_MODE}" \
    -F "sam_enable=${SAM_ENABLE_FORM}" \
    -F 'return_debug_assets=false' \
    -o "$OUT_DIR/semantics_seed.json" >"$SEMANTIC_SEED_LOG" 2>&1 &
  SEMANTIC_SEED_PID=$!

  if ! wait "$METRIC_PID"; then
    echo "[error] Metric3D request failed" >&2
    cat "$METRIC_LOG" >&2 || true
    exit 1
  fi
  if ! wait "$SEMANTIC_SEED_PID"; then
    echo "[error] semantic seed request failed" >&2
    cat "$SEMANTIC_SEED_LOG" >&2 || true
    exit 1
  fi

  echo "[step 2: align, call 3/4] Semantic seed -> Metric3D processed coordinates + geometry postprocess"
  # Alignment call consumes semantic_seed_json and intentionally skips VLM/SAM,
  # so vlm_mode and sam_enable are not sent here.
  curl -fsS -X POST "$SEMANTIC_URL/parse_semantics_mvp" \
    "${AUTH_ARGS[@]}" \
    -F "room_image=@${ROOM_IMAGE}" \
    -F "metric3d_zip=@${OUT_DIR}/room_metric3d.zip" \
    -F "semantic_seed_json=<${OUT_DIR}/semantics_seed.json" \
    -F "scene_profile=${SCENE_PROFILE}" \
    -F 'return_debug_assets=false' \
    -o "$OUT_DIR/semantics.json"
else
  echo "[1/3] Metric3D geometry"
  curl -fsS -X POST "$METRIC_URL/analyze_geometry" \
    "${AUTH_ARGS[@]}" \
    -F "file=@${ROOM_IMAGE}" \
    -F "camera_json=${CAMERA_JSON}" \
    -F "include_xyz=${INCLUDE_XYZ}" \
    -F "include_candidates=${INCLUDE_CANDIDATES}" \
    -o "$OUT_DIR/room_metric3d.zip"

  echo "[2/3] VLM semantics + optional SAM masks"
  curl -fsS -X POST "$SEMANTIC_URL/parse_semantics_mvp" \
    "${AUTH_ARGS[@]}" \
    -F "room_image=@${ROOM_IMAGE}" \
    -F "metric3d_zip=@${OUT_DIR}/room_metric3d.zip" \
    -F "artwork_image=@${ARTWORK_IMAGE}" \
    -F "artwork_vlm_enable=${ARTWORK_VLM_ENABLE}" \
    -F "scene_profile=${SCENE_PROFILE}" \
    -F "vlm_mode=${VLM_MODE}" \
    -F "sam_enable=${SAM_ENABLE_FORM}" \
    -F 'return_debug_assets=false' \
    -o "$OUT_DIR/semantics.json"
fi

if [[ "$SEMANTIC_PARALLEL" =~ ^(1|true|yes|on)$ ]]; then
  echo "[step 3: hanging, call 4/4] Hanging V1 geometry selection"
else
  echo "[3/3] Hanging V1 geometry selection"
fi
curl -fsS -X POST "$HANGING_URL/analyze_hanging_mvp" \
  "${AUTH_ARGS[@]}" \
  -F "room_image=@${ROOM_IMAGE}" \
  -F "metric3d_zip=@${OUT_DIR}/room_metric3d.zip" \
  -F "artwork_image=@${ARTWORK_IMAGE}" \
  -F "semantics_json=<${OUT_DIR}/semantics.json" \
  -F "rules_json=<${RULES_FIELD}" \
  -o "$OUT_DIR/hanging_v1_result.zip"

rm -rf "$OUT_DIR/result"
mkdir -p "$OUT_DIR/result"
unzip -q -o "$OUT_DIR/hanging_v1_result.zip" -d "$OUT_DIR/result"
echo "[done] $OUT_DIR"
find "$OUT_DIR/result" -maxdepth 3 -type f | sort
