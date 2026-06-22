#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/opt/molink/hanging}"
ROOM="${ROOM:-/root/room-exif.jpg}"
METRIC_ZIP="${METRIC_ZIP:-}"
SEMANTICS_JSON="${SEMANTICS_JSON:-}"
ARTWORK="${ARTWORK:-/root/artwork.png}"
PROVIDER="${PROVIDER:-bfl}"  # bfl by default; mock requires ALLOW_MOCK=1
RUN_ID="${RUN_ID:-room-exif_bfl_$(date +%Y%m%d_%H%M%S)}"
HANGING_RUNS_ROOT="${HANGING_RUNS_ROOT:-/root/autodl-fs/hanging_runs}"
OUT_ROOT="${OUT_ROOT:-${HANGING_RUNS_ROOT}/${RUN_ID}}"
HANGING_ZIP="$OUT_ROOT/hanging_v1_result.zip"
FLUX_OUT="$OUT_ROOT/flux_result"

cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
[ -f "$ROOT/.env.bfl" ] && source "$ROOT/.env.bfl"
PROVIDER="${PROVIDER:-bfl}"
PROVIDER_LOWER="$(printf '%s' "$PROVIDER" | tr '[:upper:]' '[:lower:]')"
RED=$'\033[31m'
RESET=$'\033[0m'
if [ "$PROVIDER_LOWER" = "mock" ] || [ "$PROVIDER_LOWER" = "dry_run" ] || [ "$PROVIDER_LOWER" = "noop" ]; then
  if [ "${ALLOW_MOCK:-0}" != "1" ]; then
    printf '%s\n' "${RED}[ERR] Refusing provider=$PROVIDER in production test script. Set ALLOW_MOCK=1 only for explicit dry-run diagnostics.${RESET}" >&2
    exit 64
  fi
  printf '%s\n' "${RED}🔴 WARNING: PROVIDER=$PROVIDER is a MOCK/NO-OP dry run. Results are copied inputs, not BFL outputs.${RESET}" >&2
fi

mkdir -p "$OUT_ROOT" "$FLUX_OUT" "$ROOT/logs" "$ROOT/run"
if [ "$PROVIDER_LOWER" = "mock" ] || [ "$PROVIDER_LOWER" = "dry_run" ] || [ "$PROVIDER_LOWER" = "noop" ]; then
  printf '%s\n' "🔴 WARNING: PROVIDER=$PROVIDER is MOCK/NO-OP. Do not interpret final images as BFL success or failure." > "$OUT_ROOT/MOCK_PROVIDER_WARNING.txt"
fi

echo "[INFO] ROOT=$ROOT"
echo "[INFO] ROOM=$ROOM"
echo "[INFO] METRIC_ZIP=$METRIC_ZIP"
echo "[INFO] SEMANTICS_JSON=$SEMANTICS_JSON"
echo "[INFO] ARTWORK=$ARTWORK"
echo "[INFO] PROVIDER=$PROVIDER"
echo "[INFO] OUT_ROOT=$OUT_ROOT"

test -f "$ROOM" || { echo "[ERR] missing room image: $ROOM"; exit 2; }
test -f "$ARTWORK" || { echo "[ERR] missing artwork: $ARTWORK"; exit 2; }

echo "[INFO] restart all MVP APIs from /opt"
bash "$ROOT/scripts/manage_all_mvp_apis.sh" restart || bash "$ROOT/manage_all_mvp_apis.sh" restart

MVP_HEALTH_ENDPOINTS=(
  "Metric3D|http://127.0.0.1:6006/health"
  "Semantic|http://127.0.0.1:6011/health"
  "Hanging|http://127.0.0.1:6010/health"
)

wait_for_mvp_health() {
  local endpoint name url ok
  for endpoint in "${MVP_HEALTH_ENDPOINTS[@]}"; do
    name="${endpoint%%|*}"
    url="${endpoint#*|}"
    ok=0
    echo "[INFO] waiting for ${name} health: ${url}"
    for i in $(seq 1 60); do
      if curl -fsS "$url" >/dev/null 2>&1; then
        ok=1
        break
      fi
      sleep 1
    done
    if [ "$ok" != "1" ]; then
      echo "[ERR] ${name} service failed health check after 60s: ${url}" >&2
      exit 3
    fi
  done
}

write_mvp_health_snapshot() {
  local endpoint name url slug
  echo "[INFO] MVP API health snapshot"
  for endpoint in "${MVP_HEALTH_ENDPOINTS[@]}"; do
    name="${endpoint%%|*}"
    url="${endpoint#*|}"
    slug="$(printf '%s' "$name" | tr '[:upper:]' '[:lower:]')"
    echo "[INFO] ${name} health"
    if [ "$slug" = "hanging" ]; then
      curl -fsS "$url" | tee "$OUT_ROOT/health_${slug}.json" "$OUT_ROOT/health.json" | python3 -m json.tool || true
    else
      curl -fsS "$url" | tee "$OUT_ROOT/health_${slug}.json" | python3 -m json.tool || true
    fi
  done
}

wait_for_mvp_health
write_mvp_health_snapshot

# Generate metric3d zip from room image when not provided (raw/web-image mode)
if [ -z "$METRIC_ZIP" ]; then
  METRIC_ZIP="$OUT_ROOT/room_metric3d_raw.zip"
  echo "[INFO] generating metric3d zip from $ROOM (raw/web-image mode, no EXIF)"
  curl -fS -X POST http://127.0.0.1:6006/analyze_geometry \
    -F "file=@${ROOM}" \
    -F 'camera_json={}' \
    -F 'include_candidates=true' \
    -o "$METRIC_ZIP"
  echo "[INFO] metric3d zip: $METRIC_ZIP ($(du -h "$METRIC_ZIP" | cut -f1))"
fi
test -f "$METRIC_ZIP" || { echo "[ERR] missing metric zip: $METRIC_ZIP"; exit 2; }

# Generate semantics from room image when not provided
if [ -z "$SEMANTICS_JSON" ]; then
  SEMANTICS_JSON="$OUT_ROOT/semantics_auto.json"
  echo "[INFO] generating semantics from $ROOM (auto mode)"
  curl -fS -X POST http://127.0.0.1:6011/parse_semantics_mvp \
    -F "room_image=@${ROOM}" \
    -F "metric3d_zip=@${METRIC_ZIP}" \
    -F 'scene_profile=gallery' \
    -F 'vlm_mode=auto' \
    -o "$SEMANTICS_JSON" 2>/dev/null || {
    echo "[WARN] semantic service unavailable, proceeding without semantics"
    SEMANTICS_JSON=""
  }
fi
if [ -n "$SEMANTICS_JSON" ] && [ -f "$SEMANTICS_JSON" ]; then
  echo "[INFO] semantics: $SEMANTICS_JSON"
fi

python3 - <<'PY'
import json, urllib.request
with urllib.request.urlopen('http://127.0.0.1:6010/health', timeout=5) as r:
    data = json.load(r)
contract = data.get('flux', {}).get('downstream_contract', '')
print('[INFO] downstream_contract:', contract)
if 'flux_inputs/items' not in contract:
    raise SystemExit('[ERR] hanging service is not patched or not started from /opt/metric3d_api')
PY

if [ -f "$SEMANTICS_JSON" ]; then
  SEM_FIELD=(-F "semantics_json=<${SEMANTICS_JSON};type=application/json")
else
  SEM_FIELD=(-F 'semantics_json={}')
fi

if [ -f "$ROOT/hanging_mvp/rules_mvp.json" ]; then
  RULES_FILE="${ROOT}/hanging_mvp/rules_mvp.json"
elif [ -f "$ROOT/configs/rules_v1.json" ]; then
  RULES_FILE="${ROOT}/configs/rules_v1.json"
else
  RULES_FILE=""
fi

# Artwork dimensions are required inputs
if [ -z "${ARTWORK_WIDTH_M:-}" ] || [ -z "${ARTWORK_HEIGHT_M:-}" ]; then
  echo "[ERR] ARTWORK_WIDTH_M and ARTWORK_HEIGHT_M environment variables are required"
  exit 1
fi

if [ -n "$RULES_FILE" ]; then
  RULES_TMP="$OUT_ROOT/rules_override.json"
  python3 - "$RULES_FILE" "$RULES_TMP" \
    "$ARTWORK_WIDTH_M" "$ARTWORK_HEIGHT_M" <<'PY'
import json, sys
rules_file, out, w, h = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
with open(rules_file, "r", encoding="utf-8") as f:
    rules = json.load(f)
rules.setdefault("artwork", {})["physical_width_m"] = float(w)
rules.setdefault("artwork", {})["physical_height_m"] = float(h)
with open(out, "w", encoding="utf-8") as f:
    json.dump(rules, f, ensure_ascii=False, indent=2)
PY
  RULES_FIELD=(-F "rules_json=<${RULES_TMP};type=application/json")
  echo "[INFO] artwork dimensions: width=${ARTWORK_WIDTH_M}m height=${ARTWORK_HEIGHT_M}m"
else
  RULES_FIELD=(-F 'rules_json={}')
fi

echo "[INFO] call /analyze_hanging_mvp"
curl -fS -X POST http://127.0.0.1:6010/analyze_hanging_mvp \
  -F "room_image=@${ROOM}" \
  -F "metric3d_zip=@${METRIC_ZIP}" \
  -F "artwork_image=@${ARTWORK}" \
  "${SEM_FIELD[@]}" \
  "${RULES_FIELD[@]}" \
  -o "$HANGING_ZIP"

echo "[OK] hanging zip: $HANGING_ZIP"
ls -lh "$HANGING_ZIP"

echo "[INFO] inspect flux export"
unzip -l "$HANGING_ZIP" | grep -E 'flux_inputs|manifest.json|hard_paste|remove_mask|art_region_mask' | head -240 || true
unzip -Z1 "$HANGING_ZIP" 2>/dev/null | grep -c '^flux_inputs/manifest\.json$' >/dev/null || { echo '[ERR] missing flux_inputs/manifest.json'; exit 4; }

echo "[INFO] run image_editing pipeline: provider=$PROVIDER"
python3 "$ROOT/scripts/run_bfl_hanging_flux_pipeline.py" \
  --hanging-zip "$HANGING_ZIP" \
  --artwork-reference "$ARTWORK" \
  --output-dir "$FLUX_OUT" \
  --provider "$PROVIDER" \
  --continue-on-error

if [ "$PROVIDER_LOWER" = "mock" ] || [ "$PROVIDER_LOWER" = "dry_run" ] || [ "$PROVIDER_LOWER" = "noop" ]; then
  printf '%s\n' "${RED}🔴 WARNING: completed with PROVIDER=$PROVIDER MOCK/NO-OP. See $FLUX_OUT/edit_trace.json and $OUT_ROOT/MOCK_PROVIDER_WARNING.txt.${RESET}" >&2
fi

echo "[DONE]"
echo "HANGING_ZIP=$HANGING_ZIP"
echo "FLUX_OUT=$FLUX_OUT"
find "$FLUX_OUT" -maxdepth 4 -type f | sort
