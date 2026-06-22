#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/opt/molink/hanging}"

###############################################################################
# Usage: run_full_hanging_pipeline.sh ROOM_IMAGE ARTWORK_IMAGE [OUT_DIR] [FLAGS...]
#
# One-command chain:
#   1. Start required MVP APIs only by default: Metric3D → Semantic → Hanging
#   2. Run v1 full pipeline
#   3. Optionally run image edit pipeline
#
# Flags:
#   --skip-bfl              skip image edit stage
#   --bfl-limit N           process only first N render manifest items
#   --dual-ab-compare       add the extra image-edit A/B render; default is off to save queue time
#   --continue-on-error     continue image-edit item loop on item errors
#   --no-start-services     assume APIs are already running
###############################################################################

usage() {
  echo "Usage: $0 ROOM_IMAGE ARTWORK_IMAGE [OUT_DIR] [--skip-bfl] [--bfl-limit N] [--dual-ab-compare] [--continue-on-error] [--no-start-services]"
}

if [ $# -lt 2 ]; then
  usage
  exit 2
fi

ROOM_IMAGE="$1"
ARTWORK_IMAGE="$2"
shift 2

if [ $# -gt 0 ] && [[ "${1:-}" != --* ]]; then
  OUT_DIR="$1"
  shift
else
  HANGING_RUNS_ROOT="${HANGING_RUNS_ROOT:-/root/autodl-fs/hanging_runs}"
  OUT_DIR="${HANGING_RUNS_ROOT}/room_full_$(date +%Y%m%d_%H%M%S)"
fi

RUN_BFL=true
BFL_LIMIT=""
CONTINUE_ON_ERROR=false
START_SERVICES=true
DUAL_AB_COMPARE=false

while [ $# -gt 0 ]; do
  case "$1" in
    --skip-bfl)
      RUN_BFL=false
      shift
      ;;
    --bfl-limit)
      BFL_LIMIT="${2:?--bfl-limit requires a value}"
      shift 2
      ;;
    --dual-ab-compare)
      DUAL_AB_COMPARE=true
      shift
      ;;
    --continue-on-error)
      CONTINUE_ON_ERROR=true
      shift
      ;;
    --no-start-services)
      START_SERVICES=false
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[error] unknown flag: $1" >&2
      usage
      exit 2
      ;;
  esac
done

mkdir -p "$OUT_DIR"
echo "=========================================="
echo "  Full Hanging Pipeline"
echo "  room:     $ROOM_IMAGE"
echo "  artwork:  $ARTWORK_IMAGE"
echo "  out_dir:  $OUT_DIR"
echo "  image_edit: $RUN_BFL"
echo "  dual_ab:  $DUAL_AB_COMPARE"
echo "=========================================="

###############################################################################
# Stage 0: Start required MVP services
###############################################################################
if [ "$START_SERVICES" = true ]; then
  echo ""
  echo "[0/3] Starting required MVP services"
  MVP_SERVICES="metric3d,semantic,hanging" bash "$ROOT/scripts/manage_all_mvp_apis.sh" start > "$OUT_DIR/00_services_start.log" 2>&1
  echo "[ok:services] required services started or already running"
else
  echo ""
  echo "[0/3] Skipping service startup (--no-start-services)"
fi

# Health wait loop
echo "[0/3] Waiting for health checks..."
for attempt in $(seq 1 30); do
  if python3 - <<'PY'
import http.client, sys
ok = 0
for port in [6006, 6011, 6010]:
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
        conn.request("GET", "/health")
        resp = conn.getresponse()
        resp.read(200)
        if resp.status == 200:
            ok += 1
    except Exception:
        pass
sys.exit(0 if ok == 3 else 1)
PY
  then
    echo "[ok:services] all three API ports healthy"
    break
  fi
  if [ "$attempt" -eq 30 ]; then
    echo "[error:services] API health check failed after 30 attempts" >&2
    exit 1
  fi
  sleep 2
done

###############################################################################
# Stage 1: v1 full pipeline (Metric3D → Semantic → Hanging)
###############################################################################
echo ""
echo "[1/3] Running v1 full pipeline (Metric3D → Semantic → Hanging)"
bash "$ROOT/scripts/run_v1_full_pipeline.sh" "$ROOM_IMAGE" "$ARTWORK_IMAGE" "$OUT_DIR" 2>&1 | tee -a "$OUT_DIR/01_v1_pipeline.log"
echo "[ok:v1] pipeline complete"

###############################################################################
# Stage 2: GPT Image 2 VIP / BFL image edit pipeline
###############################################################################
HANGING_ZIP="$OUT_DIR/hanging_v1_result.zip"
if [ ! -f "$HANGING_ZIP" ]; then
  echo "[error] hanging_v1_result.zip not found at $HANGING_ZIP" >&2
  exit 1
fi

if [ "$RUN_BFL" = false ]; then
  echo ""
  echo "[2/3] Skipping image edit stage (flag --skip-bfl)"
else
  echo ""
  echo "[2/3] Running image edit pipeline"

  BFL_ENV="$ROOT/.env.bfl"
  if [ ! -f "$BFL_ENV" ]; then
    echo "[warn] .env.bfl not found at $BFL_ENV; skipping image edit stage"
  else
    set -a
    # shellcheck disable=SC1090
    source "$BFL_ENV"
    set +a

    BFL_DIR="$OUT_DIR/flux_result_apiyi_gpt_image2_vip"
    BFL_ARGS=(
      --hanging-zip "$HANGING_ZIP"
      --output-dir "$BFL_DIR"
      --provider "${PROVIDER:-apiyi_gpt_image2_vip}"
    )
    if [ "$CONTINUE_ON_ERROR" = true ]; then
      BFL_ARGS+=(--continue-on-error)
    fi
    if [ -n "$BFL_LIMIT" ]; then
      BFL_ARGS+=(--limit "$BFL_LIMIT")
    fi
    if [ "$DUAL_AB_COMPARE" = true ]; then
      BFL_ARGS+=(--dual-ab-compare)
      if [ -n "$BFL_LIMIT" ]; then
        BFL_ARGS+=(--dual-ab-compare-limit "$BFL_LIMIT")
      fi
    fi

    python3 "$ROOT/scripts/run_bfl_hanging_flux_pipeline.py" \
      "${BFL_ARGS[@]}" 2>&1 | tee -a "$OUT_DIR/03_image_edit.log"
    echo "[ok:image-edit] pipeline complete"
  fi
fi

###############################################################################
# Summary
###############################################################################
echo ""
echo "=========================================="
echo "  Pipeline Complete"
echo "  out_dir: $OUT_DIR"
echo "=========================================="
echo ""
echo "Output files:"
find "$OUT_DIR" -maxdepth 4 -type f \( -name "*.png" -o -name "*.json" -o -name "*.zip" \) | sort | head -60
echo ""

python3 - "$OUT_DIR" <<'PY'
import json, os, sys, zipfile
out = sys.argv[1]
print("=== Quick Summary ===")
hanging_zip = os.path.join(out, "hanging_v1_result.zip")
if os.path.exists(hanging_zip):
    with zipfile.ZipFile(hanging_zip) as z:
        try:
            m = json.loads(z.read("flux_inputs/manifest.json"))
        except KeyError:
            m = {"items": []}
        print(f"Candidates: {len(m.get('items', []))}")
        for item in m.get("items", []):
            score = item.get("score", 0)
            try:
                score_text = f"{float(score):.3f}"
            except Exception:
                score_text = str(score)
            print(f"  {item.get('asset_id')}: score={score_text}")

bfl_trace = os.path.join(out, "flux_result_apiyi_gpt_image2_vip", "edit_trace.json")
if os.path.exists(bfl_trace):
    t = json.loads(open(bfl_trace, encoding="utf-8").read())
    print(f"image_edit: provider={t.get('provider')}, outputs={len(t.get('outputs', []))}, final_contract={t.get('final_render_contract')}")
    for o in t.get("outputs", []):
        print(f"  {o.get('asset_id')}: QA={o.get('render_size_qa_status', '?')}")
PY

echo ""
echo "[done]"
