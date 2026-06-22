#!/usr/bin/env bash
set -e
CMD=${1:-help}; shift || true
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_DIR=${API_DIR:-$SCRIPT_DIR}
URL=${METRIC3D_URL:-http://127.0.0.1:6006}
case "$CMD" in
  install-giant) bash "$API_DIR/install_giant_auto.sh" ;;
  start) bash "$API_DIR/start_giant_auto.sh" ;;
  health) python "$API_DIR/metric3d_cli.py" --url "$URL" health ;;
  inspect-exif) IMG=${1:?IMAGE required}; CAM=${2:-"{}"}; python "$API_DIR/metric3d_cli.py" --url "$URL" inspect-exif --image "$IMG" --camera-json "$CAM" ;;
  infer) IMG=${1:?IMAGE required}; OUT=${2:-/tmp/molink-jobs/metric3d_auto_exif_result.zip}; python "$API_DIR/metric3d_cli.py" --url "$URL" infer --image "$IMG" --include-candidates --output "$OUT" ;;
  infer-points) IMG=${1:?IMAGE required}; PTS=${2:?POINTS_JSON required}; OUT=${3:-/tmp/molink-jobs/metric3d_auto_exif_points_result.zip}; python "$API_DIR/metric3d_cli.py" --url "$URL" infer --image "$IMG" --points "$PTS" --include-candidates --output "$OUT" ;;
  infer-camera) IMG=${1:?IMAGE required}; CAM=${2:?CAMERA_JSON required}; OUT=${3:-/tmp/molink-jobs/metric3d_auto_exif_camera_result.zip}; python "$API_DIR/metric3d_cli.py" --url "$URL" infer --image "$IMG" --camera-json "$CAM" --include-candidates --output "$OUT" ;;
  meta) ZIP=${1:?ZIP required}; python "$API_DIR/metric3d_cli.py" meta --zip "$ZIP" ;;
  stop) pkill -f "uvicorn app:app" || true ;;
  help|*) cat <<'EOF'
Commands:
  metric3d_ctl.sh install-giant
  metric3d_ctl.sh start
  metric3d_ctl.sh health
  metric3d_ctl.sh inspect-exif IMAGE [CAMERA_JSON]
  metric3d_ctl.sh infer IMAGE [OUT_ZIP]
  metric3d_ctl.sh infer-points IMAGE POINTS_JSON [OUT_ZIP]
  metric3d_ctl.sh infer-camera IMAGE CAMERA_JSON [OUT_ZIP]
  metric3d_ctl.sh meta RESULT_ZIP
  metric3d_ctl.sh stop
EOF
  ;;
esac
