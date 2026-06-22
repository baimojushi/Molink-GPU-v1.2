#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/safe_update_from_clean_zip.sh /root/artwork-hanging-mvp-v1-main_optimized_safe_clean_deploy_v2.zip /opt/metric3d_api
#
# This script stages the zip, validates it, backs up the existing deployment,
# rsyncs code while preserving secrets/runtime data, restarts services,
# checks health, and automatically rolls back on failed health checks.

ZIP_PATH="${1:-}"
TARGET="${2:-/opt/metric3d_api}"
PYTHON="${PYTHON:-python3}"

if [ -z "$ZIP_PATH" ] || [ ! -f "$ZIP_PATH" ]; then
  echo "Usage: $0 /path/to/optimized_clean_deploy_v2.zip [/opt/metric3d_api]" >&2
  exit 2
fi

TS="$(date +%Y%m%d_%H%M%S)"
STAGE="/tmp/artwork_hanging_update_${TS}"
BACKUP="/opt/molink/hanging_backup_${TS}"

cleanup() {
  rm -rf "$STAGE" 2>/dev/null || true
}
trap cleanup EXIT

repair_rules_symlink() {
  local repo="$1"
  local link="$repo/hanging_mvp/rules_mvp.json"
  local target="../configs/rules_v1.json"

  if [ -L "$link" ]; then
    local actual
    actual="$(readlink "$link")"
    if [ "$actual" != "$target" ]; then
      echo "[FAIL] $link points to $actual, expected $target" >&2
      exit 3
    fi
    return 0
  fi

  if [ -f "$link" ] && [ "$(cat "$link")" = "$target" ]; then
    echo "[FIX] Replacing plain-text rules_mvp.json pointer with symlink"
    rm -f "$link"
    ln -s "$target" "$link"
    return 0
  fi

  echo "[FAIL] $link must be a symlink to $target" >&2
  exit 3
}

echo "[1/8] Unpack to staging: $STAGE"
mkdir -p "$STAGE"
unzip -oq "$ZIP_PATH" -d "$STAGE"

SRC="$(find "$STAGE" -maxdepth 1 -mindepth 1 -type d | head -n 1)"
if [ -z "$SRC" ] || [ ! -d "$SRC" ]; then
  echo "No repository directory found in zip." >&2
  exit 3
fi

echo "[2/8] Normalize scripts, permissions, and canonical symlink"
find "$SRC" -type f -name "*.sh" -exec sed -i 's/\r$//' {} +
chmod +x "$SRC"/*.sh "$SRC"/scripts/*.sh "$SRC"/services/metric3d_api/*.sh 2>/dev/null || true
repair_rules_symlink "$SRC"

echo "[3/8] Preflight staged repository"
(cd "$SRC" && $PYTHON scripts/preflight_check.py)

if command -v pytest >/dev/null 2>&1; then
  (cd "$SRC" && $PYTHON -m pytest -q)
else
  echo "[WARN] pytest is not installed; skipping optional contract tests on server"
fi

while IFS= read -r shfile; do
  bash -n "$shfile"
done < <(find "$SRC" -type f -name "*.sh")

$PYTHON - <<PY
import json, pathlib
root = pathlib.Path("$SRC")
for p in root.rglob("*.json"):
    json.load(open(p, encoding="utf-8"))
print("[OK] JSON parse passed")
PY

echo "[4/8] Stop current services"
if [ -f "$TARGET/scripts/manage_all_mvp_apis.sh" ]; then
  bash "$TARGET/scripts/manage_all_mvp_apis.sh" stop || true
fi

echo "[5/8] Backup current deployment to: $BACKUP"
if [ -d "$TARGET" ]; then
  rsync -a     --exclude logs     --exclude run     --exclude hanging_runs     --exclude outputs     --exclude "*.log"     "$TARGET/" "$BACKUP/"
else
  mkdir -p "$TARGET"
fi

echo "[6/8] Rsync staged code into target, preserving secrets/runtime"
rsync -a --delete   --exclude ".git/"   --exclude ".env"   --exclude ".env.*"   --exclude ".env.vlm_sam"   --exclude ".env.vlm_sam.bak.*"   --exclude "logs/"   --exclude "run/"   --exclude "hanging_runs/"   --exclude "outputs/"   --exclude "*.log"   "$SRC/" "$TARGET/"

repair_rules_symlink "$TARGET"
find "$TARGET" -type f -name "*.sh" -exec sed -i 's/\r$//' {} +
chmod +x "$TARGET"/*.sh "$TARGET"/scripts/*.sh "$TARGET"/services/metric3d_api/*.sh 2>/dev/null || true

echo "[7/8] Restart services"
bash "$TARGET/scripts/manage_all_mvp_apis.sh" restart
sleep 3
bash "$TARGET/scripts/manage_all_mvp_apis.sh" status || true

echo "[8/8] Health check"
ok=1
for port in 6006 6011 6010; do
  if ! curl -fsS "http://127.0.0.1:${port}/health" >/tmp/health_${port}.txt; then
    echo "[FAIL] port ${port} health failed"
    ok=0
  else
    echo "[OK] port ${port}: $(cat /tmp/health_${port}.txt | head -c 180)"
    echo
  fi
done

if [ "$ok" != "1" ]; then
  echo "[ROLLBACK] Restoring backup: $BACKUP"
  bash "$TARGET/scripts/manage_all_mvp_apis.sh" stop || true
  rsync -a --delete "$BACKUP/" "$TARGET/"
  repair_rules_symlink "$TARGET"
  find "$TARGET" -type f -name "*.sh" -exec sed -i 's/\r$//' {} +
  chmod +x "$TARGET"/*.sh "$TARGET"/scripts/*.sh "$TARGET"/services/metric3d_api/*.sh 2>/dev/null || true
  bash "$TARGET/scripts/manage_all_mvp_apis.sh" restart || true
  echo "[ROLLBACK DONE] Check logs under $TARGET/logs"
  exit 10
fi

echo "[DONE] Update succeeded. Backup kept at: $BACKUP"
