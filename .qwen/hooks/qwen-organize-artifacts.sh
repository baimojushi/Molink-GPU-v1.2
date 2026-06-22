#!/usr/bin/env bash
set -euo pipefail

PROJECT="${QWEN_PROJECT_DIR:-/opt/metric3d_api}"
STACK="${QWEN_DATA_DIR:-/root/autodl-tmp/qwen_metric3d}"

IMG_DIR="$STACK/artifacts/images"
LOG_DIR="$STACK/artifacts/logs"
PKG_DIR="$STACK/artifacts/packages"

mkdir -p "$IMG_DIR" "$LOG_DIR" "$PKG_DIR"

move_if_untracked() {
  local file="$1"
  local dest_dir="$2"

  [ -f "$file" ] || return 0

  local rel="${file#$PROJECT/}"

  if git -C "$PROJECT" ls-files --error-unmatch "$rel" >/dev/null 2>&1; then
    return 0
  fi

  mkdir -p "$dest_dir"
  mv -n "$file" "$dest_dir/"
}

find "$PROJECT" -maxdepth 1 -type f \( \
  -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.webp" -o -iname "*.gif" -o -iname "*.svg" \
\) | while read -r f; do
  move_if_untracked "$f" "$IMG_DIR"
done

find "$PROJECT" -maxdepth 1 -type f \( \
  -iname "*.log" -o -iname "*.out" \
\) | while read -r f; do
  move_if_untracked "$f" "$LOG_DIR"
done

find "$PROJECT" -maxdepth 1 -type f \( \
  -iname "*.whl" -o -iname "*.tar" -o -iname "*.tar.gz" -o -iname "*.tgz" -o -iname "*.zip" -o -iname "*.deb" -o -iname "*.run" \
\) | while read -r f; do
  move_if_untracked "$f" "$PKG_DIR"
done
