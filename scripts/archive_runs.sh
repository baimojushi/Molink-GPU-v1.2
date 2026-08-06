#!/usr/bin/env bash
# archive_runs.sh —— 将 hanging-runs（scr）中某日的运行产物归档到 MEVAL 长期产物目录。
#
# 用法: archive_runs.sh [YYYYMMDD]
#   DAY 默认取当天；只归档该前缀的 <DAY>_*.zip（由 worker diagnostics 生成）。
#
# 环境变量:
#   MEVAL_ARTIFACT_DIR       归档目标目录（WSL 路径），默认 /mnt/e/项目/art hanging/规模测试/产物
#   MOLINK_HANGING_RUNS_DIR  运行产物源（scr），默认 /var/lib/molink/hanging-runs
#
# 新约定（2026-08-07）: scr 仅作临时暂存，验证/规模测试产物统一归档到产物目录。
set -euo pipefail

SCR="${MOLINK_HANGING_RUNS_DIR:-/var/lib/molink/hanging-runs}"
ART="${MEVAL_ARTIFACT_DIR:-/mnt/e/项目/art hanging/规模测试/产物}"
DAY="${1:-$(date +%Y%m%d)}"

if [[ ! -d "$SCR" ]]; then
  echo "[archive_runs] ERROR: source dir not found: $SCR" >&2
  exit 1
fi

DEST="$ART/${DAY}_验证"
mkdir -p "$DEST"

mapfile -t ZIPS < <(compgen -G "$SCR/${DAY}_*.zip" || true)
if [[ ${#ZIPS[@]} -eq 0 ]]; then
  echo "[archive_runs] no runs archived for ${DAY}: no ${DAY}_*.zip under $SCR"
  exit 1
fi

copied=0
skipped=0
for z in "${ZIPS[@]}"; do
  if cp -n "$z" "$DEST/"; then
    copied=$((copied + 1))
  else
    skipped=$((skipped + 1))
  fi
  echo "[archive_runs] archived: $(basename "$z")"
done

echo "[archive_runs] done: ${copied} copied, ${skipped} skipped -> $DEST"
