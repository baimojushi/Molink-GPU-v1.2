#!/usr/bin/env bash
set -euo pipefail

PROJECT="${QWEN_PROJECT_DIR:-/opt/metric3d_api}"
STACK="${QWEN_DATA_DIR:-/root/autodl-tmp/qwen_metric3d}"

LOG_DIR="$STACK/artifacts/logs"
RUN_DIR="$STACK/run"
LOCK="$RUN_DIR/autocommit.lock"
LOG="$LOG_DIR/autocommit.log"

mkdir -p "$LOG_DIR" "$RUN_DIR"

{
  flock -n 9 || exit 0

  cd "$PROJECT"
  # Keep QWEN.md as data-disk symlink before staging changes.
  if [ -x "$STACK/bin/ensure-qwen-md-link.sh" ]; then
    "$STACK/bin/ensure-qwen-md-link.sh" || true
  fi


  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "$(date -Is) skip: not a git repo"
    exit 0
  fi

  BRANCH="$(git branch --show-current || true)"
  if [ "$BRANCH" != "main" ]; then
    echo "$(date -Is) skip: branch=$BRANCH"
    exit 0
  fi

  "$PROJECT/.qwen/hooks/qwen-organize-artifacts.sh" || true

  git update-index -q --refresh || true

  if [ -z "$(git status --porcelain)" ]; then
    echo "$(date -Is) skip: clean"
    exit 0
  fi

  git add -A

  git reset -q -- .qwen/.env 2>/dev/null || true
  git reset -q -- .env 2>/dev/null || true
  git reset -q -- _qwen_images _qwen_logs _qwen_cache _qwen_packages _qwen_tmp 2>/dev/null || true

  if git diff --cached --quiet; then
    echo "$(date -Is) skip: no staged changes"
    exit 0
  fi

  COMMIT_MSG="qwen-auto: $(date '+%Y-%m-%d %H:%M:%S')"

  git commit -m "$COMMIT_MSG"

  if git remote get-url origin >/dev/null 2>&1; then
    git push origin main
  else
    echo "$(date -Is) committed locally, skip push: no origin remote"
  fi

  echo "$(date -Is) committed: $COMMIT_MSG"
} 9>"$LOCK" >>"$LOG" 2>&1
