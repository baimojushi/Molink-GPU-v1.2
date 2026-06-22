#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/opt/molink/hanging}"
cd "$ROOT"

if [ ! -f "$ROOT/.last_good_sha" ]; then
  echo "[rollback][error] .last_good_sha not found"
  exit 1
fi

SHA="$(cat "$ROOT/.last_good_sha")"
echo "[rollback] checkout $SHA"

git checkout "$SHA"

pip install -r requirements.txt
pip install -e .

if [ -x "$ROOT/manage_all_mvp_apis.sh" ]; then
  bash "$ROOT/manage_all_mvp_apis.sh" restart
elif [ -x "$ROOT/scripts/manage_all_mvp_apis.sh" ]; then
  bash "$ROOT/scripts/manage_all_mvp_apis.sh" restart
fi

echo "[rollback] done"
