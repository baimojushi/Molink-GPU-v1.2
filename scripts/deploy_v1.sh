#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/opt/molink/hanging}"
BRANCH="${1:-main}"
cd "$ROOT"

mkdir -p "$ROOT/logs" "$ROOT/run"

echo "[deploy] fetch origin/$BRANCH"
git fetch origin "$BRANCH"

LOCAL_SHA="$(git rev-parse HEAD)"
REMOTE_SHA="$(git rev-parse origin/$BRANCH)"

if [ "$LOCAL_SHA" = "$REMOTE_SHA" ]; then
  echo "[deploy] already up to date: $LOCAL_SHA"
  exit 0
fi

echo "[deploy] update $LOCAL_SHA -> $REMOTE_SHA"
echo "$LOCAL_SHA" > "$ROOT/.last_good_sha"

# Stash uncommitted changes (e.g. .env files with real keys) before checkout
HAS_STASH=false
if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
  echo "[deploy] stashing local changes before checkout"
  git stash push -m "deploy_v1 pre-checkout stash" -- .env.bfl.example .env.vlm_sam .env.vlm_sam.example 2>/dev/null || git stash push -m "deploy_v1 pre-checkout stash"
  HAS_STASH=true
fi

git checkout "$REMOTE_SHA"

if [ "$HAS_STASH" = true ]; then
  echo "[deploy] restoring stashed changes after checkout"
  git stash pop --index || true
fi

echo "[deploy] install dependencies"
pip install -r requirements.txt
pip install -e .

echo "[deploy] syntax check"
python -m py_compile semantic_mvp/service.py
python -m py_compile hanging_mvp/service.py

if [ -f services/metric3d_api/app.py ]; then
  python -m py_compile services/metric3d_api/app.py
fi

if [ -f services/metric3d_api/metric3d_cli.py ]; then
  python -m py_compile services/metric3d_api/metric3d_cli.py
fi

echo "[deploy] restart APIs"
if [ -x "$ROOT/manage_all_mvp_apis.sh" ]; then
  bash "$ROOT/manage_all_mvp_apis.sh" restart
elif [ -x "$ROOT/scripts/manage_all_mvp_apis.sh" ]; then
  bash "$ROOT/scripts/manage_all_mvp_apis.sh" restart
else
  echo "[deploy][warn] manage_all_mvp_apis.sh not found; skip restart"
fi

echo "[deploy] health check"
sleep 3

python - <<'PY'
import http.client

checks = [
    ("metric3d", 6006, "/health", False),
    ("semantic", 6011, "/health", True),
    ("hanging", 6010, "/health", True),
]

failed = []
for name, port, path, required in checks:
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
        conn.request("GET", path)
        resp = conn.getresponse()
        body = resp.read(200).decode("utf-8", "ignore")
        print(name, port, resp.status, body[:120])
        if required and resp.status >= 500:
            failed.append(name)
    except Exception as e:
        print(name, port, "DOWN", repr(e))
        if required:
            failed.append(name)

if failed:
    raise SystemExit("required health check failed: " + ",".join(failed))
PY

echo "[deploy] ok: $REMOTE_SHA"
