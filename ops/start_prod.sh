#!/usr/bin/env bash
set -Eeuo pipefail

source /etc/profile.d/molink-gpu.sh
[ -f /etc/molink/gpu.env ] && set -a && source /etc/molink/gpu.env && set +a

mkdir -p /var/log/molink /tmp/molink-jobs
cd "$MOLINK_APP"

echo "[prod] starting MVP APIs..."
bash scripts/manage_all_mvp_apis.sh restart

echo "[prod] waiting local services..."
for url in \
  http://127.0.0.1:6006/health \
  http://127.0.0.1:6011/health \
  http://127.0.0.1:6010/health
do
  ok=0
  for i in $(seq 1 90); do
    if curl -fsS "$url" >/dev/null; then
      echo "[prod] ok: $url"
      ok=1
      break
    fi
    echo "[prod] waiting: $url"
    sleep 5
  done
  if [ "$ok" != "1" ]; then
    echo "[prod] service failed: $url"
    bash scripts/manage_all_mvp_apis.sh status || true
    exit 20
  fi
done

echo "[prod] starting worker..."
exec python -m worker.main
