#!/usr/bin/env bash
set -euo pipefail

# Persistent production root on AutoDL. Keep the served code under /opt, not /root.
ROOT="${ROOT:-/opt/molink/hanging}"
CONDA_ENV="${CONDA_ENV:-metric3d}"
HANGING_MVP_PORT="${HANGING_MVP_PORT:-6010}"

if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
  source /root/miniconda3/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

if command -v conda >/dev/null 2>&1; then
  conda activate "$CONDA_ENV" || true
fi

cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

# Optional BFL/VLM env. The hanging service does not call BFL, but the exit gate
# may reuse VLM env vars and downstream scripts share the same env file.
if [ -f "$ROOT/.env.bfl" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/.env.bfl"
fi

# Local dev: allow unauthenticated requests and set API_KEY from VLM_API_KEY
if [ -f "$ROOT/.env" ]; then
  source "$ROOT/.env"
fi
export ALLOW_UNAUTH_LOCAL="${ALLOW_UNAUTH_LOCAL:-1}"
if [ -z "${API_KEY:-}" ] && [ -n "${VLM_API_KEY:-}" ]; then
  export API_KEY="$VLM_API_KEY"
fi

exec python -m uvicorn hanging_mvp.service:app --host 0.0.0.0 --port "$HANGING_MVP_PORT"
