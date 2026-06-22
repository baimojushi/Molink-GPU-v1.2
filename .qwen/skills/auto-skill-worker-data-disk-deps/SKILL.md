---
name: worker-data-disk-deps
description: Install Python worker/service dependencies to data disk via pip --target, then wire PYTHONPATH in startup scripts to avoid system disk exhaustion
source: auto-skill
extracted_at: '2026-06-18T05:03:25.926Z'
---

## Problem

Installing Python packages system-wide or in the project root consumes system disk space. On this machine the system disk `/` is only 50GB with overlay filesystem — heavy packages (boto3, httpx, websockets, etc.) should go to data disk `/root/autodl-tmp/`.

## Solution

### Step 1: Install dependencies to data disk

```bash
mkdir -p /root/autodl-tmp/qwen_metric3d/artifacts/packages/<service>_deps
python3 -m pip install --target /root/autodl-tmp/qwen_metric3d/artifacts/packages/<service>_deps <package1> <package2> ...
```

Verify import:
```bash
PYTHONPATH=/root/autodl-tmp/qwen_metric3d/artifacts/packages/<service>_deps python3 -c "import pkg1; import pkg2; print('OK')"
```

### Step 2: Wire PYTHONPATH in startup script

In the service's `start_<service>.sh`, add before the exec/python3 call:

```bash
# <service> dependencies installed to data disk
<service>_DEPS_DIR="${<service>_DEPS_DIR:-/root/autodl-tmp/qwen_metric3d/artifacts/packages/<service>_deps}"
export PYTHONPATH="${<service>_DEPS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
```

The env var override `<service>_DEPS_DIR` allows relocating without editing the script.

### Step 3: Load .env file in startup script

If the service uses a `.env` file (not using python-dotenv), source it in the startup script:

```bash
# Load <service> environment variables
if [ -f "$REPO_ROOT/<service>/.env.<service>" ]; then
  set -a
  . "$REPO_ROOT/<service>/.env.<service>"
  set +a
fi
```

**Important:** Values containing spaces must be quoted in the .env file (e.g., `GPU_MODEL="TITAN Xp"` not `GPU_MODEL=TITAN Xp`), since `set -a` exports all subsequent variable assignments literally.

## Key files

| Path | Purpose |
|---|---|
| `/root/autodl-tmp/qwen_metric3d/artifacts/packages/v7_worker_deps/` | Current worker dependency installation directory |
| `worker/start_worker.sh` | Startup script with PYTHONPATH + .env.worker sourcing |
| `worker/.env.worker` | Runtime environment (gitignored by `.env.*` rule) |
| `worker/.env.worker.example` | Template (committed) |

## Verification

```bash
# Check deps exist
ls /root/autodl-tmp/qwen_metric3d/artifacts/packages/v7_worker_deps/

# Check import with PYTHONPATH
PYTHONPATH=/root/autodl-tmp/qwen_metric3d/artifacts/packages/v7_worker_deps python3 -c "import websockets; import httpx; import boto3; print('OK')"

# Check startup script syntax
bash -n worker/start_worker.sh
```
