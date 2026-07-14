---
name: phase-zip-deploy
description: Deploy a phase release zip to /opt/molink/hanging — audit via sha256sum, copy modified/new files, restart affected services, verify health
source: auto-skill
extracted_at: '2026-07-14T09:44:10.914Z'
---

# Phase Zip Deploy

## When to use

A phase release zip ships with new code for the hanging subproject (`/opt/molink/hanging`). Deploy by extracting, comparing, copying, and restarting services.

## Step 1 — Extract to safe temp location

```bash
mkdir -p /root/autodl-tmp/tmp/qwen/phase-deploy
unzip -o /path/to/release.zip -d /root/autodl-tmp/tmp/qwen/phase-deploy/extracted
EXTRACT_DIR=$(find /root/autodl-tmp/tmp/qwen/phase-deploy/extracted -maxdepth 1 -type d | grep -v extracted | head -1)
```

## Step 2 — Audit: identify NEW and MODIFIED files

Use sha256sum comparison against repo HEAD:

```bash
cd /opt/molink/hanging
for f in $(find "$EXTRACT_DIR" -type f -not -path '*/.git/*' -not -name '.gitignore' -not -name '.gitattributes' -not -name '.qwenignore' | sed "s|$EXTRACT_DIR/||" | sort); do
  repo_file="/opt/molink/hanging/$f"
  zip_file="$EXTRACT_DIR/$f"
  if [ ! -f "$repo_file" ]; then
    echo "NEW: $f"
  elif [ "$(sha256sum "$zip_file" | cut -d' ' -f1)" != "$(sha256sum "$repo_file" | cut -d' ' -f1)" ]; then
    echo "MODIFIED: $f"
  fi
done
```

Filter out local-only files (QWEN.md, .env, .qwen/, startup scripts not in zip).

## Step 3 — Copy modified files

```bash
for f in $MODIFIED_FILES; do
  cp "$EXTRACT_DIR/$f" "/opt/molink/hanging/$f"
done
```

## Step 4 — Copy new files

```bash
mkdir -p tests hanging_mvp
cp "$EXTRACT_DIR/PHASE*_DELIVERY.md" .
cp "$EXTRACT_DIR/hanging_mvp/new_module.py" hanging_mvp/
cp "$EXTRACT_DIR/tests/test_*.py" tests/
```

## Step 5 — Stop affected services

```bash
pkill -f 'semantic_mvp.service|hanging_mvp.service|worker.main|start_worker.sh' || true
sleep 2
# Verify stopped:
ps aux | grep -E 'semantic_mvp|hanging_mvp|worker.main' | grep -v grep
```

## Step 6 — Restart services

```bash
cd /opt/molink/hanging
# Semantic
bash scripts/start_semantic_v1_sam.sh &
# Hanging
bash scripts/start_hanging_v1.sh &
# Worker (targeted restart, not full manage_all)
bash worker/start_worker.sh &
```

## Step 7 — Verify health

```bash
sleep 5
ps aux | grep -E 'semantic_mvp|hanging_mvp|worker.main' | grep -v grep
curl -s http://127.0.0.1:6011/health
curl -s http://127.0.0.1:6010/health
```

Expected: both return `{"status":"ok",...}`. Worker has no HTTP endpoint — confirm via `ps`.

## Step 8 — Commit and push

```bash
cd /opt/molink/hanging
git add -A
git status  # review
git commit -m "deploy: phaseN release — <summary>"
git push origin main
```

## Key rules

- **sha256sum for comparison** — more precise than diff --brief for binary-safe comparison
- **Skip local-only files** — QWEN.md, .env.*, .qwen/, app.py, start*.sh that are repo-specific
- **Worker restart uses start_worker.sh directly** — not manage_all_mvp_apis.sh (lighter, targeted)
- **Health check semantic :6011 and hanging :6010** — worker has no HTTP health endpoint
- **Clean up temp extract** after successful deploy: `rm -rf /root/autodl-tmp/tmp/qwen/phase-deploy`

## Troubleshooting

### Semantic service fails to restart after pkill
- Port 6011 may still be held by a zombie process. Use `pkill -9 -f 'semantic_mvp'` (SIGKILL) if graceful pkill fails.
- **The startup script's `conda activate` often fails in background shells.** If `bash scripts/start_semantic_v1_sam.sh` doesn't work, use direct metric3d env Python:
  ```bash
  cd /opt/molink/hanging && nohup env SAM_ENABLE=1 SAM_BACKEND=sam1 SAM_MODEL_TYPE=vit_b SAM_CHECKPOINT=/opt/molink-cache/sam/sam_vit_b_01ec64.pth SAM_MAX_IMAGE_SIDE=1536 ALLOW_UNAUTH_LOCAL=1 OMP_NUM_THREADS=4 SEMANTIC_MVP_PORT=6011 /root/miniconda3/envs/metric3d/bin/python -m uvicorn semantic_mvp.service:app --host 0.0.0.0 --port 6011 > logs/semantic_new.log 2>&1 &
  ```
- **Important:** `nohup` PID is the wrapper, the actual uvicorn PID differs. Check `ps aux | grep semantic_mvp` and use that PID for verification.
- `setsid` also works but background shells may kill the session when they exit — `nohup` is more reliable for this use case.
- Log file: `/opt/molink/hanging/logs/semantic.log` (or `semantic_new.log`) — check for "Errno 98" (port in use) or import errors.

### Env var changes not taking effect
- Worker reads `/etc/molink/gpu.env` — changes require `pkill -f 'worker.main'` + `bash worker/start_worker.sh`.
- Semantic/hanging services read env vars at startup — any env var change in gpu.env requires a service restart.
- `SAM_MAX_IMAGE_SIDE` is read by semantic service at startup via the startup script's default export.
