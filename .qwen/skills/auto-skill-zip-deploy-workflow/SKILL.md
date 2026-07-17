---
name: zip-deploy-workflow
description: Deploy a Molink GPU release zip to /opt/molink/hanging — audit changes, backup, rsync deploy, restart services, verify health
source: auto-skill
extracted_at: '2026-07-16T04:02:29.491Z'
---

## Molink GPU Zip Deployment Workflow

### 1. Stage and Audit

```bash
# Extract to staging
rm -rf /root/autodl-tmp/qwen_metric3d/artifacts/tmp/deploy_staging
mkdir -p /root/autodl-tmp/qwen_metric3d/artifacts/tmp/deploy_staging
unzip -o "/root/<release-zip>" -d /root/autodl-tmp/qwen_metric3d/artifacts/tmp/deploy_staging/

# Audit changes against current deployment
STAGING="/root/autodl-tmp/qwen_metric3d/artifacts/tmp/deploy_staging/Molink-GPU-v1.1-main"
diff -rq --exclude='.qwen' --exclude='data_disk_*' --exclude='__pycache__' --exclude='.git' --exclude='*.pyc' --exclude='.env*' --exclude='logs' --exclude='run' --exclude='*.bak*' /opt/molink/hanging "$STAGING/"
```

**Key insight:** If diff shows no code changes (only .env, logs, bak files), the zip is identical to current deployment — confirm with user before proceeding.

### 2. Backup Current源码

```bash
BACKUP_DIR="/root/autodl-tmp/qwen_metric3d/artifacts/tmp/deploy_backups"
mkdir -p "$BACKUP_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
rsync -a --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='.env*' --exclude='logs' --exclude='run' --exclude='data_disk_*' --exclude='*.bak*' /opt/molink/hanging/ "$BACKUP_DIR/pre_deploy_$TIMESTAMP/"
```

### 3. Deploy via rsync

```bash
# Stop services first
bash /opt/molink/hanging/ops/molink-autostart.sh stop

# Deploy (preserve local .env files, logs, git metadata)
rsync -av --exclude='.qwen' --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='.env' --exclude='.env.bfl' --exclude='.env.vlm_sam' --exclude='.env.local' --exclude='logs' --exclude='run' --exclude='data_disk_*' --exclude='*.bak*' --exclude='.gitignore' --exclude='.gitattributes' "$STAGING/" /opt/molink/hanging/
```

### 4. Check Dependencies

```bash
diff /opt/molink/hanging/requirements.txt "$STAGING/requirements.txt"
diff /opt/molink/hanging/requirements-sam.txt "$STAGING/requirements-sam.txt"
```

If identical, skip pip install. If changed, run `pip install -r requirements.txt`.

### 5. Restart Services

```bash
bash /opt/molink/hanging/ops/molink-autostart.sh start
sleep 10
# Verify health
curl -s --connect-timeout 3 http://localhost:6006/health  # metric3d
curl -s --connect-timeout 3 http://localhost:6011/health  # semantic
curl -s --connect-timeout 3 http://localhost:6010/health  # hanging

# Worker is separate
bash /opt/molink/hanging/worker/start_worker.sh &
```

### 6. Verify All Services

```bash
ps aux | grep -E 'metric3d|semantic|hanging|worker.main' | grep -v grep
# Expected ports: 6006 (metric3d), 6011 (semantic), 6010 (hanging), worker has no HTTP port
```

### Port Mapping

| Service   | Port | Health Endpoint |
|-----------|------|-----------------|
| metric3d  | 6006 | /health |
| semantic  | 6011 | /health |
| hanging   | 6010 | /health |
| worker    | N/A  | WebSocket only |

### Important Notes

- **Do NOT use supervisorctl** for molink services — they run via direct bash scripts, not the AutoDL system supervisord.
- **Worker runs separately** from the three MVP services — use `worker/start_worker.sh`, not the molink-autostart.sh script.
- **Preserve local configs** during rsync: .env files, .git/, logs/, data_disk_* symlinks.
- **Backup goes to data disk** (`/root/autodl-tmp/qwen_metric3d/artifacts/tmp/deploy_backups/`), not system disk.
- **Clean up staging** after successful deployment: `rm -rf /root/autodl-tmp/qwen_metric3d/artifacts/tmp/deploy_staging`.
