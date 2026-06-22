---
name: patch-deployment-with-baseline-divergence
description: Deploy patch packages when git apply fails due to repo baseline divergence
source: auto-skill
extracted_at: '2026-06-04T08:30:00.000Z'
---

# Patch Deployment with Baseline Divergence

## When git apply fails

Patch packages are built against a specific repo baseline. When the repo has accumulated commits since that baseline, `git apply` will fail due to context mismatch.

## Step 0 — Check if patch is already applied

Before any backup or copy, verify if the patch content is already in the codebase. This avoids redundant work when a patch was previously deployed but the request recurs.

Extract unique marker strings from the patch (function names, config keys, env vars, error messages) and check each:

```bash
cd /opt/molink/hanging
# Example markers from a FLUX.2 mask patch:
markers=(
  "def build_flux2_edit_mask"
  "\"flux2_mask_required\": True"
  "BFL_FLUX2_MASK_FIELD"
  "BFL_FLUX2_REQUIRE_MASK"
  "mask=art_mask,"
  "def _qa_hard_block_reasons"
  "flux2_edit_mask_path"
  "\"mask_sent_to_flux2\""
)
for m in "${markers[@]}"; do
  count=$(grep -rl "$m" --include="*.py" . 2>/dev/null | wc -l)
  echo "$m: found in $count file(s)"
done
```

If **all** markers are present at expected locations:
- Run `python3 -m py_compile` on the patched files to confirm no syntax errors.
- Restart the affected service(s) — the code is already deployed, just needs a reload.
- Skip all remaining steps.

If any marker is missing, proceed to Step 1.

## Step 1 — Backup

```bash
BACKUP_DIR="/opt/molink/hanging/data_disk_backups/qwen/hanging_deploy.$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp -a /opt/molink/hanging/hanging_mvp /opt/molink/hanging/image_editing /opt/molink/hanging/scripts/run_bfl_hanging_flux_pipeline.py "$BACKUP_DIR/"
```

## Step 2 — Try git apply first

If the target is a git repository:

```bash
cd /opt/molink/hanging
git apply --check /path/to/molink_gpu_flux2_mask_required.patch 2>&1
```

If `git apply` succeeds, apply it. If it fails (context mismatch) or the target is **not a git repo** (`git rev-parse --show-toplevel` fails), proceed to the changed_files approach.

## Step 3 — Diff patch targets vs repo

For raw `.patch` files, extract the target file paths from the `diff --git` headers, then compare:

```bash
# Extract target paths from patch
grep "^diff --git" /root/molink_gpu_flux2_mask_required.patch | awk '{print $NF}'
# Output: hanging_mvp/flux_export.py, hanging_mvp/service.py, etc.

# For each target, check diff line count
for f in hanging_mvp/flux_export.py hanging_mvp/service.py image_editing/pipeline.py image_editing/providers/bfl.py scripts/run_bfl_hanging_flux_pipeline.py; do
  if [ -f "$f" ]; then
    echo "EXISTS: $f ($(wc -l < "$f") lines)"
  else
    echo "MISSING: $f"
  fi
done
```

## Step 4 — Marker verification before copy

Check if critical patch markers already exist in the repo:

```bash
grep -c "key_marker_string" /opt/metric3d_api/path/to/file.py
```

If markers are already present, the feature may already be deployed. If absent, the changed_files version is needed.

## Step 5 — Copy changed files, preserve repo-only updates

Apply the patch directly via `git apply` (if git repo) or use `patch` command for individual files:

```bash
# Option A: git apply (if git repo)
cd /opt/molink/hanging && git apply /root/molink_gpu_flux2_mask_required.patch

# Option B: patch command (non-git repo)
cd /opt/molink/hanging && patch -p1 < /root/molink_gpu_flux2_mask_required.patch
```

For config files that the repo may have independently updated:
1. Read both versions
2. Check if repo already contains all patch fields
3. If yes, skip copy; if no, merge missing fields

**Zip symlink corruption**: `zip` does not preserve symlinks — symlink entries become regular files containing the target path as text. After extracting a patch zip, verify that root-level files which should be symlinks are not corrupted:

```bash
# If app.py, metric3d_cli.py, or warmup_giant.py contain only a path string like
# "services/metric3d_api/app.py" or "hanging_mvp/service.py", they are corrupted symlink placeholders.
# Restore them from the pre-deploy backup:
cp "$BACKUP_DIR/app.py" /opt/molink/hanging/app.py
```

Alternatively, copy from the internal `services/` or `hanging_mvp/` directories if those were not part of the patch.

## Step 6 — Verify after copy

```bash
# Syntax check
cd /opt/molink/hanging
python3 -m py_compile hanging_mvp/flux_export.py hanging_mvp/service.py image_editing/pipeline.py image_editing/providers/bfl.py scripts/run_bfl_hanging_flux_pipeline.py

# Marker verification
python3 -c "
from pathlib import Path
checks = {
    'feature_a': ('path/to/file.py', 'unique_marker_string'),
    'feature_b': ('configs/rules_v1.json', 'config_key'),
}
missing = []
for name, (path, needle) in checks.items():
    if needle not in Path(path).read_text(encoding='utf-8', errors='ignore'):
        missing.append(f'{name}: {path} lacks {needle!r}')
if missing:
    raise SystemExit('\n'.join(missing))
print('[OK] marker verification passed')
"
```

## Step 7 — Run focused tests

```bash
python3 -m pytest tests/test_bfl_two_provider_contract.py tests/test_flux_export_v7_policy.py
```

## safe_update_from_clean_zip.sh deployment

Use `scripts/safe_update_from_clean_zip.sh` for deploying packaged patch zips (not raw patches).

```bash
PYTHON=python3 ALLOW_UNAUTH_LOCAL=1 bash scripts/safe_update_from_clean_zip.sh /path/to/patch.zip /opt/metric3d_api
```

**Critical variables:**
- `PYTHON=python3` — the script uses `python` by default; override because only `python3` is available at `/root/autodl-tmp/qwen_metric3d/bin/python3`.
- `ALLOW_UNAUTH_LOCAL=1` — the staged `preflight_check.py` now requires `API_KEY` or `ALLOW_UNAUTH_LOCAL=1`. Without this, preflight fails during local deployment.

**What the script does (8 steps):**
1. Unpack zip to `/tmp/artwork_hanging_update_<timestamp>`
2. Normalize line endings, permissions, and `rules_mvp.json` symlink
3. Preflight: rules JSON parse, hanging endpoint present, python syntax, auth check, pytest, bash syntax, JSON parse
4. Stop current services via `manage_all_mvp_apis.sh stop`
5. Backup current deployment to `/opt/metric3d_api_backup_<timestamp>` (excludes logs, run, outputs)
6. rsync staged code into target (preserves `.env`, `.env.*`, `.git/`, logs, run, hanging_runs, outputs)
7. Restart services via `manage_all_mvp_apis.sh restart`
8. Health check ports 6006, 6011, 6010 — auto-rollback on failure

**Post-deployment cleanup (always do):**
```bash
# Delete the zip package
rm -f /path/to/deployed_patch.zip

# Delete temp staging dirs
rm -rf /tmp/artwork_hanging_update_*

# Delete backup (after verifying services are stable)
rm -rf /opt/metric3d_api_backup_<timestamp>

# Remove stale files the audit identified (e.g., .bak env files, historical manifests)
rm -f /opt/metric3d_api/.env.*.bak.* /opt/metric3d_api/*_manifest.json
```

**After safe_update overwrites scripts/, re-apply the PYTHON fix:**
The zip may contain the original `safe_update_from_clean_zip.sh` without the `PYTHON` variable fix. After rsync, verify and re-apply if needed:

```bash
grep -q 'PYTHON=' /opt/metric3d_api/scripts/safe_update_from_clean_zip.sh || echo "PYTHON fix missing — re-apply"
```

If missing, add `PYTHON="${PYTHON:-python3}"` after `TARGET=` line and replace all bare `python` with `$PYTHON`.

**Commit the deployment:**
```bash
cd /opt/metric3d_api && git add -A && git commit -m "fix(audit): maintenance audit cleanup — <summary>"
```

## Key principles

- Always backup before deploying
- Try git apply first; fall back to changed_files copy
- Verify markers after copy to confirm features are present
- Preserve repo-only updates (newer config fields, independent features)
- Run py_compile and focused tests before declaring success
- safe_update_from_clean_zip.sh requires PYTHON=python3 and ALLOW_UNAUTH_LOCAL=1
- Always clean up zip, temp dirs, backups, and stale files after successful deployment
