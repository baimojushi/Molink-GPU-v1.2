---
name: manual-patch-deployment
description: Deploy a .patch file manually via direct file edits when git apply is unavailable (non-git repo or baseline divergence)
source: auto-skill
extracted_at: '2026-06-24T20:32:00Z'
---

## When to use

The target directory is not a git repository, or `git apply` fails due to baseline divergence, but a `.patch` file (or `.patch.zip`) needs to be deployed.

## Step 0 — Filter cross-repo patches (if applicable)

A patch may contain files from multiple repositories. Before deploying, identify which files belong to the current codebase:

```bash
grep -n "^--- a/\|^+++ b/" patch_file.patch
```

Examine the path prefixes. Files with paths matching other repos (e.g. `Molink-page-v1.2-main/`) should be skipped. Only deploy files whose paths exist under the current project root.

Typical pattern: a patch from a full codebase audit contains `Molink-page-v1.2-main/` (page repo, skip) and `Molink-GPU-v1.1-main/` (GPU repo, deploy). Map the prefix to the current project:

```
Molink-GPU-v1.1-main/scripts/run_bfl_hanging_flux_pipeline.py → scripts/run_bfl_hanging_flux_pipeline.py
Molink-GPU-v1.1-main/worker/main.py                           → worker/main.py
```

Only deploy files that map into the current project structure.

## Step 1 — Locate and read the patch

```bash
find /root /opt -name "<patch_name>.patch" -o -name "<patch_name>.zip" 2>/dev/null
```

If the patch is a `.zip`, extract it to a temp directory and diff against current codebase to identify changed files:

```bash
unzip -o patch.zip -d /tmp/patch_workdir_$$
diff -rq --exclude='.git' --exclude='__pycache__' --exclude='data_disk_*' . /tmp/patch_workdir_$$/<repo-root>/
```

If the patch is a `.patch` file, read it directly to identify affected files and line ranges.

## Step 2 — Find correct -p strip level

Patches generated from absolute or nested paths often need a non-standard strip level. Discover it by incrementally testing:

```bash
cd <project_root>
# Try -p 1 through -p 6 until git apply --check returns exit code 0
for p in 1 2 3 4 5 6; do
  git apply --check -p $p <patch> 2>&1 && echo "SUCCESS with -p $p" && break
  echo "FAILED with -p $p"
done
```

If `git apply --check` fails with "No such file or directory" at a given `-p`, the path prefix still has extra directories — try a higher number. Once `git apply --check` returns exit code 0 (no output), apply the patch with the same `-p` value:

```bash
git apply -p <successful_p> <patch>
```

If `git apply --check` fails with hunk errors (content mismatch) even at the correct strip level, fall back to manual deployment (Step 3+).

## Step 3 — Backup affected files

```bash
BACKUP_DIR="/opt/molink/hanging/data_disk_backups/qwen/hanging_deploy_<feature>.$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp -a <all_affected_files_and_dirs> "$BACKUP_DIR/"
```

Backup every file the patch will touch, not just directories.

## Step 4 — Check marker strings (detect prior deployment)

Grep for unique identifiers the patch would introduce:

```bash
grep -c "unique_marker_string" target_file.py
```

If all markers are already present at correct line locations, the patch is already deployed — skip to service restart.

## Step 5 — Identify file boundaries in patch

```bash
grep -n "^--- a/\|^+++ b/" patch_file.patch
```

This gives line numbers for each file section. New files show as `+++ b/path/to/new_file.py` with `--- /dev/null`.

## Step 6 — Extract new files from patch

For a new file (`+++ b/path/file.py` starting at patch line N, ending at next `---` at line M):

```bash
sed -n 'N+1,M-1p' patch_file.patch | sed 's/^\+//' > target/path/file.py
python3 -m py_compile target/path/file.py  # verify immediately
```

The `sed 's/^\+//'` strips the diff `+` prefix.

## Step 7 — Apply modifications to existing files

Read each target file, then use `edit` operations to replace the old text blocks with the new ones from the patch. Match context precisely:

1. Read the file with `read_file`
2. Use `edit` with `old_string` = the pre-patch content (include 3+ lines of surrounding context)
3. Use `new_string` = the post-patch content

For patches with multiple hunks in the same file, apply each hunk sequentially. Apply changes in dependency order: helper functions first, then callers, then prompt/config changes.

**Fallback: direct copy from patch source.** When `diff` shows many structural differences and piecemeal edits produce too many residual diffs, directly copy the patch versions:

```bash
cp /tmp/patch_workdir_$$/<repo-root>/path/to/changed_file.py /opt/molink/hanging/path/to/changed_file.py
```

Then verify with `diff` that zero differences remain, followed by `py_compile`.

## Step 8 — Syntax check all modified Python files

```bash
python3 -m py_compile file1.py file2.py file3.py 2>&1
```

Fix any syntax errors before proceeding. Common issues from manual edits:
- Mismatched parentheses (count `(` vs `)` on edited lines)
- Missing commas between dict entries
- Incorrect indentation

## Step 9 — Verify marker strings after deployment

```python
from pathlib import Path
checks = {
    'feature_a': ('path/to/file.py', 'unique_marker_string'),
    'feature_b': ('path/to/another.py', 'class NewClassName'),
}
missing = []
for name, (path, needle) in checks.items():
    if needle not in Path(path).read_text(encoding='utf-8', errors='ignore'):
        missing.append(f'{name}: {path} lacks {needle!r}')
if missing:
    raise SystemExit('\n'.join(missing))
print(f'[OK] All {len(checks)} marker verification checks passed')
```

## Step 10 — Add environment variables (if patch introduces new env vars)

Patches that introduce new tunable parameters often expect corresponding environment variables in `/etc/molink/gpu.env` (worker config) rather than project `.env` files. Worker processes load env vars via `set -a; source /etc/molink/gpu.env; set +a` in `start_worker.sh`.

Check the patch `.example` files (e.g. `.env.bfl.example`, `worker/.env.worker.example`) for new env var names and default values. Add them to `/etc/molink/gpu.env` in the appropriate section:

```bash
# Edit /etc/molink/gpu.env — add new vars under the matching comment section
# Example:
# APIYI_GPT_IMAGE_RELOCATION_SCALE_MIN=0.92
# APIYI_GPT_IMAGE_RELOCATION_SCALE_MAX=1.08
# MOLINK_FINAL_POSTPROCESS_MODE=wall_rgb_only
```

After updating `/etc/molink/gpu.env`, restart the worker for the new vars to take effect (Step 11).

## Step 11 — Restart affected services

For general service restart:
```bash
bash manage_all_mvp_apis.sh restart
# Or targeted restart if only one service changed:
pkill -f 'specific.service' && bash start_specific.sh
```

**If the patch modifies worker-executed code** (`image_editing/providers/*.py`, `scripts/run_bfl_hanging_flux_pipeline.py`, `worker/main.py`), restart the GPU worker:

```bash
pkill -f 'worker.main|worker/start_worker.sh' || true
: > /opt/molink/hanging/logs/worker.log
cd /opt/molink/hanging
bash /opt/molink/hanging/worker/start_worker.sh &
# Wait ~3s, then verify:
ps aux | grep -E 'worker.main|start_worker' | grep -v grep
```

The worker runs as a persistent background process and never reloads code automatically. Deploying without restarting means the next job still executes old bytecode.

Verify health endpoints:

```bash
curl -s http://localhost:<port>/health
```

## Key principles

- Always backup before modifying
- Check markers first — patch may already be deployed
- Try `git apply --check -p<N>` with different strip levels before falling back to manual
- Extract new files with `sed 's/^\+//'` to strip diff prefixes
- Apply existing-file changes via precise text replacement
- When piecemeal edits produce too many residual diffs, directly copy from patch source
- Verify syntax (especially parenthesis matching), zero diffs, and markers before restarting services
- Restart services after deployment to load new code
