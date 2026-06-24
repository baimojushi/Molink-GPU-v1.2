---
name: manual-patch-deployment
description: Deploy a .patch file manually via direct file edits when git apply is unavailable (non-git repo or baseline divergence)
source: auto-skill
extracted_at: '2026-06-24T20:32:00Z'
---

## When to use

The target directory is not a git repository, or `git apply` fails due to baseline divergence, but a `.patch` file (or `.patch.zip`) needs to be deployed.

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

## Step 2 — Stash uncommitted changes (if in git repo)

```bash
cd <project_root> && git stash && git apply --stat --check -p<N> <patch> 2>&1
```

Try `--strip=1`, `--strip=2`, etc. until the path prefix resolves. If `git apply --check` still fails with hunk errors, fall back to manual deployment. After the dry-run attempt, restore changes with `git stash pop`.

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

## Step 10 — Restart affected services

```bash
bash manage_all_mvp_apis.sh restart
# Or targeted restart if only one service changed:
pkill -f 'specific.service' && bash start_specific.sh
```

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
