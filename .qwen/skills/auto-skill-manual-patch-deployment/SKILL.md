---
name: manual-patch-deployment
description: Deploy a .patch file manually via direct file edits when git apply is unavailable (non-git repo or baseline divergence)
source: auto-skill
extracted_at: '2026-06-22T12:31:24.385Z'
---

## When to use

The target directory is not a git repository, or `git apply` fails due to baseline divergence, but a `.patch` file needs to be deployed.

## Step 1 — Locate and read the patch

```bash
find /root /opt -name "<patch_name>.patch" 2>/dev/null
```

Read the full patch content to identify all affected files and line ranges.

## Step 2 — Backup affected files

```bash
BACKUP_DIR="/opt/molink/hanging/data_disk_backups/qwen/hanging_deploy_<feature>.$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp -a <all_affected_dirs> "$BACKUP_DIR/"
```

## Step 3 — Check marker strings (detect prior deployment)

Grep for unique identifiers the patch would introduce:

```bash
grep -c "unique_marker_string" target_file.py
```

If all markers are already present at correct line locations, the patch is already deployed — skip to service restart.

## Step 4 — Identify file boundaries in patch

```bash
grep -n "^--- a/\|^+++ b/" patch_file.patch
```

This gives line numbers for each file section. New files show as `+++ b/path/to/new_file.py` with `--- /dev/null`.

## Step 5 — Extract new files from patch

For a new file (`+++ b/path/file.py` starting at patch line N, ending at next `---` at line M):

```bash
sed -n 'N+1,M-1p' patch_file.patch | sed 's/^\+//' > target/path/file.py
python3 -m py_compile target/path/file.py  # verify immediately
```

The `sed 's/^\+//'` strips the diff `+` prefix.

## Step 6 — Apply modifications to existing files

Read each target file, then use `edit` operations to replace the old text blocks with the new ones from the patch. Match context precisely:

1. Read the file with `read_file`
2. Use `edit` with `old_string` = the pre-patch content (include 3+ lines of surrounding context)
3. Use `new_string` = the post-patch content

For patches with multiple hunks in the same file, apply each hunk sequentially.

## Step 7 — Syntax check all modified Python files

```bash
python3 -m py_compile file1.py file2.py file3.py 2>&1
```

Fix any syntax errors before proceeding.

## Step 8 — Verify marker strings after deployment

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

## Step 9 — Restart affected services

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
- Extract new files with `sed 's/^\+//'` to strip diff prefixes
- Apply existing-file changes via precise text replacement
- Verify syntax and markers before restarting services
- Restart services after deployment to load new code
