---
name: patch-zip-audit-against-repo
description: Systematic file-by-file comparison of a packaged patch zip against the current git repository, avoiding reliance on self-reported summary JSONs.
source: auto-skill
extracted_at: '2026-06-07T07:05:00.000Z'
---

# Patch Zip Audit Against Repository

## Problem

A patch zip ships with a `summary.json` listing its own "changed files", but this self-report can omit newly added files or modifications outside its declared scope. Trusting the summary alone yields an incomplete audit.

## Procedure

### 1. Extract to a safe temp location

```bash
PATCH_DIR="/root/autodl-tmp/qwen_metric3d/artifacts/packages/<name>"
unzip -o "$PATCH_DIR/<name>.zip" -d "$PATCH_DIR/<name>"
# Patch usually has a nested subdir:
PATCH_ROOT=$(find "$PATCH_DIR/<name>" -maxdepth 2 -type d -name "<name>" | head -1)
```

### 2. Broad comparison with `diff --brief -r`

For an initial overview, run a recursive brief diff:

```bash
PROJECT="/opt/metric3d_api"
diff --brief -r "$PROJECT/" "$PATCH_ROOT/" --exclude='.git' 2>&1 | sort
```

Filter out runtime noise to surface real changes:

```bash
grep -E '^(Files|Only in)' | grep -v '__pycache__' | grep -v '.patch_backups' | grep -v '.last_good_sha' | grep -v 'data_disk_' | grep -v 'settings.json.orig' | grep -v '.qwen/.env' | grep -v 'lastfailed' | grep -v '.bak'
```

**Why:** `diff --brief -r` is faster than a per-file loop for broad comparison. The filter removes local runtime artifacts that will always differ (env files, caches, backups, symlink targets).

### 3. File-by-file scan (alternative, more granular)

```bash
PROJECT="/opt/metric3d_api"
for f in $(cd "$PATCH_ROOT" && find . -type f | sort); do
  rel="${f#./}"
  # Skip secrets, git internals, qwen config
  case "$rel" in
    .env.*|.env|.qwen/*|.git*|*.bak.*) continue ;;
  esac
  if [ ! -f "$PROJECT/$rel" ]; then
    echo "[NEW] $rel"
    continue
  fi
  if ! diff -q "$PROJECT/$rel" "$PATCH_ROOT/$rel" >/dev/null 2>&1; then
    echo "[MOD] $rel"
  fi
done
```

### 4. Diff each flagged file

For each `[NEW]` or `[MOD]` entry, run:

```bash
diff "$PROJECT/$rel" "$PATCH_ROOT/$rel"   # for scripts/docs
wc -l "$PATCH_ROOT/$rel"                   # for large files, then head/tail sections
```

### 5. Read the summary JSON for context

```bash
cat "$PATCH_ROOT/*-summary.json"
```

Use the summary to understand **why** changes were made, but do **not** use it as the authoritative list of what changed.

### 6. Report by functional group

Group findings into logical buckets (e.g., "interactive CLI", "parallel pipeline", "semantic seed+align", "health check fix", "service manager filter") rather than listing raw filenames.

## Key rules

- **Never trust self-reported changelogs** as the complete list of modifications.
- **Always enumerate** every file in the patch, compare against the repo, and flag differences.
- **Skip secrets** (`.env.*`, `.qwen/*`, `.git*`, `*.bak.*`) to avoid noise.
- For long Python files, use `wc -l` + `head`/`tail` sections rather than reading the entire file.
- Report **functional impact**, not just diffs.
- **Zip loses symlinks**: root-level files that are symlinks in the repo (e.g. `start_all_mvp_apis.sh` → `scripts/start_all_mvp_apis.sh`) become plain text files containing the symlink target path. Do not treat these as modified content.
