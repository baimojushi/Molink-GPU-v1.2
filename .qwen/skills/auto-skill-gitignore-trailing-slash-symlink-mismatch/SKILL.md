---
name: gitignore-trailing-slash-symlink-mismatch
description: Git .gitignore patterns with trailing slashes do not match symlinks — add slashless duplicates to exclude symlink entries
source: auto-skill
extracted_at: '2026-06-25T01:25:00Z'
---

## Problem

`.gitignore` entries like `data_disk_outputs/` only match directories. If `data_disk_outputs` is a symlink (mode `120000`), `git add -A` will stage it as a new file because git does not consider the trailing-slash pattern to match a symlink.

## Symptom

```
git add -A
git status --short | grep data_disk
A  data_disk_backups
A  data_disk_outputs
...
```

These appear as "new file" entries with mode `120000` (symlink) despite having slash-suffixed `.gitignore` entries.

## Fix

Duplicate each slash-suffixed pattern with a slashless variant:

```gitignore
data_disk_outputs/
data_disk_outputs
data_disk_backups/
data_disk_backups
```

The slashless pattern matches both regular files and symlinks; the slash-suffixed pattern still matches directories.

## Prevention

When adding `.gitignore` entries for paths that could be either directories or symlinks, always add both forms.
