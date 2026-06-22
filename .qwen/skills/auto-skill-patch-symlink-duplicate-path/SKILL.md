---
name: patch-symlink-duplicate-path
description: Deploy patches that reference both a git symlink and its target file — git apply rejects duplicate hunks because the symlink resolves to already-patched content
source: auto-skill
extracted_at: '2026-06-18T13:20:31.391Z'
---

# Patch Deployment with Symlink Duplicate Paths

## Problem

When a patch file contains two separate sections for the same physical file — one targeting a **git-tracked symlink** (e.g., `a/app.py`) and one targeting its **real target** (e.g., `a/services/metric3d_api/app.py`) — `git apply --check` fails even though the changes are valid:

```
error: patch failed: app.py:4
error: app.py: patch does not apply
```

**Root cause:** `git apply` resolves the symlink to the real file content. The first matching hunk (whichever `git apply` processes first) applies cleanly and modifies the real file. The second hunk targeting the other path now finds already-patched content and rejects.

## Diagnosis

1. Check if the rejected file is a git symlink:
   ```bash
   ls -la app.py                    # shows -> services/metric3d_api/app.py
   git ls-tree HEAD app.py          # shows 120000 mode (symlink)
   ```

2. Confirm both paths point to the same content:
   ```bash
   diff app.py services/metric3d_api/app.py   # empty = identical
   ```

3. Use `git apply --reject` to see which hunks succeed vs fail:
   ```
   Applied patch services/metric3d_api/app.py cleanly.
   Rejected hunk #1. (app.py)
   ```

## Resolution

1. Apply with `--reject`:
   ```bash
   cd /opt/metric3d_api
   git apply --reject /path/to/patch.patch
   ```

2. Clean up `.rej` files — they are expected artifacts when the symlink hunk was already applied via the target:
   ```bash
   rm /opt/metric3d_api/*.rej
   ```

3. Verify the real target file contains all expected changes:
   ```bash
   grep -n "new_function\|new_import" services/metric3d_api/app.py
   ```

4. Since the symlink resolves to the target file, the changes are already live through both paths. No manual merge needed.

## Verification checklist

- [ ] `git apply --reject` applied all non-symlink-file hunks cleanly
- [ ] Real target file contains all new functions/imports
- [ ] Symlink still points to the correct target (`ls -la`)
- [ ] `.rej` files removed (they contain expected duplicates, not real conflicts)

## Why not `--3way` or `--fuzz`?

`git apply --3way` attempts a three-way merge against the index, which still fails because the index entry for the symlink path is the symlink blob (not file content). `--fuzz` won't help because the context matches exactly — it's the same content being patched twice, not a context drift issue.

## General rule

When deploying patches against repos with git-tracked symlinks:
- Patches that target **both** the symlink and its real path will partially reject
- This is **expected behavior**, not a baseline divergence issue
- Apply with `--reject`, clean `.rej` files, verify the real file
