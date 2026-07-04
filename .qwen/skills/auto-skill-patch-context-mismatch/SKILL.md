---
name: patch-context-mismatch
description: When patch -p1 fails due to reversed/previously applied hunks or line number drift, manually apply the missing functions and replace target lines using edit tool
source: auto-skill
extracted_at: '2026-07-04T02:45:00.000Z'
---

## Problem

Patch application fails with "Reversed (or previously applied) patch detected" even though the target function/logic does not exist in the current code. This happens when:
1. A previous patch modified the same file with overlapping context
2. Line numbers shifted due to prior edits
3. Import statements were already added by a different patch

## Diagnosis

1. Run `patch -p1 --dry-run` to see which hunks fail
2. Check reject file: `cat file.py.rej` to see what patch tried to apply
3. For each failed hunk, grep the current file to verify:
   - Does the import already exist? (`grep 'import X' file.py`)
   - Does the function already exist? (`grep 'def func_name' file.py`)
   - Is the target line at a different number? (`grep 'target_text' file.py`)

## Manual application procedure

### For new function insertion
1. Find the insertion point (usually before a known function): `grep -n 'def known_function' file.py`
2. Read the surrounding context: `read_file` around that line
3. Use `edit` to insert the new function with exact surrounding context (3 lines before/after)

### For line replacement
1. Find the target line: `grep -n 'old_text' file.py`
2. Read surrounding context: `read_file` around that line
3. Use `edit` to replace, including 3+ lines of context before/after

### For import additions
If import already exists, skip that hunk — it was applied by a previous patch.

## Verification

After manual application:
```bash
python3 -m py_compile file.py
```

## Example (artwork VLM patch)

Patch failed on 3 hunks:
1. Import already existed → skip
2. `_analyze_artwork_image_with_vlm` function missing → manually inserted before `_semantic_light_fields`
3. `artwork_analysis` assignment at line 1078 (patch expected 1069) → manually replaced using `edit`
