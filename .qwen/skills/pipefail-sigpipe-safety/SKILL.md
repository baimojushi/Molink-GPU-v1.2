---
name: pipefail-sigpipe-safety
description: Avoid SIGPIPE false-positives in bash scripts with set -euo pipefail
source: auto-skill
extracted_at: '2026-06-04T06:21:20.833Z'
---

# Pipefail SIGPIPE Safety

## Problem

Under `set -euo pipefail`, this pattern causes a false-positive failure:

```bash
unzip -l file.zip | grep -q 'expected_file.txt' || { echo 'missing'; exit 1; }
```

`grep -q` exits immediately after first match. The upstream `unzip -l` continues writing to a broken pipe, receives `SIGPIPE` (exit code 141). `pipefail` takes the maximum non-zero exit code from the pipeline, so 141 is treated as failure even though the file was found.

## Fix

Replace `grep -q` with `grep -c` (counts matches, reads all input before exiting):

```bash
unzip -Z1 file.zip 2>/dev/null | grep -c '^expected_file\.txt$' >/dev/null || { echo 'missing'; exit 1; }
```

`grep -c` consumes all upstream input before exiting, avoiding SIGPIPE.

## Alternative

Write upstream output to a temp variable first:

```bash
listing=$(unzip -Z1 file.zip)
echo "$listing" | grep -q 'expected_file.txt'
```

## Affected scripts

- `scripts/run_full_hanging_bfl_test_opt.sh` — fixed grep -q to grep -c

## When to check

Any bash script with `set -euo pipefail` that pipes a producer (unzip, find, ls, etc.) into `grep -q`.
