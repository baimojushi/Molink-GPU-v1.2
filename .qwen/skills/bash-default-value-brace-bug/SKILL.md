---
name: bash-default-value-brace-bug
description: Bash ${VAR:-{}} produces {}} when VAR is already set to {} — always quote or use empty default for JSON literals
source: auto-skill
extracted_at: '2026-06-07T11:21:09.005Z'
---

## Problem

In bash, `${VAR:-{}}` produces `{}}` (trailing extra `}`) when `VAR` is already set to `{}`:

```bash
CAMERA_JSON="{}" bash -c 'echo "${CAMERA_JSON:-{}}"'
# → {}}   (WRONG)

CAMERA_JSON="" bash -c 'echo "${CAMERA_JSON:-{}}"'
# → {}    (correct, default applied)
```

This happens because bash parameter expansion with `${VAR:-default}` does not quote or escape the default value. When `VAR` is set, bash returns `VAR` concatenated with the unmatched closing `}` from the syntax itself.

## Affected pattern

```bash
# BUG: produces {}} when CAMERA_JSON is already {}
CAMERA_JSON="${CAMERA_JSON:-{}}"
```

This was discovered in `run_v1_full_pipeline.sh` line 14. It caused HTTP 400 from Metric3D's `/analyze_geometry` endpoint because the server received `camera_json={}}` which is invalid JSON.

## Safe alternatives

**Option 1 — empty default (recommended when empty value is valid server-side):**
```bash
CAMERA_JSON="${CAMERA_JSON:-}"
```

**Option 2 — conditional assignment:**
```bash
: "${CAMERA_JSON:={}}"    # works correctly
```

**Option 3 — explicit if:**
```bash
if [ -z "${CAMERA_JSON}" ]; then
    CAMERA_JSON="{}"
fi
```

**Option 4 — backslash escape (does NOT work for the already-set case):**
```bash
CAMERA_JSON="${CAMERA_JSON:-\{\}}"  # produces \{} when set, {} when unset
```

## Diagnosis

When a bash script passes JSON-like parameters to curl/APIs and the server returns 400 for "invalid JSON", check the shell variable expansion:

```bash
# Trace the actual value being sent
CAMERA_JSON="{}" bash -x /path/to/script.sh 2>&1 | grep camera_json
# Look for: -F 'camera_json={}}'  ← the extra } is the bug
```

## Rule for this project

Never use `${VAR:-{}}` or any `${VAR:-...}` with `{}` as the default. Use `${VAR:-}` (empty default) or explicit conditional assignment.
