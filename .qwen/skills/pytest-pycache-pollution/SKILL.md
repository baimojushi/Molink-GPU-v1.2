---
name: pytest-pycache-pollution
description: Fix pytest import file mismatch caused by __pycache__ from scratch/temp directories under project root
source: auto-skill
extracted_at: '2026-06-06T08:49:02.821Z'
---

## Problem

When extracting patch packages or any Python files to scratch directories that reside under the project root (e.g., `data_disk_tmp/scratch/` which is a symlink to `/root/autodl-tmp/tmp/scratch`), pytest encounters `import file mismatch` errors:

```
import file mismatch:
imported module 'test_something' has this __file__ attribute:
  /opt/metric3d_api/data_disk_tmp/scratch/.../tests/test_something.py
which is not the same as the test file we want to collect:
  /opt/metric3d_api/tests/test_something.py
HINT: remove __pycache__ / .pyc files and/or use a unique basename
```

This happens because Python caches `.pyc` files with absolute path references. When the same module name exists in both the scratch directory and the project's actual test directory, pytest's module discovery loads the wrong cached version.

## Symptoms

- `pytest` fails with `ERROR collecting tests/test_*.py` for ALL new test files from a patch
- `import file mismatch` pointing to a scratch/temp path under the project root
- Cleaning `__pycache__` in the project's `tests/` dir does NOT fix it

## Fix

### Step 1 — Identify scratch/temp dirs under project root

Check for directories like `data_disk_tmp/scratch/`, `data_disk_tmp/qwen/`, or any temp extraction dirs that exist as subdirectories of the project root:

```bash
ls -la /opt/metric3d_api/data_disk_tmp/
```

### Step 2 — Remove scratch dirs from project scope

```bash
rm -rf /opt/metric3d_api/data_disk_tmp/scratch
rm -rf /root/autodl-tmp/tmp/scratch
```

### Step 3 — Clean all __pycache__ in project

```bash
find /opt/metric3d_api -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find /opt/metric3d_api -name "*.pyc" -delete 2>/dev/null
```

### Step 4 — Rerun pytest

```bash
cd /opt/metric3d_api
PYTHONPATH=/root/autodl-tmp/python_envs/site-packages:$PYTHONPATH python3 -m pytest -q
```

## Prevention

- **Never extract packages to scratch dirs under project root**. Use `/root/autodl-tmp/tmp/scratch/` directly (outside project root), not via `data_disk_tmp/scratch/` symlink.
- If extraction must happen under project root, clean `__pycache__` immediately after:
  ```bash
  find /opt/metric3d_api -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
  ```

## Key details

- pytest module: located at `/root/autodl-tmp/python_envs/site-packages/`
- Must use `PYTHONPATH=/root/autodl-tmp/python_envs/site-packages:$PYTHONPATH` or `python3` with pytest in PATH
- `data_disk_tmp` is a symlink from `/opt/metric3d_api/data_disk_tmp/` → `/root/autodl-tmp/tmp/`
- Any subdirectory created under `data_disk_tmp/` becomes visible to pytest as a project subdirectory
