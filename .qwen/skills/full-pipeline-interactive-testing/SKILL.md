---
name: full-pipeline-interactive-testing
description: Build interactive CLI scripts that chain dependent MVP services (semantic → metric3d → hanging) for production testing, with automatic intermediate artifact handling
source: auto-skill
extracted_at: '2026-06-07T04:11:07.375Z'
---

# Full-Pipeline Interactive Testing

## Pattern

When a downstream service (hanging) depends on upstream outputs (semantic objects, metric3d zip), build a single script that:

1. Validates all required services are healthy before starting.
2. Calls upstream services in dependency order, saving intermediates to `data_disk_tmp/qwen/` with timestamps.
3. Extracts and transforms upstream responses into the format expected by downstream services.
4. Passes all inputs to the final service and pretty-prints the result.
5. Keeps artifacts in tmp for later inspection; does not pollute the project root.

## Script template

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/opt/metric3d_api}"
TMP_DIR="$ROOT/data_disk_tmp/qwen"
mkdir -p "$TMP_DIR"

# 1. Accept explicit args, reject missing
INPUT="${1:-}"
[ -z "$INPUT" ] && { echo "Usage: $0 <input_path>"; exit 1; }
[ ! -f "$INPUT" ] && { echo "[error] not found: $INPUT"; exit 1; }

# 2. Check service health before calling
check_port() {
    python3 -c "
import http.client, sys
try:
    c = http.client.HTTPConnection('127.0.0.1', \$1, timeout=5)
    c.request('GET', '/health')
    r = c.getresponse()
    print('[ok] port \$1: HTTP ' + str(r.status))
except Exception as e:
    print('[error] port \$1 unreachable: ' + str(e))
    sys.exit(1)
"
}

# 3. Call services in order, save intermediates with timestamp
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
UPSTREAM_OUT="$TMP_DIR/upstream_${TIMESTAMP}.json"
MIDDLE_ZIP="$TMP_DIR/middle_${TIMESTAMP}.zip"

# upstream → save JSON
curl -s -o "$UPSTREAM_OUT" -X POST "http://127.0.0.1:6011/parse_semantics_mvp" \
    -F "room_image=@${INPUT}" -F "vlm_mode=auto"

# extract and transform for downstream
EXTRACTED="$(python3 -c "
import json
with open('$UPSTREAM_OUT') as f:
    data = json.load(f)
print(json.dumps({'objects': data.get('objects', [])}))
")"

# middle → save zip (binary)
curl -s -o "$MIDDLE_ZIP" -X POST "http://127.0.0.1:6006/analyze_geometry" \
    -F "file=@${INPUT}" -F "camera_json={}"

# 4. Final service call with all inputs
curl -s -X POST "http://127.0.0.1:6010/analyze_hanging_mvp" \
    -F "room_image=@${INPUT}" \
    -F "metric3d_zip=@${MIDDLE_ZIP}" \
    -F "artwork_image=@${ART_PATH}" \
    -F "semantics_json=${EXTRACTED}" \
    -F "rules_json={}" | python3 -m json.tool

# 5. Report artifact locations
echo "[done] intermediates in $TMP_DIR"
```

## Key decisions

- **Intermediates go to `data_disk_tmp/qwen/`**, not project root, per file storage spec.
- **Timestamp filenames** prevent collision across repeated test runs.
- **`set -euo pipefail`** catches failures early — no silent partial results.
- **Health checks first** — fail fast if a dependency is down, rather than wasting time on upstream calls that downstream will reject.
- **JSON extraction via inline Python** — reliable for nested key extraction without requiring `jq`.
- **`-w "%{http_code}"` with `-o`** — captures HTTP status separately from response body for binary outputs (zip files).

## Hanging pipeline specifics

The hanging MVP API requires three inputs that must be coordinated:

| Input | Source | How to obtain |
|---|---|---|
| `room_image` | User provides | Direct `-F "room_image=@path"` |
| `artwork_image` | User provides | Direct `-F "artwork_image=@path"` |
| `metric3d_zip` | metric3d service | Call `/analyze_geometry` with room image |
| `semantics_json` | semantic service | Call `/parse_semantics_mvp` with room image, extract `objects` |

Order: semantic first (fastest), then metric3d (heaviest), then hanging (consumes both).

## Script location

Place in `scripts/` under project root: `scripts/test_hanging_interactive.sh`.
Make executable: `chmod +x scripts/test_hanging_interactive.sh`.
