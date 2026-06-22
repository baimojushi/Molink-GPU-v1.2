---
name: interactive-cli-prompt-ordering
description: hanging_mvp_cli.py _interactive_fill must only prompt when args are None/empty, not unconditionally overwrite CLI flags
source: auto-skill
extracted_at: '2026-06-14T06:07:28.851Z'
---

# Interactive CLI Prompt Ordering

`scripts/hanging_mvp_cli.py` has an `_interactive_fill()` function that prompts for missing parameters when `--interactive` is used.

## Root cause: unconditional overwrites

The original implementation unconditionally called `_prompt_text()` / `_prompt_bool()` for every parameter, overwriting values that were already set via CLI flags:

```python
# BUG: always prompts, even if user passed --skip-bfl
args.skip_bfl = not _prompt_bool("...", not args.skip_bfl)
args.camera_json = _prompt_text("...", args.camera_json or "{}")
```

When using heredoc input (e.g., `bash run_hanging_interactive.sh <<'EOF'`), the number of input lines must exactly match the number of prompts. If `_interactive_fill` prompts for more fields than the heredoc provides, input lines shift and wrong values land in wrong fields.

### Real incident

`CAMERA_JSON` received `"configs/rules_v1.json"` instead of `"{}"` because the rules_json prompt consumed the `{}` heredoc line, and the camera_json prompt consumed the next line (which was `gallery`). This caused metric3d to return `400: camera_json must be valid JSON`.

## Correct pattern

Only prompt when the argument is genuinely unset:

```python
def _interactive_fill(args: argparse.Namespace) -> argparse.Namespace:
    if not args.room:
        args.room = _prompt_text("房间图片路径", required=True)
    # ... other required fields ...
    
    # Only prompt if None/empty — respect CLI flags
    if args.camera_json is None or args.camera_json == "":
        args.camera_json = _prompt_text("相机参数 JSON", "{}")
    if not args.scene_profile:
        args.scene_profile = _prompt_text("场景 profile", "gallery")
    if args.sam_enable is None:
        args.sam_enable = _prompt_bool("启用 SAM 精修", True)
    if args.skip_bfl is None:
        args.skip_bfl = not _prompt_bool("运行 BFL/mock 图像编辑阶段", False)
    
    # Sub-prompts only shown when parent condition is True
    if not args.skip_bfl:
        if not args.provider:
            args.provider = _prompt_text("图像编辑 provider bfl/mock", "bfl")
        if args.bfl_limit is None:
            args.bfl_limit = ...
```

## Key rules

1. **String fields**: prompt only if `args.field is None or args.field == ""`
2. **Boolean flags**: prompt only if `args.field is None` (distinguish `False` from unset)
3. **Conditional sub-prompts**: wrap in `if not args.skip_bfl:` — do not prompt for BFL details when BFL is skipped
4. **CLI flag precedence**: `--skip-bfl`, `--no-start-services`, `--vlm-mode=mock` etc. must be respected and not overwritten by prompts

## Prompt count for heredoc testing

With the fix, `--skip-bfl` reduces prompt count significantly:

| Mode | Prompt count | Input lines needed |
|---|---|---|
| Full interactive (no flags) | ~15 | room, artwork, width, height, output_dir, rules_json, camera_json, scene_profile, vlm_mode, sam_enable, artwork_vlm_enable, start_services, skip_bfl, provider, bfl_limit |
| `--skip-bfl` | ~9 | room, artwork, width, height, output_dir, sam_enable, artwork_vlm_enable, start_services, (done) |
| `--no-start-services --skip-bfl` | ~8 | same minus start_services |

## Debugging prompt mismatches

When `--interactive` produces unexpected 400 errors, add a debug print in the `run()` function:

```python
print(f"[DEBUG] CAMERA_JSON in env: {env.get('CAMERA_JSON', 'NOT_SET')!r}")
```

Or inspect the pipeline script's received environment:

```bash
printf "CAMERA_JSON=[%s] len=%d" "$CAMERA_JSON" "${#CAMERA_JSON}" > "$OUT_DIR/debug_env.txt"
```
