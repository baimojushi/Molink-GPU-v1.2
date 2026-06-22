---
name: env-file-convention
description: Real API keys go in untracked .env files; .env.example are tracked templates with empty values
source: auto-skill
extracted_at: '2026-06-07T14:36:39.317Z'
---

## .env file convention

### File roles

| Pattern | Purpose | Git tracked | Real keys |
|---|---|---|---|
| `.env.bfl` | BFL service reads at runtime | No (`.gitignore`) | Yes |
| `.env.vlm_sam` | Semantic service reads at runtime | Yes (accident) | Yes |
| `.env.*.example` | Templates for new setups | Yes | No |

### Critical rules

- **Never** put real API keys in `.example` files — they are git-tracked and will be exposed or overwritten by `git checkout`.
- **Always** fill real keys in the bare `.env` files (`.env.bfl`, `.env.vlm_sam`).
- Services source the bare `.env` files, not the `.example` versions.
- `.env.vlm_sam` is accidentally tracked (not in `.gitignore`); prefer moving it to `.gitignore` pattern or keeping it but never committing key changes.

### .gitignore relevant lines

```
.env
.env.*
!.env*.example
```

This means `.env`, `.env.bfl`, `.env.vlm_sam` are all ignored. Only `.env.bfl.example`, `.env.vlm_sam.example` etc. are tracked.

### deploy_v1.sh behavior

`deploy_v1.sh` runs `git checkout "$REMOTE_SHA"` during auto-update. Native git checkout behavior:
- Remote does not modify tracked `.env` files → local uncommitted changes preserved
- Remote modifies same tracked file → checkout errors, deploy aborts, local changes preserved
- Untracked files (`.env.bfl`) → never affected

**Conclusion:** Uncommitted API keys are never silently erased by git checkout. The deploy may fail, but keys stay intact.
