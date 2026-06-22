---
name: hex-token-copy-safety
description: Never visually copy long hex tokens, keys, or hashes — always use grep+sed to extract and substitute via shell tools
source: auto-skill
extracted_at: '2026-06-19T02:20:00.000Z'
---

# Long Hex/Token Copy Safety

## The problem

LLM token-level processing cannot reliably copy long strings character-for-character. A 64-character hex key like:

```
f4e43d673a7f6eb6dfc9cf15ea3139147f432702cb3f8afdaefcb66b5ebec784
```

Is seen as multiple tokens. Visually adjacent token fragments (`3d` vs `43d`) can be transposed during generation, producing an invalid key without any obvious sign.

This has been confirmed as a real failure — an R2 secret key was corrupted with one extra character.

## The rule

**Never hand-copy long hex tokens, API keys, base64 strings, or hashes.** Always use shell tools:

```bash
# Extract a value from a source file
grep 'KEY_NAME=' source.env | sed 's/.*=//' | tr -d '"'

# Substitute into target file using sed with the extracted value
CORRECT_VAL=$(grep 'KEY_NAME=' source.env | sed 's/.*=//')
WRONG_VAL="known_wrong_value"
sed -i "s/KEY_NAME=${WRONG_VAL}/KEY_NAME=${CORRECT_VAL}/" target.env
```

## When to apply

- Copying API keys between `.env` files
- Migrating secrets to `/etc/molink/gpu.env`
- Pasting any token longer than ~20 characters
- Any hex string, base64 blob, or JWT

## What IS safe to type

- Short identifiers: `metric3d`, `6006`, `auto`
- Paths: `/opt/molink/hanging`
- Variable names: `MOLINK_CACHE`, `BFL_API_KEY`
- Short values: `1`, `true`, `cuda`

## Verification pattern

After any key migration, verify with a diff or grep comparison:

```bash
# Extract from both files and compare
diff <(grep 'KEY=' source.env | sed 's/.*=//') <(grep 'KEY=' target.env | sed 's/.*=//') && echo "MATCH"
```
