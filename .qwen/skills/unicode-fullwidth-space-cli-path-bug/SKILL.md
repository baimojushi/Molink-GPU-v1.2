---
name: unicode-fullwidth-space-cli-path-bug
description: Python `input().strip()` does not remove full-width space (U+3000) or surrogate escape bytes — causes file-not-found errors with garbled prefix in CLI path input
source: auto-skill
extracted_at: '2026-06-07T20:35:00.000Z'
---

## Problem

When users interactively enter file paths in the hanging MVP CLI, a full-width space ` ` (U+3000) typed before the path is **not removed** by `.strip()`, causing:

```
[error] room image does not exist: \udce3\udc80/root/autodl-tmp/pictures/qwen/room-exif.jpg
```

The `\udce3\udc80` bytes are the surrogate-escape representation of UTF-8 `E3 80 80` (U+3000 full-width space). When terminal encoding is non-standard, UTF-8 bytes decode to **surrogate escape** characters in the U+D800–U+DFFF range.

## Root cause

Python's `str.strip()` only removes ASCII whitespace. The `str.isspace()` method handles Unicode whitespace (U+3000, U+00A0, U+200B, etc.) but **returns `False` for surrogate characters** (U+D800–U+DFFF). So when a full-width space is decoded as surrogate escapes (`\udce3\udc80`), it bypasses both `.strip()` and `.isspace()`.

| Character | Code point | `isspace()` | Source |
|---|---|---|---|
| Full-width space ` ` | U+3000 | True | Chinese IME |
| Surrogate escape `\udce3` | U+DCE3 | **False** | UTF-8 decode error |
| Surrogate escape `\udc80` | U+DC80 | **False** | UTF-8 decode error |

## Fix

Use a shared `_clean_unicode_whitespace()` function that strips both Unicode whitespace **and** surrogate range characters from both ends:

```python
def _clean_unicode_whitespace(text: str) -> str:
    """Strip ASCII whitespace, Unicode whitespace, and surrogate escapes from both ends."""
    start = 0
    while start < len(text):
        c = text[start]
        if c.isspace() or (0xD800 <= ord(c) <= 0xDFFF):
            start += 1
        else:
            break
    end = len(text)
    while end > start:
        c = text[end - 1]
        if c.isspace() or (0xD800 <= ord(c) <= 0xDFFF):
            end -= 1
        else:
            break
    return text[start:end]
```

Apply in `_prompt_text()` and `_require_path()`:

```python
def _prompt_text(label: str, default: str | None = None, *, required: bool = False) -> str:
    while True:
        suffix = f" [{default}]" if default not in (None, "") else ""
        raw = input(f"{label}{suffix}: ")
        value = _clean_unicode_whitespace(raw)
        if value:
            return value
        # ...

def _require_path(path_text: str, label: str) -> Path:
    path_text = _clean_unicode_whitespace(path_text)
    path = Path(path_text).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path.resolve()
```
