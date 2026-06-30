---
name: painting-label-mapping-for-cleanup
description: VLM-detected paintings/posters on walls must map to existing_art role to trigger cleanup — otherwise原有挂画 won't be erased (project)
source: auto-skill
extracted_at: '2026-06-27T00:15:00.000Z'
---

## Problem

VLM may return `painting`, `poster`, or `framed painting` as object labels. If these are not mapped to `existing_art` role, the cleanup logic thinks there's no existing content to erase, causing new artwork to overlap with old artwork.

## Rule

In `semantic_mvp/service.py`, `LABEL_ALIASES` must map all wall-mounted art labels to `existing_art`:

```python
LABEL_ALIASES = {
    "painting": "existing_art",      # NOT "painting"
    "poster": "existing_art",        # NOT "poster"
    "framed painting": "existing_art",  # NOT "framed_art"
    "framed artwork": "existing_art",
    # ...
}
```

Also ensure `ROLE_BY_LABEL` maps `existing_art` → `existing_content`:
```python
ROLE_BY_LABEL = {
    "existing_art": "existing_content",
    "painting": "existing_content",  # fallback
}
```

## Why

The cleanup pipeline checks for objects with role `existing_content` on the candidate wall. If VLM detects a painting but it's labeled `painting` instead of `existing_art`, the `existing_content_ids_on_candidate_wall` list is empty, `cleanup.enabled` stays `false`, and BFL erase doesn't remove the old artwork.

## How to apply

When updating VLM prompts or object detection, verify that all art-related labels (painting, poster, framed_art, wall_art, etc.) map to `existing_art` → `existing_content`.
