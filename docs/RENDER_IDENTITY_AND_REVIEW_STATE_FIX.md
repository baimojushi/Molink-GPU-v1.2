# Render identity lock and review-state fix

## Incident

`job_453ca7c7` produced one valid low-risk candidate, but the textured-wall
render branch returned `PARTIAL_HARDPASTE_REVIEW`:

- raw core masked SSIM: `0.087719` (required `>= 0.78`)
- raw core color delta: `66.5031` (required `<= 45`)
- forbidden outer guard changed ratio: `0.0`

The provider output visibly moved and rescaled the painting toward the upper
left and repainted its content. This is a real identity/geometry failure, not a
false QA rejection.

## Root cause

The wall texture gate selected `full_space_coordinate_prompt`. The
`gpt-image-2-vip` endpoint has no multipart mask field, so the adapter disabled
its local crop/composite and relied on prompt-side compressed-image coordinates.
Those coordinates scoped the request semantically but could not guarantee pixel
identity or the candidate projection. The raw model output was then sent
directly to QA.

Separately, `Worker.send_result()` treated every non-success status as a failed
progress event. `render_review` was therefore reported as
`stage=failed/state=failed`, even though the result payload and all pipeline
stages had completed normally.

## Fix

Full-space provider output now passes
`identity_locked_shadow_composite_v2` before delivery QA:

1. Start from `02_hard_paste_after_erase.png` as the authoritative cleaned room.
2. Preserve the front and mount masks byte-for-byte at candidate geometry.
3. Preserve every pixel outside `flux2_edit_mask` byte-for-byte.
4. Reject provider relocation, repainting, chromatic ghosts and brightening.
5. If raw artwork identity passed QA, retain only connected neutral darken-only
   pixels outside the protected mount ring.
6. If raw identity failed, admit no provider pixels and project a small soft
   contact shadow from the exact mount mask; this prevents displaced content
   from being mistaken for shadow.
7. Run the original QA thresholds on the resulting production composite.
8. Record both raw-provider QA and post-lock QA in `edit_trace.json`.

For terminal progress, `render_review` now maps to
`r2_upload/done/92`; the result payload remains `status=render_review`. Fatal
statuses alone map to `failed/failed`.

## Regression evidence

The incident assets were replayed through the identity lock:

- artwork core: byte-identical to the candidate hard-paste
- outside edit mask: byte-identical to the cleaned hard-paste
- core mean absolute delta: `0.0`
- locally retained shadow pixels: non-zero

Synthetic regression tests also cover core repainting, displaced chromatic
ghosts, global full-space changes, retained neutral shadows, and the terminal
`render_review` progress mapping.
