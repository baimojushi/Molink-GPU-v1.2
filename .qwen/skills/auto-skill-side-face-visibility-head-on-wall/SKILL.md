---
name: side-face-visibility-head-on-wall
description: When wall faces camera directly, side face visibility must be computed from artwork offset relative to image center, not from front-back perspective projection shift
source: auto-skill
extracted_at: '2026-06-26T15:57:48.175Z'
---

## Problem

When a wall faces the camera directly, the 3D-to-2D perspective projection of the artwork's back face (receded by frame_depth_m along wall normal) produces sub-pixel shift relative to the front face (~0.5px at 2.5cm depth / 3m distance). The original side visibility check computed `side_visible = side_mask - front_mask`, which yielded zero visible area when front and back quads nearly overlapped.

This caused `visible_side_faces_original: []` even when the artwork was clearly offset from the image center — where perspective geometry dictates that the side facing away from the image center should be visible.

## Root cause

`back3 = front3 - n * frame_depth_m` along wall normal n. When n points toward camera (wall faces camera), back3 is farther from camera. At typical room distances (~3m), 2.5cm depth produces ~0.5px perspective shift — too small for rasterization-based visibility detection.

## Correct approach

Side visibility has two components:

1. **Perspective projection shift** — works when wall is at an angle to camera (front and back quads produce visibly different 2D projections)
2. **Artwork offset from image center** — when artwork center is offset from the image center, the side facing away from the image center is visible in perspective, regardless of wall-camera angle

## Implementation in `_compute_artwork_mount_policy_for_candidate()`

After computing `front_orig` and `back_orig` from 3D projection:

```python
front_center = np.mean(front_orig, axis=0)
image_center = np.array([ow / 2.0, oh / 2.0])
center_offset = front_center - image_center
offset_dist = np.linalg.norm(center_offset)

for face in raw_faces:
    # ... existing mask-based visible_area calculation ...

    # Fallback: when perspective shift is sub-pixel, use offset heuristic
    if visible_area < min_area and est_width < min_width and offset_dist > 10.0:
        # Determine which edge of the face is the "outer" edge
        outer_edge_pts = ...  # depends on source_edge (top/right/bottom/left)
        outer_edge_center = np.mean(outer_edge_pts, axis=0)

        dist_outer = np.linalg.norm(outer_edge_center - image_center)
        dist_front = np.linalg.norm(front_center - image_center)

        if dist_outer > dist_front + 5.0:
            # This side faces away from image center → visible
            est_visible_width = min(
                frame_depth_m * 40.0 * (offset_dist / max(100.0, dist_front)),
                edge_len * 0.15
            )
            visible_area = max(visible_area, int(est_visible_width * edge_len * 0.3))
            est_width = max(est_width, est_visible_width)
```

## Key thresholds

| Parameter | Value | Purpose |
|---|---|---|
| `offset_dist > 10.0` | 10px | Minimum artwork offset from image center to trigger fallback |
| `dist_outer > dist_front + 5.0` | 5px | Minimum edge-to-center distance advantage to qualify as "outer" |
| `frame_depth_m * 40.0` | ~1.0px | Base visible width per unit depth |
| `edge_len * 0.15` | 15% of edge | Maximum visible width capped at fraction of edge length |

## When this does NOT apply

- Artwork centered in frame: no side should be visible (correct behavior)
- Wall at significant angle to camera: perspective projection shift is sufficient, original logic works
- `precompose_frame_sides` disabled by rules: sides not generated regardless
