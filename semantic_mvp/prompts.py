from __future__ import annotations

LABEL_POLICY = {
    "support": ["sofa", "floor", "table", "cabinet"],
    "structural_forbidden": ["window", "door", "tv", "mirror", "lamp", "track_lighting", "arch", "opening"],
    "existing_content": ["existing_art", "painting", "poster", "framed_art", "artwork", "wall_clutter", "wall_label", "hook", "nail", "small_shelf", "green_board", "notice_board"],
    "non_support": ["pedestal", "plinth", "display_stand"],
    "foreground": ["foreground_object", "person", "human", "people", "plant", "chair"],
}

def system_prompt(scene_profile: str) -> str:
    common = """
You are a visual scene parser for an automated artwork hanging system.
Return JSON only. Do not include markdown.
Detect large surfaces and objects needed for hanging geometry.

Return an object:
{
  "objects": [
    {
      "label": "...",
      "role": "wall|support|forbidden|foreground|non_support|existing_content",
      "bbox_xyxy": [x1,y1,x2,y2],
      "confidence": 0.0-1.0,
      "notes": ""
    }
  ]
}

Coordinate system: pixels of the image attached in this request.
Prioritize high-recall boxes. Coarse bbox is acceptable for MVP.
Required labels when visible:
wall, floor, sofa, table, cabinet, window, door, tv, mirror, lamp,
track_lighting, arch, opening, existing_art, painting, poster, framed_art,
artwork, wall_clutter, wall_label, hook, nail, small_shelf, green_board,
notice_board, person, human, people, pedestal, plinth, display_stand,
foreground_object.

VLM-first wall proposal rules:
- Detect every visible wall plane as a separate object. Never merge walls with different orientations.
- Typical cases include left wall, front wall, right wall, and inner wall beyond an opening.
- If the model is uncertain, still propose separate wall bboxes for visually distinct wall planes instead of one scene-wide wall.
- If a large wall bbox spans multiple differently oriented wall planes, output the coarsest separate wall proposals you can infer.
- Existing displayed artworks, posters, frames, green boards, labels and wall-mounted clutter must be returned separately as existing_content objects, not merged into wall.
- Opening, arch, door and window must be separate forbidden objects, not merged into wall.
- Door panels are never hangable wall. If a door/window/opening touches a wall edge, clip the wall bbox at that structure boundary instead of including it.
- Do not stop a wall bbox at the last detected artwork/person. Include the full visible wall plane behind foreground people and wall-mounted artworks.
- For wide gallery photos, prioritize front-vs-side wall separation.

Object inventory rules for cleanup:
- Do not treat an existing painting/poster/framed artwork/green board/label as clean wall. Return each visible instance separately, even when small, far away, partially cropped, or visually similar to the wall.
- Use label existing_art for generic displayed art; use painting/poster/framed_art when visually clear. All of these must have role=existing_content.
- Use label person with role=foreground for every visible person or human body, including partial bodies on the right/left edge.
- Use wall_label, hook, nail, small_shelf, green_board, notice_board for wall-attached non-art objects when visible.
- Wall bboxes may cover the full wall plane behind objects, but object bboxes must also be returned separately so downstream erase masks are non-empty when candidates overlap old content.

Multi-wall rules:
- Detect EVERY visible wall plane as a separate object. Never merge walls with different orientations.
- Typical cases include: left wall, front wall, right wall, inner wall beyond an opening.
- Each wall must have its own independent bbox. The bbox should cover the full visible extent of that wall plane, including areas behind foreground occluders, but excluding doors/windows/openings.
- Use object_id patterns like wall_left_01, wall_front_01, wall_right_01, wall_inner_01 when applicable.
- If a wall contains an opening/arch, still return the enclosing wall as wall, and also return opening/arch as separate forbidden objects.
- Existing displayed artworks, posters, frames, green boards, labels and wall-mounted clutter must be returned separately as existing_content objects, not merged into wall.
"""
    if scene_profile == "gallery":
        return common + """
Gallery-specific rules:
- arch/opening are forbidden, never wall.
- existing displayed artworks, posters, framed art, green boards and labels are existing_art / painting / poster / framed_art / wall_label with role=existing_content.
- track lights and spotlights are track_lighting or lamp / forbidden.
- pedestal/plinth/display stand are non_support, not table/cabinet support.
- floor is support for height rule, not clutter.

- Detect separate wall planes even if they visually touch at a corner.
- Front wall and side wall must be different wall objects.
- A large single wall bbox spanning most of the image is invalid unless the scene truly has only one visible wall.

- Detect separate wall planes even if they visually touch at a corner.
- Front wall and side wall must be different wall objects.
"""
    return common + """
Home/retail rules:
- sofa/table/cabinet may be support only if visually near the wall.
- existing wall art or wall decor is existing_art or wall_clutter.
- windows, doors, TV, mirrors and lamps are forbidden.
"""

def user_prompt(scene_profile: str, width: int, height: int) -> str:
    return f"""
Scene profile: {scene_profile}
Image size: {width}x{height}

Detect wall, support, forbidden, existing_content and foreground/non_support objects.
Return JSON only.

Also enforce:
- vlm_first_multiwall: output every visible wall plane separately
- do not merge left/front/right walls into one box
- if visible, include opening/arch/window/door as separate forbidden objects
- include existing wall artworks as existing_art / painting / poster / framed_art objects
- include visible people as person foreground objects
- include wall labels, hooks, nails, small shelves, green boards and notice boards as existing_content objects

Also enforce:
- multi-wall detection: output every visible wall plane separately
- do not merge left/front/right walls into one box
- if visible, include opening/arch as separate forbidden objects inside the scene
- include existing wall artworks as existing_art / painting / poster / framed_art objects
- include visible people as person foreground objects
- include wall labels/hooks/nails/small shelves/green boards/notice boards separately as existing_content objects
"""


def artwork_frame_system_prompt() -> str:
    return """
You are an artwork product-image inspector for an automated wall-hanging renderer.
Return JSON only. Do not include markdown.

Decide whether the artwork image already contains a visible physical frame around
its painting/photo/poster content. A physical frame means a visible outer
border/frame that should be preserved and whose edge colors can be used to
synthesize deterministic side thickness.

Also classify the artwork surface/material style for side thickness: delicate or
paper-like works use normal side depth, rough/canvas works use double side depth,
and Chinese hanging scroll / vertical scroll artworks should not get generated
side faces.

Do not classify a white mat, crop margin, canvas whitespace, product background,
or image content border as a physical frame unless there is a clear frame
boundary and material-like outer rim.
"""


def artwork_frame_user_prompt(width: int, height: int) -> str:
    return f"""
Artwork image size: {int(width)}x{int(height)}

Return this JSON object exactly:
{{
  "has_frame": true|false|null,
  "confidence": 0.0-1.0,
  "frame_style": "visible_outer_frame|canvas_wrap|mat_only|border_only|none|uncertain",
  "artwork_surface_style": "refined_paper_or_delicate|rough_or_canvas|chinese_hanging_scroll|uncertain",
  "side_thickness_policy": "normal|double|none",
  "frame_width_ratio": {{
    "left": 0.0,
    "right": 0.0,
    "top": 0.0,
    "bottom": 0.0
  }},
  "dominant_edge_colors_rgb": {{
    "left": [r,g,b],
    "right": [r,g,b],
    "top": [r,g,b],
    "bottom": [r,g,b]
  }},
  "reason": ""
}}

Rules:
- has_frame=true only when a physical frame is clearly visible.
- has_frame=false when no physical frame is visible.
- has_frame=null when uncertain.
- artwork_surface_style describes the artwork material/style for deterministic side-depth synthesis:
  refined_paper_or_delicate = delicate print, photo, fine illustration, paper painting, Chinese painting sheet, or other thin/refined paper-like work;
  rough_or_canvas = coarse painting, oil/acrylic on canvas, visibly rough cloth/canvas texture, thick impasto, or rustic fabric painting;
  chinese_hanging_scroll = Chinese hanging scroll / vertical scroll with rollers or textile mounting that should stay flat in this renderer;
  uncertain = not enough evidence.
- side_thickness_policy should be normal for refined_paper_or_delicate or uncertain, double for rough_or_canvas, and none for chinese_hanging_scroll.
- frame_width_ratio is approximate visible frame width divided by image size.
- dominant_edge_colors_rgb should describe the outermost frame/edge color.
"""
