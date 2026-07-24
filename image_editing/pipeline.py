from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .providers import ImageEditProvider, ImageEditRequest, ImageEditResult


def cleanup_objects(
    provider: ImageEditProvider,
    image: Path,
    remove_mask: Path,
    *,
    output_dir: Path | None = None,
    output_name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ImageEditResult:
    """Run one mask-driven cleanup pass.

    The configured provider is expected to remove the white pixels in remove_mask
    and reconstruct the background in a single call. In the v7 hanging flow this
    function is called on original_room, before the artwork is hard-pasted.
    """
    return provider.erase(
        ImageEditRequest(
            image=image,
            mask=remove_mask,
            output_dir=output_dir,
            output_name=output_name,
            metadata={"task": "cleanup_objects", **(metadata or {})},
        )
    )


def render_hanging(
    provider: ImageEditProvider,
    target_image: Path,
    prompt: str,
    *,
    art_mask: Path | None = None,
    artwork_reference: Path | None = None,
    references: Sequence[Path] | None = None,
    output_dir: Path | None = None,
    output_name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ImageEditResult:
    """Run a render edit with either a mask or an explicit full-space mode.

    Native inpainting providers may receive ``art_mask``.  Full-space providers
    without a mask field can instead use ``metadata['full_space_edit_mode']``;
    in that contract the complete room is generated and the caller protects the
    artwork anchor after generation.  No local crop is implied by this helper.
    """
    mask_meta = {
        "final_edit_mask": str(art_mask) if art_mask else None,
        "usage": (
            "bounded_final_render_mask_for_provider_and_mask_anchored_qa"
            if art_mask
            else "full_space_or_unmasked_render_contract"
        ),
    }
    refs = list(references or [])
    if artwork_reference is not None:
        refs.append(artwork_reference)

    return provider.edit(
        ImageEditRequest(
            image=target_image,
            mask=art_mask,
            prompt=prompt,
            references=tuple(refs),
            output_dir=output_dir,
            output_name=output_name,
            metadata={
                "task": "render_hanging",
                "masks": mask_meta,
                "flux2_mask_sent": bool(art_mask),
                "final_edit_mask_available": bool(art_mask),
                **(metadata or {}),
            },
        )
    )
