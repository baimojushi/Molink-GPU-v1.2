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
    """Run final render with an explicit bounded-edit mask.

    The production contract is: final rendering may only affect the provided
    mask region. Providers that support native inpainting may send the mask to
    the remote model. Providers without a mask field, such as API易 GPT Image 2
    VIP image-edit, must enforce the same contract locally by crop/composite.
    """
    mask_meta = {
        "final_edit_mask": str(art_mask) if art_mask else None,
        "usage": "bounded_final_render_mask_for_provider_and_mask_anchored_qa" if art_mask else "missing_required_final_edit_mask",
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
