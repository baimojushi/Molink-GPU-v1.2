from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class ImageEditRequest:
    image: Path
    mask: Path | None = None
    prompt: str | None = None
    references: Sequence[Path] = field(default_factory=tuple)
    output_dir: Path | None = None
    output_name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ImageEditResult:
    image: Path
    trace: Mapping[str, Any] = field(default_factory=dict)


class ImageEditProvider(Protocol):
    """Two-capability provider used by the production hanging pipeline.

    Erase owns mask-driven cleanup and contextual reconstruction. FLUX.2 edit owns
    final photorealistic rendering of the already hard-pasted artwork. Wall fill is
    intentionally absent because erase-v1 performs the required cleanup + inpaint in
    one operation for this product flow.
    """

    def erase(self, request: ImageEditRequest) -> ImageEditResult:
        ...

    def edit(self, request: ImageEditRequest) -> ImageEditResult:
        ...
