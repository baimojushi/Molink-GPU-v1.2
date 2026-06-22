from __future__ import annotations

from .providers.base import ImageEditProvider
from .providers.bfl import BflImageEditProvider, MockImageEditProvider


def build_image_edit_provider(name: str) -> ImageEditProvider:
    key = (name or "bfl").strip().lower()
    if key in {"bfl", "bfl_official", "bfl_flux2"}:
        return BflImageEditProvider.from_env()
    if key in {"mock", "dry_run", "noop"}:
        return MockImageEditProvider()
    raise ValueError(f"unknown image edit provider: {name}")
