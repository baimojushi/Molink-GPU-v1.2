from __future__ import annotations

import base64
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import requests
from PIL import Image

from .base import ImageEditRequest, ImageEditResult


READY_STATUS = "Ready"
FAILED_STATUSES = {"Error", "Failed", "Request Moderated", "Content Moderated", "Task not found"}


MOCK_PROVIDER_WARNING = (
    "🔴 WARNING: provider=mock is a local no-op dry run. "
    "The output image is copied from the input image and is not a BFL result."
)


def _read_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _image_value(path: Path, *, as_data_url: bool = False) -> str:
    b64 = _read_base64(path)
    if not as_data_url:
        return b64
    suffix = path.suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    return f"data:image/{mime};base64,{b64}"


def _ensure_same_image_size(image: Path, mask: Path) -> None:
    with Image.open(image) as img, Image.open(mask) as m:
        if img.size != m.size:
            raise ValueError(f"mask size {m.size} must match image size {img.size}: {mask}")


def _mask_nonempty(mask: Path) -> bool:
    with Image.open(mask) as m:
        gray = m.convert("L")
        return bool(gray.getbbox())


def _default_output_path(request: ImageEditRequest, suffix: str) -> Path:
    out_dir = request.output_dir or request.image.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    name = request.output_name or f"{request.image.stem}_{suffix}.png"
    return out_dir / name


@dataclass(frozen=True)
class BflClient:
    api_key: str | None = None
    base_url: str = "https://api.bfl.ai"
    timeout_s: float = 180.0
    poll_interval_s: float = 1.0
    max_retries: int = 3

    @classmethod
    def from_env(cls) -> "BflClient":
        return cls(
            api_key=os.getenv("BFL_API_KEY", "").strip() or None,
            base_url=os.getenv("BFL_API_BASE", "https://api.bfl.ai").rstrip("/"),
            timeout_s=max(10.0, float(os.getenv("BFL_TIMEOUT_S", "180"))),
            poll_interval_s=max(0.25, float(os.getenv("BFL_POLL_INTERVAL_S", "1"))),
            max_retries=int(os.getenv("BFL_MAX_RETRIES", "3")),
        )

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError("BFL_API_KEY is required for BFL provider")
        return {
            "accept": "application/json",
            "x-key": self.api_key,
            "Content-Type": "application/json",
        }

    def _request_json(self, method: str, url: str, **kwargs: Any) -> Mapping[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.request(method, url, timeout=self.timeout_s, **kwargs)
                if resp.status_code == 429 and attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                if resp.status_code >= 400:
                    raise RuntimeError(f"BFL {method} {url} -> {resp.status_code}: {resp.text[:1000]}")
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.max_retries - 1:
                    break
                time.sleep(2**attempt)
        raise RuntimeError(f"BFL API request failed after {self.max_retries} attempts: {last_error}")

    def submit(self, endpoint: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        url = endpoint if endpoint.startswith("http") else f"{self.base_url}{endpoint}"
        return self._request_json("POST", url, headers=self._headers(), json=payload)

    def poll(self, initial: Mapping[str, Any]) -> Mapping[str, Any]:
        task_id = str(initial.get("id") or "")
        poll_url = str(initial.get("polling_url") or f"{self.base_url}/v1/get_result?id={task_id}")
        deadline = time.time() + max(0.0, float(self.timeout_s))
        first_poll = True
        while first_poll or time.time() < deadline:
            first_poll = False
            result = self._request_json("GET", poll_url, headers={"accept": "application/json", "x-key": self.api_key or ""})
            status = str(result.get("status") or "")
            if status == READY_STATUS:
                return result
            if status in FAILED_STATUSES:
                raise RuntimeError(f"BFL task failed with status={status}: {json.dumps(result, ensure_ascii=False)}")
            if time.time() >= deadline:
                break
            time.sleep(min(float(self.poll_interval_s), max(0.0, deadline - time.time())))
        raise TimeoutError(f"BFL task timed out after {self.timeout_s}s: id={task_id}")

    def download_sample(self, result: Mapping[str, Any], output_path: Path) -> None:
        sample_url = (((result.get("result") or {}) if isinstance(result.get("result"), dict) else {}).get("sample"))
        if not sample_url:
            raise RuntimeError(f"BFL result has no result.sample URL: {json.dumps(result, ensure_ascii=False)}")
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.get(str(sample_url), timeout=self.timeout_s)
                resp.raise_for_status()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(resp.content)
                return
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.max_retries - 1:
                    break
                time.sleep(2**attempt)
        raise RuntimeError(f"BFL sample download failed after {self.max_retries} attempts: {last_error}")


class BflImageEditProvider:
    """BFL official API adapter with exactly two production capabilities.

    - erase: POST /v1/flux-tools/erase-v1
    - edit:  POST /v1/flux-2-max by default, configurable to flux-2-pro
    """

    def __init__(
        self,
        client: BflClient | None = None,
        *,
        erase_endpoint: str | None = None,
        flux2_endpoint: str | None = None,
        encode_flux2_inputs_as_data_url: bool | None = None,
    ) -> None:
        self.client = client or BflClient.from_env()
        self.erase_endpoint = erase_endpoint or os.getenv("BFL_ERASE_ENDPOINT", "/v1/flux-tools/erase-v1")
        self.flux2_endpoint = flux2_endpoint or os.getenv("BFL_FLUX2_ENDPOINT", "/v1/flux-2-max")
        self.flux2_mask_field = os.getenv("BFL_FLUX2_MASK_FIELD", "mask").strip() or "mask"
        self.require_flux2_mask = os.getenv("BFL_FLUX2_REQUIRE_MASK", "1").strip().lower() not in {"", "0", "false", "no", "off"}
        self.encode_flux2_inputs_as_data_url = (
            os.getenv("BFL_FLUX2_IMAGE_AS_DATA_URL", "0") == "1"
            if encode_flux2_inputs_as_data_url is None
            else encode_flux2_inputs_as_data_url
        )

    @classmethod
    def from_env(cls) -> "BflImageEditProvider":
        return cls(BflClient.from_env())

    def erase(self, request: ImageEditRequest) -> ImageEditResult:
        if request.mask is None:
            raise ValueError("erase requires request.mask")
        image = Path(request.image)
        mask = Path(request.mask)
        _ensure_same_image_size(image, mask)
        out_path = _default_output_path(request, "erase")
        dilate_pixels = int(request.metadata.get("dilate_pixels", os.getenv("BFL_ERASE_DILATE_PIXELS", "10")))

        # Downscale oversized images before base64 encoding to avoid BFL 422.
        max_long_edge = int(os.getenv("BFL_ERASE_MAX_LONG_EDGE", "2048"))
        payload_image = image
        payload_mask = mask
        dilate_scale = 1.0
        if max_long_edge > 0:
            with Image.open(image) as img:
                longest = max(img.size)
                if longest > max_long_edge:
                    dilate_scale = max_long_edge / longest
                    new_size = (max(1, int(img.size[0] * dilate_scale)), max(1, int(img.size[1] * dilate_scale)))
                    tmp_dir = out_path.parent / ".bfl_tmp"
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    payload_image = tmp_dir / f"{out_path.stem}_downscaled_input.png"
                    payload_mask = tmp_dir / f"{out_path.stem}_downscaled_mask.png"
                    img.resize(new_size, Image.Resampling.LANCZOS).save(payload_image)
                    with Image.open(mask) as m:
                        m.resize(new_size, Image.Resampling.NEAREST).save(payload_mask)

        payload = {
            "image": _read_base64(payload_image),
            "mask": _read_base64(payload_mask),
            "dilate_pixels": max(0, min(25, int(dilate_pixels * dilate_scale))),
            "output_format": str(request.metadata.get("output_format", "png")),
        }
        started = time.time()
        submit_meta = self.client.submit(self.erase_endpoint, payload)
        result = self.client.poll(submit_meta)
        self.client.download_sample(result, out_path)
        trace = {
            "provider": "bfl",
            "capability": "erase",
            "endpoint": self.erase_endpoint,
            "task_id": submit_meta.get("id"),
            "polling_url_present": bool(submit_meta.get("polling_url")),
            "status": result.get("status"),
            "elapsed_s": round(time.time() - started, 3),
            "output_path": str(out_path),
            "metadata": dict(request.metadata),
        }
        return ImageEditResult(image=out_path, trace=trace)

    def edit(self, request: ImageEditRequest) -> ImageEditResult:
        if not request.prompt:
            raise ValueError("edit requires request.prompt")
        image = Path(request.image)
        out_path = _default_output_path(request, "flux2")
        references = [Path(p) for p in request.references]
        if len(references) > 7:
            raise ValueError("FLUX.2 API supports input_image plus up to 7 extra references in this adapter")
        mask = Path(request.mask) if request.mask is not None else None
        if mask is None:
            if self.require_flux2_mask:
                raise ValueError("FLUX.2 render requires request.mask because BFL_FLUX2_REQUIRE_MASK=1")
        else:
            _ensure_same_image_size(image, mask)
            if self.require_flux2_mask and not _mask_nonempty(mask):
                raise ValueError(f"FLUX.2 render mask is empty: {mask}")

        payload: dict[str, Any] = {
            "prompt": request.prompt,
            "input_image": _image_value(image, as_data_url=self.encode_flux2_inputs_as_data_url),
            "output_format": str(request.metadata.get("output_format", "png")),
        }
        if mask is not None:
            payload[self.flux2_mask_field] = _image_value(mask, as_data_url=self.encode_flux2_inputs_as_data_url)
        for idx, ref in enumerate(references, start=2):
            payload[f"input_image_{idx}"] = _image_value(ref, as_data_url=self.encode_flux2_inputs_as_data_url)
        for key in ("seed", "width", "height", "aspect_ratio", "safety_tolerance", "prompt_upsampling"):
            if key in request.metadata and request.metadata[key] is not None:
                payload[key] = request.metadata[key]
        started = time.time()
        submit_meta = self.client.submit(self.flux2_endpoint, payload)
        result = self.client.poll(submit_meta)
        self.client.download_sample(result, out_path)
        trace = {
            "provider": "bfl",
            "capability": "flux2_edit",
            "endpoint": self.flux2_endpoint,
            "task_id": submit_meta.get("id"),
            "polling_url_present": bool(submit_meta.get("polling_url")),
            "status": result.get("status"),
            "elapsed_s": round(time.time() - started, 3),
            "output_path": str(out_path),
            "reference_count": len(references),
            "payload_image_keys": [k for k in payload.keys() if k.startswith("input_image")],
            "payload_mask_field": self.flux2_mask_field if mask is not None else None,
            "mask_path": str(mask) if mask is not None else None,
            "mask_sent_to_flux2": mask is not None,
            "mask_parameter_supported_by_adapter": True,
            "metadata": {k: v for k, v in dict(request.metadata).items() if k not in {"prompt"}},
        }
        return ImageEditResult(image=out_path, trace=trace)


class MockImageEditProvider:
    """Local no-op provider for CI, deployment dry runs and contract tests."""

    def _trace(self, request: ImageEditRequest, *, capability: str, out: Path, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        trace = {
            "provider": "mock",
            "capability": capability,
            "output_path": str(out),
            "warning": MOCK_PROVIDER_WARNING,
            "warning_label": "🔴 MOCK PROVIDER / NO-OP RESULT",
            "noop_result": True,
            "production_usable": False,
            "metadata": dict(request.metadata),
        }
        if extra:
            trace.update(dict(extra))
        return trace

    def erase(self, request: ImageEditRequest) -> ImageEditResult:
        out = _default_output_path(request, "mock_erase")
        shutil.copyfile(request.image, out)
        return ImageEditResult(image=out, trace=self._trace(request, capability="erase", out=out))

    def edit(self, request: ImageEditRequest) -> ImageEditResult:
        out = _default_output_path(request, "mock_flux2")
        shutil.copyfile(request.image, out)
        return ImageEditResult(image=out, trace=self._trace(
            request,
            capability="flux2_edit",
            out=out,
            extra={
                "reference_count": len(tuple(request.references)),
                "mask_path": str(request.mask) if request.mask else None,
                "mask_sent_to_flux2": request.mask is not None,
            },
        ))
