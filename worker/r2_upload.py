"""worker/r2_upload.py —— 把 final_hd/ 与 partial_review/ 直传 Cloudflare R2。

复用业务端同一套 R2 配置（账号/桶/密钥）。返回公网可访问 URL。
"""
from __future__ import annotations

import json
import mimetypes
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import boto3
from botocore.config import Config

R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID", "").strip()
R2_BUCKET = os.getenv("R2_BUCKET", "artworks").strip()
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "").strip()
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
R2_PUBLIC_BASE_URL = os.getenv("R2_PUBLIC_BASE_URL", "").rstrip("/")
R2_ENDPOINT = os.getenv(
    "R2_ENDPOINT", f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
).rstrip("/")

_client = None


PARTIAL_REVIEW_SUFFIXES = (
    "__frame_side_hardpaste_preview",
    "__hardpaste_preview",
)


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client(
            "s3",
            endpoint_url=R2_ENDPOINT,
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            region_name="auto",
            config=Config(s3={"addressing_style": "path"}),
        )
    return _client


def _public_url(key: str) -> str:
    key = key.lstrip("/")
    return f"{R2_PUBLIC_BASE_URL}/{key}" if R2_PUBLIC_BASE_URL else key


def _base_asset_id_from_partial_stem(stem: str) -> str:
    """把 partial_review 文件名还原成 render_aggregate 里的 asset_id。

    run_bfl_hanging_flux_pipeline.py 会生成：
      <asset_id>__hardpaste_preview.png
      <asset_id>__frame_side_hardpaste_preview.png

    旧代码直接把 png.stem 作为 asset_id，导致 result_builder 无法把 partial
    图匹配回候选：render 输出是 <asset_id>，partial 图却是 <asset_id>__xxx。
    """
    value = str(stem or "")
    for suffix in PARTIAL_REVIEW_SUFFIXES:
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def upload_file(local_path: Path, key: str) -> str:
    content_type = mimetypes.guess_type(str(local_path))[0] or "application/octet-stream"
    _get_client().upload_file(
        str(local_path),
        R2_BUCKET,
        key,
        ExtraArgs={"ContentType": content_type, "CacheControl": "public, max-age=31536000, immutable"},
    )
    return _public_url(key)


def upload_terminal_result_envelope(
    *,
    prefix: str,
    result_id: str,
    job_id: str,
    order_id: str | None,
    status: str | None,
    payload: Dict[str, Any],
) -> Dict[str, str]:
    """Persist a terminal result before WebSocket delivery.

    The business server can recover this small control object directly from R2
    when the WebSocket dies after image upload.  A versioned object is retained
    for audit, while ``control/result_payload.json`` is the stable recovery key.
    """
    clean_prefix = str(prefix or "").strip().strip("/")
    if not clean_prefix:
        raise ValueError("terminal result upload requires r2_output_prefix")

    envelope = {
        "schema_version": 1,
        "result_id": str(result_id),
        "job_id": str(job_id),
        "order_id": str(order_id or ""),
        "status": str(status or ""),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    }
    body = json.dumps(envelope, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    client = _get_client()
    common = {
        "Bucket": R2_BUCKET,
        "Body": body,
        "ContentType": "application/json; charset=utf-8",
        "CacheControl": "no-store, max-age=0",
    }
    versioned_key = f"{clean_prefix}/control/results/{result_id}.json"
    latest_key = f"{clean_prefix}/control/result_payload.json"
    client.put_object(Key=versioned_key, **common)
    client.put_object(Key=latest_key, **common)
    return {
        "result_key": latest_key,
        "result_url": _public_url(latest_key),
        "versioned_result_key": versioned_key,
        "versioned_result_url": _public_url(versioned_key),
    }


def upload_all(output_dir: Path, prefix: str) -> Dict[str, Any]:
    """上传 final_hd、Stage A 留档与 partial_review。"""
    output_dir = Path(output_dir)
    final_map: Dict[str, str] = {}
    pre_styling_map: Dict[str, str] = {}
    pre_styling_upload_errors: List[dict] = []
    partials: List[dict] = []

    final_dir = output_dir / "final_hd"
    if final_dir.is_dir():
        for png in sorted(final_dir.glob("*.png")):
            asset_id = png.stem
            key = f"{prefix}/final_hd/{png.name}"
            final_map[asset_id] = upload_file(png, key)

    pre_styling_dir = output_dir / "pre_styling_hd"
    if pre_styling_dir.is_dir():
        for png in sorted(pre_styling_dir.glob("*.png")):
            asset_id = png.stem
            key = f"{prefix}/pre_styling_hd/{png.name}"
            try:
                pre_styling_map[asset_id] = upload_file(png, key)
            except Exception as exc:
                # The Stage A archive is useful for comparison/future re-render,
                # but its upload must never block delivery of the final image.
                pre_styling_upload_errors.append({"asset_id": asset_id, "error": str(exc)})

    partial_dir = output_dir / "partial_review"
    if partial_dir.is_dir():
        for png in sorted(partial_dir.glob("*.png")):
            partial_asset_id = png.stem
            asset_id = _base_asset_id_from_partial_stem(partial_asset_id)
            key = f"{prefix}/partial_review/{png.name}"
            partials.append({
                # 匹配 render_aggregate.outputs[].asset_id / candidate 的主 id
                "asset_id": asset_id,
                # 保留旧文件 stem，方便排障和兼容前端展示
                "partial_asset_id": partial_asset_id,
                "file_name": png.name,
                "url": upload_file(png, key),
                "reason": "render_qa_failed",
                "delivery_role": "hardpaste_preview_fallback",
            })

    return {
        "final_by_asset_id": final_map,
        "pre_styling_by_asset_id": pre_styling_map,
        "pre_styling_upload_errors": pre_styling_upload_errors,
        "partial_review_images": partials,
    }
