"""worker/pipeline.py —— 串联四段流水线。

Metric3D(:6006) → Semantic(:6011) → Hanging(:6010) → BFL 渲染。
前三段通过 localhost HTTP 调用同机服务；第四段封装离线脚本 run_pipeline。
"""
from __future__ import annotations

import io
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from PIL import Image

METRIC3D_URL = os.getenv("METRIC3D_URL", "http://127.0.0.1:6006")
SEMANTIC_URL = os.getenv("SEMANTIC_URL", "http://127.0.0.1:6011")
HANGING_URL = os.getenv("HANGING_URL", "http://127.0.0.1:6010")

HTTP_TIMEOUT = float(os.getenv("PIPELINE_HTTP_TIMEOUT_S", "600"))
API_KEY = os.getenv("API_KEY", "")


def _auth_headers() -> dict:
    return {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}


def _download(url: str, dest: Path) -> Path:
    with httpx.stream("GET", url, timeout=120, follow_redirects=True) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_bytes():
                f.write(chunk)
    return dest


def download_to_path(url: str, dest: Path) -> Path:
    """Download a persisted artifact such as hanging_result.zip for supplement render jobs."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    return _download(url, dest)


def fetch_inputs(job: Dict[str, Any], work_dir: Path) -> Dict[str, Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    room = _download(job["room_image_url"], work_dir / "room.jpg")
    artwork = _download(job["artwork_image_url"], work_dir / "artwork.png")
    return {"room": room, "artwork": artwork}


def call_metric3d(room: Path, work_dir: Path) -> Path:
    with open(room, "rb") as f:
        files = {"file": (room.name, f, "image/jpeg")}
        data = {"camera_json": "{}", "include_candidates": "true"}
        r = httpx.post(f"{METRIC3D_URL}/analyze_geometry", files=files, data=data, timeout=HTTP_TIMEOUT, headers=_auth_headers())
    if r.status_code >= 400:
        raise RuntimeError(f"Metric3D HTTP {r.status_code}: {r.text[:1000]}")
    metric_zip = work_dir / "metric3d.zip"
    metric_zip.write_bytes(r.content)
    return metric_zip


def call_semantic(
    room: Path,
    metric_zip: Path,
    artwork: Path,
    scene_profile: str,
    work_dir: Path,
    *,
    styling_masks_requested: bool = False,
) -> Path:
    with open(room, "rb") as fr, open(metric_zip, "rb") as fz, open(artwork, "rb") as fa:
        files = {
            "room_image": (room.name, fr, "image/jpeg"),
            "metric3d_zip": (metric_zip.name, fz, "application/zip"),
            "artwork_image": (artwork.name, fa, "image/png"),
        }
        data = {
            "scene_profile": scene_profile or "home",
            "vlm_mode": os.getenv("VLM_MODE", "auto"),
            "artwork_vlm_enable": os.getenv("ARTWORK_VLM_ENABLE", "1"),
        }
        if styling_masks_requested:
            data["sam_enable"] = os.getenv("STYLING_SAM_ENABLE", "1")
            data["sam_refine_labels"] = os.getenv(
                "STYLING_SAM_REFINE_LABELS",
                "wall,floor,sofa,table,cabinet,sideboard,chair",
            )
        r = httpx.post(f"{SEMANTIC_URL}/parse_semantics_mvp", files=files, data=data, timeout=HTTP_TIMEOUT, headers=_auth_headers())
    if r.status_code == 422:
        diag_path = work_dir / "semantic_422_response.json"
        try:
            diag_path.write_text(r.text, encoding="utf-8")
        except Exception:
            diag_path.write_text(json.dumps({"raw_text": r.text[:2000]}), encoding="utf-8")
    r.raise_for_status()
    semantics_path = work_dir / "semantics.json"
    semantics_path.write_text(r.text, encoding="utf-8")
    return semantics_path


def call_hanging(
    room: Path,
    metric_zip: Path,
    artwork: Path,
    semantics_path: Path,
    rules_json: Optional[str],
    work_dir: Path,
) -> Path:
    with open(room, "rb") as fr, open(metric_zip, "rb") as fz, open(artwork, "rb") as fa:
        files = {
            "room_image": (room.name, fr, "image/jpeg"),
            "metric3d_zip": (metric_zip.name, fz, "application/zip"),
            "artwork_image": (artwork.name, fa, "image/png"),
        }
        data = {"semantics_json": semantics_path.read_text(encoding="utf-8")}
        if rules_json:
            data["rules_json"] = rules_json
        r = httpx.post(f"{HANGING_URL}/analyze_hanging_mvp", files=files, data=data, timeout=HTTP_TIMEOUT, headers=_auth_headers())
    r.raise_for_status()
    hanging_zip = work_dir / "hanging_result.zip"
    hanging_zip.write_bytes(r.content)
    return hanging_zip


def extract_client_result(hanging_zip: Path) -> Dict[str, Any]:
    with zipfile.ZipFile(hanging_zip) as zf:
        name = next((n for n in zf.namelist() if n.endswith("client_result.json")), None)
        if not name:
            raise RuntimeError("hanging zip 缺少 client_result.json")
        return json.loads(zf.read(name).decode("utf-8"))


def prepare_pre_styling_render_aggregate(
    hanging_zip: Path,
    output_dir: Path,
    *,
    stage_a_base_by_asset_id: Optional[Dict[str, str]] = None,
    stage_a_base_by_wall_id: Optional[Dict[str, str]] = None,
    target_wall_ids: Optional[list[str]] = None,
    target_candidate_ids: Optional[list[str]] = None,
    provider: str = "apiyi_gpt_image2_vip",
) -> Dict[str, Any]:
    """Seed ``final_hd`` from the persisted clean Stage A image.

    Primary-wall wallpaper opt-in must start from the image archived before the
    first Stage B edit.  This helper deliberately makes zero Stage A provider
    calls, but emits the same minimal aggregate contract consumed by
    ``result_builder`` and the Stage B pipeline.
    """
    started = time.time()
    asset_urls = {str(k): str(v).strip() for k, v in (stage_a_base_by_asset_id or {}).items() if str(v).strip()}
    wall_urls = {str(k): str(v).strip() for k, v in (stage_a_base_by_wall_id or {}).items() if str(v).strip()}
    if not asset_urls and not wall_urls:
        raise RuntimeError("primary supplement render requires a persisted Stage A base image")

    wall_filter = {str(value) for value in (target_wall_ids or []) if str(value).strip()}
    candidate_filter = {str(value) for value in (target_candidate_ids or []) if str(value).strip()}
    with zipfile.ZipFile(Path(hanging_zip)) as archive:
        try:
            manifest = json.loads(archive.read("flux_inputs/manifest.json").decode("utf-8"))
        except KeyError as exc:
            raise RuntimeError("hanging zip lacks flux_inputs/manifest.json") from exc

    selected = []
    for item in manifest.get("items") or []:
        if not isinstance(item, dict):
            continue
        asset_id = str(item.get("asset_id") or "").strip()
        wall_id = str(item.get("wall_id") or "").strip()
        candidate_id = str(item.get("candidate_id") or "").strip()
        if not asset_id:
            continue
        if wall_filter and wall_id not in wall_filter:
            continue
        if candidate_filter and candidate_id not in candidate_filter:
            continue
        source_url = asset_urls.get(asset_id) or wall_urls.get(wall_id)
        if source_url:
            selected.append((item, source_url))

    if not selected:
        raise RuntimeError("persisted Stage A base did not match the requested manifest candidate")

    final_dir = Path(output_dir) / "final_hd"
    final_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for item, source_url in selected:
        asset_id = str(item["asset_id"])
        downloaded = final_dir / f".{asset_id}.source"
        destination = final_dir / f"{asset_id}.png"
        try:
            _download(source_url, downloaded)
            with Image.open(downloaded) as source:
                source.convert("RGB").save(destination, format="PNG")
        finally:
            downloaded.unlink(missing_ok=True)
        outputs.append({
            "asset_id": asset_id,
            "wall_id": item.get("wall_id"),
            "candidate_id": item.get("candidate_id"),
            "exit_code": "REUSED_PRE_STYLING_STAGE_A_BASE",
            "production_usable": True,
            "final_path": str(destination),
            "stage_a_base_source": "pre_styling_image_url",
        })

    return {
        "provider": str(provider),
        "render_provider": str(provider),
        "stage_a_base_source": "pre_styling_image_url",
        "stage_a_render_call_count": 0,
        "outputs": outputs,
        "errors": [],
        "production_usable_output_count": len(outputs),
        "elapsed_s": round(time.time() - started, 3),
    }


def build_rules_overrides(job: Dict[str, Any]) -> Optional[str]:
    """把订单作品物理尺寸与 scene_profile 注入 rules_json 覆盖。
    服务端按 rules_json 内 artwork 尺寸做硬贴与高度规则；缺失则服务端报
    FAILED_MISSING_ARTWORK_DIMENSIONS，因此尺寸必须随作业传入。"""
    artwork = job.get("artwork") or {}
    overrides = dict(job.get("rules_overrides") or {})
    if artwork.get("physical_width_m") and artwork.get("physical_height_m"):
        overrides["artwork"] = {
            "physical_width_m": float(artwork["physical_width_m"]),
            "physical_height_m": float(artwork["physical_height_m"]),
            "has_frame": bool(artwork.get("has_frame", True)),
        }
    return json.dumps(overrides) if overrides else None
