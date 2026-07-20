"""worker/main.py —— GPU worker 主入口。

建立到业务端的 WebSocket 反隧道连接 → 注册 → 监听 job_dispatch →
串四段流水线 → 直传 R2 → 上报 result → 空闲 5 分钟自关机。
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
import os
import tempfile
import time
import traceback
from pathlib import Path

import websockets

from worker import pipeline, render_api, r2_upload, result_builder, autodl_api, diagnostics

try:
    from molink_causal import CausalEngine, Event
except Exception:  # pragma: no cover - optional integration
    CausalEngine = None
    Event = None

MOLINK_WS_URL = os.getenv("MOLINK_WS_URL", "wss://www.molink.art/api/gpu/worker-connect")
WORKER_SECRET = os.getenv("WORKER_SECRET", "")
INSTANCE_ID = os.getenv("AUTODL_INSTANCE_ID", os.getenv("INSTANCE_ID", "autodl-local"))

# --- molink worker config debug begin ---
import hashlib

def _molink_fp(value):
    value = str(value or "")
    return hashlib.sha256(value.encode()).hexdigest()[:12] if value else "(empty)"

print(
    "[worker] config "
    f"instance_id={INSTANCE_ID} "
    f"ws_url={MOLINK_WS_URL} "
    f"secret_len={len(WORKER_SECRET)} "
    f"secret_fp={_molink_fp(WORKER_SECRET)}",
    flush=True,
)

if not WORKER_SECRET:
    raise RuntimeError("WORKER_SECRET is empty; refuse to connect without token")
# --- molink worker config debug end ---

GPU_MODEL = os.getenv("GPU_MODEL", "TITAN Xp")
GPU_COUNT = int(os.getenv("GPU_COUNT", "1"))
IDLE_TIMEOUT_S = int(os.getenv("IDLE_TIMEOUT_S", "300"))
WORKER_VERSION = "1.0.1"


def _truthy_env(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default)
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def _causal_enabled() -> bool:
    return _truthy_env("MOLINK_CAUSAL_ENABLED", "0") and CausalEngine is not None and Event is not None


def _safe_json_load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _count_semantic_space(semantics: dict) -> dict:
    objects = semantics.get("objects") if isinstance(semantics, dict) else []
    if not isinstance(objects, list):
        objects = []

    def label_is(obj, names):
        label = str((obj or {}).get("label") or "").strip().lower()
        role = str((obj or {}).get("role") or "").strip().lower()
        return label in names or role in names

    return {
        "wall_count": sum(1 for o in objects if label_is(o, {"wall"})),
        "window_count": sum(1 for o in objects if label_is(o, {"window"})),
        "door_count": sum(1 for o in objects if label_is(o, {"door"})),
        "furniture_count": sum(1 for o in objects if str((o or {}).get("label") or "").strip().lower() in {"sofa", "table", "cabinet", "chair"}),
        "existing_art_count": sum(1 for o in objects if str((o or {}).get("role") or "").strip().lower() == "existing_content"),
        "person_count": sum(1 for o in objects if str((o or {}).get("label") or "").strip().lower() in {"person", "human", "people"}),
    }


def _best_plan_delta(client_res: dict) -> dict:
    best = client_res.get("best_plan") if isinstance(client_res, dict) else None
    if not isinstance(best, dict):
        return {}
    install = best.get("install_reference") or {}
    comp = best.get("composition_context") or {}
    light_sem = best.get("light_semantics") or {}
    artwork_analysis = best.get("artwork_analysis") or {}
    axis_center = comp.get("axis_centering")
    delta = {
        "best_plan": {
            "wall_id": best.get("wall_id") or "",
            "wall_position_zh": best.get("wall_position_zh") or comp.get("above_furniture_label_zh") or "",
            "is_primary_wall": int(best.get("rank") or 0) == 1,
            "artwork_wall_area_ratio": comp.get("artwork_wall_area_ratio") or 0.0,
            "center_height_cm": install.get("center_height_cm") or install.get("provisional_center_height_cm"),
            "bottom_edge_cm": install.get("bottom_edge_cm"),
            "top_edge_cm": install.get("top_edge_cm"),
            "height_finalized": bool(install.get("metric_height_finalized")),
            "above_furniture_label_zh": comp.get("above_furniture_label_zh"),
            "bottom_gap_to_furniture_cm": comp.get("bottom_gap_to_furniture_cm"),
            "axis_centering": True if axis_center in (True, "居中", "center", "centered") else False,
            "composition_zh": "，".join(str(x) for x in [comp.get("above_furniture_label_zh"), comp.get("axis_centering"), comp.get("height_rule_label")] if x),
            "risk_level": best.get("risk_level") or "unknown",
            "window_edge_clearance_cm": comp.get("window_edge_clearance_cm"),
            "surface_active_light": light_sem.get("surface_active_light") if light_sem.get("surface_active_light_scope") == "wall" else None,
            "landscape_competition_score": light_sem.get("landscape_competition_score") if light_sem.get("landscape_competition_scope") == "wall" else None,
            "light_components": best.get("light_components"),
            "light_risk_band": best.get("light_risk_band"),
            "light_penalty_band": best.get("light_penalty_band") or (best.get("light_components") or {}).get("penalty_band"),
            "needs_review": bool(best.get("needs_review")),
            "production_usable": True,
        },
        "light": {
            **(best.get("light_facts") or {}),
            **{k: v for k, v in light_sem.items() if k in {"theta", "exterior_brightness_class", "ambient_light_mode", "shadow_presence", "temporal_light_quality", "ceiling_height"}},
        },
    }
    art_delta = {
        "mood_register": artwork_analysis.get("mood_register"),
        "visual_complexity": artwork_analysis.get("visual_complexity") or artwork_analysis.get("complexity_score"),
    }
    art_delta = {k: v for k, v in art_delta.items() if v is not None}
    if art_delta:
        delta["artwork"] = art_delta
    return delta


def _result_delivery_delta(payload: dict) -> dict:
    candidates = payload.get("candidates") or []
    partials = payload.get("partial_review_images") or []
    final_count = sum(1 for c in candidates if isinstance(c, dict) and c.get("final_image_url"))
    return {
        "delivery": {
            "final_url_count": int(final_count),
            "partial_review_count": int(len(partials)),
        },
        "qa": {
            "status": "passed" if final_count else ("review" if partials else "failed"),
        },
    }


DEFAULT_RENDER_PROVIDER = "apiyi_gpt_image2_vip"
RENDER_PROVIDER_ALIASES = {
    "apiyi": "apiyi_gpt_image2_vip",
    "apiyi_gpt_image2_vip": "apiyi_gpt_image2_vip",
    "gpt_image2_vip": "apiyi_gpt_image2_vip",
    "gpt-image-2-vip": "apiyi_gpt_image2_vip",
    "bfl": "bfl",
    "bfl_official": "bfl",
    "bfl_flux2": "bfl",
    "mock": "mock",
    "dry_run": "mock",
    "noop": "mock",
}


def normalize_render_provider(value) -> str:
    raw = str(value or os.getenv("HANGING_RENDER_PROVIDER") or DEFAULT_RENDER_PROVIDER).strip().lower()
    normalized = RENDER_PROVIDER_ALIASES.get(raw)
    if normalized:
        return normalized
    print(f"[worker] unknown render provider={raw!r}; fallback to {DEFAULT_RENDER_PROVIDER}", flush=True)
    return DEFAULT_RENDER_PROVIDER


def resolve_render_provider(job: dict) -> tuple[str, str]:
    render = job.get("render") if isinstance(job.get("render"), dict) else {}
    candidates = [
        ("render.provider", render.get("provider")),
        ("render.render_provider", render.get("render_provider")),
        ("render_provider", job.get("render_provider")),
        ("hanging_provider", job.get("hanging_provider")),
        ("provider", job.get("provider")),
        ("env.HANGING_RENDER_PROVIDER", os.getenv("HANGING_RENDER_PROVIDER")),
    ]
    for source, value in candidates:
        if value is not None and str(value).strip():
            return normalize_render_provider(value), source
    return DEFAULT_RENDER_PROVIDER, "default"


def _stage_b_request(job: dict) -> tuple[bool, dict | None]:
    """Read the two independent Stage B switches from a job payload."""
    soft = bool(job.get("soft_furnishing_requested"))
    wallpaper = render_api.normalize_wallpaper_recolor(job.get("wallpaper_recolor"))
    return soft, wallpaper


def _round_cm(value) -> int | None:
    try:
        return round(float(value))
    except Exception:
        return None


def _advisor_progress_from_client_result(client_res: dict) -> str:
    """把 hanging 结构化候选转为等待页可读的顾问式短句。

    worker 不依赖 LLM；这里作为 LLM 未接入时的确定性兜底，避免等待页只看到阶段枚举。
    """
    best = client_res.get("best_plan") if isinstance(client_res, dict) else None
    if not best:
        return "我没有贸然选择风险位置，正在整理更稳妥的处理建议…"

    install = best.get("install_reference") or {}
    comp = best.get("composition_context") or {}
    clearance = best.get("structural_clearance") or {}

    wall = best.get("wall_position_zh") or comp.get("above_furniture_label_zh") or "主墙面"
    tags = [str(x) for x in (best.get("reason_tags") or []) if x][:2]
    h = _round_cm(install.get("center_height_cm") or install.get("provisional_center_height_cm"))
    parts = [f"我倾向把作品放在{wall}"]
    if h is not None:
        parts.append(f"画心离地约{h}厘米")
    if tags:
        parts.append("这里" + "、".join(tags))

    elements = clearance.get("elements") if isinstance(clearance, dict) else []
    safe_notes = []
    for item in elements or []:
        label = item.get("label_zh")
        gap = _round_cm(item.get("edge_clearance_cm"))
        if label and gap is not None:
            safe_notes.append(f"距{label}约{gap}厘米")
        if len(safe_notes) >= 2:
            break
    if safe_notes:
        parts.append("并保留" + "、".join(safe_notes) + "的安全间距")

    return "，".join(parts) + "。正在进入最终效果图渲染…"


class Worker:
    def __init__(self) -> None:
        self.idle_since: float | None = None
        self._shutdown_task: asyncio.Task | None = None
        self._current_job_id: str | None = None
        self._executed_jobs: set[str] = set()
        self._causal_engine = None
        self._causal_base_delta: dict = {}

    async def _create_ws(self):
        """Create a fresh WebSocket connection to the business server."""
        url = f"{MOLINK_WS_URL}?token={WORKER_SECRET}"
        return await websockets.connect(url, ping_interval=30, ping_timeout=10, max_size=None)

    async def register(self, ws) -> None:
        await ws.send(json.dumps({
            "type": "register",
            "instance_id": INSTANCE_ID,
            "gpu_model": GPU_MODEL,
            "gpu_count": GPU_COUNT,
            "worker_version": WORKER_VERSION,
            "services": ["metric3d", "semantic", "hanging", "bfl"],
        }))
        print(f"[worker] 已注册 {INSTANCE_ID} ({GPU_MODEL})")

    async def send_result(self, ws, job, payload) -> None:
        result_stage, result_state, result_pct, result_message = result_builder.terminal_progress_contract(
            payload.get("status") if isinstance(payload, dict) else None
        )
        if isinstance(payload, dict) and payload.get("narration_bundle") is None:
            bundle = self._narration_bundle(
                result_stage, result_state, result_pct,
                result_message,
                delta=_result_delivery_delta(payload),
            )
            if bundle:
                payload["narration_bundle"] = bundle
        print(
            f"[worker-debug] send_result job_id={job.get('job_id')} order_id={job.get('order_id')} "
            f"status={payload.get('status')} exit_code={payload.get('exit_code')} "
            f"candidates={len(payload.get('candidates') or [])} "
            f"final_urls={sum(1 for c in (payload.get('candidates') or []) if c.get('final_image_url'))} "
            f"partial_images={len(payload.get('partial_review_images') or [])} "
            f"error={payload.get('error')}",
            flush=True,
        )
        try:
            await ws.send(json.dumps({
                "type": "result", "job_id": job["job_id"],
                "status": payload.get("status"), "order_id": job.get("order_id"),
                "payload": payload,
            }))
            await self.progress(
                ws, job, result_stage, result_state, result_pct, result_message,
                causal_delta=_result_delivery_delta(payload),
            )
        except Exception as exc:
            # Connection died — reconnect and resend via a fresh connection.
            print(f"[worker] send_result failed ({exc}), reconnecting to resend result…", flush=True)
            fresh_ws = None
            try:
                fresh_ws = await self._create_ws()
                await self.register(fresh_ws)
                await fresh_ws.send(json.dumps({
                    "type": "result", "job_id": job["job_id"],
                    "status": payload.get("status"), "order_id": job.get("order_id"),
                    "payload": payload,
                }))
                await self.progress(
                    fresh_ws, job, result_stage, result_state, result_pct, result_message,
                    causal_delta=_result_delivery_delta(payload),
                )
                print(f"[worker-debug] send_result retransmitted via fresh connection job_id={job.get('job_id')}", flush=True)
            except Exception as retry_exc:
                print(f"[worker] send_result retransmit also failed ({retry_exc}), result saved locally only", flush=True)
            finally:
                if fresh_ws:
                    try:
                        await fresh_ws.close()
                    except Exception:
                        pass

    async def heartbeat(self, ws) -> None:
        while True:
            await asyncio.sleep(30)
            try:
                await ws.send(json.dumps({
                    "type": "heartbeat", "instance_id": INSTANCE_ID,
                    "status": "busy" if self._current_job_id else "idle",
                    "current_job_id": self._current_job_id,
                    "uptime_s": int(time.time() - START_TS),
                }))
            except Exception:
                return

    async def progress(self, ws, job, stage, state, pct, message, advisor_message=None, causal_delta=None) -> None:
        payload = {
            "type": "progress",
            "job_id": job["job_id"],
            "order_id": job.get("order_id"),
            "stage": stage,
            "stage_state": state,
            "pct": pct,
            "message": message,
        }
        if advisor_message:
            payload["advisor_message"] = advisor_message
        bundle = self._narration_bundle(stage, state, pct, message, advisor_message, causal_delta)
        if bundle:
            payload["narration_bundle"] = bundle

        print(
            "[worker-progress:send] "
            f"job_id={job.get('job_id')} "
            f"order_id={job.get('order_id')} "
            f"stage={stage} state={state} pct={pct} "
            f"msg_len={len(message or '')} "
            f"advisor_len={len(advisor_message or '')} "
            f"advisor_preview={(advisor_message or '')[:80]}",
            flush=True,
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                await ws.send(json.dumps(payload, ensure_ascii=False))
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                print(f"[progress] failed to send after {max_retries} attempts: {e}", flush=True)
                print(f"[progress] progress message dropped, pipeline continues", flush=True)
                return

    def _reset_causal_engine(self, job: dict | None = None) -> None:
        self._causal_engine = CausalEngine() if _causal_enabled() else None
        artwork = (job or {}).get("artwork") or {}
        self._causal_base_delta = {
            "artwork": {
                "width_cm": round(float(artwork.get("physical_width_m") or 0.0) * 100.0, 1) if artwork else 0.0,
                "height_cm": round(float(artwork.get("physical_height_m") or 0.0) * 100.0, 1) if artwork else 0.0,
                "has_frame": bool(artwork.get("has_frame", False)) if artwork else False,
            }
        } if self._causal_engine is not None else {}

    def _narration_bundle(self, stage, state, pct, message, advisor_message=None, delta=None):
        if self._causal_engine is None or Event is None:
            return None
        try:
            merged = dict(self._causal_base_delta or {})
            for group, value in (delta or {}).items():
                if isinstance(value, dict) and isinstance(merged.get(group), dict):
                    merged[group] = {**merged[group], **value}
                else:
                    merged[group] = value
            merged["progress"] = {"message": message or "", "advisor_message": advisor_message}
            bundle = self._causal_engine.on_event(Event(stage, state, float(pct or 0), merged))
            return asdict(bundle)
        except Exception as exc:
            print(f"[causal] narration bundle skipped: {exc}", flush=True)
            return None

    async def on_job_received(self, job_id) -> None:
        if self._shutdown_task:
            self._shutdown_task.cancel()
            self._shutdown_task = None
        self.idle_since = None
        self._current_job_id = job_id

    async def on_job_complete(self, ws) -> None:
        self._current_job_id = None
        self._causal_engine = None
        self._causal_base_delta = {}
        try:
            await ws.send(json.dumps({"type": "worker_idle", "instance_id": INSTANCE_ID}))
        except Exception as e:
            print(f"[on_job_complete] idle message failed: {e}, countdown starts anyway", flush=True)
        self.idle_since = time.time()
        self._shutdown_task = asyncio.create_task(self._idle_countdown(ws))

    async def _idle_countdown(self, ws) -> None:
        if IDLE_TIMEOUT_S <= 0:
            return  # idle shutdown disabled
        try:
            await asyncio.sleep(IDLE_TIMEOUT_S)
        except asyncio.CancelledError:
            return
        print(f"[worker] 空闲 {IDLE_TIMEOUT_S}s 无新任务，准备关机")
        try:
            await ws.send(json.dumps({
                "type": "shutting_down", "instance_id": INSTANCE_ID, "reason": "idle_timeout"
            }))
        except Exception:
            pass
        await asyncio.sleep(2)
        await autodl_api.stop_self()

    async def run_supplement_job(self, ws, job) -> None:
        """Render selected walls from a persisted hanging_result.zip.

        This job intentionally skips Metric3D/Semantic/Hanging and only reuses the
        existing render path with manifest filters. It is triggered by waiting-page
        wall preferences and does not block the main recommendation delivery.
        A primary-wall wallpaper rerender seeds the pipeline from the archived
        pre-styling Stage A image instead of repainting the already styled final.
        """
        job_id = job.get("job_id")
        print(f"[worker-debug] supplement_job_received job_id={job_id} order_id={job.get('order_id')} targets={job.get('target_wall_ids')}", flush=True)
        await self.on_job_received(job["job_id"])
        self._reset_causal_engine(job)
        await ws.send(json.dumps({"type": "job_accepted", "job_id": job["job_id"], "instance_id": INSTANCE_ID}))
        render_provider, render_provider_source = resolve_render_provider(job)
        diag = diagnostics.DiagnosticsRun.create(job)
        diag_tasks: list[asyncio.Task] = []

        def diag_later(method, *args) -> None:
            if not getattr(diag, "enabled", False):
                return
            diag_tasks.append(asyncio.create_task(asyncio.to_thread(method, *args)))

        async def flush_diag_tasks() -> None:
            if not diag_tasks:
                return
            results = await asyncio.gather(*diag_tasks, return_exceptions=True)
            diag_tasks.clear()
            for item in results:
                if isinstance(item, Exception):
                    print(f"[diagnostics] background export failed: {item}", flush=True)

        with tempfile.TemporaryDirectory(prefix="hanging_supplement_") as td:
            work_dir = Path(td)
            payload_for_diag: dict | None = None
            try:
                await self.progress(ws, job, "render", "running", 70, "正在根据您点选的墙面补充生成效果图…")
                artifact_url = str(job.get("hanging_result_zip_url") or "").strip()
                if not artifact_url:
                    raise RuntimeError("supplement job missing hanging_result_zip_url")
                hanging_zip = await asyncio.to_thread(pipeline.download_to_path, artifact_url, work_dir / "hanging_result.zip")
                diag_later(diag.copy_file, hanging_zip, "04_hanging/hanging_result.zip")
                client_res = await asyncio.to_thread(pipeline.extract_client_result, hanging_zip)
                diag_later(diag.write_json, "04_hanging/client_result.json", client_res)
                flux_out = work_dir / "flux_out"
                target_wall_ids = [str(x) for x in (job.get("target_wall_ids") or []) if str(x).strip()]
                target_candidate_ids = [str(x) for x in (job.get("target_candidate_ids") or []) if str(x).strip()]
                limit = (job.get("render") or {}).get("global_limit") if isinstance(job.get("render"), dict) else None
                stage_a_base_by_asset_id = job.get("stage_a_base_by_asset_id") if isinstance(job.get("stage_a_base_by_asset_id"), dict) else {}
                stage_a_base_by_wall_id = job.get("stage_a_base_by_wall_id") if isinstance(job.get("stage_a_base_by_wall_id"), dict) else {}
                if stage_a_base_by_asset_id or stage_a_base_by_wall_id:
                    render_aggregate = await asyncio.to_thread(
                        pipeline.prepare_pre_styling_render_aggregate,
                        hanging_zip, flux_out,
                        stage_a_base_by_asset_id=stage_a_base_by_asset_id,
                        stage_a_base_by_wall_id=stage_a_base_by_wall_id,
                        target_wall_ids=target_wall_ids,
                        target_candidate_ids=target_candidate_ids,
                        provider=render_provider,
                    )
                else:
                    render_aggregate = await asyncio.to_thread(
                        render_api.run_pipeline,
                        str(hanging_zip), str(flux_out), provider=render_provider, limit=limit,
                        target_wall_ids=target_wall_ids, target_candidate_ids=target_candidate_ids,
                    )
                soft_furnishing, wallpaper_recolor = _stage_b_request(job)
                if soft_furnishing or wallpaper_recolor:
                    await self.progress(ws, job, "styling", "running", 84, "正在优化空间风格与色彩…")
                stage_b_aggregate = await asyncio.to_thread(
                    render_api.run_stage_b_pipeline,
                    str(hanging_zip), str(flux_out),
                    provider=render_provider,
                    soft_furnishing_requested=soft_furnishing,
                    wallpaper_recolor=wallpaper_recolor,
                )
                render_aggregate["stage_b"] = stage_b_aggregate
                diag_later(diag.write_json, "05_render/render_aggregate.json", render_aggregate)
                diag_later(diag.copy_tree, flux_out, "05_render/flux_out")
                await self.progress(ws, job, "r2_upload", "running", 92, "正在上传您点选墙面的补充效果图…")
                r2_result = await asyncio.to_thread(r2_upload.upload_all, flux_out, job["r2_output_prefix"])
                diag_later(diag.write_json, "06_r2_upload/r2_result.json", r2_result)
                payload_for_diag = result_builder.build(client_res, render_aggregate, r2_result, job)
                payload_for_diag["job_kind"] = "hanging_supplement_render"
                payload_for_diag["supplement"] = True
                payload_for_diag["selected_wall_ids"] = target_wall_ids
                payload_for_diag["target_candidate_ids"] = target_candidate_ids
                payload_for_diag["primary_wall_rerender"] = bool(job.get("primary_wall_rerender"))
                payload_for_diag["primary_wall_id"] = job.get("primary_wall_id")
                payload_for_diag["wallpaper_recolor_requested"] = bool(job.get("wallpaper_recolor_requested"))
                payload_for_diag["stage_a_base_source"] = render_aggregate.get("stage_a_base_source") or job.get("stage_a_base_source")
                diag_later(diag.write_json, "99_result/result_payload_pre_send.json", payload_for_diag)
                await self.send_result(ws, job, payload_for_diag)
                await flush_diag_tasks()
                await asyncio.to_thread(diag.finalize_zip, payload_for_diag, None)
                await self.on_job_complete(ws)
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                payload_for_diag = {
                    "job_id": job["job_id"], "order_id": job.get("order_id"),
                    "job_kind": "hanging_supplement_render", "supplement": True,
                    "selected_wall_ids": job.get("target_wall_ids") or [],
                    "primary_wall_rerender": bool(job.get("primary_wall_rerender")),
                    "primary_wall_id": job.get("primary_wall_id"),
                    "wallpaper_recolor_requested": bool(job.get("wallpaper_recolor_requested")),
                    "stage_a_base_source": job.get("stage_a_base_source"),
                    "status": "failed", "exit_code": "FAILED_SUPPLEMENT_RENDER",
                    "candidates": [], "not_recommended": [], "partial_review_images": [],
                    "error": str(exc), "metrics": {"render_provider": render_provider},
                }
                await self.send_result(ws, job, payload_for_diag)
                await flush_diag_tasks()
                await asyncio.to_thread(diag.finalize_zip, payload_for_diag, exc)
                await self.on_job_complete(ws)
            finally:
                if job_id:
                    self._executed_jobs.add(job_id)

    async def run_job(self, ws, job) -> None:
        job_id = job.get("job_id")
        if job_id and job_id in self._executed_jobs:
            print(f"[worker] duplicate job_id={job_id} — skipping (already executed)", flush=True)
            await ws.send(json.dumps({"type": "job_skipped", "job_id": job_id, "instance_id": INSTANCE_ID, "reason": "duplicate"}))
            return
        if job.get("job_kind") == "hanging_supplement_render" or job.get("supplement") is True:
            await self.run_supplement_job(ws, job)
            return
        print(f"[worker-debug] job_received job_id={job.get('job_id')} order_id={job.get('order_id')} r2_prefix={job.get('r2_output_prefix')}", flush=True)
        await self.on_job_received(job["job_id"])
        self._reset_causal_engine(job)
        await ws.send(json.dumps({"type": "job_accepted", "job_id": job["job_id"], "instance_id": INSTANCE_ID}))
        print(f"[worker-debug] job_accepted job_id={job.get('job_id')}", flush=True)

        diag = diagnostics.DiagnosticsRun.create(job)
        render_provider, render_provider_source = resolve_render_provider(job)
        soft_furnishing, wallpaper_recolor = _stage_b_request(job)
        print(
            f"[worker-debug] render_provider={render_provider} source={render_provider_source} "
            f"raw_render={(job.get('render') or {}).get('provider') if isinstance(job.get('render'), dict) else None}",
            flush=True,
        )
        diag_tasks: list[asyncio.Task] = []
        payload_for_diag: dict | None = None
        run_error: BaseException | None = None

        def diag_later(method, *args) -> None:
            if not getattr(diag, "enabled", False):
                return
            diag_tasks.append(asyncio.create_task(asyncio.to_thread(method, *args)))

        async def flush_diag_tasks() -> None:
            if not diag_tasks:
                return
            results = await asyncio.gather(*diag_tasks, return_exceptions=True)
            diag_tasks.clear()
            for item in results:
                if isinstance(item, Exception):
                    print(f"[diagnostics] background export failed: {item}", flush=True)

        with tempfile.TemporaryDirectory(prefix="hanging_job_") as td:
            work_dir = Path(td)
            try:
                diag_later(diag.write_json, "00_job/runtime_context.json", {
                    "temp_work_dir": str(work_dir),
                    "instance_id": INSTANCE_ID,
                    "gpu_model": GPU_MODEL,
                    "worker_version": WORKER_VERSION,
                    "render_provider": render_provider,
                    "render_provider_source": render_provider_source,
                    "render_provider_raw": (job.get("render") or {}).get("provider") if isinstance(job.get("render"), dict) else None,
                })

                await self.progress(ws, job, "metric3d", "running", 15, "正在测量空间深度…")
                inputs = await asyncio.to_thread(pipeline.fetch_inputs, job, work_dir)
                diag_later(diag.copy_file, inputs.get("room"), "01_inputs/room.jpg")
                diag_later(diag.copy_file, inputs.get("artwork"), "01_inputs/artwork.png")
                metric_zip = await asyncio.to_thread(pipeline.call_metric3d, inputs["room"], work_dir)

                await self.progress(ws, job, "semantic", "running", 35, "正在识别墙面与结构信息…")
                scene_profile = (job.get("rules_overrides") or {}).get("scene_profile", "home")
                semantics = await asyncio.to_thread(
                    pipeline.call_semantic,
                    inputs["room"], metric_zip, inputs["artwork"], scene_profile, work_dir,
                    styling_masks_requested=bool(soft_furnishing or wallpaper_recolor),
                )
                diag_later(diag.copy_file, semantics, "03_semantic/semantics.json")

                semantic_payload = _safe_json_load(semantics)
                semantic_delta = {"space": _count_semantic_space(semantic_payload)}
                if isinstance(semantic_payload.get("light"), dict):
                    semantic_delta["light"] = semantic_payload.get("light")
                await self.progress(ws, job, "hanging", "running", 55, "正在筛选安全挂画位置…", causal_delta=semantic_delta)
                rules_json = pipeline.build_rules_overrides(job)
                if rules_json:
                    diag_later(diag.write_text, "04_hanging/rules_overrides.json", rules_json)
                hanging_zip = await asyncio.to_thread(
                    pipeline.call_hanging,
                    inputs["room"], metric_zip, inputs["artwork"], semantics, rules_json, work_dir
                )
                diag_later(diag.copy_file, hanging_zip, "04_hanging/hanging_result.zip")
                diag_later(diag.extract_zip, hanging_zip, "04_hanging/extracted")
                hanging_artifact_url = None
                hanging_artifact_key = None
                try:
                    hanging_artifact_key = f"{job['r2_output_prefix'].rstrip('/')}/artifacts/hanging_result.zip"
                    hanging_artifact_url = await asyncio.to_thread(r2_upload.upload_file, hanging_zip, hanging_artifact_key)
                    print(f"[worker-debug] persisted hanging_result_zip={hanging_artifact_url}", flush=True)
                except Exception as artifact_exc:
                    print(f"[worker] hanging_result.zip persist skipped: {artifact_exc}", flush=True)
                client_res = await asyncio.to_thread(pipeline.extract_client_result, hanging_zip)
                diag_later(diag.write_json, "04_hanging/client_result.json", client_res)
                await self.progress(
                    ws, job, "hanging", "running", 60,
                    "已完成挂画方案筛选，正在进入效果图渲染…",
                    advisor_message=_advisor_progress_from_client_result(client_res),
                    causal_delta=_best_plan_delta(client_res),
                )

                exit_code = str(client_res.get("exit_code", ""))
                if exit_code.startswith("FAILED") or exit_code == "PARTIAL_ALL_REJECTED_WITH_REASONS":
                    payload_for_diag = result_builder.build_failed_payload(job, client_res)
                    await self.send_result(ws, job, payload_for_diag)
                    await flush_diag_tasks()
                    await asyncio.to_thread(diag.finalize_zip, payload_for_diag, None)
                    await self.on_job_complete(ws)
                    return

                await self.progress(ws, job, "render", "running", 65, "几何分析完成，正在生成高保真效果图…")
                limit = (job.get("render") or {}).get("global_limit") if isinstance(job.get("render"), dict) else None
                flux_out = work_dir / "flux_out"
                render_aggregate = await asyncio.to_thread(
                    render_api.run_pipeline,
                    str(hanging_zip), str(flux_out), provider=render_provider, limit=limit
                )
                if soft_furnishing or wallpaper_recolor:
                    await self.progress(ws, job, "styling", "running", 82, "正在优化空间风格与色彩…")
                stage_b_aggregate = await asyncio.to_thread(
                    render_api.run_stage_b_pipeline,
                    str(hanging_zip), str(flux_out),
                    provider=render_provider,
                    soft_furnishing_requested=soft_furnishing,
                    wallpaper_recolor=wallpaper_recolor,
                )
                render_aggregate["stage_b"] = stage_b_aggregate
                diag_later(diag.write_json, "05_render/render_aggregate.json", render_aggregate)
                diag_later(diag.copy_tree, flux_out, "05_render/flux_out")

                await self.progress(ws, job, "r2_upload", "running", 90, "上传效果图中…")
                r2_result = await asyncio.to_thread(r2_upload.upload_all, flux_out, job["r2_output_prefix"])
                diag_later(diag.write_json, "06_r2_upload/r2_result.json", r2_result)

                payload_for_diag = result_builder.build(client_res, render_aggregate, r2_result, job)
                if hanging_artifact_url:
                    payload_for_diag["hanging_result_zip_url"] = hanging_artifact_url
                if hanging_artifact_key:
                    payload_for_diag["hanging_result_zip_key"] = hanging_artifact_key
                diag_later(diag.write_json, "99_result/result_payload_pre_send.json", payload_for_diag)
                await self.send_result(ws, job, payload_for_diag)
                await flush_diag_tasks()
                await asyncio.to_thread(diag.finalize_zip, payload_for_diag, None)
                await self.on_job_complete(ws)
            except Exception as exc:  # noqa: BLE001
                run_error = exc
                traceback.print_exc()
                payload_for_diag = {
                    "job_id": job["job_id"], "order_id": job.get("order_id"),
                    "status": "failed", "exit_code": "FAILED_WORKER_EXCEPTION",
                    "candidates": [], "not_recommended": [], "partial_review_images": [],
                    "error": str(exc), "metrics": {},
                }
                await self.send_result(ws, job, payload_for_diag)
                await flush_diag_tasks()
                await asyncio.to_thread(diag.finalize_zip, payload_for_diag, exc)
                await self.on_job_complete(ws)
            finally:
                if job_id:
                    self._executed_jobs.add(job_id)

    async def listen(self, ws) -> None:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            mtype = msg.get("type")
            if mtype == "job_dispatch":
                await self.run_job(ws, msg)
            elif mtype == "ping":
                await ws.send(json.dumps({"type": "pong", "instance_id": INSTANCE_ID}))
            elif mtype == "cancel":
                print(f"[worker] 收到取消（暂不中断在途作业）: {msg.get('job_id')}")


async def connect_with_retry() -> None:
    worker = Worker()
    while True:
        try:
            url = f"{MOLINK_WS_URL}?token={WORKER_SECRET}"
            async with websockets.connect(url, ping_interval=30, ping_timeout=10, max_size=None) as ws:
                await worker.register(ws)
                hb = asyncio.create_task(worker.heartbeat(ws))
                try:
                    await worker.listen(ws)
                finally:
                    hb.cancel()
        except Exception as exc:  # noqa: BLE001
            print(f"[worker] ws 断开: {exc}，5s 后重连")
            await asyncio.sleep(5)


START_TS = time.time()

if __name__ == "__main__":
    asyncio.run(connect_with_retry())
