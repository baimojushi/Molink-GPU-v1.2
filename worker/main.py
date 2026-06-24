"""worker/main.py —— GPU worker 主入口。

建立到业务端的 WebSocket 反隧道连接 → 注册 → 监听 job_dispatch →
串四段流水线 → 直传 R2 → 上报 result → 空闲 5 分钟自关机。
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import traceback
from pathlib import Path

import websockets

from worker import pipeline, render_api, r2_upload, result_builder, autodl_api, diagnostics

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
WORKER_VERSION = "1.0.0"

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

    async def progress(self, ws, job, stage, state, pct, message, advisor_message=None) -> None:
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

        await ws.send(json.dumps(payload, ensure_ascii=False))

    async def on_job_received(self, job_id) -> None:
        if self._shutdown_task:
            self._shutdown_task.cancel()
            self._shutdown_task = None
        self.idle_since = None
        self._current_job_id = job_id

    async def on_job_complete(self, ws) -> None:
        self._current_job_id = None
        await ws.send(json.dumps({"type": "worker_idle", "instance_id": INSTANCE_ID}))
        self.idle_since = time.time()
        self._shutdown_task = asyncio.create_task(self._idle_countdown(ws))

    async def _idle_countdown(self, ws) -> None:
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

    async def run_job(self, ws, job) -> None:
        print(f"[worker-debug] job_received job_id={job.get('job_id')} order_id={job.get('order_id')} r2_prefix={job.get('r2_output_prefix')}", flush=True)
        await self.on_job_received(job["job_id"])
        await ws.send(json.dumps({"type": "job_accepted", "job_id": job["job_id"], "instance_id": INSTANCE_ID}))
        print(f"[worker-debug] job_accepted job_id={job.get('job_id')}", flush=True)

        diag = diagnostics.DiagnosticsRun.create(job)
        render_provider, render_provider_source = resolve_render_provider(job)
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
                semantics = await asyncio.to_thread(pipeline.call_semantic, inputs["room"], metric_zip, scene_profile, work_dir)
                diag_later(diag.copy_file, semantics, "03_semantic/semantics.json")

                await self.progress(ws, job, "hanging", "running", 55, "正在筛选安全挂画位置…")
                rules_json = pipeline.build_rules_overrides(job)
                if rules_json:
                    diag_later(diag.write_text, "04_hanging/rules_overrides.json", rules_json)
                hanging_zip = await asyncio.to_thread(
                    pipeline.call_hanging,
                    inputs["room"], metric_zip, inputs["artwork"], semantics, rules_json, work_dir
                )
                diag_later(diag.copy_file, hanging_zip, "04_hanging/hanging_result.zip")
                diag_later(diag.extract_zip, hanging_zip, "04_hanging/extracted")
                client_res = await asyncio.to_thread(pipeline.extract_client_result, hanging_zip)
                diag_later(diag.write_json, "04_hanging/client_result.json", client_res)
                await self.progress(
                    ws, job, "hanging", "running", 60,
                    "已完成挂画方案筛选，正在进入效果图渲染…",
                    advisor_message=_advisor_progress_from_client_result(client_res),
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
                diag_later(diag.write_json, "05_render/render_aggregate.json", render_aggregate)
                diag_later(diag.copy_tree, flux_out, "05_render/flux_out")

                await self.progress(ws, job, "r2_upload", "running", 90, "上传效果图中…")
                r2_result = await asyncio.to_thread(r2_upload.upload_all, flux_out, job["r2_output_prefix"])
                diag_later(diag.write_json, "06_r2_upload/r2_result.json", r2_result)

                payload_for_diag = result_builder.build(client_res, render_aggregate, r2_result, job)
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

    async def send_result(self, ws, job, payload) -> None:
        print(
            f"[worker-debug] send_result job_id={job.get('job_id')} order_id={job.get('order_id')} "
            f"status={payload.get('status')} exit_code={payload.get('exit_code')} "
            f"candidates={len(payload.get('candidates') or [])} "
            f"final_urls={sum(1 for c in (payload.get('candidates') or []) if c.get('final_image_url'))} "
            f"partial_images={len(payload.get('partial_review_images') or [])} "
            f"error={payload.get('error')}",
            flush=True,
        )
        await ws.send(json.dumps({
            "type": "result", "job_id": job["job_id"],
            "status": payload.get("status"), "order_id": job.get("order_id"),
            "payload": payload,
        }))
        if payload.get("status") in {"succeeded", "succeeded_partial"}:
            await self.progress(ws, job, "done", "done", 100, "完成")
        else:
            await self.progress(ws, job, "failed", "failed", 92, "本次自动生成遇到异常，团队会继续为您处理…")

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
