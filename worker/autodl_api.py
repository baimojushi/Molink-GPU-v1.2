"""worker/autodl_api.py —— worker 自调 AutoDL 关机接口。

worker 完成作业后启动 idle 倒计时，超时无新作业则调用本模块自关机，
业务端无需轮询或持有实例状态。
"""
from __future__ import annotations

import os

import httpx

AUTODL_MODE = os.getenv("AUTODL_MODE", "pro")
AUTODL_BASE = os.getenv(
    "AUTODL_API_BASE",
    "https://api.autodl.com" if AUTODL_MODE == "pro" else "https://www.autodl.com/api/v1",
).rstrip("/")
AUTODL_TOKEN = os.getenv("AUTODL_TOKEN", "")
AUTODL_AUTH_PREFIX = os.getenv("AUTODL_AUTH_PREFIX", "")
INSTANCE_ID = os.getenv("AUTODL_INSTANCE_ID", "")
PATH_STOP = os.getenv(
    "AUTODL_PATH_STOP",
    "/api/v1/dev/instance/pro/power_off" if AUTODL_MODE == "pro" else "/instance/shutdown",
)
UUID_FIELD = os.getenv("AUTODL_PRO_UUID_FIELD", "instance_uuid")


def _auth_header_value() -> str:
    if not AUTODL_AUTH_PREFIX:
        return AUTODL_TOKEN
    return f"{AUTODL_AUTH_PREFIX} {AUTODL_TOKEN}"


async def stop_self() -> bool:
    """关闭当前实例。缺少 token/instance_id 时跳过（本地/调试环境）。"""
    if not AUTODL_TOKEN or not INSTANCE_ID:
        print("[autodl] 未配置 AUTODL_TOKEN/AUTODL_INSTANCE_ID，跳过自关机")
        return False
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{AUTODL_BASE}{PATH_STOP}",
                headers={"Authorization": _auth_header_value(), "Content-Type": "application/json"},
                json={UUID_FIELD: INSTANCE_ID},
            )
            print(f"[autodl] shutdown 请求返回 HTTP {resp.status_code}")
            return resp.status_code < 400
    except Exception as exc:  # noqa: BLE001
        print(f"[autodl] 自关机失败: {exc}")
        return False
