# WebSocket send_result 超时中断诊断与修复文档

## 1. 问题现象

Worker 完成全部 pipeline（metric3d → semantic → hanging → render → r2_upload）后，在向业务服务器发送最终结果（`send_result`）时失败：

```
websockets.exceptions.ConnectionClosedError: no close frame received or sent
send_result job_id=job_xxx status=failed exit_code=FAILED_WORKER_EXCEPTION
```

**影响：**
- 本地 pipeline 全部完成，渲染结果已上传 R2
- 但业务服务器未收到 `result` 消息，订单状态停留在"处理中"
- 业务服务器重复派发同一 job（观察到 `dd30a2f9` 被重发 3 次），浪费 GPU 资源

## 2. 根因分析

### 2.1 WebSocket 空闲超时

业务服务器对 WebSocket 连接设置了**空闲超时**（idle timeout），当连接上没有消息传输时自动断开。

完整 pipeline 耗时：
- metric3d: ~30s
- semantic: ~20s
- hanging: ~60s
- render: ~120-300s（含 full-space 纹理渲染时更长）
- r2_upload: ~10s
- **总计: 230-430s（约 4-7 分钟）**

在 render 阶段（120-300s），worker 只向业务服务器发送了一次 progress 消息（65%）。如果业务服务器的 WebSocket 空闲超时 < 120s，连接会在 render 完成前被服务端静默关闭。

### 2.2 连接断开后的行为链

```
1. progress(ws, "r2_upload", 90%) → ws.send() → ConnectionClosedError
2. except 捕获 → payload = build_failed_payload()
3. send_result(ws, failed_payload) → ws.send() → ConnectionClosedError（同一死连接）
4. on_job_complete(ws) → ws.send("worker_idle") → ConnectionClosedError
5. 外层 connect_with_retry 捕获 → 5s 后重连
6. 业务服务器未收到 result → 重发同一 job
```

### 2.3 业务服务器重复派发的原因

业务服务器在 WebSocket 断开后：
- 没有收到 `result` 消息
- 可能将该 job 标记为"超时"或"失败"
- 自动重发机制将该 job 重新放入队列
- Worker 重连后收到新的 `job_dispatch`，**重新执行同一 job**

## 3. Worker 端修复（已部署）

### 3.1 修复内容

`worker/main.py` — `Worker.send_result()` 方法增加了**重连重发**逻辑：

```python
async def send_result(self, ws, job, payload) -> None:
    try:
        await ws.send(result_message)
        await self.progress(ws, ...)
    except Exception as exc:
        # 连接已死 → 创建新连接重发
        fresh_ws = await self._create_ws()  # 新建 WebSocket
        await self.register(fresh_ws)        # 重新注册
        await fresh_ws.send(result_message)  # 重发结果
        await self.progress(fresh_ws, ...)   # 重发进度
        await fresh_ws.close()               # 清理临时连接
```

### 3.2 修复效果

- `send_result` 失败时自动创建新 WebSocket 连接
- 通过新连接重新发送结果 payload
- 确保业务服务器能收到最终结果（即使原连接已断）

### 3.3 限制

- 重连重发的是 `result` 消息，不是 `progress` 消息
- 业务服务器可能已经将该 job 标记为"超时"，收到延迟的 result 后需要正确处理
- 如果业务服务器在 result 到达前已重发 job，worker 会**再次执行**同一 job

## 4. 业务服务器需要修复的内容

### 4.1 延长 WebSocket 空闲超时（关键）

**问题：** 当前空闲超时 < 120s，短于 render 阶段耗时。

**建议：** 将空闲超时设置为 **≥ 600s（10 分钟）**，或根据 render 最大预期耗时动态调整。

```python
# 示例：websockets 服务端配置
async with websockets.serve(
    handler,
    host, port,
    ping_interval=30,      # 每 30s 发 ping
    ping_timeout=10,       # 10s 无 pong 视为断开
    close_timeout=10,      # 关闭超时
    # 关键：增加空闲超时
    # 如果服务端使用自定义 idle_timeout，设置为 ≥ 600
) as server:
```

### 4.2 添加 WebSocket 心跳/Keepalive

**问题：** 连接上没有周期性消息时，中间网络设备（NAT、负载均衡器）可能静默断开连接。

**建议：** 业务服务器定期向 worker 发送 `ping` 消息，worker 回复 `pong`。当前 worker 已支持：

```python
# worker/main.py — Worker.listen()
elif mtype == "ping":
    await ws.send(json.dumps({"type": "pong", "instance_id": INSTANCE_ID}))
```

业务服务器应每 **15-30s** 发送一次 `ping`：

```python
# 业务服务器端伪代码
async def keepalive(ws):
    while True:
        await asyncio.sleep(20)
        await ws.send(json.dumps({"type": "ping"}))
```

### 4.3 改进 job 重发逻辑

**问题：** 当前业务服务器在 WebSocket 断开后无条件重发 job，导致重复执行。

**建议：**
1. 收到 `result` 消息后，检查 `job_id` 是否已有成功记录
2. 如果已有成功记录（含 R2 URL），**不重发**，直接返回缓存结果
3. 重发前等待一段冷却时间（如 30s），给 worker 的 `send_result` 重连重发留出时间

```python
# 业务服务器端伪代码
async def on_websocket_disconnect(job_id):
    # 等待 30s，给 worker 重连重发 result 的时间
    await asyncio.sleep(30)
    
    # 检查是否已收到 result
    if result_store.has_result(job_id):
        return  # 不重发
    
    # 检查是否已有 R2 上传记录
    if r2_store.has_files(job_id):
        return  # 结果已上传，不重发
    
    # 真正需要重发
    await redispatch_job(job_id)
```

### 4.4 正确处理延迟到达的 result

**问题：** worker 的 `send_result` 重连重发可能导致 result 延迟到达。

**建议：** 业务服务器收到 `result` 时：
1. 检查 job 当前状态
2. 如果已在"处理中"或"已完成"，更新状态而非创建新任务
3. 如果已被重发，取消重发的任务

```python
# 业务服务器端伪代码
async def on_result_received(job_id, result):
    job = job_store.get(job_id)
    
    if job.status == "completed":
        # 重复 result，忽略
        return
    
    if job.status == "dispatched_retry":
        # result 先到，取消重发的任务
        cancel_retry(job_id)
    
    job.mark_completed(result)
```

## 5. 排查 checklist

| 检查项 | 当前状态 | 目标 |
|---|---|---|
| WebSocket 空闲超时 | < 120s（推测） | ≥ 600s |
| 服务端 ping 心跳 | 无 | 每 15-30s |
| job 重发冷却时间 | 无（立即重发） | ≥ 30s |
| 重复 result 处理 | 未知 | 幂等处理 |
| R2 文件去重 | 未知 | 同 job_id 不重复上传 |

## 6. 已发生的案例

| job_id | 发生时间 | 现象 | 结果 |
|---|---|---|---|
| `1bfacbe3` | 2026-06-25 23:29 | send_result 断开，重连后重发 | 第 2 次执行成功 |
| `dd30a2f9` | 2026-06-26 00:21-00:37 | send_result 断开，被重发 3 次 | 每次都完成但 result 延迟到达 |

## 7. 临时缓解

在业务服务器修复前，可通过以下方式减少影响：

1. **减少 render 耗时：** 对纹理墙面使用 local crop 而非 full-space 渲染（设置 `MOLINK_RENDER_TEXTURE_GATE_ENABLED=0`）
2. **增加 worker 空闲超时：** `IDLE_TIMEOUT_S=600`（默认 300s）
3. **业务服务器端：** 重发前检查 R2 是否已有该 job 的文件
