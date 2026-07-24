from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

os.environ.setdefault("WORKER_SECRET", "test-worker-secret")

from worker import main as worker_main  # noqa: E402


class FakeWebSocket:
    def __init__(self, incoming=None, on_send=None):
        self.incoming = list(incoming or [])
        self.sent: list[dict] = []
        self.on_send = on_send
        self.closed = False

    def __aiter__(self):
        self._iter = iter(self.incoming)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def send(self, raw):
        payload = json.loads(raw)
        self.sent.append(payload)
        if self.on_send:
            self.on_send(payload)

    async def close(self):
        self.closed = True


def _new_worker(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(worker_main, "RESULT_SPOOL_DIR", tmp_path / "spool")
    monkeypatch.setattr(worker_main, "collect_runtime_resource", lambda **kwargs: {
        "instance_id": kwargs["instance_id"],
        "gpu_model": kwargs["gpu_model"],
        "gpu_count": kwargs["gpu_count"],
        "worker_version": kwargs["worker_version"],
        "precision": "test",
        "benchmark_score": None,
    })
    return worker_main.Worker()


def test_websocket_client_disables_aggressive_library_keepalive(monkeypatch, tmp_path):
    captured = {}
    sentinel = object()

    async def fake_connect(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return sentinel

    worker = _new_worker(tmp_path, monkeypatch)
    monkeypatch.setattr(worker_main.websockets, "connect", fake_connect)
    result = asyncio.run(worker._create_ws())

    assert result is sentinel
    assert captured["ping_interval"] is None
    assert captured["ping_timeout"] is None
    assert captured["max_queue"] == 128


def test_listener_keeps_consuming_ping_while_job_runs(monkeypatch, tmp_path):
    worker = _new_worker(tmp_path, monkeypatch)
    started = asyncio.Event()
    release = asyncio.Event()

    async def long_job(job):
        worker._current_job_id = job["job_id"]
        started.set()
        await release.wait()
        worker._current_job_id = None

    monkeypatch.setattr(worker, "run_job", long_job)
    ws = FakeWebSocket([
        json.dumps({"type": "job_dispatch", "job_id": "job-1", "order_id": "order-1"}),
        json.dumps({"type": "ping"}),
    ])

    async def scenario():
        worker._attach_ws(ws)
        await worker.listen(ws)
        await asyncio.wait_for(started.wait(), timeout=1)
        assert any(message.get("type") == "pong" for message in ws.sent)
        assert worker._current_job_task is not None
        assert not worker._current_job_task.done()
        release.set()
        await asyncio.wait_for(worker._current_job_task, timeout=1)

    asyncio.run(scenario())


def test_reconnect_registration_preserves_inflight_busy_state(monkeypatch, tmp_path):
    worker = _new_worker(tmp_path, monkeypatch)
    worker._current_job_id = "job-inflight"
    ws = FakeWebSocket()

    asyncio.run(worker.register(ws))
    register = ws.sent[-1]
    assert register["type"] == "register"
    assert register["status"] == "busy"
    assert register["current_job_id"] == "job-inflight"
    assert register["resume_inflight_job"] is True
    assert register["transport_protocol_version"] == 2


def test_terminal_result_is_spooled_retried_and_removed_only_after_ack(monkeypatch, tmp_path):
    worker = _new_worker(tmp_path, monkeypatch)
    monkeypatch.setattr(worker_main, "RESULT_ACK_RETRY_S", 0.05)
    monkeypatch.setattr(worker_main, "RESULT_ACK_MAX_WAIT_S", 1.0)
    monkeypatch.setattr(
        worker_main.r2_upload,
        "upload_terminal_result_envelope",
        lambda **kwargs: {
            "result_key": f"prefix/control/result_payload.json",
            "result_url": "https://example/result.json",
        },
    )

    def on_send(message):
        if message.get("type") == "result":
            worker._ack_result({
                "type": "result_ack",
                "accepted": True,
                "result_id": message["result_id"],
                "job_id": message["job_id"],
            })

    ws = FakeWebSocket(on_send=on_send)
    worker._attach_ws(ws)
    job = {
        "job_id": "job-result",
        "order_id": "order-result",
        "r2_output_prefix": "orders/order-result/job-result",
    }
    payload = {
        "job_id": "job-result",
        "order_id": "order-result",
        "status": "succeeded",
        "exit_code": "SUCCESS",
        "candidates": [{"final_image_url": "https://example/final.png"}],
        "partial_review_images": [],
    }

    asyncio.run(worker.send_result(job, payload))

    result_messages = [message for message in ws.sent if message.get("type") == "result"]
    assert len(result_messages) == 1
    result_id = result_messages[0]["result_id"]
    assert payload["transport_result_id"] == result_id
    assert payload["transport_result_key"].endswith("control/result_payload.json")
    assert result_id not in worker._pending_results
    assert result_id not in worker._result_ack_waiters
    assert list((tmp_path / "spool").glob("*.json")) == []
