"""Versioned progress facts for server-authoritative ETA calculation."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import time
import uuid


TERMINAL_STATES = {"completed", "failed", "cancelled"}
STAGE_ALIASES = {"r2_upload": "upload", "complete": "delivery"}
PAUSE_CAPABILITY = {
    "worker_queue": "immediate",
    "metric3d": "boundary_only",
    "semantic": "boundary_only",
    "hanging": "boundary_only",
    "render": "boundary_only",
    "styling": "boundary_only",
    "upload": "checkpoint",
    "delivery": "unsupported",
}


@dataclass
class WorkProgress:
    done: float | None = None
    total: float | None = None
    unit: str | None = None
    quality: float = 0.0


class ProgressEmitter:
    def __init__(self, job: dict, resource: dict | None = None) -> None:
        self.job = dict(job or {})
        self.resource = dict(resource or {})
        self.run_id = f"run_{uuid.uuid4().hex}"
        self.seq = 0
        self.job_started_mono = time.monotonic()
        self.current_stage: str | None = None
        self.current_stage_started_mono: float | None = None
        self.current_stage_version = "v1"
        self.last_event: dict | None = None

    def _stage_id(self, value: str) -> str:
        raw = str(value or "unknown").strip().lower()
        return STAGE_ALIASES.get(raw, raw)

    def _build(
        self,
        *,
        stage: str,
        state: str,
        now: float,
        stage_started_mono: float,
        work: WorkProgress | None,
        message: str,
        metadata: dict | None = None,
    ) -> dict:
        self.seq += 1
        stage_id = self._stage_id(stage)
        payload = {
            "type": "progress_v2",
            "protocol_version": 2,
            "event_id": f"evt_{uuid.uuid4().hex}",
            "job_id": self.job["job_id"],
            "order_id": self.job.get("order_id"),
            "run_id": self.run_id,
            "seq": self.seq,
            "pipeline_version": self.job.get("pipeline_version", "hanging-main-v1"),
            "job_kind": self.job.get("job_kind", "hang_in_home"),
            "stage": {
                "id": stage_id,
                "type": stage_id,
                "version": self.current_stage_version,
                "attempt": 1,
                "state": str(state or "running").lower(),
                "pause_capability": PAUSE_CAPABILITY.get(stage_id, "boundary_only"),
            },
            "timing": {
                "worker_monotonic_ms": int(now * 1000),
                "stage_active_elapsed_ms": max(0, int((now - stage_started_mono) * 1000)),
                "job_active_elapsed_ms": max(0, int((now - self.job_started_mono) * 1000)),
            },
            "work": asdict(work) if work else None,
            "resource": self.resource,
            "input_features": self.job.get("eta_features", {}),
            "metadata": dict(metadata or {}),
            "message": str(message or ""),
        }
        self.last_event = payload
        return payload

    def transition(
        self,
        stage: str,
        state: str,
        *,
        message: str = "",
        work: WorkProgress | None = None,
        metadata: dict | None = None,
    ) -> list[dict]:
        now = time.monotonic()
        stage_id = self._stage_id(stage)
        events: list[dict] = []

        if self.current_stage and self.current_stage != stage_id and self.current_stage_started_mono is not None:
            events.append(self._build(
                stage=self.current_stage,
                state="completed",
                now=now,
                stage_started_mono=self.current_stage_started_mono,
                work=None,
                message="",
                metadata={"closed_by_stage_transition": True},
            ))

        if self.current_stage != stage_id or self.current_stage_started_mono is None:
            self.current_stage = stage_id
            self.current_stage_started_mono = now

        events.append(self._build(
            stage=stage_id,
            state=state,
            now=now,
            stage_started_mono=self.current_stage_started_mono,
            work=work,
            message=message,
            metadata=metadata,
        ))

        if str(state).lower() in TERMINAL_STATES:
            self.current_stage = None
            self.current_stage_started_mono = None
        return events

    def heartbeat_snapshot(self) -> dict | None:
        return dict(self.last_event) if self.last_event else None
