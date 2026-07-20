"""Stable runtime resource facts used for ETA cohorting."""
from __future__ import annotations

import os
import subprocess


def _nvidia_value(field: str) -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        value = (result.stdout or "").strip().splitlines()
        return value[0].strip() if value else None
    except Exception:
        return None


def collect_runtime_resource(*, instance_id: str, gpu_model: str, gpu_count: int, worker_version: str) -> dict:
    memory = _nvidia_value("memory.total")
    return {
        "instance_id": instance_id,
        "gpu_model": gpu_model or _nvidia_value("name") or "unknown",
        "gpu_count": int(gpu_count or 0),
        "gpu_memory_mb": int(float(memory)) if memory else None,
        "precision": os.getenv("INFERENCE_PRECISION", "unknown"),
        "worker_version": worker_version,
        "model_warm": False,
        "benchmark_score": _float_env("ETA_BENCHMARK_SCORE"),
        "cuda_version": os.getenv("CUDA_VERSION", ""),
    }


def _float_env(name: str) -> float | None:
    try:
        value = float(os.getenv(name, ""))
        return value if value > 0 else None
    except Exception:
        return None
