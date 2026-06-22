"""worker/diagnostics.py —— per-job diagnostic bundle exporter.

This module is intentionally optional.  It only becomes active when
`/root/autodl-fs/hanging_runs` exists on the GPU instance, so production jobs
without that data-disk mount keep the old behavior.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any

DEFAULT_ROOT = "/root/autodl-fs/hanging_runs"


def _safe_name(value: Any, fallback: str = "item") -> str:
    text = str(value or "").strip() or fallback
    return "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in text).strip("_") or fallback


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except Exception:
        return repr(value)


class DiagnosticsRun:
    """Collect existing pipeline artifacts into a zip for offline inspection."""

    def __init__(self, job: dict[str, Any], root: Path) -> None:
        self.job = job
        self.root = root
        self.enabled = bool(root.exists() and root.is_dir())
        self.run_dir: Path | None = None
        self.zip_path: Path | None = None
        self._started_at = time.time()
        if not self.enabled:
            return
        order_id = _safe_name(job.get("order_id") or job.get("job_id"), "order")
        job_id = _safe_name(job.get("job_id"), "job")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = root / f"{stamp}_{order_id}_{job_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.write_json("00_job/job_payload.json", job)
        self.write_json("00_job/diagnostics_meta.json", {
            "enabled": True,
            "root": str(root),
            "run_dir": str(self.run_dir),
            "started_at_epoch": self._started_at,
            "order_id": job.get("order_id"),
            "job_id": job.get("job_id"),
            "r2_output_prefix": job.get("r2_output_prefix"),
            "note": "Artifacts are copied from the temporary worker pipeline before cleanup.",
        })

    @classmethod
    def create(cls, job: dict[str, Any]) -> "DiagnosticsRun":
        root = Path(os.getenv("MOLINK_HANGING_RUNS_DIR", DEFAULT_ROOT))
        inst = cls(job, root)
        if inst.enabled:
            print(f"[diagnostics] enabled run_dir={inst.run_dir}", flush=True)
        else:
            print(f"[diagnostics] skipped; root not found: {root}", flush=True)
        return inst

    def _dest(self, rel: str) -> Path | None:
        if not self.enabled or self.run_dir is None:
            return None
        clean = str(rel).replace("\\", "/").lstrip("/")
        return self.run_dir / clean

    def write_json(self, rel: str, value: Any) -> None:
        dst = self._dest(rel)
        if dst is None:
            return
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            print(f"[diagnostics] write_json failed rel={rel}: {exc}", flush=True)

    def write_text(self, rel: str, text: str) -> None:
        dst = self._dest(rel)
        if dst is None:
            return
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(str(text or ""), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            print(f"[diagnostics] write_text failed rel={rel}: {exc}", flush=True)

    def copy_file(self, src: str | Path | None, rel: str) -> None:
        dst = self._dest(rel)
        if dst is None or src is None:
            return
        try:
            p = Path(src)
            if not p.exists() or not p.is_file():
                self.write_json(f"_missing/{_safe_name(rel)}.json", {"source": str(src), "reason": "not_found_or_not_file"})
                return
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
        except Exception as exc:  # noqa: BLE001
            print(f"[diagnostics] copy_file failed src={src} rel={rel}: {exc}", flush=True)

    def copy_tree(self, src: str | Path | None, rel: str) -> None:
        dst = self._dest(rel)
        if dst is None or src is None:
            return
        try:
            p = Path(src)
            if not p.exists() or not p.is_dir():
                self.write_json(f"_missing/{_safe_name(rel)}.json", {"source": str(src), "reason": "not_found_or_not_dir"})
                return
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(p, dst)
        except Exception as exc:  # noqa: BLE001
            print(f"[diagnostics] copy_tree failed src={src} rel={rel}: {exc}", flush=True)

    def extract_zip(self, src: str | Path | None, rel: str) -> None:
        dst = self._dest(rel)
        if dst is None or src is None:
            return
        try:
            p = Path(src)
            if not p.exists() or not p.is_file():
                self.write_json(f"_missing/{_safe_name(rel)}.json", {"source": str(src), "reason": "zip_not_found"})
                return
            dst.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(p, "r") as zf:
                zf.extractall(dst)
        except Exception as exc:  # noqa: BLE001
            print(f"[diagnostics] extract_zip failed src={src} rel={rel}: {exc}", flush=True)
            self.write_text(f"{rel.rstrip('/')}_extract_error.txt", traceback.format_exc())

    def finalize_zip(self, payload: dict[str, Any] | None = None, error: BaseException | None = None) -> str | None:
        if not self.enabled or self.run_dir is None:
            return None
        try:
            self.write_json("99_result/result_payload.json", payload or {})
            if error is not None:
                self.write_text("99_result/worker_exception.txt", "".join(traceback.format_exception(type(error), error, error.__traceback__)))
            finished_at = time.time()
            self.write_json("99_result/diagnostics_finished.json", {
                "finished_at_epoch": finished_at,
                "elapsed_s": round(finished_at - self._started_at, 3),
                "zip_policy": "created only when diagnostics root exists",
            })
            zip_path = self.run_dir.with_suffix(".zip")
            if zip_path.exists():
                zip_path.unlink()
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
                for path in sorted(self.run_dir.rglob("*")):
                    if path.is_file():
                        zf.write(path, path.relative_to(self.run_dir.parent))
            self.zip_path = zip_path
            print(f"[diagnostics] zip_created path={zip_path} size={zip_path.stat().st_size}", flush=True)
            return str(zip_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[diagnostics] finalize_zip failed: {exc}", flush=True)
            return None
