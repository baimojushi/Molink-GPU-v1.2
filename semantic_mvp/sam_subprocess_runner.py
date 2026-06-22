"""Subprocess runner for SAM refinement — isolates native crashes from uvicorn worker."""
from __future__ import annotations

import json
import multiprocessing
import os
import sys
import traceback
from typing import Any, Dict, List, Tuple

import numpy as np


def _worker_entry(args_dict: dict, result_queue: multiprocessing.Queue):
    """Entry point for the SAM subprocess. Loads SAM, runs refinement, puts result on queue."""
    try:
        # Re-import in subprocess to get a clean module state
        from semantic_mvp.sam_refiner import refine_objects_with_sam

        image_rgb = np.frombuffer(args_dict["image_bytes"], dtype=np.uint8).reshape(
            args_dict["image_shape"]
        )
        objects = args_dict["objects"]
        processed_size = tuple(args_dict["processed_size"])
        original_size = tuple(args_dict["original_size"])
        enabled = args_dict["enabled"]
        refine_labels_csv = args_dict["refine_labels_csv"]
        max_side = args_dict.get("max_side")

        refined_objects, sam_debug = refine_objects_with_sam(
            image_rgb,
            objects,
            processed_size=processed_size,
            original_size=original_size,
            enabled=enabled,
            refine_labels_csv=refine_labels_csv,
            max_side=max_side,
        )

        # Serialize: convert numpy types in debug dict
        def _sanitize(obj):
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_sanitize(v) for v in obj]
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        result_queue.put({
            "status": "ok",
            "refined_objects": refined_objects,
            "sam_debug": _sanitize(sam_debug),
        })
    except Exception as e:
        result_queue.put({
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        })


def refine_objects_with_sam_subprocess(
    image_rgb: np.ndarray,
    objects: List[Dict[str, Any]],
    processed_size: Tuple[int, int],
    original_size: Tuple[int, int],
    enabled: bool = False,
    refine_labels_csv: str = "",
    max_side: int = None,
    timeout_seconds: int = 120,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run SAM refinement in a separate process so native crashes don't kill uvicorn.

    Returns (refined_objects, sam_debug) — on subprocess crash/failure, falls
    back to original objects with debug explaining the failure.
    """
    if not enabled:
        fallback_debug = {"enabled": False, "reason": "sam_disabled"}
        return objects, fallback_debug

    # Serialize image to bytes for IPC
    image_bytes = image_rgb.tobytes()
    image_shape = image_rgb.shape

    args_dict = {
        "image_bytes": image_bytes,
        "image_shape": image_shape,
        "objects": objects,
        "processed_size": list(processed_size),
        "original_size": list(original_size),
        "enabled": enabled,
        "refine_labels_csv": refine_labels_csv,
        "max_side": max_side,
    }

    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(target=_worker_entry, args=(args_dict, result_queue), daemon=True)
    proc.start()

    try:
        proc.join(timeout=timeout_seconds)
    except Exception:
        proc.terminate()
        fallback_debug = {
            "enabled": True,
            "reason": "subprocess_join_failed",
            "error": "SAM subprocess join timed out or raised",
        }
        return objects, fallback_debug

    exit_code = proc.exitcode

    if exit_code != 0:
        # Subprocess crashed — fallback to bbox
        fallback_debug = {
            "enabled": True,
            "reason": f"subprocess_crashed_exit_code_{exit_code}",
            "error": f"SAM subprocess exited with code {exit_code}",
        }
        return objects, fallback_debug

    # Subprocess exited cleanly — read result
    try:
        result = result_queue.get_nowait()
    except Exception as e:
        fallback_debug = {
            "enabled": True,
            "reason": "subprocess_result_queue_empty",
            "error": str(e),
        }
        return objects, fallback_debug

    if result["status"] == "ok":
        return result["refined_objects"], result["sam_debug"]

    # Subprocess returned an error
    fallback_debug = {
        "enabled": True,
        "reason": "subprocess_error",
        "error": result.get("error", "unknown"),
        "traceback": result.get("traceback", ""),
    }
    return objects, fallback_debug
