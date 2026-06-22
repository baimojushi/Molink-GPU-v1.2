#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.client
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PORTS = (6006, 6011, 6010)


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _clean_path_input(raw: str) -> str:
    """Strip Unicode whitespace and remove surrogate-escaped bytes from terminal input."""
    # Filter out surrogate escapes (U+DC80-U+DFFF) that come from malformed UTF-8
    cleaned = "".join(ch for ch in raw if not (0xDC80 <= ord(ch) <= 0xDFFF))
    # Strip all Unicode whitespace from both ends
    start = 0
    while start < len(cleaned) and cleaned[start].isspace():
        start += 1
    end = len(cleaned)
    while end > start and cleaned[end - 1].isspace():
        end -= 1
    return cleaned[start:end]


def _prompt_text(label: str, default: str | None = None, *, required: bool = False) -> str:
    while True:
        suffix = f" [{default}]" if default not in (None, "") else ""
        raw = input(f"{label}{suffix}: ")
        value = _clean_path_input(raw)
        if value:
            return value
        if default not in (None, ""):
            return str(default)
        if not required:
            return ""
        print("该项必填。")


def _prompt_bool(label: str, default: bool = True) -> bool:
    default_text = "Y/n" if default else "y/N"
    value = input(f"{label} [{default_text}]: ").strip().lower()
    if not value:
        return default
    return value in {"y", "yes", "1", "true", "on"}


def _require_path(path_text: str, label: str) -> Path:
    path_text = _clean_path_input(path_text)
    path = Path(path_text).expanduser()
    # Common typo: "root/..." instead of "/root/..."
    if not path.exists() and str(path).startswith("root/"):
        path = Path("/" + str(path))
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path.resolve()


def _default_output_dir() -> Path:
    hanging_runs_root = os.environ.get("HANGING_RUNS_ROOT", "/root/autodl-fs/hanging_runs")
    return Path(hanging_runs_root) / time.strftime("room_full_%Y%m%d_%H%M%S")


def _port_healthy(port: int, timeout_s: float = 2.0) -> bool:
    try:
        conn = http.client.HTTPConnection("127.0.0.1", int(port), timeout=timeout_s)
        conn.request("GET", "/health")
        resp = conn.getresponse()
        resp.read(200)
        return resp.status == 200
    except Exception:
        return False


def _health_status() -> dict[int, bool]:
    return {port: _port_healthy(port) for port in REQUIRED_PORTS}


def _wait_for_health(max_attempts: int = 30, sleep_s: float = 2.0) -> None:
    for attempt in range(1, max_attempts + 1):
        status = _health_status()
        if all(status.values()):
            print("[ok:services] all required API ports healthy")
            return
        if attempt == max_attempts:
            details = ", ".join(f"{port}={'UP' if ok else 'DOWN'}" for port, ok in status.items())
            raise TimeoutError(f"API health check failed: {details}")
        time.sleep(sleep_s)


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, env=env or os.environ.copy(), check=True)


def _load_bfl_env(env: dict[str, str]) -> dict[str, str]:
    bfl_env = ROOT / ".env.bfl"
    if not bfl_env.exists():
        return env
    for raw_line in bfl_env.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            env[key] = value
    return env


def describe(_: argparse.Namespace) -> None:
    print(
        """
Artwork Hanging MVP 服务链路

1. Metric3D API :6006
   输入：房间图、相机参数 JSON、采样点、include_xyz/include_candidates。
   输出：room_metric3d.zip，包含 meta.json、depth_m.npy、normal_xyz.npy、可视化 PNG。

2. Semantic MVP API :6011
   输入：房间图、Metric3D ZIP、scene_profile、vlm_mode、SAM 参数。
   输出：semantics.json，包含墙面/地面/障碍物/既有内容 objects。

3. Hanging MVP API :6010
   输入：房间图、Metric3D ZIP、艺术品图、semantics_json、rules_json。
   输出：hanging_v1_result.zip，包含 client_result.json、plan.json、flux_inputs/manifest.json、hard-paste/mask/debug 资产。

4. GPT Image 2 VIP / BFL / mock 图像编辑脚本
   输入：hanging_v1_result.zip、provider、QA 参数、可选 original-room/artwork-reference。
   默认 GPT Image 2 VIP 接口不接收 mask 字段，adapter 会用 flux2_edit_mask 做本地裁剪与回贴，避免整图重绘。
   输出：flux_result_apiyi_gpt_image2_vip、flux_result_bfl 或 mock 目录，包含 edit_trace.json、final_hd/、partial_review/。

推荐：python3 scripts/hanging_mvp_cli.py run --interactive
""".strip()
    )
    doc = ROOT / "docs" / "PRODUCT_SERVICES_IO.md"
    if doc.exists():
        print(f"\n详细文档：{doc}")


def services(args: argparse.Namespace) -> None:
    manager = ROOT / "scripts" / "manage_all_mvp_apis.sh"
    env = os.environ.copy()
    env["MVP_SERVICES"] = "metric3d,semantic,hanging,auto_update" if args.with_auto_update else "metric3d,semantic,hanging"
    _run(["bash", str(manager), args.action], env=env)


def _interactive_fill(args: argparse.Namespace) -> argparse.Namespace:
    print("交互式运行 Artwork Hanging MVP。直接回车使用默认值。")
    if not args.room:
        args.room = _prompt_text("房间图片路径", required=True)
    if not args.artwork:
        args.artwork = _prompt_text("艺术品图片路径", required=True)
    if not args.art_width_cm:
        w = _prompt_text("艺术品宽度 (cm)", required=True)
        args.art_width_cm = _parse_cm(w, "width")
    if not args.art_height_cm:
        h = _prompt_text("艺术品高度 (cm)", required=True)
        args.art_height_cm = _parse_cm(h, "height")
    if not args.output_dir:
        args.output_dir = _prompt_text("输出目录", str(_default_output_dir()))
    if not args.rules_json:
        args.rules_json = _prompt_text("规则 JSON", "configs/rules_v1.json")
    args.camera_json = _prompt_text("相机参数 JSON", args.camera_json or "{}")
    args.scene_profile = _prompt_text("场景 profile", args.scene_profile or "gallery")
    args.vlm_mode = _prompt_text("VLM 模式 auto/mock/openai_compatible", args.vlm_mode or "auto")
    args.sam_enable = _prompt_bool("启用 SAM 精修", True if args.sam_enable is None else args.sam_enable)
    args.start_services = _prompt_bool("运行前启动/复用本地三项 API", True if args.start_services is None else args.start_services)
    args.skip_bfl = not _prompt_bool("运行 BFL/mock 图像编辑阶段", not args.skip_bfl)
    if not args.skip_bfl:
        args.provider = _prompt_text("图像编辑 provider apiyi_gpt_image2_vip/bfl/mock", args.provider or "apiyi_gpt_image2_vip")
        limit_text = _prompt_text("BFL 处理候选数量限制，留空表示全部", "" if args.bfl_limit is None else str(args.bfl_limit))
        args.bfl_limit = int(limit_text) if limit_text else None
        args.dual_ab_compare = _prompt_bool("额外运行 dual A/B 对比渲染", bool(args.dual_ab_compare))
    return args


def _parse_cm(text: str, label: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"{label} must be a number, got: {text}")
    if value <= 0:
        raise ValueError(f"{label} must be positive, got: {value}")
    return value


def _summarize_output(out_dir: Path) -> None:
    hanging_zip = out_dir / "hanging_v1_result.zip"
    if hanging_zip.exists():
        try:
            with zipfile.ZipFile(hanging_zip) as z:
                manifest = json.loads(z.read("flux_inputs/manifest.json"))
            items = manifest.get("items", [])
            print(f"[summary] approved flux items: {len(items)}")
            for item in items[:10]:
                print(f"  - {item.get('asset_id')} wall={item.get('wall_id')} score={item.get('score')}")
        except Exception as exc:
            print(f"[summary] could not read manifest: {exc}")
    bfl_trace = out_dir / "flux_result_bfl" / "edit_trace.json"
    if bfl_trace.exists():
        try:
            trace = json.loads(bfl_trace.read_text(encoding="utf-8"))
            print(f"[summary] image edit outputs: {len(trace.get('outputs', []))}, errors: {len(trace.get('errors', []))}")
        except Exception as exc:
            print(f"[summary] could not read BFL trace: {exc}")


def run(args: argparse.Namespace) -> None:
    if args.interactive:
        args = _interactive_fill(args)

    if not args.art_width_cm:
        raise ValueError("--art-width-cm is required (cm, positive number)")
    if not args.art_height_cm:
        raise ValueError("--art-height-cm is required (cm, positive number)")

    room = _require_path(args.room or "", "room image")
    artwork = _require_path(args.artwork or "", "artwork image")
    output_dir = Path(args.output_dir or _default_output_dir()).expanduser().resolve()
    rules_json = Path(args.rules_json or "configs/rules_v1.json").expanduser()
    if not rules_json.is_absolute():
        rules_json = ROOT / rules_json
    rules_json = _require_path(str(rules_json), "rules JSON")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-call rules: inject artwork physical dimensions (cm -> m)
    art_width_m = float(args.art_width_cm) / 100.0
    art_height_m = float(args.art_height_cm) / 100.0
    rules = json.loads(rules_json.read_text(encoding="utf-8"))
    rules.setdefault("artwork", {})["physical_width_m"] = art_width_m
    rules.setdefault("artwork", {})["physical_height_m"] = art_height_m
    rules_tmp = output_dir / "rules_call.json"
    rules_tmp.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "ROOT": str(ROOT),
            "CAMERA_JSON": args.camera_json or "{}",
            "SCENE_PROFILE": args.scene_profile or "gallery",
            "VLM_MODE": args.vlm_mode or "auto",
            "SAM_ENABLE_FORM": "1" if (args.sam_enable is None or args.sam_enable) else "0",
        }
    )

    start_services = True if args.start_services is None else bool(args.start_services)
    if start_services:
        env["MVP_SERVICES"] = "metric3d,semantic,hanging,auto_update" if args.start_auto_update else "metric3d,semantic,hanging"
        _run(["bash", str(ROOT / "scripts" / "manage_all_mvp_apis.sh"), "start"], env=env)
    _wait_for_health()

    _run(
        [
            "bash",
            str(ROOT / "scripts" / "run_v1_full_pipeline.sh"),
            str(room),
            str(artwork),
            str(output_dir),
            str(rules_tmp),
        ],
        env=env,
    )

    if args.skip_bfl:
        print("[skip:bfl] local geometry/semantic/hanging pipeline complete")
        _summarize_output(output_dir)
        print(f"[done] {output_dir}")
        return

    provider = (args.provider or "apiyi_gpt_image2_vip").strip().lower()
    bfl_out = output_dir / ("flux_result_bfl" if provider == "bfl" else ("flux_result_mock" if provider == "mock" else "flux_result_apiyi_gpt_image2_vip"))
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_bfl_hanging_flux_pipeline.py"),
        "--hanging-zip",
        str(output_dir / "hanging_v1_result.zip"),
        "--output-dir",
        str(bfl_out),
        "--provider",
        provider,
    ]
    if args.bfl_limit is not None:
        cmd += ["--limit", str(args.bfl_limit)]
    if args.dual_ab_compare:
        cmd += ["--dual-ab-compare"]
        if args.bfl_limit is not None:
            cmd += ["--dual-ab-compare-limit", str(args.bfl_limit)]
    if args.continue_on_error:
        cmd.append("--continue-on-error")
    env = _load_bfl_env(env)
    _run(cmd, env=env)
    _summarize_output(output_dir)
    print(f"[done] {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Artwork Hanging MVP terminal entrypoint")
    sub = parser.add_subparsers(dest="command", required=True)

    p_describe = sub.add_parser("describe", help="Print product purpose and service I/O summary")
    p_describe.set_defaults(func=describe)

    p_services = sub.add_parser("services", help="Manage required local MVP APIs")
    p_services.add_argument("action", choices=["start", "stop", "restart", "status", "logs"], nargs="?", default="status")
    p_services.add_argument("--with-auto-update", action="store_true", help="Include auto_update daemon")
    p_services.set_defaults(func=services)

    p_run = sub.add_parser("run", help="Run Metric3D → Semantic → Hanging → optional final image edit")
    p_run.add_argument("--interactive", action="store_true", help="Prompt for missing paths and parameters")
    p_run.add_argument("--room", help="Room image path")
    p_run.add_argument("--artwork", help="Artwork image path")
    p_run.add_argument("--art-width-cm", type=float, help="Artwork width in centimeters (required)")
    p_run.add_argument("--art-height-cm", type=float, help="Artwork height in centimeters (required)")
    p_run.add_argument("--output-dir", help="Output directory")
    p_run.add_argument("--rules-json", default="configs/rules_v1.json", help="Rules JSON path")
    p_run.add_argument("--camera-json", default="{}", help="Metric3D camera_json form value")
    p_run.add_argument("--scene-profile", default="gallery", help="gallery/home/office/retail/unknown")
    p_run.add_argument("--vlm-mode", default="auto", help="auto/mock/openai_compatible")
    sam = p_run.add_mutually_exclusive_group()
    sam.add_argument("--sam-enable", dest="sam_enable", action="store_true", help="Enable SAM refinement")
    sam.add_argument("--no-sam", dest="sam_enable", action="store_false", help="Disable SAM refinement")
    p_run.set_defaults(sam_enable=None)
    services_group = p_run.add_mutually_exclusive_group()
    services_group.add_argument("--start-services", dest="start_services", action="store_true", help="Start/reuse required local APIs before running")
    services_group.add_argument("--no-start-services", dest="start_services", action="store_false", help="Do not start services")
    p_run.set_defaults(start_services=None)
    p_run.add_argument("--start-auto-update", action="store_true", help="Also start auto_update daemon")
    p_run.add_argument("--skip-bfl", action="store_true", help="Stop after local v1 pipeline")
    p_run.add_argument("--provider", default="apiyi_gpt_image2_vip", choices=["apiyi_gpt_image2_vip", "gpt_image2_vip", "bfl", "mock"], help="Image edit provider")
    p_run.add_argument("--bfl-limit", type=int, default=None, help="Process only first N manifest items")
    p_run.add_argument("--dual-ab-compare", action="store_true", help="Run extra dual A/B comparison render")
    p_run.add_argument("--continue-on-error", action="store_true", help="Continue image edit loop on item errors")
    p_run.set_defaults(func=run)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n[aborted] user interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
