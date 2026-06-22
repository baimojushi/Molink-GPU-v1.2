# Artwork Hanging MVP 产品目的与服务输入输出

## 产品目的

该仓库实现一条“把指定艺术品安全、真实地挂到室内墙面”的 MVP/V1 流水线。系统先用 Metric3D 估计房间深度、法线和相机尺度，再用 VLM/SAM 生成墙面、地面、窗门、既有装饰等语义结构，随后在 Hanging MVP 中选择几何安全的挂画候选点，最后把通过安全门的候选导出给图像编辑阶段做清理和真实渲染。

核心产物面向两个使用者：

- 前端/业务端：读取 `client_result.json`，展示候选墙面、安装高度、风险等级和可用预览。
- 渲染/运营端：读取 `flux_inputs/manifest.json` 和配套图像资产，交给 BFL 或 mock provider 生成最终图。

## 服务与脚本输入输出

| 模块 | 入口 | 输入 | 输出 | 说明 |
|---|---|---|---|---|
| Metric3D API | `GET /health`，端口 `6006` | 无 | JSON：状态、模型名、设备、EXIF 能力 | 服务健康检查。 |
| Metric3D API | `POST /inspect_exif` | `file` 房间图；`camera_json` 可选相机参数 | JSON：原图/处理尺寸、EXIF 字段、选用内参、尺度状态 | 只做 EXIF/相机尺度检查，不跑完整深度推理。 |
| Metric3D API | `POST /analyze_geometry` | `file` 房间图；`camera_json`；`points`；`include_xyz`；`include_candidates` | ZIP：`meta.json`、`depth_m.npy`、`depth_mm_uint16.png`、`depth_vis.png`、`normal_xyz.npy`、`normal_rgb.png`、可选 `points_xyz_m.npy` | 产出深度、法线、内参和采样点，是后续 Hanging 的几何基础。 |
| Semantic MVP API | `GET /health`，端口 `6011` | 无 | JSON：状态、VLM 模式、SAM 运行状态 | 语义服务健康检查。 |
| Semantic MVP API | `POST /parse_semantics_mvp` | `room_image` 房间图；可选 `metric3d_zip`；可选 `semantic_seed_json`；`scene_profile`；`vlm_mode`；`sam_enable`；`sam_refine_labels`；`sam_max_side`；`return_debug_assets` | 默认 JSON：`objects`、`object_inventory`、尺寸、警告；开启 debug 时返回 ZIP：`semantics.json`、`semantic_wall_debug.png` | 单端点支持两种调用：无 `metric3d_zip` 时生成原图坐标的 VLM+SAM seed；带 `metric3d_zip` 和 seed 时跳过 VLM/SAM，只做 Metric3D 坐标对齐和墙面几何后处理。 |
| Hanging MVP API | `GET /health`，端口 `6010` | 无 | JSON：状态和导出策略 | 悬挂规划服务健康检查。 |
| Hanging MVP API | `POST /analyze_hanging_mvp` | `room_image`；`metric3d_zip`；`artwork_image`；可选 `artwork_mask`；`semantics_json`；`rules_json` | ZIP：`plan.json`、`client_result.json`、`flux_inputs/manifest.json`、候选 hard-paste 图、mask、prompt、debug 资产 | 选择安全挂画位置，生成前端可读结果和 BFL 输入包。 |
| BFL 图像编辑流水线 | `python3 scripts/run_bfl_hanging_flux_pipeline.py` | `--hanging-zip`；`--output-dir`；`--provider bfl|mock`；可选 `--original-room`、`--artwork-reference`、QA 参数、`--limit`、`--dual-ab-compare` | 目录：`edit_trace.json`、每个候选的 `edit_trace.json`、`final_hd/`、`partial_review/` | 调用 erase 和 FLUX.2 edit。adapter 对上层是同步阻塞返回；BFL 官方任务本身仍可能在服务端排队。 |
| 一体化 CLI | `python3 scripts/hanging_mvp_cli.py run` | `--room`；`--artwork`；`--output-dir`；`--rules-json`；`--scene-profile`；`--vlm-mode`；`--sam-enable/--no-sam`；`--skip-bfl`；`--provider`；交互模式可逐项输入 | 同 `run_v1_full_pipeline.sh` 和可选 BFL 输出 | 推荐人机入口。显式路径和参数可通过命令行传入；缺省值可在交互模式中确认。 |

## 本次启动链路优化

1. 一次性推理默认只启动 `metric3d,semantic,hanging` 三个 API。
2. `run_v1_full_pipeline.sh` 默认启用局部同步优化：Metric3D `/analyze_geometry` 和 Semantic `/parse_semantics_mvp` 的 VLM+SAM seed 同时发起。Metric3D 返回后，同一个 semantic 端点用 `semantic_seed_json` 快速完成坐标对齐和几何墙面后处理。关闭方式：`SEMANTIC_PARALLEL=false bash scripts/run_v1_full_pipeline.sh ...`。
3. semantic 服务没有新增 VLM-only/finalize 路由。接口仍是 `/parse_semantics_mvp`，新增可选表单字段 `semantic_seed_json`。无 `metric3d_zip` 的调用输出 `coordinate_space=original`、`semantic_pipeline_stage=raw_original`；带 `metric3d_zip` 和 seed 的调用输出 `coordinate_space=processed`、`semantic_pipeline_stage=final_metric_aligned`。
4. `run_full_hanging_pipeline.sh` 增加 `--no-start-services`、`--dual-ab-compare`。默认跳过 dual A/B 额外渲染，减少一次 BFL 排队任务。
5. BFL provider 保持同步阻塞语义：`erase()` 和 `edit()` 提交任务后轮询到结果再返回。这样上层脚本不需要管理异步任务状态。服务端队列由 BFL 控制，客户端不能直接消除；本仓库通过减少非必要任务和把必要任务直连串起来降低总耗时。
6. 新增 `scripts/hanging_mvp_cli.py`，支持交互式补全房间图、艺术品图、输出目录、VLM/SAM/BFL 参数，减少手写 curl 和多脚本切换。

## 推荐命令

交互式：

```bash
python3 scripts/hanging_mvp_cli.py run --interactive
```

非交互式，先只跑本地三段：

```bash
python3 scripts/hanging_mvp_cli.py run \
  --room /path/to/room.jpg \
  --artwork /path/to/artwork.png \
  --output-dir /root/hanging_runs/demo \
  --skip-bfl
```

非交互式，跑 BFL 但不跑额外 dual A/B：

```bash
python3 scripts/hanging_mvp_cli.py run \
  --room /path/to/room.jpg \
  --artwork /path/to/artwork.png \
  --output-dir /root/hanging_runs/demo_bfl \
  --provider bfl
```

需要 A/B 验证时再显式开启：

```bash
python3 scripts/hanging_mvp_cli.py run \
  --room /path/to/room.jpg \
  --artwork /path/to/artwork.png \
  --output-dir /root/hanging_runs/demo_ab \
  --provider bfl \
  --dual-ab-compare \
  --bfl-limit 1
```
