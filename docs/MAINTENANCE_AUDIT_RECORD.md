# Artwork Hanging MVP v1 — 持久化维护记录

> 记录版本：2026-06-17-fix-audit-cleanup  
> 来源报告：`artwork_hanging_mvp_maintenance_audit.md`  
> 落地范围：当前仓库代码、配置、测试、根目录历史文档、环境文件  
> 维护策略：项目详细记录集中保存在本文件，根目录历史补丁/报告/部署类文档不再分散保留。

---

## 本次修复日志

### 触发背景

审计报告指出仓库存在根目录历史补丁文档分散、明文环境文件残留、鉴权默认免认证、若干死代码和硬编码阈值等维护风险。本次维护将报告中的项目详细记录集中归档到本文件，并对可直接落地的代码问题进行修复。

### 修复范围

| 文件/目录 | 处理内容 |
|---|---|
| `hanging_mvp/service.py` | 直接导入并使用 `resize_mask_to_original`，移除 `globals()` 死分支；读取可配置地面法线阈值；鉴权改为显式本地豁免。 |
| `semantic_mvp/service.py` | 鉴权改为显式本地豁免。 |
| `services/metric3d_api/app.py` | 鉴权改为显式本地豁免。 |
| `scripts/preflight_check.py` | 增加 `API_KEY` / `ALLOW_UNAUTH_LOCAL` 显式配置检查。 |
| `hanging_mvp/candidate_safety.py` | 对 `cleanup_labels` 和风险艺术标签统一做语义规范化。 |
| `hanging_mvp/geometry.py` | 将地面法线过滤阈值、最小像素数、方差软阈值参数化并输出到 debug。 |
| `hanging_mvp/support_context.py` | 将 floor 支撑点数门槛改为规则配置。 |
| `image_editing/providers/bfl.py` | `from_env` 增加超时/轮询间隔下限，`poll` 至少执行一次轮询。 |
| `hanging_mvp/exit_review.py` | 区分 VLM 网络/HTTP 错误、响应解析/配置错误和未知错误，并默认阻塞异常复核结果。 |
| `hanging_mvp/masks.py` | 移除不会被直接返回的 `human` / `people` canonical 条目。 |
| `hanging_mvp/response_assets.py` | 候选覆盖调试竖线改为按图高比例绘制。 |
| `configs/rules_v1.json` | 新增 `camera.floor_normal_filter` 和 `support_surface_constraints.min_floor_valid_points`。 |
| `tests/` | 补充别名归一化与 BFL 轮询安全下限测试。 |
| 根目录历史文档 | 删除 `[PATCH]`、`[REPORT]`、`[DEPLOY]`、`[CONFIG]`、`[CONTRACT]`、`[README]` 前缀 Markdown 和历史 manifest JSON。 |
| 环境文件 | 删除 `.env.vlm_sam` 与 `.env.vlm_sam.bak.20260530_222644`，仅保留 `.example` 模板。 |

### 关键决策

1. 鉴权采用 `API_KEY` 必填、`ALLOW_UNAUTH_LOCAL=1` 显式本地豁免。部署环境缺少 `API_KEY` 时立即失败，本地调试通过明确环境变量进入免鉴权模式。
2. `cleanup_semantic_types` 的 mask resize 改为单一路径调用 `resize_mask_to_original`。重复手写 resize 逻辑已删除，避免后续维护时产生两套行为。
3. 地面法线阈值保留当前数值，并进入 `rules_v1.json`。当前模型行为不变，后续替换深度模型时可通过配置调整。
4. VLM 出口复核异常默认阻塞。高风险候选进入最终渲染前需要可解释的复核结果，网络错误和响应格式错误写入 `error_type` 供排障。
5. 根目录历史文档集中归档。项目细节、历史修复与日志规范统一维护在 `docs/MAINTENANCE_AUDIT_RECORD.md`，根目录保留运行、合约和源码入口文件。

### 规则变更

| 路径 | 旧值 | 新值 | 原因 |
|---|---:|---:|---|
| `camera.floor_normal_filter.normal_y_max` | 硬编码 `-0.75` | `-0.75` | 将 Metric3D 地面法线 y 过滤阈值配置化。 |
| `camera.floor_normal_filter.abs_z_max` | 硬编码 `0.45` | `0.45` | 将前向法线排除阈值配置化。 |
| `camera.floor_normal_filter.min_pixels` | 硬编码 `1000` | `1000` | 将地面法线最小样本数配置化。 |
| `camera.floor_normal_filter.angular_variance_soft_limit_rad` | 硬编码 `0.08` | `0.08` | 将重力方向置信度软阈值配置化。 |
| `camera.floor_normal_filter.full_confidence_pixels` | 硬编码 `5000` | `5000` | 将满置信样本数配置化。 |
| `support_surface_constraints.min_floor_valid_points` | 硬编码 `50` | `50` | 将 floor 支撑语境最小点数独立配置。 |

### 测试基线

```bash
pytest -q
```

本次维护新增测试：

- `tests/test_candidate_safety_and_render_qa.py::test_class_aware_dual_policy_uses_erased_room_only_for_non_art_wall_attachment` 增加 `wall art` 别名归一化断言。
- `tests/test_bfl_two_provider_contract.py::test_bfl_client_from_env_enforces_poll_safety_floor` 验证 BFL 超时和轮询间隔下限。

### 已知限制

- `semantic_mvp/service.py` 与 `hanging_mvp/masks.py` 的标签表仍是平行维护，本次仅修复挂画侧策略入口的别名归一化。共享标签模块适合作为后续结构性重构。
- `analyze_hanging_mvp` 主流程仍然偏长。本次维护聚焦审计报告中的确定性缺陷和仓库整理，未拆分核心编排函数。
- `ALLOW_UNAUTH_LOCAL=1` 仅用于本地调试。部署脚本和服务环境需要设置非空 `API_KEY`。

---

## 审计问题处置表

| 编号 | 处置状态 | 本次处理 |
|---|---|---|
| M-01 | 已修复 | `resize_mask_to_original` 直接导入并调用。 |
| M-02 | 已修复 | 三个服务统一改为 `API_KEY` 必填与 `ALLOW_UNAUTH_LOCAL=1` 显式本地豁免。 |
| M-03 | 已修复 | FLUX.2 输入策略统一语义规范化 cleanup 标签和风险艺术标签。 |
| M-04 | 已修复 | VLM 出口复核错误分类并默认阻塞异常结果。 |
| M-05 | 已修复 | 地面法线过滤参数进入规则配置。 |
| M-06 | 已修复 | floor 支撑点数门槛进入规则配置。 |
| M-07 | 已修复 | BFL 轮询至少执行一次，环境读取设置安全下限。 |
| M-08 | 已修复 | 移除 `human` / `people` 死 canonical 条目。 |
| M-09 | 已修复 | 调试 overlay 竖线改为图高比例。 |
| M-10 | 后续重构 | 两服务共享标签模块未在本次拆分，已在已知限制记录。 |
| M-11 | 已验证 | 当前 `test_height_fit_contract.py` 已覆盖 P2 契约字段。 |
| M-12 | 已处理 | `.qwenignore` 已含 `.env.*`，本次删除残留真实 env 文件。 |

---

## 项目详细记录归档

以下内容来自本次审计报告，作为后续维护的集中引用基线。

# Artwork Hanging MVP v1 — 代码库维护审计文档

> 文档版本：2026-06-17  
> 审核范围：artwork-hanging-mvp-v1-main（含所有 `.qwen/skills` 经验沉淀）  
> 文档目的：代码库细节设计说明 + 历史修复日志整合 + 后续迭代日志撰写规范

---

## 目录

1. [系统总体架构与服务拓扑](#1-系统总体架构与服务拓扑)
2. [各模块详细设计](#2-各模块详细设计)
   - 2.1 `hanging_mvp/service.py` — 核心挂画分析服务
   - 2.2 `hanging_mvp/geometry.py` — 三维几何原语
   - 2.3 `hanging_mvp/hardpaste.py` — 艺术品透视粘贴
   - 2.4 `hanging_mvp/candidate_safety.py` — 候选安全门
   - 2.5 `hanging_mvp/flux_export.py` — BFL 输出包构建
   - 2.6 `hanging_mvp/client_result.py` — 业务结果封装
   - 2.7 `hanging_mvp/exit_review.py` — VLM 出口复核
   - 2.8 `hanging_mvp/structural_clearance.py` — 结构安全间距
   - 2.9 `hanging_mvp/support_context.py` — 支撑语境与高度规则
   - 2.10 `hanging_mvp/masks.py` — 语义掩码规范化
   - 2.11 `hanging_mvp/schemas.py` — 核心数据类
   - 2.12 `hanging_mvp/rules_access.py` — 规则 JSON 遍历
   - 2.13 `hanging_mvp/numeric_utils.py` — 数值工具
   - 2.14 `hanging_mvp/metric3d_io.py` — Metric3D 包读取
   - 2.15 `hanging_mvp/height_fit_contract.py` — 高度契约（P2 规格）
   - 2.16 `hanging_mvp/response_assets.py` — 响应资产序列化
   - 2.17 `hanging_mvp/render_qa.py` — 渲染质量校验
   - 2.18 `semantic_mvp/` — 语义分割服务
   - 2.19 `image_editing/` — 图像编辑抽象层
   - 2.20 `services/metric3d_api/` — Metric3D 深度估计服务
   - 2.21 `scripts/` — 编排脚本层
   - 2.22 `configs/rules_v1.json` — 规则配置体系
   - 2.23 `tests/` — 测试套件
3. [接口契约](#3-接口契约)
4. [历史修复日志整合](#4-历史修复日志整合)
   - 4.1 V1-P1：调试资产与客户端结果输出
   - 4.2 V7 BFL 流程修复（v7-bfl-flow-fixes）
   - 4.3 V7 I1/I2/I3 补丁
   - 4.4 Patch4：脚本编排修复
   - 4.5 优化与对齐报告提炼
5. [维护性、鲁棒性与逻辑漏洞审核](#5-维护性鲁棒性与逻辑漏洞审核)
6. [后续迭代日志撰写规范](#6-后续迭代日志撰写规范)

---

## 1. 系统总体架构与服务拓扑

该仓库实现一条"将指定艺术品安全挂到室内墙面"的端到端 MVP/V1 流水线，由三个独立 FastAPI 微服务和一个离线 BFL 编排脚本组成。

```
房间图 ─────┬──────────────────────────────────────────────────────────────┐
             │                                                              │
     ┌───────▼──────────┐      ┌──────────────────────┐      ┌─────────────▼──────────┐
     │  Metric3D API    │      │   Semantic MVP API   │      │   Hanging MVP API      │
     │  port :6006      │─────▶│   port :6011         │─────▶│   port :6010           │
     │  深度+法线估计   │      │   VLM+SAM 语义分割   │      │   几何候选+硬贴+导出   │
     └──────────────────┘      └──────────────────────┘      └─────────────┬──────────┘
                                                                            │
                                                              hanging_v1_result.zip
                                                                            │
                                                              ┌─────────────▼──────────┐
                                                              │  BFL Pipeline Script   │
                                                              │  run_bfl_hanging_flux_ │
                                                              │  pipeline.py           │
                                                              │  erase → FLUX.2 edit   │
                                                              └────────────────────────┘
```

**并行优化（`run_v1_full_pipeline.sh` 默认开启）：** Metric3D `/analyze_geometry` 与 Semantic `/parse_semantics_mvp`（VLM+SAM seed 阶段）同步发出。Metric3D 返回后，同一个 `/parse_semantics_mvp` 端点带 `semantic_seed_json` 完成坐标对齐，无需新增路由。

**数据磁盘挂载约定（`data_disk_*` symlink 文件）：** 根目录的 `data_disk_*` 文件记录了宿主机数据盘路径（绝对路径文本），供 CI 或远端部署环境将外部挂载磁盘的各子目录映射到仓库已知位置。这些文件在 `.gitignore` 中标记为主机特定，不应提交真实路径。

---

## 2. 各模块详细设计

### 2.1 `hanging_mvp/service.py` — 核心挂画分析服务（2572 行）

**职责：** 提供 `POST /analyze_hanging_mvp` 和 `GET /health` 两个端点，以及在同一进程里托管的 `POST /parse_semantics_mvp`（语义服务 fallback 入口）。

**`analyze_hanging_mvp` 端点主流程：**

1. **鉴权**：`require_auth(authorization)` 检查 `Authorization: Bearer <API_KEY>` Header。`API_KEY` 为空时免鉴权，方便本地测试。

2. **输入解析**：加载 Metric3D ZIP（深度、法线、内参、原图/处理尺寸缩放比）、房间 RGB 图、艺术品 RGBA 图、语义 JSON、规则 JSON。

3. **全局几何初始化**：
   - `backproject(depth, intr)` 将深度图转为点云 `xyz`（[H, W, 3]，相机坐标系）。
   - `estimate_g_hat_from_floor_normals(normal, floor_mask)` 从语义地面掩码的法线像素估计重力方向 `g_hat_cam`，用于后续墙面坐标系的"上"轴定义。

4. **多墙枚举**：遍历语义对象中所有 `role=="wall"` 的实例（`wall_obj`），逐墙执行：
   - 墙面掩码过滤（面积、连通性、边缘裁剪）。
   - `fit_plane_ransac` 拟合墙面平面 `WallPlane(n, d)`，`make_wall_basis` 建立以 `(origin, u, v, n)` 为基的墙面局部坐标系。
   - 生成以下四类轴（axis）候选：固体连通组件中心轴、固体连通组件三等分轴、语义清晰子区域轴、间隙中心轴。
   - `semantic_axis_policy` 以整组丢弃方式处理低覆盖率语义轴（`semantic_axis_safe_mask_coverage < 0.5`），不裁剪 s 范围。
   - `_estimate_floor_wall_intersection` 估计地-墙交线高度（三级优先：地面掩码投影 → solid_wall 底部 → 可见墙底部）。
   - `_estimate_axis_support_context` 在轴附近探测支撑物（sofa/table/cabinet/floor）并按规则确定高度，输出 `height_solution`。

5. **候选过滤与评分**（每个 axis × 每组 artwork 尺寸）：
   - `_candidate_structural_clearance`：墙面局部水平间距门（artwork_width/3）。
   - `evaluate_wall_support_gate`：solid\_wall\_inside\_ratio 单门决策（hard\_reject/review/pass）。
   - `compute_edge_corner_support_debug`：投影四边形所有边和角点的支撑比率。
   - 通过所有硬门的候选进入评分，计算 `score`、`risk_level`、`soft_risk_reasons`、`requires_hardpaste_vlm_review`。

6. **艺术品投影与硬贴**：
   - `project_quad` 将三维艺术品矩形投影到 2D 处理坐标。
   - `scale_points_to_original` 缩放到原始图像坐标。
   - `_compute_artwork_mount_policy_for_candidate`：检测是否需要预合成帧侧面（`precompose_frame_sides`），计算最多 `max_visible_side_faces=2` 个可见面的 quad。
   - `warp_artwork_mount_to_geometry`（预合成模式）或 `warp_artwork_to_quad`（平面模式）执行透视粘贴，生成 `hard_paste.png` 和 `art_region_mask.png`。
   - `review_hardpaste_exit` 对高风险候选触发 VLM 出口复核。

7. **重复候选抑制**：`_suppress_projected_duplicate_candidates` 按墙面分组，用投影四边形 IoU 和中心距两重条件丢弃重复。

8. **Flux 导出打包**：
   - `build_remove_mask_for_hardpaste`：构建候选级清除掩码（候选墙范围内的现有内容 + 全局前景人物），本地形态学膨胀 6px。
   - `_cleanup_semantic_types_from_remove_mask`：分析导出的清除掩码实际覆盖了哪些语义对象。
   - `choose_flux2_input_policy`：按清除内容类型决定 FLUX.2 输入模式（single/dual\_erased）。
   - `build_render_prompt`：生成保守渲染 Prompt，注入 bbox/polygon，禁止移动/缩放/重裁。
   - `select_flux_manifest_items`：per-wall cap（2）→ 同墙 NMS（IoU/中心距）→ 全局 cap（4）。

9. **ZIP 响应组装**：打包 `plan.json`、`client_result.json`、`flux_inputs/manifest.json`、`wall_assets/<id>/*`、`rejected_wall_assets/<id>/*`、`debug_assets/` 等。

**重要设计决策：**
- `opening_or_recess_overlap_ratio` 在 `opening_recess_used_in_solid_mask=true` 时仅作说明字段，不构成第二个硬门，避免双重拒绝。
- 帧侧面 quad 构建时，visible front face 锁定在 candidate projection 上，side face 向后退缩 `frame_depth_m`，防止视觉过冲。
- P2 多墙地-角拟合（`room_corner_floor_fit`）已定义契约但未启用（`enabled: false`）。

---

### 2.2 `hanging_mvp/geometry.py` — 三维几何原语

| 函数 | 设计要点 |
|------|----------|
| `backproject(depth, intr)` | 标准针孔模型反投影，输出 `[H,W,3]` 点云，z 即深度值。 |
| `fit_plane_ransac` | 500 次迭代 RANSAC（threshold=0.06m），内点精化用 SVD（最小奇异向量为法线），返回内点比率。点数 < 50 时直接抛出。 |
| `estimate_g_hat_from_floor_normals` | 过滤地面像素条件：`ny < -0.75 && |nz| < 0.45`，选取 ≥ 1000 个符合条件的像素做均值，角度方差 > 0.08 rad 时按比例降低置信度。这两个阈值是数据集特定常量，未暴露到规则配置。 |
| `make_wall_basis` | `u = normalize(cross(g_hat, n))`，`v = normalize(cross(n, u))`，强制 `n` 朝向相机（dot(n, -origin) > 0），强制 `v` 与 `g_hat` 同向（"上"方向）。 |
| `project_points` | z 用 `max(z, 1e-6)` 防零除。 |
| `ray_plane_depth_for_mask` | 对掩码区域逐像素做射线-平面求交，填充光线深度，分母 < 1e-6 时置 NaN。 |
| `scale_points_to_original` | 简单线性缩放，返回 `list[list[float]]` 而非 numpy 数组，供 JSON 序列化。 |

---

### 2.3 `hanging_mvp/hardpaste.py` — 艺术品透视粘贴

**核心设计：局部 ROI 高倍超采样。**

`_warp_rgba_overlay_local` 仅在目标四边形包围盒（加 pad）范围内分配超采样画布，`_effective_supersample_scale` 防止超采样后的单边超过 `max_canvas_side=8192px`。超采样后用 `_downsample_rgba_premultiplied` 做预乘 Alpha 的 AREA 插值降采样，避免 Alpha 边缘的黑边晕染。

**帧侧面合成（`warp_artwork_mount_to_geometry`）：**
1. 侧面先于正面渲染，保证正面覆盖侧面（不被侧面遮挡）。
2. `_side_face_front_span_clip_polygon` 用 Sutherland-Hodgman 算法把侧面四边形裁剪到正面边的切线范围（±0.35px pad），防止斜视角时背面边略长于前面边产生的黑色帧线露出。
3. `_edge_side_texture` 从艺术品对应边采样像素并应用线性渐变阴影（`side_shading_strength=0.18`），靠近正面一侧亮，靠近墙面一侧暗（物理约定）。
4. 侧面掩码用 `bitwise_or` 合并，正面掩码区域从侧面掩码中清除，最终输出 `front_mask`、`side_mask`、`mount_mask`。

**重要函数：**
- `alpha_blend_rgba`：标准 alpha 混合，输入 base 若含 alpha 通道则取前三通道。
- `_mask_from_quad` / `_mask_from_polygon`：cv2.fillPoly 填充，无限制几何有效性检查（`pts.shape[0] >= 3 and np.isfinite(pts).all()`）。

---

### 2.4 `hanging_mvp/candidate_safety.py` — 候选安全门

**`evaluate_wall_support_gate`：** 单门决策替代旧的多条件交叉硬门。  
- `ratio = effective` when `removable > 0` else `solid`（可清除的现有墙面内容计入有效支撑）。  
- 阈值：`ratio < 0.90` → hard\_reject；`[0.90, 0.97)` → review；`≥ 0.97` → pass。  
- 孔洞/凹槽重叠比率在 `opening_recess_used_in_solid_mask=true` 时仅记录为 `explanation_reasons`，不构成独立拒绝门。

**`semantic_axis_policy`：** 以整组丢弃取代 s 范围裁剪——如果语义轴的安全掩码覆盖率 < `min_safe_coverage=0.50`，整组语义轴全部丢弃，回退到几何轴。

**`compute_edge_corner_support_debug`：** 用 `cv2.line` 和 `cv2.circle` 对投影四边形的四边和四角采样，计算各自的 safe\_wall 支撑比率，输出 `min_edge_support_ratio` 和 `min_corner_support_ratio`。

**`choose_flux2_input_policy`：** 三级分流：
- 前景人物存在 → `single_after_erase_hardpaste`（禁止双参考）
- 现有艺术品存在 → `single_after_erase_hardpaste`（禁止参考内容回流）
- 仅有非艺术品壁挂杂物 → `dual_erased_room_plus_target`（清洁房间图作为第二参考）
- 其余 → `single_after_erase_hardpaste`（默认）

`raw_original_dual_allowed` 永远为 `false`，彻底禁止未清洁原图作为双参考。

---

### 2.5 `hanging_mvp/flux_export.py` — BFL 输出包构建

**`build_remove_mask_for_hardpaste`：**  
- `existing_masks` 与 `current_wall_mask` 做 AND（只清除当前候选墙上的现有内容），`foreground_masks` 全局合并（人物是全图投递缺陷）。  
- 本地形态学膨胀 `dilate_px=6`（默认），`protect_art_region=False` 用于 v7 的"先清除后粘贴"顺序（旧版为 `True`，防止清除刚贴上的艺术品）。

**`build_render_prompt`：** 三段式结构：
1. 输入模式合约（single/dual\_erased/legacy 各有不同措辞）
2. 区域文本（`front_bbox_px` + `front_polygon_px` + 可选 `mount_bbox_px`）
3. 帧处理规则（预合成侧面 / 已有帧 / 无帧三种分支）+ 严格尺寸锁定语句。

**`select_flux_manifest_items`：** 三阶段：per-wall cap（默认 2）→ 同墙中心距或 polygon IoU NMS → global cap（默认 4）。`_polygon_iou_px` 在面积超过 12M 像素时退化为 bbox IoU，避免大图内存爆炸。

**`safe_asset_id`：** 用正则把非安全字符替换为 `_`，防止路径注入。

---

### 2.6 `hanging_mvp/client_result.py` — 业务结果封装

`build_client_result` 将内部候选数据映射为小程序/业务端可直接消费的 `client_result.json`：
- `best_plan` / `alternative_plans`：包含 `score`（×100 整数）、`risk_level`、`reason_tags`（中文）、`asset_paths`、`install_reference`（安装高度、是否已确认、height\_mode 等）。
- `not_recommended`：被拒绝的墙面，**仅**当该墙面没有任何通过候选时才出现（避免 right 墙同时出现在 plans 和 not\_recommended）。
- `_risk_level`：若候选已有明确 `risk_level` 字段则直接读取，否则用 score/plane/solid/foreground/recess 五维阈值规则推导。
- `_reason_tags`：读取 `hard_constraints` 各子字段，生成最多 5 个中文标签。

---

### 2.7 `hanging_mvp/exit_review.py` — VLM 出口复核

**三层短路：**
1. 候选不需要复核（低风险）且 `review_all=false` → `approved_by_geometry_gate`。
2. VLM 模式为 `off`/`disabled`/`metadata_only`，或 VLM 未配置 → 按 `block_unreviewed` 规则决定是否允许通过。
3. VLM 已配置 → 调用 OpenAI 兼容接口，返回 `{"approved": bool, "confidence": float, "reasons": [...]}`，用 `min_confidence=0.55` 过滤。

VLM 调用失败时同样按 `block_unreviewed` 决定，不抛出到上层。Prompt 要求模型只返回 JSON（`temperature=0`），`_extract_json` 会尝试从 Markdown 代码块或正则提取。

---

### 2.8 `hanging_mvp/structural_clearance.py` — 结构安全间距

在墙面局部坐标系内对每个结构禁区对象（门/窗/拱门/开口等）做三维投影，只考虑与艺术品在 t 轴（垂直方向）有重叠的对象（`±vertical_overlap_margin_m=0.05`），计算水平 s 方向的间距。

- 间距 < `artwork_width * 1/3` → 阻塞（硬拒绝）。  
- 距墙平面 > `max_structural_wall_plane_distance_m=0.75` 的点排除（防止房间另一侧的对象误判为同墙阻塞）。  
- 少于 `min_structural_points_for_clearance=20` 个点的对象跳过。

输出 `blocking_objects` 列表和 `min_clearance_m`，供调用方在硬拒绝时提供详细说明。

---

### 2.9 `hanging_mvp/support_context.py` — 支撑语境与高度规则

**`_estimate_floor_wall_intersection`：** 三级优先（依次尝试）：

1. 语义地面掩码点云投影到当前墙面平面，按候选墙 s 范围过滤，取 `floor_projected_t_percentile=50`。该来源被 `floor_source_is_metric_trusted` 标记为受信任（`metric_height_finalized=true`）。
2. solid\_wall\_mask 底部百分位（p2）。
3. 可见墙面底部百分位（p2）作为最终 fallback。

**`_estimate_axis_support_context`：** 在 `axis_s ± margin_m` 范围内按优先级（sofa > table > cabinet > floor）找最佳支撑物，用各自的高度规则计算挂画中心高度。unknown 支撑时输出 `mode="provisional_visual_review_required"`，前端必须显示临时预览提示而非确定安装高度。

地面 label 的有效点数门槛为 50（其余标签为 `min_valid_points=200`），这是唯一一处硬编码而非规则驱动的支撑对象点数阈值。

---

### 2.10 `hanging_mvp/masks.py` — 语义掩码规范化

**实例标签剥离（`_INSTANCE_SUFFIX_RE`）：** 用正则 `^(?P<base>.+?)[_\-\s]+\d{1,4}$` 把 `painting_01`、`window-2` 等 VLM 实例化标签的数字后缀去除，再查别名表和已知规范标签集。

**别名表（`_LABEL_ALIASES`）：** 把 VLM 常见表达（`"wall art"`, `"framed artwork"`, `"people"`, `"man"` 等）映射到规范标签。

**`canonical_semantic_label`：** `"human"` 和 `"people"` 在 `_KNOWN_CANONICAL_LABELS` 中虽有条目但会被单独处理为 `"person"`，`label_matches` 两端均规范化，确保查询时不需要调用者了解规范形式。

**`load_semantic_masks`：** 从多边形构建 uint8 掩码，存储 `{"object_id", "label", "raw_label", "role", "confidence", "mask", "polygon"}`，`role` 直接沿用 semantic service 传来的字段而非由 label 推导（保持来源可追溯）。

---

### 2.11 `hanging_mvp/schemas.py` — 核心数据类

| 类 | 字段 |
|---|---|
| `Intrinsics` | fx, fy, cx, cy, source, fov\_deg |
| `WallPlane` | n (3-tuple), d (float) |
| `WallBasis` | origin, u, v, n, g\_hat\_cam（均 3-tuple）|
| `Candidate` | candidate\_id, wall\_id, axis\_s\_m, center\_t\_m, center\_height\_m, projected quads, reject\_reasons, hard\_constraints, score |

`to_dict` 对 dataclass 调用 `asdict`，对其他类型直接返回，供 JSON 序列化。

---

### 2.12 `hanging_mvp/rules_access.py` — 规则 JSON 遍历

**`rule_value(rules, path, default)`：** 递归路径查找，遇到 `{"value": ..., "notes": "..."}` 形式的叶节点时返回 `cur["value"]`，否则直接返回节点值。这个约定允许规则文件在数值旁附加注释字段。

**`parse_json`：** JSON 解析失败时抛出 HTTP 400，而非 500，防止错误格式的 `rules_json` 触发内部错误。

---

### 2.13 `hanging_mvp/numeric_utils.py` — 数值工具

- `_safe_percentile`：过滤 NaN/Inf 后做 percentile，空数组返回 `default`，任何异常静默返回 `default`。
- `_mask_bbox`：`np.where(mask > 0)` 取边界框，空掩码返回 `None`。
- `_clamp01`：强制 float 转换后夹到 [0,1]，转换失败返回 0.0。

---

### 2.14 `hanging_mvp/metric3d_io.py` — Metric3D 包读取

从 ZIP 读取 `meta.json`、`depth_m.npy`、`normal_xyz.npy`，构建 `Metric3DBundle`。`scale_to_original` 由 `original_size / processed_size` 计算，用于后续将 2D 坐标从处理分辨率映射回原始分辨率。内参来源优先 `selected_intrinsics`，回退 `intrinsics`，缺失必要字段时抛出 `ValueError`。

---

### 2.15 `hanging_mvp/height_fit_contract.py` — 高度契约（P2 规格）

`room_corner_floor_fit_contract` 固定返回 `enabled: false`，作为 P2 多墙角点地面拟合功能的规格声明占位。该函数当前未被任何测试覆盖。

`floor_source_is_metric_trusted` 只在来源为 `"floor_mask_projected_to_current_wall_plane"` 时返回 `True`，用于 service.py 决定 `metric_height_finalized` 字段。

---

### 2.16 `hanging_mvp/response_assets.py` — 响应资产序列化

- `_png_bytes`：PIL 转 PNG 字节，输入为 2D（灰度）或 3D（RGB/RGBA）数组。
- `_draw_overlay`：在 room\_rgb 上用红色多边形和绿色竖线画候选框，竖线硬编码 ±200px，不含画布边界感知。
- `_zip_response`：FastAPI `Response` 包装，MIME 为 `application/zip`。

---

### 2.17 `hanging_mvp/render_qa.py` — 渲染质量校验

`evaluate_mask_anchored_render_qa` 以 `art_region_mask` 为唯一几何锚，不使用 SAM/VLM 重检测。三层评判区：

| 区域 | 计算方式 | 用途 |
|------|---------|------|
| `core`（内缩 2px） | 核心绘画区 SSIM（全通道均值，非空间窗口） | 检测画作被重绘 |
| `allowed`（外扩 6px）| 帧阴影允许区 | 允许少量边缘调整 |
| `forbidden_guard`（外扩 20px，减 allowed） | 外部禁止区 | 检测 Flux 是否改动了墙面 |

失败条件：`core_masked_ssim < 0.96` 或 `core_color_delta > 10.0` 或 `guard_changed_ratio > 0.12`。

---

### 2.18 `semantic_mvp/` — 语义分割服务

**`service.py`（1172 行）：** 提供 `/parse_semantics_mvp`（1000 行核心逻辑）和 `/health`。

- 无 `metric3d_zip` + 无 `semantic_seed_json` → VLM + SAM seed 模式（`coordinate_space=original`）。
- 带 `metric3d_zip` + 带 `semantic_seed_json` → 跳过 VLM/SAM，只做坐标缩放+墙面几何后处理（`coordinate_space=processed`）。

`ROLE_BY_LABEL` 和 `LABEL_ALIASES` 与 `hanging_mvp/masks.py` 中的定义**平行维护但未共享**，存在潜在漂移风险。

**`sam_subprocess_runner.py`：** 将 SAM 推理封装在独立子进程（`multiprocessing.Process`），通过 `Queue` 传递序列化结果，防止 SAM native 崩溃杀死 uvicorn worker。超时 120 秒后返回错误，原始 objects 原样返回（降级）。

**`vlm_client.py`（1172 行）：** OpenAI 兼容 VLM 调用，结构化输出解析，带重试和错误分类。

---

### 2.19 `image_editing/` — 图像编辑抽象层

**`providers/base.py`：** `ImageEditRequest`（frozen dataclass）和 `ImageEditResult`（frozen dataclass）定义协议接口。`ImageEditProvider` 为 `Protocol`，只声明 `erase` 和 `edit` 两种能力。

**`providers/bfl.py`：**
- `BflClient`：负责 HTTP 通信、提交（`submit`）、轮询（`poll`）、下载（`download_sample`）三步流程。轮询 deadline 基于提交前的 `time.time()` 计算。指数退避最多 3 次重试（`max_retries=3`），HTTP 429 只在非最后一次尝试时退避。
- `BflImageEditProvider`：`erase` 调用 `/v1/flux-tools/erase-v1`，`edit` 调用 `/v1/flux-2-max`（可配置）。`edit` 支持最多 7 个额外参考图（`input_image_2` ... `input_image_8`），不向 FLUX.2 传送 mask（`mask_sent_to_flux2=False`，遵循 BFL API 规格）。
- `MockImageEditProvider`：用 `shutil.copyfile` 复制输入图，所有 trace 注入 `noop_result: true` 和 `production_usable: false` 警告字段，防止 mock 结果被误认为真实 BFL 产物。

**`pipeline.py`：**
- `cleanup_objects`：v7 流中在硬贴之前调用，清除 `original_room`。
- `render_hanging`：`art_mask` 明确标注为不发送到 FLUX.2（QA 元数据），`references` 支持可变长度（dual 模式传入已清洁房间图）。

---

### 2.20 `services/metric3d_api/` — Metric3D 深度估计服务

**`app.py`（14274 字节）：** 提供 `/health`、`/inspect_exif`、`/analyze_geometry` 三个端点。

- `/analyze_geometry`：推理 Metric3D ViT-L 模型（giant），返回 ZIP（`meta.json` + `depth_m.npy` + `normal_xyz.npy` + 可视化图像）。`meta.json` 包含处理尺寸、原始尺寸、选用内参（来源：EXIF/规则/默认）、scale 状态。
- 服务以 `start_giant.sh` 启动，需要 `metric3d_ctl.sh` 进行进程管理（pid 文件 + healthcheck 轮询）。
- `warmup_giant.py` 在服务启动后做一次空白图推理预热。

---

### 2.21 `scripts/` — 编排脚本层

| 脚本 | 职责 |
|------|------|
| `run_bfl_hanging_flux_pipeline.py`（33k） | BFL 完整编排：加载 hanging ZIP → 按 manifest 逐条 erase → hard-paste → flux2 edit → QA → 输出 |
| `hanging_mvp_cli.py`（15k） | 交互式/非交互式一体化入口，封装三服务调用 + BFL 流程 |
| `run_v1_full_pipeline.sh` | 并行调度 Metric3D + Semantic → 串行 Hanging，可用 `SEMANTIC_PARALLEL=false` 关闭并行 |
| `manage_all_mvp_apis.sh` | 三服务（Metric3D/Semantic/Hanging）的启停/状态管理 |
| `preflight_check.py` | 环境就绪性检查：规则 JSON 可解析、Python 模块可导入、服务端口可达 |
| `run_full_hanging_bfl_test_opt.sh`（7.5k） | 集成测试脚本，含三服务健康检查（Patch4 后修复） |
| `parse_artwork_dimensions.py` | 从用户输入字符串解析艺术品尺寸（带单位转换） |
| `safe_update_from_clean_zip.sh` | 从干净 ZIP 安全更新仓库（备份 + 对比 + 应用） |

`run_bfl_hanging_flux_pipeline.py` 的 v7 流程：  
`original_room → erase（清除掩码）→ hard_paste_after_erase（清洁房间上贴艺术品）→ flux2_edit（最终渲染）`  
旧 legacy 流程（`legacy_hardpaste_erase_flux2`）在 manifest 缺少必要字段时自动回退，I1 补丁后会向 stderr 和 trace 输出警告。

---

### 2.22 `configs/rules_v1.json` — 规则配置体系

规则文件为规范化的单一来源（`hanging_mvp/rules_mvp.json` 是到 `../configs/rules_v1.json` 的 symlink）。所有数值均通过 `rule_value` 按路径读取，叶节点格式为 `{"value": ..., "notes": "..."}` 或直接标量。

**关键规则分组：**

| 分组路径 | 说明 |
|----------|------|
| `artwork` | 尺寸、has\_frame（auto/true/false）、fallback 宽度规则 |
| `height_rules` | sofa/floor/table/cabinet/unknown 各支撑类型的挂画中心高度 |
| `hard_constraints` | 硬门阈值：wall\_inside\_ratio、edge/corner 支撑比率、结构间距等 |
| `forbidden_classes` | 禁止挂画的语义类别列表 |
| `existing_content_classes` | 候选墙清除范围的语义类别 |
| `structural_clearance_policy` | 间距分数、风险阈值、soft pass 开关 |
| `cleanup_policy` | 清除掩码定义、两级膨胀文档（local 6px + BFL 10px ≈ 16px） |
| `floor_wall_intersection` | 三级地-墙交线估计的各项参数 |
| `existing_content_overlap_policy` | 候选墙范围限定、off-wall 保护 |
| `candidate_diversity` | 投影重复抑制的 IoU/中心距阈值 |
| `exit_review` | VLM 出口复核的模式、置信度阈值、未复核阻塞开关 |
| `artwork_mount` | 帧侧面预合成的深度/面数/阴影参数 |
| `room_corner_floor_fit` | P2 规格（`enabled: false`） |

---

### 2.23 `tests/` — 测试套件（7 个文件，预期 22+ passed）

| 文件 | 覆盖范围 |
|------|---------|
| `test_candidate_safety_and_render_qa.py` | 支撑门决策、语义轴策略、dual 模式选择、render QA pass/fail、超采样硬贴 |
| `test_flux_export_v7_policy.py` | manifest 选择 NMS、polygon IoU 重复抑制、render prompt 内容、规则文件膨胀参数文档、实例标签规范化 |
| `test_hanging_mvp_contract.py` | service.py 导入契约、API 契约字段、投影重复候选抑制 |
| `test_artwork_mount_policy.py` | 帧侧面挂载策略 |
| `test_bfl_two_provider_contract.py` | BFL/mock 双 provider 契约、v7/legacy manifest 区分 |
| `test_height_fit_contract.py` | 高度契约字段 |
| `test_semantic_wall_guards.py` | 语义墙面安全规则 |

---

## 3. 接口契约

### `POST /analyze_hanging_mvp`（port 6010）

**输入（multipart/form-data）：**

| 字段 | 类型 | 必须 | 说明 |
|------|------|------|------|
| `room_image` | File | ✓ | 房间图（JPEG/PNG） |
| `metric3d_zip` | File | ✓ | Metric3D 输出 ZIP |
| `artwork_image` | File | ✓ | 艺术品图（透明背景 PNG 或 JPEG） |
| `artwork_mask` | File | — | 可选艺术品掩码 |
| `semantics_json` | Form str | — | 语义 JSON（默认 `{}`） |
| `rules_json` | Form str | — | 规则覆盖 JSON（默认 `{}`） |

**输出（ZIP）：**

```
plan.json                                  # 完整内部几何计划
client_result.json                         # 业务端消费结构
flux_inputs/manifest.json                  # BFL 编排 manifest
flux_inputs/original_room.png              # 原始房间图
flux_inputs/artwork.png                    # 艺术品图
wall_assets/<wall_id>/hard_paste.png
wall_assets/<wall_id>/art_region_mask.png
wall_assets/<wall_id>/remove_mask.png
wall_assets/<wall_id>/debug_overlay.png
wall_assets/<wall_id>/candidates.json
...
debug_assets/semantic_wall_debug.png
```

**追加兼容字段（downstream 必须容忍未知嵌套键）：**  
`risk_level`、`soft_risk_reasons`、`requires_hardpaste_vlm_review`、`structural_clearance`、`flux2_input_policy`、`render_size_qa_status`、`exit_code`、`production_usable`。

**P1 安全墙支撑字段：**  
`solid_wall_inside_ratio`、`safe_wall_support_gate_ratio`、`wall_support_gate`、`edge_support_debug`、`corner_support_debug`、`center_t_normalized`。

---

## 4. 历史修复日志整合

### 4.1 V1-P1：调试资产与客户端结果输出

**问题：** 原始 MVP 输出缺乏业务端可消费结构，没有统一的候选调试资产。

**新增：**
- `hanging_mvp/client_result.py`：业务友好 JSON，下游小程序读取 `client_result.json` 而非原始 debug schema。
- `hanging_mvp/debug_assets.py`：`wall_debug_overlay.png`、各种墙面掩码 PNG。
- `semantic_mvp/debug_assets.py`：`semantic_wall_debug.png`。
- hanging ZIP 扩展：`wall_assets/<id>/` 分目录结构、`rejected_wall_assets/<id>/`。
- `/parse_semantics_mvp` 的 `return_debug_assets=true` 参数：返回含调试图的 ZIP，默认仍返回 JSON。

**影响范围：** `hanging_mvp/client_result.py`、`hanging_mvp/debug_assets.py`、`semantic_mvp/debug_assets.py`、`hanging_mvp/service.py`（ZIP 打包）。

---

### 4.2 V7 BFL 流程修复（v7-bfl-flow-fixes）

**P1 清除顺序修复：**  
旧顺序 `hard_paste → erase → flux2` 导致 BFL erase 看到的是已贴艺术品的图，背景修复受污染。修复后为 `original_room → erase → hard_paste_after_erase → flux2`。`protect_art_region` 参数在 v7 流程中设为 `False`，允许清除艺术品目标区域下方的旧内容。legacy manifest（缺 `original_room_path` 或 `artwork_path`）回退旧流程，trace 标记 `legacy_hardpaste_erase_flux2`。

**P4 manifest 渲染上限：**  
`select_flux_manifest_items` 实施 per-wall NMS（IoU 阈值 0.25 / 中心距分数 0.65）、per\_wall\_limit=2、global\_limit=4。被 NMS 或上限丢弃的候选写入 `blocked_or_pending_reviews.json` 和 `exit_review_summary.json`，保留调试资产。

**P3 FLUX.2 Prompt 保守化：**  
移除产品交付语言（`"Return a clean high-resolution product delivery image"`）和强制帧语言。注入 `bbox_px`/`polygon_px`。加入 `STRICT SIZE LOCK` 和 `FINAL SIZE LOCK AFTER FRAME RULE` 双重尺寸锁定。三分支帧处理规则（预合成侧面 / 已有帧 / 无帧）。

**N1 art\_mask 语义修正：**  
`art_region_mask` 从发送给 FLUX.2 的参数中移除，改为 QA-only 资产。`pipeline.py::render_hanging` 签名中 `art_mask` 参数仅作本地 trace 记录，`mask_sent_to_flux2=False`。

**N2 Exit review 配置化：**  
`rules_v1.json` 中新增 `exit_review` 分组，显式定义 VLM 模式默认值和缺失 VLM 时的行为。

**N3 遮挡类型分流：**  
manifest 条目携带 `cleanup_semantic_types` 和 `flux2_input_mode`。`choose_flux2_input_policy` 按人物/现有艺术品/非艺术品壁挂三类决定 single/dual\_erased 模式。

**测试基线：** 10 passed。

---

### 4.3 V7 I1/I2/I3 补丁

**I1 — legacy fallback 警告：**  
`run_bfl_hanging_flux_pipeline.py` 在 `--continue-on-error` 模式下，legacy manifest 不再静默降级。警告包含 asset\_id、缺失字段列表、v7 manifest 所需字段提示、质量警告（erase 看到已贴艺术品）。写入 stderr + `edit_trace.json::warnings[]` + 条目 trace `legacy_fallback_reasons`。

**I2 — 双模式 A/B 对比验证：**  
新增 `--dual-ab-compare` / `--dual-ab-compare-limit N` CLI 参数。启用时对 `flux2_input_mode=dual_original_plus_target` 的条目额外跑一次 single-input baseline。输出到 `final_hd_ab_compare/<asset_id>__single_baseline.png`，`edit_trace.json` 包含对比摘要。决策规则：如 dual 模式在位置/尺寸保持上不优于 single baseline，则撤回 dual 覆盖。

**I3 — 清除掩码两级膨胀文档化：**  
`build_remove_mask_for_hardpaste` docstring 说明本地 6px 膨胀 + BFL server `dilate_pixels=10`（近似总计 16px）的设计意图和调试建议。`rules_v1.json::cleanup_policy::remove_mask_dilation_policy` 结构化记录 `local_export_dilate_px`、`bfl_erase_api_dilate_pixels_default`、`effective_total_dilate_px_approx`，供 `test_flux_export_v7_policy.py::test_rules_document_remove_mask_dilation_stack` 验证。

**测试基线：** 12 passed。

---

### 4.4 Patch4：脚本编排修复

| 修复项 | 文件 | 问题描述 | 修复方式 |
|--------|------|---------|---------|
| 并行日志步骤编号 | `run_v1_full_pipeline.sh` | 并行模式日志显示三个逻辑步骤但未标注四次服务调用的编号 | 明确标注 call 1-2/4 并行、call 3/4 alignment、call 4/4 hanging |
| 冗余 alignment 参数 | `run_v1_full_pipeline.sh` | 第二次 Semantic alignment 调用错误携带 `vlm_mode` 和 `sam_enable` | 移除，添加注释说明 semantic\_seed\_json 对齐跳过 VLM/SAM |
| 健康检查不完整 | `run_full_hanging_bfl_test_opt.sh` | API 重启后只检查 Hanging :6010，未等待 Metric3D :6006 和 Semantic :6011 | 三服务健康端点均纳入等待循环 |
| 重复健康轮询块 | `run_full_hanging_bfl_test_opt.sh` | 存在两个 Hanging-only 轮询块 | 移除第二个重复块，健康快照只写一次 |

**测试基线：** 22 passed（含前序所有测试）。

---

### 4.5 优化与对齐报告提炼

**优化阶段（[REPORT].optimized-repository）：**

从 `service.py`（1697 行）中拆分独立辅助模块：
- `hanging_mvp/rules_access.py`（规则遍历）
- `hanging_mvp/numeric_utils.py`（数值工具）
- `hanging_mvp/structural_clearance.py`（结构间距）
- `hanging_mvp/support_context.py`（支撑语境+高度）
- `hanging_mvp/response_assets.py`（响应序列化）

拆分后 `service.py` 降至 1206 行，后续功能迭代后当前为 2572 行，`analyze_hanging_mvp` 函数体本身仍超过 1500 行。

根目录的一行文本指针文件（`install_giant.sh` 等 7 个）替换为真正的 launcher 脚本（解析仓库根目录、转发所有参数、exec 目标脚本）。

`HANGING_MVP_API_CONTRACT.md` 作为追加兼容字段的显式合约文档添加。

**安全整改（[REPORT].alignment-and-code-audit）：**

`.env.vlm_sam` 和 `.env.vlm_sam.bak.20260530_222644` 含明文 VLM API Key，已从可下载仓库 ZIP 中移除，替换为 `.env.vlm_sam.example`。对应密钥应视为泄露并轮换。

---

## 5. 维护性、鲁棒性与逻辑漏洞审核

以下为本次代码审核新发现的问题，按严重程度排列。未在历史修复日志中出现。

---

### 【严重】M-01：`_cleanup_semantic_types_from_remove_mask` 中 `globals()` 检查永远为 `False`

**文件：** `hanging_mvp/service.py` 第 335 行

**代码：**
```python
obj_mask = resize_mask_to_original(mask, original_size_wh) if 'resize_mask_to_original' in globals() else None
```

**问题：** `service.py` 从 `flux_export` 的导入语句为：
```python
from .flux_export import build_remove_mask_for_hardpaste, build_render_prompt, safe_asset_id, select_flux_manifest_items
```
`resize_mask_to_original` 未被导入到 `service.py` 的 globals 命名空间中，因此该条件**永远为 `False`**，fallback 代码块（内联手写的 resize 逻辑）永远执行。

**影响：** 功能正确性未受损（fallback 实现等价），但 `globals()` 检查是死代码，掩盖了"resize 辅助函数未导入"这一隐含假设。若未来开发者直接移除 fallback 块（以为条件分支会被激活），会导致跳过 resize 时 `obj_mask=None`，进而跳过所有掩码重叠计算，使 `cleanup_semantic_types` 结果静默为空。

**修复方式：** 从 `flux_export` 直接导入 `resize_mask_to_original`，移除 `globals()` 检查，改为直接调用。

---

### 【严重】M-02：三个服务的 `require_auth` 在 `API_KEY` 未设置时完全免鉴权

**文件：** `hanging_mvp/service.py` L47、`semantic_mvp/service.py` L987、`services/metric3d_api/app.py` L43

**代码：**
```python
def require_auth(a):
    if API_KEY and a != f"Bearer {API_KEY}":
        raise HTTPException(401, "Unauthorized")
```

**问题：** `API_KEY=""` 时短路为无鉴权，任何未携带 Header 的请求均可访问。在开发环境便利，但生产部署若忘记设置 `API_KEY` 则完全开放。三个服务的 `API_KEY` 均来自同一个 `os.getenv("API_KEY", "")` 调用，意味着**一个环境变量控制所有服务的鉴权**，无法细粒度控制。

**修复方式：** 生产启动脚本中强制要求 `API_KEY` 非空（`preflight_check.py` 可增加此项检查），或增加 `ALLOW_UNAUTH_LOCAL=1` 显式豁免标志。

---

### 【中等】M-03：`candidate_safety.py::choose_flux2_input_policy` 标签匹配未经规范化

**文件：** `hanging_mvp/candidate_safety.py` L96-109

**代码：**
```python
risky = {x.lower() for x in (risky_existing_art_labels or {"existing_art", "wall_art", "painting", ...})}
labels = sorted({str(x or "").strip().lower() for x in cleanup_labels ...})
existing_art_present = any(label in risky for label in labels)
```

**问题：** `cleanup_labels` 来自 `_cleanup_semantic_types_from_remove_mask` 的 `obj.get("label")`，该值已经是 `canonical_semantic_label` 输出（如 `"painting"`、`"existing_art"`），而 `risky` 集合的默认值包含原始字符串 `"existing_art"` 和 `"painting"` 等——两侧实际上均为小写规范标签，逻辑可以工作。  
但 `"wall_art"` 在 `canonical_semantic_label` 中会被映射为 `"existing_art"`，所以 `risky` 集合中 `"wall_art"` 的条目永远不会被命中（已规范化的标签不含 `"wall_art"`）。这是死代码而非功能错误，但注释和命名可能误导后续维护者。

**修复方式：** `risky` 集合只保留规范标签，移除别名（`"wall_art"` 等），并补充注释说明 `cleanup_labels` 已规范化。

---

### 【中等】M-04：`exit_review.py` 整体异常捕获可能掩盖配置错误

**文件：** `hanging_mvp/exit_review.py` L106-120

**代码：**
```python
except Exception as exc:
    approved = not (requires_review and block_unreviewed)
    return {..., "reasons": [str(exc)]}
```

**问题：** `requests.post` 的网络超时、DNS 解析失败、JSON 解析错误均被同一 `except` 块处理，统一降级为 `vlm_review_error_blocked` 或 `vlm_review_error_allowed_by_rule`。若 `VLM_API_URL` 配置错误（如端口号错误），每次调用都会失败，但服务不会发出任何告警，只是每次返回 `approved` 基于 `block_unreviewed` 的值。在 `block_unreviewed=false` 的配置下，这意味着 VLM 配置错误时所有高风险候选都会通过。

**修复方式：** 区分网络错误（可重试）和配置错误（需人工介入），在连续失败超过阈值时写入健康状态或告警。

---

### 【中等】M-05：`geometry.py::estimate_g_hat_from_floor_normals` 的过滤常量不可配置

**文件：** `hanging_mvp/geometry.py` L34

**代码：**
```python
sel = n[(n[:, 1] < -0.75) & (np.abs(n[:, 2]) < 0.45)]
```

**问题：** `-0.75` 和 `0.45` 是针对特定数据集中 Metric3D 法线约定的硬编码经验阈值（`ny < -0.75` 对应地面朝上，`|nz| < 0.45` 排除前向表面）。若法线约定改变或换用不同深度模型，地面重力方向估计会静默失效（退回默认值 `[0,-1,0]`，置信度 0.0），但不会报错。

**修复方式：** 将这两个阈值暴露到 `rules_v1.json` 的 `camera.floor_normal_filter` 分组，或至少作为 `estimate_g_hat_from_floor_normals` 的关键字参数。

---

### 【中等】M-06：`support_context.py` 地面对象点数门槛部分硬编码

**文件：** `hanging_mvp/support_context.py` L86

**代码：**
```python
if count >= (50 if label == "floor" else min_valid_points)
```

**问题：** `floor` 标签的最低有效点数门槛（50）是硬编码的，而 `min_valid_points`（默认 200）来自规则配置。在点云稀疏的场景下，地面往往有足够点数，但若场景中有裁切的地面视野，50 点可能对应非常小的区域，导致地面参考高度不稳定。

**修复方式：** 在 `support_surface_constraints` 规则分组中增加 `min_floor_valid_points` 字段。

---

### 【中等】M-07：`BflClient.poll` 在超时极短时可能跳过全部轮询

**文件：** `image_editing/providers/bfl.py` L76

**代码：**
```python
deadline = time.time() + self.timeout_s
while time.time() < deadline:
    ...
    time.sleep(self.poll_interval_s)
```

**问题：** `deadline` 在 `submit` 成功后才开始计时，不含提交耗时。若 `BFL_TIMEOUT_S` 被意外设为极小值（如 0.5s），deadline 条件在首次 `time.time()` 检查时可能已超时，循环体一次都不执行，抛出 `TimeoutError("BFL task timed out after 0.5s: id=...")`，这个错误信息让人误以为任务已提交并排队很久，实际上轮询从未发生。

**修复方式：** 保证至少执行一次轮询（`do-while` 语义），或在 `from_env` 时对 `timeout_s` 设置最小值（如 10 秒），并在配置读取时记录警告。

---

### 【低优先级】M-08：`masks.py` 中 `"human"` 和 `"people"` 在 `_KNOWN_CANONICAL_LABELS` 中的条目是死代码

**文件：** `hanging_mvp/masks.py` L21

**代码：**
```python
_KNOWN_CANONICAL_LABELS = {
    ..., "human", "people", ...
}
...
if key in {\"human\", \"people\"}:
    return \"person\"
return key
```

**问题：** `"human"` 和 `"people"` 虽在集合中，但代码在 `return key` 之前先检查 `if key in {"human", "people"}` 并返回 `"person"`，所以这两个键从未被 `return key` 返回。功能正确，但集合条目有歧义，可能让人误以为 `canonical_semantic_label("human")` 返回 `"human"`。

**修复方式：** 从 `_KNOWN_CANONICAL_LABELS` 移除 `"human"` 和 `"people"`（它们只需在 `_LABEL_ALIASES` 中保留）。

---

### 【低优先级】M-09：`response_assets.py::_draw_overlay` 的竖线高度硬编码

**文件：** `hanging_mvp/response_assets.py` L18

**代码：**
```python
cv2.line(room_rgb, (int(c[0]), max(0, int(c[1]) - 200)), (int(c[0]), min(room_rgb.shape[0]-1, int(c[1]) + 200)), ...)
```

**问题：** ±200px 竖线高度对于小图（如 400px 高）会占据整个图高，对于高分辨率图（如 3000px 高）几乎不可见。OpenCV 会自动裁剪超出画布的线段，不会崩溃。

**修复方式：** 将竖线高度改为 `room_rgb.shape[0] * 0.5`（图高的一半）或绝对值取图高百分比。

---

### 【低优先级】M-10：`semantic_mvp/service.py` 与 `hanging_mvp/masks.py` 的标签定义平行维护

**文件：** `semantic_mvp/service.py` L24-83（`ROLE_BY_LABEL`、`LABEL_ALIASES`）与 `hanging_mvp/masks.py` L11-53（`_LABEL_ALIASES`、`_KNOWN_CANONICAL_LABELS`）

**问题：** 两处标签映射表独立维护，若 semantic service 增加新标签（如 `"artwork_shelf"`），hanging service 可能不会同步处理，导致下游分类静默回落到 unknown。

**修复方式：** 提取共享 `labels.py` 模块，两服务均从其导入，单点维护标签定义。

---

### 【设计缺陷】M-11：`height_fit_contract.py::room_corner_floor_fit_contract` 未被任何测试覆盖

**文件：** `hanging_mvp/height_fit_contract.py`

**问题：** P2 功能的契约函数固定返回 `enabled: false`，当 P2 功能实现时该函数必须修改，但当前没有测试验证其返回格式的稳定性，也没有测试验证 P2 实现后的正确性路径。

**修复方式：** 在 `test_height_fit_contract.py` 中增加对返回结构字段的断言（`implementation_status`、`enabled`、`min_valid_wall_planes` 等）。

---

### 【安全遗留】M-12：`.qwenignore` 排除了 `.env.vlm_sam`，但未排除可能的新 env 文件

**文件：** `.qwenignore`

**问题：** `.qwenignore` 列出了 `.env.vlm_sam` 等特定文件，但采用白名单排除而非通配符（如 `.env.*`），若后续新增 `.env.bfl` 等文件且包含 API Key，可能被意外提交。当前 `.env.bfl.example` 已在仓库中，但实际的 `.env.bfl` 依赖 `.gitignore` 的 `*.env` 模式保护。

**修复方式：** `.qwenignore` 增加 `*.env` 和 `.env.*` 通配符覆盖，与 `.gitignore` 保持一致。

---

## 6. 后续迭代日志撰写规范

以下规范适用于本仓库所有后续补丁、修复和功能迭代的日志文档。

---

### 6.1 文件命名与存放

| 类型 | 文件名格式 | 存放位置 |
|------|----------|---------|
| 补丁记录 | `[PATCH].v{版本}-{简述}.md` | 仓库根目录 |
| 部署记录 | `[DEPLOY].{目标描述}.md` | 仓库根目录 |
| 审计报告 | `[REPORT].{主题}.md` | 仓库根目录 |
| 合约说明 | `[CONTRACT].{接口名}.md` | 仓库根目录 |
| 配置说明 | `[CONFIG].{主题}.md` | 仓库根目录 |
| 经验沉淀 | `SKILL.md` | `.qwen/skills/{主题名}/` |

---

### 6.2 补丁日志（`[PATCH]`）必含字段

```markdown
# [PATCH].v{N}-{简述}.md

## 触发背景
<!-- 1-3 句话，说明问题的观测现象，不做根因分析 -->

## 修复范围
<!-- 列出修改文件及简单说明，不超过 10 行 -->

## 关键决策
<!-- 对本补丁中的非显然选择，说明"选了什么"和"为什么不选备选方案" -->
<!-- 1-5 条，每条 1-3 句 -->

## 测试基线
<!-- 新的 pytest 预期结果，如 "22 passed" 或新增测试名 -->

## 已知限制
<!-- 本补丁明确排除的问题，防止他人重复尝试 -->
```

---

### 6.3 决策说明的写作要求

**正确示例：**
> `opening_or_recess_overlap_ratio` 在 `opening_recess_used_in_solid_mask=true` 时标记为说明字段，不构成第二个硬门。solid\_wall\_mask 已排除孔洞像素，重复门等价于用同一信号两次拒绝。

**错误示例（勿用）：**
> 之所以不把 opening\_overlap 作为硬门，是因为这样做可能会误拒，应该用更好的方式。

规则：说明**选了什么**，说明**为什么该选项比另一个选项更符合产品约束**，不说"可能""应该""更好"等模糊措辞。

---

### 6.4 测试基线更新规范

每次补丁后在日志末尾记录：

```
## 测试基线
pytest -q → N passed [, M warnings]
新增：test_{功能}.py::test_{用例名}
```

若测试数量减少（如合并或删除用例），需单独说明删除原因。

---

### 6.5 规则配置变更规范

凡修改 `configs/rules_v1.json` 中的数值，需在补丁日志中记录：

```markdown
## 规则变更
| 路径 | 旧值 | 新值 | 原因 |
|------|------|------|------|
| structural_clearance_policy.min_edge_clearance_fraction_of_artwork_width | 0.25 | 0.333 | 实测窗边 25% 间距不足，3 件真实场景挂画被拒绝 |
```

凡新增规则字段，必须同步：① `configs/rules_v1.json` 条目，② `rule_value` 调用处默认值，③ 相关测试断言。

---

### 6.6 接口变更（追加兼容）规范

所有对 `POST /analyze_hanging_mvp` 输出结构的变更遵循追加兼容原则：

- 只能新增字段，不能删除或重命名已有字段。
- 新字段添加后，同步更新 `HANGING_MVP_API_CONTRACT.md` 的"P{N} 扩展字段"节。
- 若字段语义发生变化（即使名称不变），视为破坏兼容性，需要版本号递增和迁移说明。
- `business_server_payload_version` 字段在接口语义变化时更新（当前 `v1.3-safe-wall-mask-qa-height-review`）。

---

### 6.7 不应记录的内容

日志文档禁止包含：
- API Key 或任何凭证明文（含备份文件中的残留）
- 宿主机绝对路径（`/root/autodl-tmp/xxx`）作为规范路径引用
- "修复了 bug"等无信息量描述（应说明 bug 的具体表现和修复的具体位置）
- 已在代码注释中完整说明的内容（日志与注释不应重复）

---

### 6.8 `.qwen/skills/` 经验沉淀规范

每次解决了一个新类型的工程问题，在 `.qwen/skills/{主题名}/SKILL.md` 中记录：

```markdown
---
name: {主题名}
description: {一句话触发条件描述，供 AI 助手识别适用场景}
source: auto-skill | manual
extracted_at: '{ISO 8601 时间}'
---

# {标题}

## 问题
<!-- 现象描述，不超过 5 行 -->

## 根因
<!-- 技术根因 -->

## 解法
<!-- 代码或命令 -->

## 验证方式
<!-- 如何确认解法有效 -->
```

已有 skill 主题参考：`bash-default-value-brace-bug`、`metric3d-zombie-process-diagnosis`、`semantic-sam-lazy-load-crash`、`unicode-fullwidth-space-cli-path-bug`、`mock-provider-safety-guards` 等。

---

*文档生成于 2026-06-17，覆盖截至当前提交的全部代码和修复日志。*

