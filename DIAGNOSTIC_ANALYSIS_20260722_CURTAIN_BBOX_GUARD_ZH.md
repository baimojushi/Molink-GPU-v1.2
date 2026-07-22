# 窗帘被 bbox 当作墙面的复核与修复

## 结论

这类错误不能只靠 Metric3D 的“竖直墙面”判定解决。窗帘在深度图里经常是连续、近似竖直的面；SAM 也可能对它给出高质量 mask。因此“`role=wall` → SAM 成功 → 平面/支撑率通过”会把错误语义放大成可渲染墙面。

本轮同时保留了一个重要区分：当前上传的 `job_478c26ae` 诊断快照最终只公开了 `wall_front_01` 三个候选，快照本身没有出现 curtain wall 候选；但语义对象清单也没有 curtain 对象，不能把“当前单未选到帘布”误认为“帘布误标问题已根治”。修复针对的是已复现的直接误标路径：模型返回 `label=wall, role=wall`，帘布没有独立 forbidden 记录。

## 新增防线

1. **自由文本优先级**：若模型结构化字段仍为 wall，但 `notes` 明确包含 curtain/drape/blind/sheer/window covering，则在进入 SAM/几何前改为 `label=curtain, role=forbidden`。
2. **bbox 上下文 fail-closed**：在墙面结构裁剪前检查墙框与 window/opening/door/curtain 的重叠、边缘相邻关系。高覆盖开口留下的窄条直接丢弃，不再把残片当墙。
3. **无 forbidden 标签的帘布兜底**：对贴图像左右边缘、窄而高的 wall bbox，计算原图 crop 的竖向褶皱信号（x 梯度/ y 梯度与列能量变化）。只有同时满足窄条、边缘和褶皱条件才硬拒绝，避免误伤普通大面积纹理墙。
4. **候选层二次拦截**：`evaluate_wall_mask_lineage` 读取 `wall_bbox_curtain_guard`；即使旧缓存或调用方绕过语义服务，带有 hard-reject 标记的墙仍不能生成挂画候选。
5. **坐标一致性**：纹理 crop 按 processed→original 的 x/y 比例换算，避免 Metric3D 缩放后在错误图像区域取样。

## 失败策略

可疑墙框不再降级为 geometry-only wall，也不继续尝试“看起来数学自洽”的透视投影；该墙进入 rejected 诊断，其他独立语义墙仍可继续处理。所有拒绝原因和相关 forbidden 对象会写入 `multiwall_split_debug.edge_structural_trim`，便于核验。

## 验证

- Python 编译：`semantic_mvp/service.py`、`hanging_mvp/candidate_safety.py`、新增回归测试通过。
- 新增回归覆盖：
  - 无 curtain/window 标签但窄边缘竖褶 wall bbox；
  - window 覆盖大部分 wall bbox 后留下的帘布残条；
  - `notes` 明确写出 sheer curtain、结构化 role 却错误为 wall；
  - 原有 door/window 减法和 SAM lineage 规则保持兼容。
- 当前工作区未安装 OpenCV/FastAPI/pytest，无法在此环境启动 GPU 服务或运行完整 pytest；正式 GPU 镜像应使用原任务图像复跑，并检查 `edge_structural_trim` 是否包含 `wall_bbox_curtain_guard_rejected`。
