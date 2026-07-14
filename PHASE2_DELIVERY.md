# 第二阶段交付说明

本阶段实现 Stage A 之后的单一 Stage B 合并通道骨架。

- `soft_furnishing_requested` 与 `wallpaper_recolor` 是两个独立输入；四种组合统一进入 `build_stage_b_prompt()` / `build_stage_b_mask()`。
- 两项都未启用时不调用生图；任意一项或两项同时启用时，每张 Stage A 成图最多调用一次 Stage B。
- Stage A 图先保存到 `pre_styling_hd`，Stage B 成功后覆盖 `final_hd`；失败不重试并保留 Stage A 交付图。
- 作品/画框保护 mask 缺字段、文件不存在或内容为空时一律 fail closed，不调用 Stage B，直接回退 Stage A 图。
- 结果候选新增 `pre_styling_image_url` 与 `styling_status`，并记录 Stage B 调用次数。
- 本阶段使用确定性的占位编辑区域，仅用于打通调用与回退契约。风格 VLM、SAM 支撑面精修、轻量 QA 与墙纸建议判断属于后续阶段。
