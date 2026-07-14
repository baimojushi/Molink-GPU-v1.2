# 第三阶段交付说明

本阶段把 Stage B 的占位分析与占位区域替换为真实风格分析和语义区域。

- Stage B 被请求时，对 `flux_inputs/artwork.png` 发起最多一次结构化 styling VLM 请求；远端未配置或失败时走确定性算法配色兜底。
- `styling_analysis` 包含作品抽象性、风格、质感、抽象强调色、中性色、视觉张力、线条方向和建议墙色；固定墙色会作为分析前置条件。
- hanging zip 为每个候选导出 `stage_b_decor_zone_mask`、`stage_b_wall_zone_mask` 和 `stage_b_protect_zone_mask`。
- decor 区域使用 `role=support` 的 floor/sofa/table/cabinet/sideboard/chair SAM 多边形并集；SAM 不可用时使用 bbox；无支撑物时回退到非墙面的画面下半区。每个候选还会按作品位置收窄到近墙走廊，避免开放整间地面。
- protect 区域按作品物理尺度估算约 10cm 留白并膨胀；缺字段、坏路径或空 mask 均 fail closed。
- Stage B 输出会再次做本地 mask 合成和内容质量 QA：保护区与 mask 外像素逐像素不变；decor 区消费 masked SSIM 与改动比例阈值；墙面重上色消费亮度梯度相关性与梯度能量阈值。任一门槛不通过都回退 Stage A。
- 软装数量由视觉张力和作品实际物理尺寸共同决定，prompt 同时映射风格、材质、色彩和线条方向。
