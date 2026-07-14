# 第四阶段交付说明

本阶段只实现深色墙纸建议的判断、下发与等待页预览，不触发墙面重上色。

- 作品有效 alpha 区域计算 `artwork_mean_luminance`、`artwork_contrast` 与 `is_dark_or_high_contrast`。
- 每个候选使用原空间图和候选墙 mask 计算 `wall_mean_luminance`，并排除膨胀后的作品投影四边形。
- 建议规则固定为：作品平均亮度 `< 0.42` 或对比度 `> 0.55`，同时候选墙平均亮度 `> 0.62`。
- 建议墙色来自作品低明度像素的抽象深色调，并通过 `install.suggested_wall_tone_rgb` 下发。
- 本阶段不构造 `wallpaper_recolor`，不新增 Stage B 调用，也不修改 `final_image_url`。
