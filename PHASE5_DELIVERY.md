# 第五阶段 GPU 交付说明

- 补充渲染继续跳过 Metric3D、Semantic 与 Hanging，复用持久化的 `hanging_result.zip`。
- 非主墙先走既有 Stage A，再按订单软装开关和墙纸 opt-in 进入一次合并 Stage B。
- 主推荐墙墙纸补渲染不重新调用 Stage A provider；worker 从业务端保存的 `pre_styling_image_url` 恢复干净底图，并按 manifest 的 asset/wall id 精确填入 `final_hd`。
- 底图缺失或与 manifest 不匹配时直接失败，不会拿当前已软装的交付终图继续叠加生成。
- Stage B 的保护区、mask 外逐像素锁定、内容 QA 与失败回退保持不变；主墙失败状态由业务端解释为“保留原交付图”。
- result payload 新增 `primary_wall_rerender`、`primary_wall_id`、`wallpaper_recolor_requested`、`stage_a_base_source`，并保留 `stage_b_call_count` 诊断字段。
- 新增 `tests/test_phase5_primary_stage_a_reuse.py`；既有四组合测试继续保证软装与墙纸同时请求时每资产只调用一次 provider。
