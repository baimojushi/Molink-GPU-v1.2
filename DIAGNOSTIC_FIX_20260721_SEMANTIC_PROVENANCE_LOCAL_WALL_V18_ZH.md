# V18：语义来源与实体墙局部几何修复

## 本次失败的确定根因

最近失败任务并非单纯由 `12°` 法线阈值过严造成，而是两条链路叠加：

1. Semantic 在真实 VLM 未配置或调用失败时，用两个 mock bbox 返回 `status=ok`；第二阶段又信任该 seed。
2. 单个粗墙被 Metric3D 法线桶拆成四个 geometry proposal，窗帘因近似竖直、近似平面而被当成墙。
3. 中央实体墙进入 v17 的局部深度 fallback 后，代码读取不存在的 `WallPlane.normal`，实际字段为 `WallPlane.n`；异常被 `except: pass` 静默吞掉，所以 `segmented_depth_plane_yaw` 永远缺失。
4. v17 的所谓 4×4 图像分块没有使用像素位置，而是对打散后的 3D 点按 camera x/y 范围分桶，沙发、帘布和墙点仍可混入同一局部票。
5. quorum 还把低置信深度数值算作一票，并混用了“墙法线 yaw”和“墙切向 yaw”。

因此，放宽法线偏差阈值会让三个窗帘伪墙重新通过，同时不能可靠恢复中央墙。

## 修复契约

- `auto` 模式没有真实 VLM 时返回结构化 503，不再产生 mock 假成功。
- 显式 `vlm_mode=mock` 仅在 `SEMANTIC_ALLOW_MOCK_MODE=1` 的测试环境可用。
- 对齐阶段复验 seed provenance；历史 mock seed 返回 422。
- geometry proposal 只写入诊断，不进入 `objects`；Hanging 对旧缓存再做一次 lineage 硬拒绝。
- 局部深度拟合接收密集 `HxWx3` 点云，以实际像素网格和清洁墙 mask 分块。
- 清洁墙 mask 会扣除 forbidden、foreground、support、existing_content、non_support，并腐蚀边界。
- 修复 `WallPlane.normal` 字段错误；异常写入诊断，不再静默吞掉。
- direct normal 使用独立清洁墙 mask，不由有争议的全墙 RANSAC 预选样本。
- 2-of-3 quorum 只允许高置信来源投票，并统一为墙法线 yaw；相机俯仰/横滚通过重力正交约束保留。

## 部署与重跑

1. 配置 `VLM_PROVIDER=openai_compatible`、`VLM_API_URL`、`VLM_API_KEY`、`VLM_MODEL`。
2. 保持 `SEMANTIC_ALLOW_MOCK_MODE=0`。
3. 从第一次 Semantic 调用重新执行；不要复用旧 `semantics_seed.json`、geometry proposal、wall basis 或 projected quad。
4. 若 Semantic 返回 503，检查任务目录 `semantic_error_response.json`。

期望：没有真实 VLM 时明确失败；有真实 VLM 时只让独立识别的中央实体墙参与局部几何仲裁，窗帘继续被拒绝。
