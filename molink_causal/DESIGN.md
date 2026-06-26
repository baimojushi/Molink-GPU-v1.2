# Molink 挂画 · 因果层方案与事件驱动落地

日期：2026-06-22
子项目代号：`molink-causal`
状态：方案 + 可运行代码骨架

---

## 0. 子项目概述

这个子项目把 GPU 服务端已经产出的精准字段，组织成一个**因果层（causal layer）**，让流式 LLM 在生图过程中输出的不是泛泛安抚，而是真实的空间认知翻译。

核心主张：当前 LLM 文案的问题不在模型能力，在于输入太薄、时序太早、复核字段缺失。补齐字段、对齐时序、引入建筑光哲学，文案就能从"等一下哦"变成一次空间认知启蒙，同时增加艺术品购买动机。

它做三件事：

1. 把字段按**真实挂画的认知顺序**分成 L0–L4 五层，每层对应顾问会说的一类话。
2. 用**事件驱动状态机**把 worker 进度事件翻译成"此刻允许说哪一层、该触发哪个语义意图"。
3. 把建筑光哲学（来自 Studio Junction、Duchaufour-Lawrance、ASK、Oppenheim、Torres、Kahn、Ando 等案例）落地为**可计算的光线约束**，其中视觉噪声用图形学统计而非 VLM，保证可复现。

边界原则贯穿全程：像资深心理咨询师，**揭示关系，翻译体验，止于观察，绝不替用户下结论**。

---

## 1. 思考过程记录

### 1.1 为什么是"因果层"而不是"字段清单"

字段清单回答"有什么数据"，因果层回答"在什么时刻、对谁、说哪一句话才有意义"。真实挂画是一个连续认知过程：先识别作品、再理解空间、建立尺度、筛选挂点、判断安全、渲染交付、QA 复核。LLM 的输入应该覆盖这个认知链，而不是只塞作品名和当前步骤。

五层的划分对应顾问与人的关系建立顺序：

- L0 原始信号：像素坐标、置信度分值、鉴权字段。存在但沉默，让其他层可信，自己不被说出来。
- L1 空间词汇：墙/窗/门/家具计数。事实陈述，让用户感到"被这台机器具体看见了"。
- L2 构图因果：高度、家具锚点、安全间距。推理语气，建立"为什么这样挂"的判断依据。
- L3 感官预测：光线、窗距、阴影、时间性。经验翻译，触碰未来的身体感受。
- L4 开放邀请：作品情绪、景观竞争、留白。开一扇门，把结论留给用户。

### 1.2 为什么先做光线、且光线先于美学

第一轮做了字段串联和触发器；第二轮意识到光线被当前置信度逻辑完全忽略，而光是挂画体验的物理基础——人眼不感知绝对亮度，只校准到视野中最亮的区域，所以窗的相对亮度决定作品的感知质量。第三轮发现只立足知觉不够，建筑师对光的抽象思考（阴影即材料、天花即漫射器、景观即竞争、寂静墙原则）提供了知觉之上的因果维度。

顺序是：知觉约束（能不能看清）→ 哲学约束（看清之后体验如何）。前者是硬约束，后者是加分/惩罚和叙事原料。

### 1.3 为什么 surface_visual_noise 从 VLM 移到图形学

最初把视觉噪声设计成 VLM 字段。复盘后改为图形学统计，三个理由：

1. 可复现：同一组像素永远得到同一个数；VLM 有采样方差，跨版本会漂移。图形统计可单元测试、可回归。
2. 可解释：触发时能回看究竟是边缘密度、高频能量还是局部方差导致，便于校准与排障。
3. 与光照正交：会干扰画作的视觉噪声位于高空间频带，光照梯度位于低频带。高通分解把两者彻底分离，noise 不会把窗边平滑亮度渐变误判为"墙面很乱"。`demo.py` 里"平整白墙含光照梯度"判 low、"砖墙纹理"判 high，验证了这个分离。

判断标准：当一个语义判断本质上是统计量（纹理、亮度、对比度、频率），用图形学；当它需要世界知识或意图理解（这是窗还是镜子、作品情绪是沉静还是欢快），用 VLM。

### 1.4 为什么用事件驱动 + 状态机

worker 的进度是事件流，字段是逐步到位的。状态机回答"现在到哪一步"，因果层回答"这一步允许说哪层话"，触发器回答"在允许的层里哪个关系值得说"。三者解耦后，新增一个触发器或调一个阈值，不会牵动时序逻辑。

关键设计：字段快照单调增长（已知事实只增不减），层级解锁累积式（高层解锁不撤销低层），状态不回退（worker 偶发乱序时保护时序）。失败/复核的判断基于交付事实（有没有 final URL、有没有 partial 图），不基于单一 stage 字符串——这是当前链路最常见的误报来源。

---

## 2. 因果层模型（人读版）

| 层 | 触发时机 | 语气 | 关键字段 | 顾问会说的一类话 |
|---|---|---|---|---|
| L0 原始信号 | 永不进入正文 | 沉默 | bbox_px、ssim、delivery_token | （不说，只支撑可信度） |
| L1 空间词汇 | 10–45% | 事实陈述 | wall/window/door/furniture/existing_art_count | 识别到 4 面墙、2 扇窗、6 处既有装饰，参照网格足够。 |
| L2 构图因果 | 55–75% | 推理 | center_height_cm、above_furniture、clearance | 沙发中轴正上方 16cm，画心 152cm，坐下来视线落在画的中部。 |
| L3 感官预测 | 65–90% | 经验翻译 | window_clearance、ambient_light_mode、shadow_presence | 距窗 82cm，超出直射光竞争区，作品不会因时段产生颜色漂移。 |
| L4 开放邀请 | 交付时 | 不下结论 | mood_register、landscape_competition、留白比 | 它在这个空间里选择了留白——这是它的本性，效果图出来你会感受到。 |

---

## 3. 机器逻辑：事件驱动架构

### 3.1 数据流（每个进度事件一轮）

```
Event(stage, state, pct, delta)
  │
  ├─▶ FieldSnapshot.merge(delta)          单调增长的知识库
  │
  ├─▶ light_model.evaluate(...)           算 R_light / B / P / delta（字段就绪才算）
  │
  ├─▶ state_machine.next_state(...)       基于交付事实转移，终态不回退
  │
  ├─▶ unlocked_layers(state)              本状态允许的因果层
  │
  ├─▶ 遍历 ALL_TRIGGERS，在允许层内评估 guard，收集命中
  │      每层取最高 priority，再全局 top-k
  │
  └─▶ compose → NarrationBundle           task + char_limit + facts + copy_hints + constraints
          │
          └─▶ 流式 LLM 生成最终文案（人格由 constraints + copy_hint 约束）
```

引擎不生成文案本身，只生成"喂给 LLM 的扎实语料"。文案的顾问人格、绝不下结论的边界，由 `constraints` 与 `copy_hint` 在 LLM 侧共同成形。

### 3.2 状态与解锁层

| 状态 | stage | pct | 解锁层 | task |
|---|---|---|---|---|
| INIT | init | <10 | 无 | waiting_progress |
| MEASURING | metric3d | 0–20 | L1 | waiting_progress |
| UNDERSTANDING | semantic | 20–45 | L1 | waiting_progress |
| EVALUATING | hanging | 45–60 | L1+L2 | waiting_progress |
| RENDERING | render | 60–90 | L1+L2+L3 | waiting_progress |
| UPLOADING | r2_upload | 90–99 | L1+L2+L3 | waiting_progress |
| DELIVERED | — | final>0 | L1+L2+L3+L4 | delivery_main |
| REVIEW | — | partial>0,final=0 | L1+L2+L3 | render_review |
| FAILED | failed | — | L1 | render_review |

REVIEW 不解锁 L4：还没真正交付，不该做开放邀请式的总结，只做"正在复核效果图"的安全表达。

### 3.3 触发器命中规则

- 只评估当前状态已解锁层内的触发器。
- 每层最多保留一个（最高 priority），再全局取 top-k（默认 2），避免文案塞满。
- T5 复核触发器 priority 最高（70），保证复核态优先说复核。
- guard 抛异常时静默跳过（字段缺失降级，不报错）。

---

## 4. 数学方法

记号：`L_*` 为区域亮度均值，`rho = L_window/L_room`，`lambda = L_wall/L_room`，`lambda_ceil = L_ceiling/L_room`，`d_w` 为作品到最近窗边距离（cm），`theta ∈ {same, opposite, perpendicular, none}` 为窗相对候选墙关系，`S` 为室外亮度类型。JPEG/HDR 压缩修正系数 `kappa ≈ 0.6`。

### 4.1 风险分量（取最大值合成）

逆光风险（仅 opposite）：

\[
R_{bl} = \begin{cases}
0 & \theta \neq \text{opp 或 } \rho/\kappa \leq 1.5 \\
\dfrac{\rho/\kappa - 1.5}{3.5} & 1.5 < \rho/\kappa \leq 5 \\
1 & \rho/\kappa > 5
\end{cases}
\]

同墙侧光（仅 same）：

\[
R_{sw} = \max\!\left(0,\; 1 - \dfrac{d_w}{60}\right)
\]

直射热点（仅 bright_direct）：

\[
R_{hs} = \begin{cases} 0.2 & \theta=\text{perp} \\ 0.5 & \theta=\text{opp} \\ 1.0 & \theta=\text{same} \end{cases}
\]

暗区（所有配置）：

\[
R_{dk} = \max\!\left(0,\; 1 - \dfrac{\lambda}{0.6}\right)
\]

合成：\( R_{light} = \max(R_{bl}, R_{sw}, R_{hs}, R_{dk}) \)。取最大值而非加权和——任一严重条件都是结构性限制，不应被低风险项稀释。

### 4.2 奖励分量

\[
B_{light} = \begin{cases}
+0.15 & \theta=\text{opp},\; \rho/\kappa \leq 1.5,\; S \neq \text{bright\_direct} \\
+0.10 & \theta=\text{perp},\; 40 \leq d_w \leq 150,\; S=\text{overcast} \\
0 & \text{otherwise}
\end{cases}
\]

\[
B_{ceil} = \begin{cases}
+0.12 & \lambda_{ceil} > 0.9,\; \text{ceiling} \in \{\text{high, dramatic}\} \\
+0.05 & \lambda_{ceil} > 0.8,\; \text{ceiling} = \text{standard} \\
0 & \text{otherwise}
\end{cases}
\]

### 4.3 惩罚分量

寂静墙（Torres）：

\[
P_{backdrop} = \begin{cases}
0 & \text{noise}=\text{low} \\
0.05 & \text{noise}=\text{medium},\; c_{art} > 0.5 \\
0.12 & \text{noise}=\text{high},\; c_{art} \geq 0.4 \\
0.22 & \text{noise}=\text{high},\; c_{art} < 0.4
\end{cases}
\]

景观竞争（Oppenheim，仅小画幅）：

\[
P_{view} = 0.12 \cdot \text{lcs} \cdot \mathbf{1}\!\left[\dfrac{A_{art}}{A_{wall}} < 0.15\right]
\]

### 4.4 合成进墙面评分

\[
\text{score}^{final}_{wall} = \text{score}^{base}_{wall} + (B_{light}+B_{ceil}) - \alpha R_{light} - (P_{backdrop}+P_{view})
\]

`alpha = 0.20`，上线后用真实偏好数据校准。风险等级映射：`R_light < 0.25` 无标记；`0.25–0.55` 软风险 `light_caution`；`≥ 0.55` reason_tag `light_concern` 且候选降序。

### 4.5 视觉噪声（图形学）

候选墙 mask 区域 W 的亮度通道 L 上：

1. 低频（光照）：\( L_{lf} = \text{GaussianBlur}(L, \sigma) \)，`sigma = 9px`
2. 高频（纹理）：\( L_{hf} = L - L_{lf} \)
3. 三个归一化统计：边缘密度 \( e \)（Canny 边缘像素比）、高频能量 \( hf = \text{mean}(L_{hf}^2)/255^2 \)、局部标准差 \( sd \)
4. 合成：\( N = \text{clip}(0.45\,\hat e + 0.35\,\widehat{hf} + 0.20\,\widehat{sd},\; 0, 1) \)，各项按经验上限截断归一化
5. 分箱：\( N < 0.28 \to \text{low} \)；\( 0.28 \leq N < 0.55 \to \text{medium} \)；\( N \geq 0.55 \to \text{high} \)

高通分解（步骤 1–2）是关键：它把光照梯度（rho/lambda 的事）从纹理噪声里彻底剔除，两个字段因此正交。

---

## 5. 触发器全表

| ID | 层 | guard 关键条件 | 来源原则 |
|---|---|---|---|
| T_SPACE_VOCAB | L1 | wall_count>0 | — |
| T4 清场 | L1 | existing_art_count≥2 | — |
| T1 家具锚点 | L2 | above_furniture + axis_centering + gap∈[12,28] | 坐视标准 |
| T13 阴影映衬 | L2 | shadow∈{moderate,dramatic} + surface_active_light=false | Kahn / Duchaufour-Lawrance |
| T2 窗距光线 | L3 | window_clearance>60 + S≠bright_direct | 知觉 |
| T11 天花漫射 | L3 | ambient=gallery_ambient + lambda_ceil>0.9 | Kimbell / ASK Grand Dame |
| T12 时间性光质 | L3 | temporal=dynamic + shadow∈{moderate,dramatic} | Studio Junction |
| T15 寂静背景 | L3 | noise=low + ambient∈{soft_diffuse,gallery_ambient} | Torres |
| T5 复核在途 | L3 | partial>0 + final=0 | 安全表达 |
| T3 小画幅克制 | L4 | area_ratio<0.12 + risk=low + wall_count≥2 | 博物馆克制 |
| T14 景观竞争 | L4 | lcs>0.6 + area_ratio<0.15 | Oppenheim |
| T_INVITE 作品本性 | L4 | mood_register 存在 + final>0 | — |

---

## 6. 新增字段注释（意义 + 数据来源管线）

来源管线缩写：`pix` 图像统计/图形学（新增，无 VLM）｜`vlm+` VLM 语义新增返回｜`geom` Metric3D｜`sem` semantics.json｜`rank` 候选排序｜`qa` 渲染 QA｜`rt` 运行时。

### 6.1 像素统计（pix，无需 VLM）

| 字段 | 意义 | 来源 |
|---|---|---|
| `light.rho` | 窗室亮度比，逆光/热点风险基础。受 HDR 压缩，阈值乘 kappa | 窗口 bbox 内外像素均值（低频） |
| `light.wall_lambda` | 候选墙照度比，暗区风险 R_dk | 墙面 mask 像素均值 |
| `light.lambda_ceil` | 天花漫射比，B_ceil 加分（Kimbell/Grand Dame） | 天花 bbox 像素均值（低频） |
| `light.surface_visual_noise_score` | 0–1 墙面视觉噪声，寂静墙原则。高通频带统计，排除光照梯度 | `imaging.surface_visual_noise()` |

### 6.2 VLM 语义新增（vlm+）

| 字段 | 意义 | 来源 |
|---|---|---|
| `light.theta` | 窗相对候选墙 same/opposite/perpendicular/none。仅相对方位，无需绝对指南针 | VLM 空间图语义 |
| `light.exterior_brightness_class` | bright_direct/overcast/artificial | VLM 空间图语义 |
| `light.ambient_light_mode` | gallery_ambient/soft_diffuse/direct_dramatic/flat_artificial/mixed，美术馆模式判断（T11） | VLM 空间图语义（亦可由 lambda_ceil+shadow 推断） |
| `light.shadow_presence` | none/subtle/moderate/dramatic，空间阴影强度（阴影即设计） | VLM 空间图语义（pix 阴影锐利度辅助） |
| `light.temporal_light_quality` | static/dynamic，光的时间性（Studio Junction） | VLM 空间图语义（pix 阴影锐利度推断） |
| `artwork.mood_register` | calm/contemplative/dynamic/melancholic/joyful，L4 开放邀请原料 | VLM 艺术品图语义 |
| `artwork.visual_complexity` | 0–1，与 noise 交叉判断寂静墙 | VLM 艺术品图语义（pix 高频能量后备） |
| `plan.surface_active_light` | 墙面是否受直射光，阴影映衬区（T13） | VLM 空间图语义 |
| `plan.landscape_competition_score` | 0–1 观看方向景观竞争度（Oppenheim） | VLM 对面窗内容复杂度评分 |

---

## 7. 代码骨架（molink_causal/）

| 文件 | 职责 |
|---|---|
| `fields.py` | 字段快照模型，每字段标注来源管线与所属层，单调增长 merge |
| `imaging.py` | 图形学管线：`surface_visual_noise()` + `light_statistics()`，确定性可复现 |
| `light_model.py` | 光线数学：风险/奖励/惩罚分量 + 合成 delta，全部纯函数 |
| `triggers.py` | T1–T15 触发器注册表：guard + copy_hint + priority |
| `state_machine.py` | 状态、转移、每状态解锁层、task 映射 |
| `engine.py` | 缝合器：on_event → merge → 光线 → 转移 → 触发 → 装配 bundle |
| `demo.py` | 端到端事件流模拟 + imaging 可复现性验证 |

运行 `python demo.py` 可见：事件流逐轮装配的 bundle、复核分支不解锁 L4、交付分支解锁 L4，以及视觉噪声对砖墙/平墙/光照梯度的正确分箱。

### 7.1 待校准项

- `light_model` 全部阈值（kappa、alpha、各分量边界）：用真实偏好/转化数据回归。
- `imaging` 噪声合成权重与饱和上限（W_EDGE/W_HF/W_SD、CAP_*、T_LO/T_HI）：用标注墙面集校准。
- `surface_active_light` 与 `landscape_competition_score` 的 VLM 推断置信度：低置信时降级为中性，不触发强惩罚。

### 7.2 接入点

引擎输出的 `NarrationBundle` 直接序列化为现有 `llm-stream?task=` 的输入 JSON。`facts` 对应第一版方案里的 `waiting_progress` / `delivery_main` / `render_review` 结构，`copy_hints` 与 `constraints` 作为系统提示注入。光线分量明细可进入 operator/客服侧调试视图，不进用户正文。
