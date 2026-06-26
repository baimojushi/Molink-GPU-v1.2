"""
fields.py
=========
Molink 挂画因果层 —— 字段快照模型 (Field Snapshot Model)。

设计原则
--------
1. 单调增长 (monotonic growth)：字段快照是一个随事件流不断被填充的知识库。
   一个字段一旦被某个事件写入，后续事件只允许覆盖更新，不允许"丢失"。
   这保证了 LLM 在任意时刻拿到的都是"目前已知的最完整事实"。
2. 来源可追溯 (provenance)：每个字段都标注了它来自哪条管线。四条来源：
       - geom    : Metric3D / 几何置信度管线 (已有)
       - sem     : VLM semantics.json 语义管线 (已有)
       - pix     : 纯图像统计 / 图形学管线 (本子项目新增，无需 VLM)
       - vlm+    : VLM 语义管线新增字段 (本子项目要求 VLM 补充返回)
       - rank    : 候选排序 / hanging 引擎 (已有)
       - qa      : 渲染 QA 聚合 (已有)
       - rt      : worker 运行时进度 (已有)
   provenance 既是文档，也是运行时校验：pix / vlm+ 字段缺失时引擎降级，不报错。
3. 层级归属 (layer)：每个语义字段标注它服务于哪个因果层 (L0–L4)，
   决定它在哪个进度阶段才被允许进入 LLM 正文。
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields as dc_fields
from enum import Enum
from typing import Optional


class Provenance(str, Enum):
    GEOM = "geom"      # Metric3D / 几何置信度 (已有)
    SEM = "sem"        # VLM semantics.json (已有)
    PIX = "pix"        # 图像统计 / 图形学 (新增, 无 VLM)
    VLM_PLUS = "vlm+"  # VLM 语义新增返回字段 (新增)
    RANK = "rank"      # 候选排序 / hanging (已有)
    QA = "qa"          # 渲染 QA (已有)
    RT = "rt"          # worker 运行时进度 (已有)


class Layer(int, Enum):
    L0_RAW = 0          # 原始信号, 永不进入 LLM 正文
    L1_VOCAB = 1        # 空间词汇, 事实陈述
    L2_CAUSAL = 2       # 构图因果, 推理语气
    L3_SENSORY = 3      # 感官预测, 经验翻译
    L4_INVITATION = 4   # 开放邀请, 绝不下结论


# 一个轻量的字段元数据描述符。仅用于自省 / 文档生成 / 运行时降级判断。
def f(default, *, src: Provenance, layer: Layer, note: str):
    return field(default=default, metadata={"src": src, "layer": layer, "note": note})


@dataclass
class ArtworkFacts:
    name: str = f("", src=Provenance.RANK, layer=Layer.L1_VOCAB,
                  note="作品名, 顾问文案主体。来源: 订单 / job payload。")
    author: str = f("", src=Provenance.RANK, layer=Layer.L1_VOCAB,
                    note="作者, 为空时不得编造。来源: 订单。")
    width_cm: float = f(0.0, src=Provenance.RANK, layer=Layer.L2_CAUSAL,
                        note="厘米宽度, 用于构图比例计算。来源: job payload 派生。")
    height_cm: float = f(0.0, src=Provenance.RANK, layer=Layer.L2_CAUSAL,
                         note="厘米高度。来源: job payload 派生。")
    has_frame: bool = f(False, src=Provenance.RANK, layer=Layer.L3_SENSORY,
                        note="是否带框, 影响阴影深度叙事 (阴影即设计原则)。来源: job payload。")
    aspect_ratio: float = f(1.0, src=Provenance.RANK, layer=Layer.L2_CAUSAL,
                            note="横竖幅判断。来源: 派生。")
    # —— 以下为本子项目新增, 服务 L4 开放邀请 ——
    mood_register: Optional[str] = f(None, src=Provenance.VLM_PLUS, layer=Layer.L4_INVITATION,
        note="作品情绪基调 calm/contemplative/dynamic/melancholic/joyful。"
             "L4 开放邀请的核心原料, 让'作品本性'的表达有视觉依据而非 LLM 揣测。"
             "来源: VLM 艺术品图语义管线新增返回。")
    visual_complexity: Optional[float] = f(None, src=Provenance.VLM_PLUS, layer=Layer.L2_CAUSAL,
        note="0–1 视觉复杂度。与 surface_visual_noise 交叉判断寂静墙原则 (Torres)。"
             "来源: VLM 艺术品图语义管线 (亦可用 pix 高频能量作为后备估计)。")


@dataclass
class SpaceVocab:
    """L1 空间词汇 —— 全部来自已有 semantic 管线的派生计数。"""
    wall_count: int = f(0, src=Provenance.SEM, layer=Layer.L1_VOCAB,
                        note="候选墙数量。等待页解释'正在筛选墙面'。")
    window_count: int = f(0, src=Provenance.SEM, layer=Layer.L1_VOCAB,
                          note="窗户数量。安全间距 + 光线约束依据。")
    door_count: int = f(0, src=Provenance.SEM, layer=Layer.L1_VOCAB,
                        note="门数量。安全间距依据。")
    furniture_count: int = f(0, src=Provenance.SEM, layer=Layer.L1_VOCAB,
                             note="家具数量。构图锚点参照。")
    existing_art_count: int = f(0, src=Provenance.SEM, layer=Layer.L1_VOCAB,
                                note="既有挂画数量。清理叙事 (T4) 依据。")
    person_count: int = f(0, src=Provenance.SEM, layer=Layer.L1_VOCAB,
                          note="前景人物数量。erase 策略依据。")


@dataclass
class LightFacts:
    """
    光线事实 —— 本子项目的核心。
    低频部分 (rho, lambda_ceil, wall_lambda) 来自 pix 像素统计;
    方位与类型 (theta, exterior_class 等) 来自 vlm+ 相对语义;
    surface_visual_noise 是高频部分, 来自 pix 图形学统计 (而非 VLM, 见 imaging.py 说明)。
    """
    # —— 像素统计 (pix), 无需 VLM ——
    rho: Optional[float] = f(None, src=Provenance.PIX, layer=Layer.L0_RAW,
        note="rho = L_window / L_room, 窗室亮度比。手机 JPEG 的 HDR 压缩会压低实测值, "
             "阈值需乘修正系数 kappa~=0.6。来源: 窗口 bbox 内外像素均值 (低频)。")
    wall_lambda: Optional[float] = f(None, src=Provenance.PIX, layer=Layer.L0_RAW,
        note="lambda = L_wall / L_room, 候选墙照度比。暗区风险 R_dk 依据。来源: 墙面 mask 像素均值。")
    lambda_ceil: Optional[float] = f(None, src=Provenance.PIX, layer=Layer.L0_RAW,
        note="lambda_ceil = L_ceiling / L_room, 天花漫射加分 B_ceil 依据 (Kimbell/Grand Dame 原则)。"
             "来源: 天花区域 bbox 像素均值 (低频)。")
    surface_visual_noise_score: Optional[float] = f(None, src=Provenance.PIX, layer=Layer.L0_RAW,
        note="0–1 墙面视觉噪声标量。高通频带的边缘密度 + 高频能量 + 局部方差合成。"
             "刻意排除低频光照梯度 (那是 rho/lambda 的事), 只保留纹理噪声。"
             "寂静墙原则 (Torres) 的可复现操作化。来源: imaging.surface_visual_noise()。")
    # —— VLM 相对语义 (vlm+) ——
    theta: Optional[str] = f(None, src=Provenance.VLM_PLUS, layer=Layer.L0_RAW,
        note="窗相对候选墙的关系 same/opposite/perpendicular/none。"
             "无需绝对方位, VLM 仅给相对描述即可。来源: VLM 空间图语义新增。")
    exterior_brightness_class: Optional[str] = f(None, src=Provenance.VLM_PLUS, layer=Layer.L0_RAW,
        note="bright_direct/overcast/artificial。直射光/阴天/无窗。来源: VLM 空间图语义新增。")
    ambient_light_mode: Optional[str] = f(None, src=Provenance.VLM_PLUS, layer=Layer.L3_SENSORY,
        note="gallery_ambient/soft_diffuse/direct_dramatic/flat_artificial/mixed。"
             "美术馆漫射模式判断 (T11)。来源: VLM 空间图语义新增 (亦可由 lambda_ceil + shadow 推断)。")
    shadow_presence: Optional[str] = f(None, src=Provenance.VLM_PLUS, layer=Layer.L3_SENSORY,
        note="none/subtle/moderate/dramatic。空间阴影强度 (阴影即设计原则, Kahn/Duchaufour-Lawrance)。"
             "来源: VLM 空间图语义新增 (亦可由 pix 阴影锐利度辅助)。")
    temporal_light_quality: Optional[str] = f(None, src=Provenance.VLM_PLUS, layer=Layer.L3_SENSORY,
        note="static/dynamic。光的时间性 (Studio Junction 原则)。锐利定向阴影 -> dynamic。"
             "来源: VLM 空间图语义新增 (亦可由 pix 阴影锐利度推断)。")


@dataclass
class WallCandidate:
    """单个候选墙 + 其安装/构图/光线评估。"""
    wall_id: str = f("", src=Provenance.RANK, layer=Layer.L0_RAW, note="墙面 ID, 不给用户。")
    wall_position_zh: str = f("", src=Provenance.RANK, layer=Layer.L2_CAUSAL,
                              note="用户可理解的位置描述, 如'主墙面'。")
    is_primary_wall: bool = f(False, src=Provenance.RANK, layer=Layer.L2_CAUSAL, note="是否主视觉墙。")
    artwork_wall_area_ratio: float = f(0.0, src=Provenance.RANK, layer=Layer.L2_CAUSAL,
        note="作品面积 / 墙面可用面积。小画幅克制美学 (T3) 与景观竞争 (T14) 依据。")
    # —— 安装高度 ——
    center_height_cm: Optional[float] = f(None, src=Provenance.RANK, layer=Layer.L2_CAUSAL,
                                          note="画心离地高度, 真实挂画最核心字段。")
    bottom_edge_cm: Optional[float] = f(None, src=Provenance.RANK, layer=Layer.L2_CAUSAL, note="下沿离地。")
    top_edge_cm: Optional[float] = f(None, src=Provenance.RANK, layer=Layer.L2_CAUSAL, note="上沿离地。")
    height_finalized: bool = f(False, src=Provenance.RANK, layer=Layer.L2_CAUSAL,
                               note="高度是否可直接执行, false 时文案必须提示现场确认。")
    # —— 构图 ——
    above_furniture_label_zh: Optional[str] = f(None, src=Provenance.RANK, layer=Layer.L2_CAUSAL,
                                                note="位于何家具上方, 如'沙发'。家具锚点 (T1) 依据。")
    bottom_gap_to_furniture_cm: Optional[float] = f(None, src=Provenance.RANK, layer=Layer.L2_CAUSAL,
                                                    note="作品下沿到家具距离。T1 依据。")
    axis_centering: Optional[bool] = f(None, src=Provenance.RANK, layer=Layer.L2_CAUSAL,
                                       note="是否与家具/墙面轴线对齐。T1 依据。")
    composition_zh: str = f("", src=Provenance.RANK, layer=Layer.L2_CAUSAL, note="自然语言构图描述。")
    # —— 安全 ——
    risk_level: str = f("unknown", src=Provenance.RANK, layer=Layer.L2_CAUSAL, note="low/medium/high。")
    window_edge_clearance_cm: Optional[float] = f(None, src=Provenance.RANK, layer=Layer.L3_SENSORY,
                                                  note="到最近窗边距离 (cm)。窗距光线 (T2) + 同墙侧光 (R_sw) 依据。")
    # —— 光线 (per-wall, vlm+) ——
    surface_active_light: Optional[bool] = f(None, src=Provenance.VLM_PLUS, layer=Layer.L3_SENSORY,
        note="墙面是否处于直射/近直射光。阴影映衬区 (T13) 依据。来源: VLM 空间图语义新增。")
    landscape_competition_score: Optional[float] = f(None, src=Provenance.VLM_PLUS, layer=Layer.L4_INVITATION,
        note="0–1 观看方向上的景观竞争度 (Oppenheim 原则)。来源: VLM 对面窗内容复杂度评分。")
    light_components: Optional[dict] = f(None, src=Provenance.RANK, layer=Layer.L3_SENSORY,
        note="hanging 服务端逐候选计算出的权威光线分量。叙事层只读取该结果, 不重复计算。")
    light_risk_band: Optional[str] = f(None, src=Provenance.RANK, layer=Layer.L3_SENSORY,
        note="光线风险 band: none/caution/concern。")
    light_penalty_band: Optional[str] = f(None, src=Provenance.RANK, layer=Layer.L3_SENSORY,
        note="寂静墙/景观竞争等降序惩罚 band: none/caution/concern。")
    # —— 状态 ——
    needs_review: bool = f(False, src=Provenance.RANK, layer=Layer.L0_RAW, note="是否需人工复核。")
    production_usable: bool = f(False, src=Provenance.RANK, layer=Layer.L0_RAW, note="是否可生产交付。")


@dataclass
class RenderState:
    flux2_mask_sent: Optional[bool] = f(None, src=Provenance.QA, layer=Layer.L0_RAW, note="mask 是否进入供应商 payload。")
    status: str = f("pending", src=Provenance.RT, layer=Layer.L0_RAW, note="rendering/render_review/done。")
    elapsed_s: Optional[float] = f(None, src=Provenance.RT, layer=Layer.L0_RAW, note="渲染耗时, 异常长时可转化。")


@dataclass
class QaState:
    status: Optional[str] = f(None, src=Provenance.QA, layer=Layer.L0_RAW, note="passed/failed/review。")
    hard_block_reasons: list = field(default_factory=list,
        metadata={"src": Provenance.QA, "layer": Layer.L0_RAW, "note": "自动交付硬拦截原因。"})
    core_masked_ssim: Optional[float] = f(None, src=Provenance.QA, layer=Layer.L0_RAW, note="画芯结构相似度, 不给用户。")
    core_color_delta: Optional[float] = f(None, src=Provenance.QA, layer=Layer.L0_RAW, note="画芯颜色偏差, 不给用户。")


@dataclass
class DeliveryState:
    final_url_count: int = f(0, src=Provenance.QA, layer=Layer.L0_RAW, note="最终交付图数量, 为 0 不得说已完成。")
    partial_review_count: int = f(0, src=Provenance.QA, layer=Layer.L0_RAW, note="复核图数量, >0 时说复核中。")


@dataclass
class RuntimeProgress:
    stage: str = f("init", src=Provenance.RT, layer=Layer.L0_RAW,
                   note="metric3d/semantic/hanging/render/r2_upload/failed。")
    state: str = f("running", src=Provenance.RT, layer=Layer.L0_RAW, note="running/done/failed。")
    pct: float = f(0.0, src=Provenance.RT, layer=Layer.L0_RAW, note="真实进度百分比, LLM 启动门槛依赖它。")
    message: str = f("", src=Provenance.RT, layer=Layer.L0_RAW, note="当前普通进度文案。")
    advisor_message: Optional[str] = f(None, src=Provenance.RT, layer=Layer.L2_CAUSAL,
                                       note="worker 已生成的顾问摘要, 作为事实输入。")


@dataclass
class FieldSnapshot:
    """随事件流单调增长的知识库。"""
    artwork: ArtworkFacts = field(default_factory=ArtworkFacts)
    space: SpaceVocab = field(default_factory=SpaceVocab)
    light: LightFacts = field(default_factory=LightFacts)
    best_plan: Optional[WallCandidate] = None
    render: RenderState = field(default_factory=RenderState)
    qa: QaState = field(default_factory=QaState)
    delivery: DeliveryState = field(default_factory=DeliveryState)
    progress: RuntimeProgress = field(default_factory=RuntimeProgress)

    def merge(self, delta: dict) -> "FieldSnapshot":
        """
        将一个事件携带的字段增量合并进快照 (覆盖更新, 不丢失既有非空值)。
        delta 形如 {"light": {"rho": 3.2}, "best_plan": {...}}。
        """
        for group_name, group_delta in delta.items():
            if group_name == "best_plan":
                if self.best_plan is None:
                    self.best_plan = WallCandidate()
                _apply(self.best_plan, group_delta)
            elif hasattr(self, group_name):
                _apply(getattr(self, group_name), group_delta)
        return self


def _apply(obj, group_delta: dict):
    valid = {fd.name for fd in dc_fields(obj)}
    for k, v in group_delta.items():
        if k in valid and v is not None:
            setattr(obj, k, v)
