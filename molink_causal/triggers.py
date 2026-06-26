"""
triggers.py
===========
触发器注册表。每个触发器是 (id, layer, priority, guard, copy_hint)。

guard      : 给定 FieldSnapshot + LightScore, 返回 bool, 判定条件是否满足。
copy_hint  : 给 LLM 的"语义意图"提示 —— 不是成品文案, 是一句方向指引,
             LLM 据此用顾问口吻生成最终句子。绝不替用户下结论的边界写在
             constraints 里 (见 compose.py), 这里只给"该说什么关系"。
priority   : 同一层内多个触发器命中时的取舍顺序 (大者优先)。

层级与触发器的对应:
  L1 空间词汇   : T_SPACE_VOCAB (统一计数陈述)
  L2 构图因果   : T1 家具锚点, T13 阴影映衬区
  L3 感官预测   : T2 窗距光线, T11 天花漫射, T12 时间性光质, T15 寂静背景
  L4 开放邀请   : T3 小画幅克制, T14 景观竞争, T_INVITE (作品本性)
  跨层运行时    : T4 清场, T5 复核在途
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .fields import FieldSnapshot, Layer


@dataclass(frozen=True)
class Trigger:
    id: str
    layer: Layer
    priority: int
    guard: Callable[[FieldSnapshot, object], bool]
    copy_hint: str


def _plan(s: FieldSnapshot):
    return s.best_plan


# ---------------- L1 ----------------

def _g_space_vocab(s, ls):
    return s.space.wall_count > 0

T_SPACE_VOCAB = Trigger(
    "T_SPACE_VOCAB", Layer.L1_VOCAB, 10, _g_space_vocab,
    "事实陈述识别到的墙/窗/门/家具/既有装饰计数, 说明参照网格已建立, 几何分析继续。")


# ---------------- L2 ----------------

def _g_t1(s, ls):
    p = _plan(s)
    return bool(p and p.above_furniture_label_zh and p.axis_centering
               and p.bottom_gap_to_furniture_cm is not None
               and 12 <= p.bottom_gap_to_furniture_cm <= 28)

T1_FURNITURE_ANCHOR = Trigger(
    "T1_FURNITURE_ANCHOR", Layer.L2_CAUSAL, 60, _g_t1,
    "解释作品位于家具中轴正上方、画心落在坐视高度区间 (约150–160cm), "
    "把数字翻译成'坐下来视线落在画的中部'的身体感受。")

def _g_t13(s, ls):
    p = _plan(s)
    return bool(s.light.shadow_presence in ("moderate", "dramatic")
               and p and p.surface_active_light is False
               and s.light.ambient_light_mode != "flat_artificial")

T13_SHADOW_FOIL = Trigger(
    "T13_SHADOW_FOIL", Layer.L2_CAUSAL, 50, _g_t13,
    "空间有建筑性阴影在构层次, 候选墙处于漫射光区域 —— 作品可自主发光, "
    "不必和空间阴影竞争。阴影即设计原则 (Kahn / Duchaufour-Lawrance)。")


# ---------------- L3 ----------------

def _g_t2(s, ls):
    p = _plan(s)
    if not p or s.light.exterior_brightness_class == "bright_direct":
        return False
    if p.window_edge_clearance_cm is not None:
        return p.window_edge_clearance_cm > 60
    # 对面/垂直墙窗不会进入同墙 structural_clearance。VLM 已确认非同墙窗且非直射时,
    # 允许触发保守光线叙事, 但 copy_hint 会要求避免编造具体距离。
    return s.light.theta in ("opposite", "perpendicular")

T2_WINDOW_LIGHT = Trigger(
    "T2_WINDOW_LIGHT", Layer.L3_SENSORY, 55, _g_t2,
    "若有 window_edge_clearance_cm 就给出窗边距离; 若没有具体距离, 只说明窗与候选墙不是同墙直射关系, "
    "光线进入方式更接近漫射, 不要编造距离。")

def _g_t11(s, ls):
    return bool(s.light.ambient_light_mode == "gallery_ambient"
               and (s.light.lambda_ceil or 0) > 0.9)

T11_GALLERY_AMBIENT = Trigger(
    "T11_GALLERY_AMBIENT", Layer.L3_SENSORY, 58, _g_t11,
    "候选墙被高天花漫射光均匀照亮 —— 美术馆照明原理, 无需人工布光, "
    "作品在最接近原作意图的光线下被观看。Kimbell / Grand Dame 原则。")

def _g_t12(s, ls):
    return bool(s.light.temporal_light_quality == "dynamic"
               and s.light.shadow_presence in ("moderate", "dramatic"))

T12_TEMPORAL = Trigger(
    "T12_TEMPORAL", Layer.L3_SENSORY, 45, _g_t12,
    "空间光线有强方向性, 作品在早晨和下午呈现不同面貌 —— 表述为这面墙的"
    "时间特质而非不确定性。Studio Junction 原则。")

def _g_t15(s, ls):
    comp = getattr(ls, "components", {}) or {}
    return bool(
        (float(comp.get("P_backdrop") or 0.0) >= 0.05)
        or (_noise(s) == "low" and s.light.ambient_light_mode in ("soft_diffuse", "gallery_ambient"))
    )

T15_QUIET_BACKDROP = Trigger(
    "T15_QUIET_BACKDROP", Layer.L3_SENSORY, 40, _g_t15,
    "围绕'安静背景'解释墙面与作品的关系: 若 light_analysis.P_backdrop>0, 说明墙面纹理/视觉噪声已进入筛选, "
    "避免背景抢走简洁作品的表达; 若 P_backdrop=0, 说明墙面光线平静均匀, 作品在这里是唯一的表达。Torres 寂静墙原则。")


# ---------------- L4 ----------------

def _g_t3(s, ls):
    p = _plan(s)
    return bool(p and p.artwork_wall_area_ratio
               and p.artwork_wall_area_ratio < 0.12
               and p.risk_level == "low" and s.space.wall_count >= 2)

T3_RESTRAINT = Trigger(
    "T3_RESTRAINT", Layer.L4_INVITATION, 35, _g_t3,
    "作品在墙上留出大量呼吸空间 —— 博物馆的选择, 让观看成为需要靠近的邀请。"
    "止于观察, 不下'好/不好'结论。")

def _g_t14(s, ls):
    p = _plan(s)
    comp = getattr(ls, "components", {}) or {}
    return bool(p and p.artwork_wall_area_ratio is not None
               and p.artwork_wall_area_ratio < 0.15
               and ((p.landscape_competition_score is not None and p.landscape_competition_score > 0.6)
                    or float(comp.get("P_view") or 0.0) >= 0.05))

T14_LANDSCAPE = Trigger(
    "T14_LANDSCAPE", Layer.L4_INVITATION, 38, _g_t14,
    "这面墙主要视线方向有开阔景观 —— 作品尺度在此需要一点分量才能建立对话, "
    "而非被视野淹没。必须以'效果图出来你会感受到这个关系'收尾, 不替用户判断。"
    "Oppenheim 原则。")

def _g_invite(s, ls):
    return bool(s.delivery.final_url_count > 0
                and (s.artwork.mood_register or s.artwork.visual_complexity is not None
                     or (s.artwork.name and str(s.artwork.name).strip())))

T_INVITE = Trigger(
    "T_INVITE", Layer.L4_INVITATION, 30, _g_invite,
    "结合作品情绪基调 mood_register; 若 mood_register 缺失, 只使用 visual_complexity、作品尺幅和空间密度这些已知事实, "
    "开一扇门让用户自己走进去, 把结论留给用户。")


# ---------------- 跨层运行时 ----------------

def _g_t4(s, ls):
    return bool(s.space.existing_art_count >= 2)

T4_CLEANUP = Trigger(
    "T4_CLEANUP", Layer.L1_VOCAB, 48, _g_t4,
    "系统正在移除 N 处既有装饰, 让墙面只保留和作品之间的对话。带叙事期待感。")

def _g_t5(s, ls):
    return bool(s.delivery.partial_review_count > 0 and s.delivery.final_url_count == 0)

T5_REVIEW = Trigger(
    "T5_REVIEW", Layer.L3_SENSORY, 70, _g_t5,
    "效果图已生成预览, 系统正在复核画芯色彩与墙面融合质量 —— 最后一步确认。"
    "禁止'系统失败/重新拍摄'等表述。")


def _noise(s):
    sc = s.light.surface_visual_noise_score
    if sc is None:
        return None
    return "low" if sc < 0.28 else ("medium" if sc < 0.55 else "high")


ALL_TRIGGERS = [
    T_SPACE_VOCAB, T4_CLEANUP,
    T1_FURNITURE_ANCHOR, T13_SHADOW_FOIL,
    T2_WINDOW_LIGHT, T11_GALLERY_AMBIENT, T12_TEMPORAL, T15_QUIET_BACKDROP, T5_REVIEW,
    T3_RESTRAINT, T14_LANDSCAPE, T_INVITE,
]
