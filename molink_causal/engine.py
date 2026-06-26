"""
engine.py
=========
把状态机、触发器、光线数学缝合成一个事件驱动引擎。

数据流 (每个进度事件一轮)
-------------------------
  Event ──▶ merge 进 FieldSnapshot (单调增长)
        ──▶ 读取 hanging 端已算出的 light_components 作为权威光线分量
        ──▶ state_machine.next_state 转移状态
        ──▶ unlocked_layers 得到本状态允许的因果层
        ──▶ 在允许层内评估全部触发器 guard, 收集命中
        ──▶ 按 (priority) 取每层最高 + 全局 top-k
        ──▶ compose 装配 LLM 输入 bundle (task, char_limit, facts, copy_hints, constraints)
        ──▶ 交给流式 LLM 生成文案

引擎不生成文案本身, 只生成"喂给 LLM 的扎实语料"。文案的人格 (顾问口吻、
绝不下结论) 由 constraints + copy_hint 共同约束, 在 LLM 侧成形。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List, Optional

from .fields import FieldSnapshot, Layer
from .state_machine import (Event, State, next_state, unlocked_layers, STATE_TASK)
from .triggers import ALL_TRIGGERS, Trigger


# 各 task 的输出长度上限 (字符)
CHAR_LIMIT = {
    "waiting_progress": 260,
    "delivery_main": 320,
    "render_review": 220,
}

# 全局每轮最多并入几个触发器 (避免文案塞太满)
TOP_K = 2

# 绝不替用户下结论 + 安全表达的恒定约束


def _truthy_env(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default)
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def _light_in_narration_enabled() -> bool:
    return _truthy_env("MOLINK_CAUSAL_LIGHT_IN_NARRATION", "0")


BASE_CONSTRAINTS = [
    "用私人艺术顾问口吻, 揭示空间关系并翻译成感官体验, 止于观察, 不做'好/不好/适合'的评价性收尾。",
    "不要编造没有出现在输入中的家具、光线、方位、距离或颜色。",
    "不要暴露算法、像素、模型、SSIM、mask、Flux2 等技术词。",
    "出现 height_finalized=false 时必须提示现场卷尺确认最终钉点。",
    "final_url_count=0 且 partial_review_count>0 时表述为'正在复核效果图', 禁止说系统失败或重新拍摄。",
]


@dataclass
class NarrationBundle:
    task: str
    char_limit: int
    state: str
    pct: float
    facts: dict                       # 进入本轮的事实字段 (按层过滤后)
    fired: List[str]                  # 命中并采用的触发器 id
    copy_hints: List[str]             # 对应的语义意图提示
    light_components: Optional[dict]  # 光线分量明细 (调试 + 部分进入 copy)
    constraints: List[str]


class CausalEngine:
    def __init__(self):
        self.snap = FieldSnapshot()
        self.state = State.INIT
        self.light_score = None

    def on_event(self, ev: Event) -> NarrationBundle:
        self.snap.merge(ev.delta)
        # 同步运行时进度到快照
        self.snap.progress.stage = ev.stage
        self.snap.progress.state = ev.state
        self.snap.progress.pct = ev.pct

        # 光线数学只在 hanging 服务端逐候选计算一次。这里读取 best_plan 透传的
        # light_components, 避免叙事层与排序层因入参不同而发散。
        if self.snap.best_plan is not None and self.snap.best_plan.light_components:
            comp = dict(self.snap.best_plan.light_components or {})
            self.light_score = SimpleNamespace(
                components=comp,
                risk_band=self.snap.best_plan.light_risk_band or comp.get("risk_band"),
                penalty_band=self.snap.best_plan.light_penalty_band or comp.get("penalty_band"),
                delta=comp.get("delta"),
                penalty=comp.get("penalty"),
                r_light=comp.get("r_light"),
            )

        self.state = next_state(self.state, ev, self.snap)
        allowed = unlocked_layers(self.state)

        fired = self._fire(allowed)
        return self._compose(fired)

    # ---- 触发器评估: 每层取最高 priority, 再全局 top-k ----
    def _fire(self, allowed) -> List[Trigger]:
        hits: List[Trigger] = []
        for t in ALL_TRIGGERS:
            if t.layer not in allowed:
                continue
            try:
                if t.guard(self.snap, self.light_score):
                    hits.append(t)
            except Exception:
                continue
        # 每层只留最高 priority
        by_layer = {}
        for t in hits:
            cur = by_layer.get(t.layer)
            if cur is None or t.priority > cur.priority:
                by_layer[t.layer] = t
        ranked = sorted(by_layer.values(), key=lambda x: x.priority, reverse=True)
        return ranked[:TOP_K]

    # ---- 装配 LLM 输入 ----
    def _compose(self, fired: List[Trigger]) -> NarrationBundle:
        task = STATE_TASK[self.state]
        facts = self._facts_for_layers(unlocked_layers(self.state))
        comp = None
        if self.snap.best_plan is not None and self.snap.best_plan.light_components:
            comp = self.snap.best_plan.light_components
        elif self.light_score is not None:
            comp = self.light_score.components
        if comp and _light_in_narration_enabled():
            facts["light_analysis"] = comp
        return NarrationBundle(
            task=task,
            char_limit=CHAR_LIMIT.get(task, 260),
            state=self.state.value,
            pct=self.snap.progress.pct,
            facts=facts,
            fired=[t.id for t in fired],
            copy_hints=[t.copy_hint for t in fired],
            light_components=comp,
            constraints=list(BASE_CONSTRAINTS),
        )

    def _facts_for_layers(self, allowed) -> dict:
        """只导出 <= 已解锁最高层 的字段, 且永不导出 L0 原始信号正文。"""
        out = {}
        s = self.snap
        if Layer.L1_VOCAB in allowed:
            out["artwork"] = {"name": s.artwork.name, "author": s.artwork.author,
                              "width_cm": s.artwork.width_cm, "height_cm": s.artwork.height_cm,
                              "has_frame": s.artwork.has_frame}
            out["space"] = {"wall_count": s.space.wall_count, "window_count": s.space.window_count,
                            "door_count": s.space.door_count, "furniture_count": s.space.furniture_count,
                            "existing_art_count": s.space.existing_art_count,
                            "person_count": s.space.person_count}
            out["progress"] = {"pct": s.progress.pct, "message": s.progress.message,
                               "advisor_message": s.progress.advisor_message}
        if Layer.L2_CAUSAL in allowed and s.best_plan is not None:
            p = s.best_plan
            out["plan"] = {"wall_position_zh": p.wall_position_zh,
                           "center_height_cm": p.center_height_cm,
                           "bottom_edge_cm": p.bottom_edge_cm, "top_edge_cm": p.top_edge_cm,
                           "height_finalized": p.height_finalized, "risk_level": p.risk_level,
                           "above_furniture_label_zh": p.above_furniture_label_zh,
                           "bottom_gap_to_furniture_cm": p.bottom_gap_to_furniture_cm,
                           "composition_zh": p.composition_zh}
        if Layer.L3_SENSORY in allowed and s.best_plan is not None:
            out.setdefault("plan", {})
            out["plan"]["window_edge_clearance_cm"] = s.best_plan.window_edge_clearance_cm
            out["light"] = {"ambient_light_mode": s.light.ambient_light_mode,
                            "shadow_presence": s.light.shadow_presence,
                            "temporal_light_quality": s.light.temporal_light_quality}
            out["render_review"] = {"final_url_count": s.delivery.final_url_count,
                                    "partial_review_count": s.delivery.partial_review_count}
        if Layer.L4_INVITATION in allowed:
            out["invitation"] = {"mood_register": s.artwork.mood_register,
                                 "visual_complexity": s.artwork.visual_complexity,
                                 "landscape_competition_score":
                                     s.best_plan.landscape_competition_score if s.best_plan else None,
                                 "artwork_wall_area_ratio":
                                     s.best_plan.artwork_wall_area_ratio if s.best_plan else None}
        return out
