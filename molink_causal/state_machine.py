"""
state_machine.py
================
事件驱动状态机。把 worker 进度事件翻译成 LLM 叙事意图。

状态 (从 LLM 视角看的订单生命周期)
-----------------------------------
  INIT          初始, pct<10, 只能输出无字段过程短句
  MEASURING     metric3d, 测量空间纵深        ~0–20%
  UNDERSTANDING semantic, 识别墙/窗/门/家具    ~20–45%
  EVALUATING    hanging, 比较安全挂点+定方案   ~45–60%
  RENDERING     flux2, 局部光影融合            ~60–90%
  UPLOADING     r2_upload, 整理交付图          ~90–99%
  DELIVERED     终态: 有 final URL
  REVIEW        准终态: 有 partial, 无 final (render_review)
  FAILED        终态: no_safe_wall / 崩溃

每个状态"解锁"哪些因果层 —— 这是时序门槛的核心。
字段就算提前到位, 未解锁的层也不会进入 LLM 正文 (防止 pct<60 就开始
做交付承诺)。

事件 (Event)
------------
worker 每次进度推送是一个 Event(stage, state, pct, delta)。
delta 是本次新到位的字段增量 (dict), 会被 merge 进 FieldSnapshot。
状态机据 (stage, state, pct, delivery) 决定转移, 不信任单一信号。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Set

from .fields import FieldSnapshot, Layer


class State(str, Enum):
    INIT = "INIT"
    MEASURING = "MEASURING"
    UNDERSTANDING = "UNDERSTANDING"
    EVALUATING = "EVALUATING"
    RENDERING = "RENDERING"
    UPLOADING = "UPLOADING"
    DELIVERED = "DELIVERED"
    REVIEW = "REVIEW"
    FAILED = "FAILED"


TERMINAL = {State.DELIVERED, State.FAILED}

# 每个状态解锁的因果层 (累积式: 高层解锁不撤销低层)
UNLOCKED_LAYERS = {
    State.INIT:          set(),
    State.MEASURING:     {Layer.L1_VOCAB},
    State.UNDERSTANDING: {Layer.L1_VOCAB},
    State.EVALUATING:    {Layer.L1_VOCAB, Layer.L2_CAUSAL},
    State.RENDERING:     {Layer.L1_VOCAB, Layer.L2_CAUSAL, Layer.L3_SENSORY},
    State.UPLOADING:     {Layer.L1_VOCAB, Layer.L2_CAUSAL, Layer.L3_SENSORY},
    State.DELIVERED:     {Layer.L1_VOCAB, Layer.L2_CAUSAL, Layer.L3_SENSORY, Layer.L4_INVITATION},
    State.REVIEW:        {Layer.L1_VOCAB, Layer.L2_CAUSAL, Layer.L3_SENSORY},
    State.FAILED:        {Layer.L1_VOCAB},
}

# 每个状态对应的 LLM task
STATE_TASK = {
    State.INIT: "waiting_progress",
    State.MEASURING: "waiting_progress",
    State.UNDERSTANDING: "waiting_progress",
    State.EVALUATING: "waiting_progress",
    State.RENDERING: "waiting_progress",
    State.UPLOADING: "waiting_progress",
    State.DELIVERED: "delivery_main",
    State.REVIEW: "render_review",
    State.FAILED: "render_review",
}


@dataclass
class Event:
    stage: str           # metric3d/semantic/hanging/render/r2_upload/failed
    state: str           # running/done/failed
    pct: float
    delta: dict = field(default_factory=dict)


# stage -> 候选状态 (再用 pct / delivery 细化)
_STAGE_STATE = {
    "init": State.INIT,
    "metric3d": State.MEASURING,
    "semantic": State.UNDERSTANDING,
    "hanging": State.EVALUATING,
    "render": State.RENDERING,
    "r2_upload": State.UPLOADING,
}


def next_state(prev: State, ev: Event, snap: FieldSnapshot) -> State:
    """根据事件 + 当前快照决定下一个状态。终态不回退。"""
    if prev in TERMINAL:
        return prev

    # 失败 / 复核优先判断 (基于交付事实, 不基于单一 stage 字符串)
    if ev.stage == "failed" or ev.state == "failed":
        if snap.delivery.partial_review_count > 0 and snap.delivery.final_url_count == 0:
            return State.REVIEW
        return State.FAILED

    if snap.delivery.final_url_count > 0:
        return State.DELIVERED
    if snap.delivery.partial_review_count > 0 and ev.stage in ("render", "r2_upload") \
            and ev.state == "done":
        return State.REVIEW

    cand = _STAGE_STATE.get(ev.stage, prev)
    # INIT 门槛: pct<10 留在 INIT
    if cand == State.MEASURING and ev.pct < 10:
        return State.INIT
    # 不允许状态倒退 (worker 偶发乱序)
    order = [State.INIT, State.MEASURING, State.UNDERSTANDING,
             State.EVALUATING, State.RENDERING, State.UPLOADING]
    if prev in order and cand in order and order.index(cand) < order.index(prev):
        return prev
    return cand


def unlocked_layers(state: State) -> Set[Layer]:
    return UNLOCKED_LAYERS[state]
