"""
light_model.py
==============
光线置信度数学层。把第二、三轮提出的全部约束落地为纯函数。

输入来自 FieldSnapshot.light + FieldSnapshot.best_plan, 输出一组 [0,1] 分量
和一个并入墙面评分的增量。所有阈值集中在此, 便于上线后用偏好数据校准。

约束总览 (取最大值合成风险, 加权求和合成奖励/惩罚)
--------------------------------------------------
风险 R_light = max(R_bl, R_sw, R_hs, R_dk)
  R_bl  逆光      仅 theta=opposite, rho 调制
  R_sw  同墙侧光  仅 theta=same, 距离 d_w 调制
  R_hs  直射热点  仅 exterior=bright_direct, theta 调制
  R_dk  暗区      所有配置, lambda 调制
奖励 B = B_light + B_ceil
  B_light 正面/漫射侧光的优质自然光 (美术馆等效)
  B_ceil  高亮天花漫射 (Kimbell / Grand Dame)
惩罚 P = P_backdrop + P_view
  P_backdrop 寂静墙原则 (Torres): 噪声墙 x 简洁作品
  P_view     景观竞争 (Oppenheim): 小画幅 x 高景观竞争

score_final = score_light_base + B - alpha * R_light - P
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# JPEG/HDR 压缩修正: 实测 rho 普遍低于物理真实值
KAPPA = 0.6
ALPHA = 0.20  # 风险权重


def _g(x: Optional[float], default: float = 0.0) -> float:
    return default if x is None else x


# ---------- 风险分量 ----------

def r_backlight(theta: Optional[str], rho: Optional[float]) -> float:
    if theta != "opposite" or rho is None:
        return 0.0
    rho_c = rho / KAPPA  # 反压缩到物理尺度
    if rho_c <= 1.5:
        return 0.0
    if rho_c > 5.0:
        return 1.0
    return (rho_c - 1.5) / 3.5


def r_sidewall(theta: Optional[str], d_w: Optional[float]) -> float:
    if theta != "same" or d_w is None:
        return 0.0
    return max(0.0, 1.0 - d_w / 60.0)


def r_hotspot(exterior_class: Optional[str], theta: Optional[str]) -> float:
    if exterior_class != "bright_direct":
        return 0.0
    return {"perpendicular": 0.2, "opposite": 0.5, "same": 1.0}.get(theta or "", 0.3)


def r_dark(wall_lambda: Optional[float]) -> float:
    if wall_lambda is None:
        return 0.0
    return max(0.0, 1.0 - wall_lambda / 0.6)


# ---------- 奖励分量 ----------

def b_light(theta: Optional[str], rho: Optional[float],
            exterior_class: Optional[str], d_w: Optional[float]) -> float:
    if theta == "opposite" and rho is not None and rho / KAPPA <= 1.5 \
            and exterior_class != "bright_direct":
        return 0.15
    if theta == "perpendicular" and exterior_class == "overcast":
        # 对面/垂直墙窗在 structural_clearance 中会被过滤掉, d_w 可能为空。
        # 仍给一个保守的漫射侧光奖励, 避免 VLM 已确认垂直窗时该分支空转。
        if d_w is None:
            return 0.06
        if 40.0 <= d_w <= 150.0:
            return 0.10
    return 0.0


def b_ceiling(lambda_ceil: Optional[float], ceiling_height: Optional[str]) -> float:
    if lambda_ceil is None:
        return 0.0
    if lambda_ceil > 0.9 and ceiling_height in ("high", "dramatic"):
        return 0.12
    if lambda_ceil > 0.8 and ceiling_height == "standard":
        return 0.05
    return 0.0


# ---------- 惩罚分量 ----------

def p_backdrop(noise_band: Optional[str], artwork_complexity: Optional[float]) -> float:
    ac = _g(artwork_complexity, 0.5)
    if noise_band == "low" or noise_band is None:
        return 0.0
    if noise_band == "medium":
        return 0.05 if ac > 0.5 else 0.0
    # high
    return 0.22 if ac < 0.4 else 0.12


def p_view(landscape_competition: Optional[float], area_ratio: Optional[float]) -> float:
    if landscape_competition is None or area_ratio is None:
        return 0.0
    return 0.12 * landscape_competition if area_ratio < 0.15 else 0.0


# ---------- 合成 ----------

@dataclass
class LightScore:
    r_light: float
    bonus: float
    penalty: float
    delta: float            # 并入 score 的总增量 = bonus - alpha*r_light - penalty
    risk_band: str          # none / caution / concern, 只描述光线风险 R_light
    penalty_band: str       # none / caution / concern, 描述降序惩罚 P_backdrop/P_view
    components: dict        # 明细, 便于排障与触发器查询


def evaluate(light, plan, *, ceiling_height: Optional[str] = None,
             artwork_complexity: Optional[float] = None) -> LightScore:
    """
    light               : FieldSnapshot.light (LightFacts)
    plan                : FieldSnapshot.best_plan (WallCandidate)
    artwork_complexity  : FieldSnapshot.artwork.visual_complexity (寂静墙交叉判断用)
    """
    theta = getattr(light, "theta", None)
    rho = getattr(light, "rho", None)
    ext = getattr(light, "exterior_brightness_class", None)
    wl = getattr(light, "wall_lambda", None)
    lc = getattr(light, "lambda_ceil", None)
    noise_band = _noise_band(getattr(light, "surface_visual_noise_score", None))

    d_w = getattr(plan, "window_edge_clearance_cm", None)
    lcs = getattr(plan, "landscape_competition_score", None)
    ar = getattr(plan, "artwork_wall_area_ratio", None)

    comp = {
        "R_bl": round(r_backlight(theta, rho), 3),
        "R_sw": round(r_sidewall(theta, d_w), 3),
        "R_hs": round(r_hotspot(ext, theta), 3),
        "R_dk": round(r_dark(wl), 3),
        "B_light": round(b_light(theta, rho, ext, d_w), 3),
        "B_ceil": round(b_ceiling(lc, ceiling_height), 3),
        "P_backdrop": round(p_backdrop(noise_band, artwork_complexity), 3),
        "P_view": round(p_view(lcs, ar), 3),
    }
    r = max(comp["R_bl"], comp["R_sw"], comp["R_hs"], comp["R_dk"])
    bonus = comp["B_light"] + comp["B_ceil"]
    penalty = comp["P_backdrop"] + comp["P_view"]
    delta = bonus - ALPHA * r - penalty

    band = "none" if r < 0.25 else ("caution" if r < 0.55 else "concern")
    penalty_band = _penalty_band(penalty, delta)
    comp["penalty_band"] = penalty_band
    return LightScore(round(r, 3), round(bonus, 3), round(penalty, 3),
                      round(delta, 3), band, penalty_band, comp)


def _penalty_band(penalty: float, delta: float) -> str:
    if penalty >= 0.20 or delta <= -0.20:
        return "concern"
    if penalty >= 0.08 or delta <= -0.10:
        return "caution"
    return "none"


def _noise_band(score: Optional[float]) -> Optional[str]:
    if score is None:
        return None
    return "low" if score < 0.28 else ("medium" if score < 0.55 else "high")
