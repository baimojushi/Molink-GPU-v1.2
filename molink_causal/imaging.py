"""
imaging.py
==========
图形学 / 图像统计管线 (provenance = pix)。无需 VLM, 完全确定性。

为什么 surface_visual_noise 用图形学而不是 VLM
----------------------------------------------
1. 可复现 (reproducible): 同一组像素永远得到同一个数。VLM 有采样方差,
   且跨模型版本会漂移; 图形统计可单元测试、可回归。
2. 可解释 (interpretable): 触发时可回看究竟是边缘密度、高频能量还是局部
   方差导致的, 便于校准阈值与排障。
3. 与光照正交 (orthogonal to lighting): "会干扰画作的视觉噪声"位于高空间
   频带, 而光照梯度 (rho / lambda) 位于低频带。高通分解把两者彻底分离,
   noise 不会把窗边的平滑亮度渐变误判为'墙面很乱'。

核心数学
--------
在候选墙 mask 区域 W 的亮度通道 L 上:
  1. 低频分量 (光照):  L_lf = GaussianBlur(L, sigma)
  2. 高频分量 (纹理):  L_hf = L - L_lf
  3. 三个归一化统计:
       e   = 边缘密度      = (Canny 边缘像素数 in W) / |W|
       hf  = 高频能量      = mean(L_hf^2) / 255^2
       sd  = 局部标准差    = mean_W( local_std(L, k) ) / 128
  4. 合成噪声标量:
       N = clip( w_e*e_n + w_hf*hf_n + w_sd*sd_n , 0, 1 )
     其中 *_n 为按经验上限的截断归一化。
  5. 分箱:  N < t_lo -> low ; t_lo<=N<t_hi -> medium ; N>=t_hi -> high

低频光照统计 rho / lambda / lambda_ceil 同样在此模块计算 (都是区域亮度均值比)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except Exception:  # 降级: 无 cv2 时用 numpy/scipy 近似
    _HAS_CV2 = False
    from scipy import ndimage as _ndi


# ----- 噪声合成权重与归一化上限 (上线后用标注数据校准) -----
W_EDGE, W_HF, W_SD = 0.45, 0.35, 0.20
CAP_EDGE, CAP_HF, CAP_SD = 0.18, 0.020, 0.45   # 经验饱和上限
T_LO, T_HI = 0.28, 0.55                        # low/medium/high 分箱阈值
GAUSS_SIGMA = 9.0                              # 低频/高频分界尺度 (像素)
LOCAL_STD_K = 7                                # 局部标准差窗口


def _to_luma(img_rgb: np.ndarray) -> np.ndarray:
    """RGB(uint8) -> 亮度通道 float32 (Rec.601)。"""
    img = img_rgb.astype(np.float32)
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def _gaussian(L: np.ndarray, sigma: float) -> np.ndarray:
    if _HAS_CV2:
        k = int(sigma * 6) | 1
        return cv2.GaussianBlur(L, (k, k), sigma)
    return _ndi.gaussian_filter(L, sigma=sigma)


def _local_std(L: np.ndarray, k: int) -> np.ndarray:
    """逐像素局部标准差: sqrt(E[L^2] - E[L]^2), 用均值滤波实现。"""
    if _HAS_CV2:
        mean = cv2.blur(L, (k, k))
        sq = cv2.blur(L * L, (k, k))
    else:
        mean = _ndi.uniform_filter(L, size=k)
        sq = _ndi.uniform_filter(L * L, size=k)
    var = np.clip(sq - mean * mean, 0, None)
    return np.sqrt(var)


def _edges(L: np.ndarray) -> np.ndarray:
    u8 = np.clip(L, 0, 255).astype(np.uint8)
    if _HAS_CV2:
        return cv2.Canny(u8, 60, 160) > 0
    # 降级: Sobel 梯度阈值
    gx = _ndi.sobel(L, axis=1)
    gy = _ndi.sobel(L, axis=0)
    g = np.hypot(gx, gy)
    return g > (g.mean() + g.std())


def _mask_mean(L: np.ndarray, mask: Optional[np.ndarray]) -> float:
    if mask is None:
        return float(L.mean())
    m = mask.astype(bool)
    return float(L[m].mean()) if m.any() else float(L.mean())


@dataclass
class NoiseResult:
    score: float          # 0–1
    band: str             # low / medium / high
    edge_density: float
    hf_energy: float
    local_std: float


def surface_visual_noise(img_rgb: np.ndarray, wall_mask: np.ndarray) -> NoiseResult:
    """
    候选墙面视觉噪声 (寂静墙原则的可复现操作化)。

    img_rgb   : HxWx3 uint8 原始空间图
    wall_mask : HxW   bool   候选墙面 mask (polygon 栅格化得到)
    返回 NoiseResult, 其中 .score 写入 light.surface_visual_noise_score, .band 用于触发器。
    """
    L = _to_luma(img_rgb)
    m = wall_mask.astype(bool)
    if not m.any():
        return NoiseResult(0.0, "low", 0.0, 0.0, 0.0)

    L_lf = _gaussian(L, GAUSS_SIGMA)
    L_hf = L - L_lf                              # 去掉光照低频, 只剩纹理

    edge = _edges(L)
    e = float(edge[m].mean())                    # 边缘密度
    hf = float((L_hf[m] ** 2).mean()) / (255.0 ** 2)
    sd = float(_local_std(L, LOCAL_STD_K)[m].mean()) / 128.0

    e_n = min(e / CAP_EDGE, 1.0)
    hf_n = min(hf / CAP_HF, 1.0)
    sd_n = min(sd / CAP_SD, 1.0)

    score = float(np.clip(W_EDGE * e_n + W_HF * hf_n + W_SD * sd_n, 0.0, 1.0))
    band = "low" if score < T_LO else ("medium" if score < T_HI else "high")
    return NoiseResult(round(score, 4), band, round(e, 4), round(hf, 6), round(sd, 4))


@dataclass
class LightStats:
    rho: Optional[float]
    wall_lambda: Optional[float]
    lambda_ceil: Optional[float]


def light_statistics(img_rgb: np.ndarray,
                     room_mask: np.ndarray,
                     window_mask: Optional[np.ndarray],
                     wall_mask: Optional[np.ndarray],
                     ceiling_mask: Optional[np.ndarray]) -> LightStats:
    """
    低频光照统计 (区域亮度均值比)。全部来自像素, 无需 VLM。
    room_mask 应排除窗口区域以稳定室内基准。
    """
    L = _to_luma(img_rgb)
    L_room = _mask_mean(L, room_mask)
    L_room = max(L_room, 1e-3)
    rho = _mask_mean(L, window_mask) / L_room if window_mask is not None and window_mask.any() else None
    wl = _mask_mean(L, wall_mask) / L_room if wall_mask is not None and wall_mask.any() else None
    lc = _mask_mean(L, ceiling_mask) / L_room if ceiling_mask is not None and ceiling_mask.any() else None
    return LightStats(
        rho=round(rho, 3) if rho is not None else None,
        wall_lambda=round(wl, 3) if wl is not None else None,
        lambda_ceil=round(lc, 3) if lc is not None else None,
    )
