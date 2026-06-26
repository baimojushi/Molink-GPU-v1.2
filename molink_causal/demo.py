"""
demo.py
=======
端到端演示: 模拟一条 worker 进度事件流, 打印每轮装配出的 LLM 输入 bundle。
同时对合成墙面验证 imaging.surface_visual_noise 的可复现性与分箱。
运行: python demo.py
"""

import json
import numpy as np

from molink_causal import CausalEngine, Event
from molink_causal.imaging import surface_visual_noise, light_statistics


def banner(t):
    print("\n" + "=" * 64 + f"\n{t}\n" + "=" * 64)


def show(b):
    print(f"[{b.state}] pct={b.pct:.0f}  task={b.task}  fired={b.fired}")
    for h in b.copy_hints:
        print("   hint:", h[:70] + ("…" if len(h) > 70 else ""))
    if b.light_components:
        print("   light:", b.light_components)


def run_stream():
    banner("事件流模拟 (一件 20x28cm 带框作品《圆》)")
    eng = CausalEngine()

    stream = [
        Event("init", "running", 5, {"artwork": {"name": "圆", "width_cm": 21.2,
              "height_cm": 28, "has_frame": True}}),
        Event("metric3d", "running", 15, {}),
        Event("semantic", "running", 35, {"space": {"wall_count": 4, "window_count": 2,
              "door_count": 1, "furniture_count": 3, "existing_art_count": 6, "person_count": 2}}),
        Event("hanging", "running", 55, {"best_plan": {
              "wall_position_zh": "主墙面", "is_primary_wall": True,
              "center_height_cm": 152, "bottom_edge_cm": 138, "top_edge_cm": 166,
              "height_finalized": False, "risk_level": "low",
              "above_furniture_label_zh": "沙发", "bottom_gap_to_furniture_cm": 16,
              "axis_centering": True, "artwork_wall_area_ratio": 0.10,
              "window_edge_clearance_cm": 82, "surface_active_light": False,
              "landscape_competition_score": 0.2},
              "light": {"rho": 1.2, "wall_lambda": 0.95, "lambda_ceil": 0.92,
              "theta": "opposite", "exterior_brightness_class": "overcast",
              "ambient_light_mode": "gallery_ambient", "shadow_presence": "moderate",
              "temporal_light_quality": "static", "ceiling_height": "high",
              "surface_visual_noise_score": 0.18}}),
        Event("render", "running", 70, {}),
        Event("render", "running", 90, {"render": {"flux2_mask_sent": True}}),
        # 复核分支: 有 partial 无 final
        Event("r2_upload", "done", 95, {"delivery": {"partial_review_count": 1,
              "final_url_count": 0}, "qa": {"hard_block_reasons": ["hard_core_masked_ssim_too_low"]}}),
    ]
    for ev in stream:
        show(eng.on_event(ev))

    banner("最后一轮完整 bundle (复核态)")
    last = eng._compose(eng._fire(
        __import__("molink_causal.state_machine", fromlist=["unlocked_layers"]).unlocked_layers(eng.state)))
    print(json.dumps({"task": last.task, "char_limit": last.char_limit, "state": last.state,
                      "facts": last.facts, "fired": last.fired, "constraints": last.constraints},
                     ensure_ascii=False, indent=2))

    banner("交付成功分支 (final_url_count=1, 解锁 L4)")
    eng2 = CausalEngine()
    for ev in stream[:6]:
        eng2.on_event(ev)
    b = eng2.on_event(Event("r2_upload", "done", 100,
        {"delivery": {"final_url_count": 1}, "artwork": {"mood_register": "contemplative"}}))
    show(b)
    print("   facts.invitation:", b.facts.get("invitation"))


def run_imaging():
    banner("imaging: surface_visual_noise 可复现性 (合成墙面)")
    rng = np.random.default_rng(0)
    h = w = 256
    mask = np.ones((h, w), bool)

    # 1) 平整白墙 + 低频光照梯度 (应判 low, 不被梯度污染)
    grad = np.linspace(120, 210, w)[None, :].repeat(h, 0)
    plain = np.stack([grad] * 3, -1).astype(np.uint8)
    r1 = surface_visual_noise(plain, mask)

    # 2) 砖墙纹理 (高频, 应判 high)
    brick = grad.copy()
    brick[::16, :] = 60
    brick[:, ::32] = 60
    brick += rng.normal(0, 18, (h, w))
    brick = np.stack([np.clip(brick, 0, 255)] * 3, -1).astype(np.uint8)
    r2 = surface_visual_noise(brick, mask)

    # 3) 中等: 轻微壁纸纹理
    mid = grad + 12 * np.sin(np.linspace(0, 40, w))[None, :]
    mid = np.stack([np.clip(mid, 0, 255)] * 3, -1).astype(np.uint8)
    r3 = surface_visual_noise(mid, mask)

    for name, r in [("平整白墙(含光照梯度)", r1), ("砖墙纹理", r2), ("轻纹壁纸", r3)]:
        print(f"  {name:18s} score={r.score:.3f} band={r.band:6s} "
              f"(edge={r.edge_density:.3f} hf={r.hf_energy:.5f} std={r.local_std:.3f})")

    # 复现性: 同输入两次完全相等
    r1b = surface_visual_noise(plain, mask)
    print("  可复现性 (两次同输入相等):", r1.score == r1b.score)

    banner("imaging: 低频光照统计 rho / lambda / lambda_ceil")
    room = np.full((h, w), 150, np.uint8)
    img = np.stack([room] * 3, -1)
    window = np.zeros((h, w), bool); window[:, :40] = True
    img[:, :40] = 240  # 亮窗
    ceil = np.zeros((h, w), bool); ceil[:40, :] = True
    img[:40, :] = 175  # 略亮天花
    wall = np.zeros((h, w), bool); wall[80:, 80:] = True
    ls = light_statistics(img, room_mask=~window, window_mask=window,
                          wall_mask=wall, ceiling_mask=ceil)
    print(f"  rho={ls.rho}  wall_lambda={ls.wall_lambda}  lambda_ceil={ls.lambda_ceil}")


if __name__ == "__main__":
    run_stream()
    run_imaging()
