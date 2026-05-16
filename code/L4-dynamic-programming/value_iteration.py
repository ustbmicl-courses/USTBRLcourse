"""价值迭代 (Value Iteration) —— L4 教学示例

直接对应 L4-DP.tex 中的 Bellman 最优算子 T 与算法 4.1。
复用 L1 已建的 GridWorld（3x3 网格）。

算法（同步备份）：
    v_{k+1}(s) = max_a [ r(s,a) + γ * v_k(s') ]
直到 ||v_{k+1} - v_k||_∞ < tol。

收敛性来源：T 是 γ-收缩映射（教材命题 4.1）。
"""
from __future__ import annotations

import os
import sys

import numpy as np

# 复用 L1 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "L1-mdp-foundations"))
from grid_world import ACTIONS, GridConfig, GridWorld  # noqa: E402


def value_iteration(
    env: GridWorld,
    tol: float = 1e-8,
    max_iter: int = 1000,
) -> tuple[np.ndarray, dict, list[float]]:
    """同步价值迭代。

    返回：
        V          —— 形状 (n_rows, n_cols) 的最优价值
        policy     —— {state: action} 贪婪策略
        residuals  —— 每轮 ||v_{k+1} - v_k||_∞，用于画收敛曲线
    """
    cfg = env.cfg
    V = np.zeros((cfg.n_rows, cfg.n_cols))
    residuals = []

    for k in range(max_iter):
        V_new = np.zeros_like(V)
        for s in env.all_states():
            best = -np.inf
            for a in ACTIONS:
                s_next, r, _ = env.step(s, a)
                q = r + cfg.gamma * V[s_next]
                if q > best:
                    best = q
            V_new[s] = best
        delta = float(np.max(np.abs(V_new - V)))
        residuals.append(delta)
        V = V_new
        if delta < tol:
            break

    # 抽取贪婪策略
    policy = {}
    for s in env.all_states():
        best_a, best_q = ACTIONS[0], -np.inf
        for a in ACTIONS:
            s_next, r, _ = env.step(s, a)
            q = r + cfg.gamma * V[s_next]
            if q > best_q:
                best_q, best_a = q, a
        policy[s] = best_a

    return V, policy, residuals


def print_value(V: np.ndarray) -> None:
    for r in range(V.shape[0]):
        print("  " + "  ".join(f"{V[r, c]:+6.3f}" for c in range(V.shape[1])))


def print_policy(env: GridWorld, policy: dict) -> None:
    arrow = {"up": "↑", "down": "↓", "left": "←", "right": "→", "stay": "·"}
    for r in range(env.cfg.n_rows):
        line = []
        for c in range(env.cfg.n_cols):
            if (r, c) == env.cfg.target:
                line.append("T")
            elif (r, c) in env.cfg.forbidden:
                line.append("X")
            else:
                line.append(arrow[policy[(r, c)]])
        print("  " + " ".join(line))


def main() -> None:
    env = GridWorld(GridConfig(gamma=0.9))
    print("===== 3x3 网格 · 价值迭代 (γ=0.9) =====")
    V, policy, residuals = value_iteration(env)
    print(f"\n收敛于第 {len(residuals)} 轮  (tol=1e-8)")
    print(f"最终残差 ||v_{{K}} - v_{{K-1}}||_∞ = {residuals[-1]:.2e}\n")

    print("最优价值 V*：")
    print_value(V)

    print("\n最优策略 π*：")
    print_policy(env, policy)

    print("\n前 10 轮残差（验证 γ-收缩，每轮约衰减 0.9 倍）：")
    for k, d in enumerate(residuals[:10]):
        ratio = residuals[k] / residuals[k - 1] if k > 0 and residuals[k - 1] > 0 else 0.0
        print(f"  k={k:2d}  Δ={d:.6f}  Δ_k/Δ_{{k-1}}={ratio:.4f}")

    # γ 改变最优策略示例（教材 §3.x）
    print("\n===== γ=0.5 时再跑一次（观察策略是否改变）=====")
    env2 = GridWorld(GridConfig(gamma=0.5))
    V2, policy2, _ = value_iteration(env2)
    print("V* (γ=0.5)：")
    print_value(V2)
    print("π* (γ=0.5)：")
    print_policy(env2, policy2)


if __name__ == "__main__":
    main()
