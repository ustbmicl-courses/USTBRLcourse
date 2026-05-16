"""Q-Value Iteration (Q-VI) —— L4 教学示例（V-VI 的动作价值版本）

直接对应 L4-DP.tex §"动作价值版本：Q^π 评估与 Q* 价值迭代"。
复用 L1 GridWorld。

算法（同步备份，对应 PPT line 891 的 Q-VI 伪代码）：
    Q_{k+1}(s, a) = r(s, a) + γ * max_{a'} Q_k(s', a')
直到 ||Q_{k+1} - Q_k||_∞ < tol。

与 V-VI 的关键对比（教学诊断）：
  - 存储量：Q 是 |S||A|，V 是 |S|；Q 贵 |A| 倍
  - 单步代价：两者都 O(|S|^2 |A|)（同源 γ-压缩证明）
  - 取最优策略：V-VI 需重做一次 P/R 期望；Q-VI 直接 argmax_a Q —— 无须模型
  - 这是 L7 Q-learning / L8 DQN 都选 Q 而非 V 的根本理由
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "L1-mdp-foundations"))
from grid_world import ACTIONS, GridConfig, GridWorld  # noqa: E402

from value_iteration import value_iteration  # noqa: E402


def q_value_iteration(
    env: GridWorld,
    tol: float = 1e-8,
    max_iter: int = 1000,
) -> tuple[np.ndarray, dict, list[float]]:
    """同步 Q-value iteration。

    返回：
        Q         —— 形状 (n_rows, n_cols, |A|) 的最优动作价值
        policy    —— {state: action} 贪婪策略（直接 argmax_a Q）
        residuals —— 每轮 ||Q_{k+1} - Q_k||_∞
    """
    cfg = env.cfg
    nA = len(ACTIONS)
    Q = np.zeros((cfg.n_rows, cfg.n_cols, nA))
    residuals = []

    for k in range(max_iter):
        Q_new = np.zeros_like(Q)
        for s in env.all_states():
            for ai, a in enumerate(ACTIONS):
                s_next, r, _ = env.step(s, a)
                Q_new[s[0], s[1], ai] = r + cfg.gamma * Q[s_next].max()
        delta = float(np.max(np.abs(Q_new - Q)))
        residuals.append(delta)
        Q = Q_new
        if delta < tol:
            break

    # 直接 argmax_a Q —— 无须再过一次 P/R
    policy = {}
    for s in env.all_states():
        best_idx = int(np.argmax(Q[s[0], s[1]]))
        policy[s] = ACTIONS[best_idx]

    return Q, policy, residuals


def print_q_table(Q: np.ndarray, env: GridWorld) -> None:
    """逐状态打印 Q(s, ·) —— 5 个动作的值。"""
    arrow = {"up": "↑", "down": "↓", "left": "←", "right": "→", "stay": "·"}
    print(f"  {'s':>6s} | " + "  ".join(f"{arrow[a]:>6s}" for a in ACTIONS) + " | argmax")
    print("  " + "-" * 60)
    for r in range(env.cfg.n_rows):
        for c in range(env.cfg.n_cols):
            row = Q[r, c]
            best = ACTIONS[int(np.argmax(row))]
            tag = f"({r},{c})"
            print(f"  {tag:>6s} | " + "  ".join(f"{v:+6.3f}" for v in row) + f" | {arrow[best]}")


def main() -> None:
    env = GridWorld(GridConfig(gamma=0.9))
    print("===== 3x3 网格 · Q-value iteration (γ=0.9) =====")
    Q, policy_q, residuals_q = q_value_iteration(env)
    print(f"\n收敛于第 {len(residuals_q)} 轮  (tol=1e-8)")
    print(f"最终残差 ||Q_K - Q_{{K-1}}||_∞ = {residuals_q[-1]:.2e}\n")

    print("最优动作价值 Q*(s, ·)：")
    print_q_table(Q, env)

    # 一致性检查：V* = max_a Q*；π* 与 V-VI 一致
    print("\n===== 与 V-VI 对比（一致性核对）=====")
    V_vi, policy_v, residuals_v = value_iteration(env)

    V_from_q = Q.max(axis=2)
    diff = float(np.max(np.abs(V_from_q - V_vi)))
    print(f"||max_a Q* - V*||_∞ = {diff:.2e}   （应 ≈ 0）")

    policy_match = all(policy_q[s] == policy_v[s] for s in env.all_states())
    print(f"贪婪策略一致： {policy_match}")

    print(f"\nV-VI 收敛步数 = {len(residuals_v)}")
    print(f"Q-VI 收敛步数 = {len(residuals_q)}   （理论应同源，每轮 γ-压缩）")

    print("\n前 10 轮残差对比：")
    print(f"  {'k':>3s} | {'V-VI Δ':>12s} | {'Q-VI Δ':>12s} | {'Q/V 比':>8s}")
    for k in range(min(10, len(residuals_v), len(residuals_q))):
        ratio = residuals_q[k] / residuals_v[k] if residuals_v[k] > 1e-15 else 1.0
        print(f"  {k:>3d} | {residuals_v[k]:>12.6f} | {residuals_q[k]:>12.6f} | {ratio:>8.4f}")

    print("\n===== 教学诊断 =====")
    print(f"|S|  = {env.cfg.n_rows * env.cfg.n_cols}")
    print(f"|A|  = {len(ACTIONS)}")
    print(f"V 存储 = {V_vi.size}    （|S|）")
    print(f"Q 存储 = {Q.size}    （|S| × |A|，贵 {len(ACTIONS)} 倍）")
    print()
    print("关键论点：")
    print("  · Q-VI 与 V-VI 收敛速率同源（都来自 γ-压缩 + L4 共用引理）")
    print("  · Q-VI 存储贵 |A| 倍 —— 表格 RL 中这点代价换来了 model-free 选动作")
    print("  · L7 Q-learning / L8 DQN 都选 Q 形式，正是因为 argmax_a Q 不需要 P/R")


if __name__ == "__main__":
    main()
