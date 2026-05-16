"""L3 演示：Bellman 最优方程、Q 表与压缩映射。

核心目标：
1. 从 v_k 计算每个动作的 q_k(s,a)。
2. 用 max_a q_k(s,a) 实现 Bellman 最优备份。
3. 打印最优价值、最优策略和几个代表状态的 Q 表。
4. 观察残差下降：Bellman 最优算子在 gamma < 1 时是压缩映射。

运行：
    python3 code/L3-bellman-optimality/bellman_optimality.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

REPO_CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_CODE)

from shared.grid_world import GridConfig, GridWorld  # noqa: E402


ARROW = {"up": "^", "down": "v", "left": "<", "right": ">", "stay": "."}


def print_grid_values(env: GridWorld, V: np.ndarray, title: str) -> None:
    print(title)
    values = V.reshape(env.cfg.n_rows, env.cfg.n_cols)
    for r in range(env.cfg.n_rows):
        print("  " + " ".join(f"{values[r, c]:+6.2f}" for c in range(env.cfg.n_cols)))


def print_policy(env: GridWorld, policy: dict[tuple[int, int], str]) -> None:
    print("最优策略 pi*(s)：")
    for r in range(env.cfg.n_rows):
        row = []
        for c in range(env.cfg.n_cols):
            s = (r, c)
            if s == env.cfg.target:
                row.append("T")
            elif s in env.cfg.forbidden:
                row.append("X")
            else:
                row.append(ARROW[policy[s]])
        print("  " + " ".join(row))


def print_q_for_state(env: GridWorld, Q: np.ndarray, s: tuple[int, int]) -> None:
    idx = env.state_index(s)
    print(f"s{env.display_index(s)} 的 Q*(s,a)：")
    for action, value in zip(env.actions, Q[idx]):
        print(f"  {action:>5}: {value:+.4f}")


def run_value_iteration(env: GridWorld, tol: float = 1e-10) -> tuple[np.ndarray, dict, list[float]]:
    """调用共享 GridWorld 的最优 Bellman 备份。"""
    return env.value_iteration(tol=tol)


def main() -> None:
    env = GridWorld(GridConfig(gamma=0.9))

    print("===== L3: Bellman 最优方程 =====")
    print(f"|S|={env.n_states}, |A|={env.n_actions}, gamma={env.cfg.gamma}")
    print()
    print(env.render())
    print()

    V_star, policy_star, residuals = run_value_iteration(env)
    Q_star = env.compute_q_table(V_star)

    print_grid_values(env, V_star, "最优价值 V*：")
    print()
    print_policy(env, policy_star)
    print()

    for state in [(0, 0), (0, 2), (2, 1), (3, 2)]:
        print_q_for_state(env, Q_star, state)
        print()

    print("Bellman 最优备份残差 ||T V_k - V_k||_inf：")
    for k, delta in enumerate(residuals[:12]):
        ratio = delta / residuals[k - 1] if k > 0 and residuals[k - 1] > 0 else 0.0
        print(f"  k={k:2d}: residual={delta:.6f}, ratio={ratio:.4f}")
    print()
    print(
        "解释：gamma=0.9 时，Bellman 最优算子是 gamma-压缩映射，"
        "两次价值函数之间的误差会被最多乘以 0.9。"
    )
    print(f"实际收敛轮数: {len(residuals)}, 最终残差: {residuals[-1]:.2e}")


if __name__ == "__main__":
    main()
