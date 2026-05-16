"""L2 演示：固定策略下的 Bellman 期望方程。

核心目标：
1. 在同一个 GridWorld 上定义一个确定性策略 pi(a|s)。
2. 构造 P^pi 和 r^pi。
3. 直接解 v^pi = (I - gamma P^pi)^(-1) r^pi。
4. 用迭代备份验证同一个 Bellman 方程。

运行：
    python3 code/L2-bellman-equation/policy_evaluation.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

REPO_CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_CODE)

from shared.grid_world import GridConfig, GridWorld  # noqa: E402


ARROW = {"up": "^", "down": "v", "left": "<", "right": ">", "stay": "."}


def teaching_policy(env: GridWorld) -> dict[tuple[int, int], str]:
    """一个手工策略：优先向目标列移动，再向目标行移动。

    这不是最优策略，只是为了让 P^pi、r^pi、v^pi 的含义更容易看清。
    """
    target_r, target_c = env.cfg.target
    policy: dict[tuple[int, int], str] = {}
    for s in env.all_states():
        r, c = s
        if s == env.cfg.target:
            policy[s] = "stay"
        elif c < target_c:
            policy[s] = "right"
        elif c > target_c:
            policy[s] = "left"
        elif r < target_r:
            policy[s] = "down"
        elif r > target_r:
            policy[s] = "up"
        else:
            policy[s] = "stay"
    return policy


def iterative_policy_evaluation(
    env: GridWorld,
    policy: dict[tuple[int, int], str],
    theta: float = 1e-10,
    max_iter: int = 1000,
) -> tuple[np.ndarray, list[float]]:
    """同步策略评估：v_{k+1}=r^pi+gamma P^pi v_k。"""
    V = np.zeros(env.n_states)
    P = env.transition_matrix(policy)
    r = env.reward_vector(policy)
    residuals: list[float] = []

    for _ in range(max_iter):
        V_next = r + env.cfg.gamma * P @ V
        delta = float(np.max(np.abs(V_next - V)))
        residuals.append(delta)
        V = V_next
        if delta < theta:
            break
    return V, residuals


def print_grid_values(env: GridWorld, V: np.ndarray, title: str) -> None:
    print(title)
    values = V.reshape(env.cfg.n_rows, env.cfg.n_cols)
    for r in range(env.cfg.n_rows):
        print("  " + " ".join(f"{values[r, c]:+6.2f}" for c in range(env.cfg.n_cols)))


def print_policy(env: GridWorld, policy: dict[tuple[int, int], str]) -> None:
    print("策略 pi(s)：")
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


def explain_one_state(env: GridWorld, policy: dict[tuple[int, int], str], s: tuple[int, int]) -> None:
    a = policy[s]
    s_next, reward, done = env.step(s, a)
    print(
        f"s{env.display_index(s)} 执行动作 {a}: "
        f"s' = s{env.display_index(s_next)}, r = {reward:+.1f}, done = {done}"
    )


def main() -> None:
    env = GridWorld(GridConfig(gamma=0.9))
    policy = teaching_policy(env)

    print("===== L2: Bellman 期望方程 / 策略评估 =====")
    print(f"|S|={env.n_states}, |A|={env.n_actions}, gamma={env.cfg.gamma}")
    print()
    print(env.render())
    print()
    print_policy(env, policy)
    print()

    print("固定策略后，MDP 变成 Markov reward process:")
    P_pi = env.transition_matrix(policy)
    r_pi = env.reward_vector(policy)
    print(f"  P^pi shape = {P_pi.shape}")
    print(f"  r^pi shape = {r_pi.shape}")
    print()

    for state in [(0, 0), (1, 0), (2, 1), (3, 2)]:
        explain_one_state(env, policy, state)
    print()

    V_direct = env.solve_bellman(policy)
    V_iter, residuals = iterative_policy_evaluation(env, policy)

    print_grid_values(env, V_direct, "直接解线性方程得到 v^pi：")
    print()
    print_grid_values(env, V_iter, "迭代 Bellman 备份得到 v^pi：")
    print()
    print(f"两种方法最大差异: {np.max(np.abs(V_direct - V_iter)):.2e}")
    print(f"策略评估迭代轮数: {len(residuals)}")
    print("前 8 轮 Bellman 残差 ||v_{k+1}-v_k||_inf：")
    for k, delta in enumerate(residuals[:8]):
        print(f"  k={k:2d}: {delta:.6f}")


if __name__ == "__main__":
    main()
