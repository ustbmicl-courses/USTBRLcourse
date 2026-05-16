"""策略迭代 (Policy Iteration) —— L4 教学示例

对应 L4-DP.tex 中的算法 4.2 和 GPI 框架。
两步交替：
  (1) 策略评估：求解 v^π = r^π + γ P^π v^π   —— 这里用迭代求解（截断 PI）
  (2) 策略提升：π_{k+1}(s) = argmax_a [ r(s,a) + γ v^π(s') ]
直到策略稳定。

对比演示：完整 PI（评估到收敛） vs 截断 PI（每轮评估固定步数）—— 后者就是 VI 的极限。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "L1-mdp-foundations"))
from grid_world import ACTIONS, GridConfig, GridWorld  # noqa: E402


def policy_evaluation(
    env: GridWorld,
    policy: dict,
    V_init: np.ndarray | None = None,
    eval_iters: int | None = None,
    tol: float = 1e-8,
) -> tuple[np.ndarray, int]:
    """策略评估。eval_iters=None 表示评估到收敛；否则做截断 PI。"""
    cfg = env.cfg
    V = np.zeros((cfg.n_rows, cfg.n_cols)) if V_init is None else V_init.copy()
    k = 0
    while True:
        V_new = np.zeros_like(V)
        for s in env.all_states():
            a = policy[s]
            s_next, r, _ = env.step(s, a)
            V_new[s] = r + cfg.gamma * V[s_next]
        delta = float(np.max(np.abs(V_new - V)))
        V = V_new
        k += 1
        if eval_iters is not None and k >= eval_iters:
            break
        if eval_iters is None and delta < tol:
            break
    return V, k


def policy_improvement(env: GridWorld, V: np.ndarray) -> dict:
    policy = {}
    for s in env.all_states():
        best_a, best_q = ACTIONS[0], -np.inf
        for a in ACTIONS:
            s_next, r, _ = env.step(s, a)
            q = r + env.cfg.gamma * V[s_next]
            if q > best_q:
                best_q, best_a = q, a
        policy[s] = best_a
    return policy


def policy_iteration(
    env: GridWorld,
    eval_iters: int | None = None,
    max_outer: int = 100,
) -> tuple[np.ndarray, dict, list[int], list[float]]:
    """策略迭代主循环。

    返回：
        V, policy, eval_steps_per_outer, value_norms
    """
    # 初始策略：全部 stay
    policy = {s: "stay" for s in env.all_states()}
    V = np.zeros((env.cfg.n_rows, env.cfg.n_cols))
    eval_steps, value_norms = [], []

    for outer in range(max_outer):
        V, k_eval = policy_evaluation(env, policy, V_init=V, eval_iters=eval_iters)
        eval_steps.append(k_eval)
        value_norms.append(float(np.linalg.norm(V)))

        new_policy = policy_improvement(env, V)
        if new_policy == policy:
            # 策略稳定 → 最后做一次完整评估，给出真正的 v^π
            V, _ = policy_evaluation(env, policy, V_init=V, eval_iters=None)
            return V, policy, eval_steps, value_norms
        policy = new_policy

    return V, policy, eval_steps, value_norms


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

    print("===== 完整策略迭代（评估到收敛）=====")
    V, policy, ev, _ = policy_iteration(env, eval_iters=None)
    print(f"外层迭代 {len(ev)} 轮收敛")
    print(f"每轮内层评估步数: {ev}")
    print("V*：")
    print_value(V)
    print("π*：")
    print_policy(env, policy)

    print("\n===== 截断策略迭代 (每轮评估 1 步) — 等价于 VI =====")
    V_t, pi_t, ev_t, _ = policy_iteration(env, eval_iters=1)
    print(f"外层迭代 {len(ev_t)} 轮收敛  (期望: 与 VI 相同量级)")
    print("V*：")
    print_value(V_t)

    print("\n===== 截断策略迭代 (每轮评估 5 步) =====")
    V_t5, pi_t5, ev_t5, _ = policy_iteration(env, eval_iters=5)
    print(f"外层迭代 {len(ev_t5)} 轮收敛")

    # 三者最优价值应一致
    print("\n===== 三种 PI 的 V* 一致性检验 =====")
    print(f"  ||V_full - V_trunc1||_∞ = {np.max(np.abs(V - V_t)):.6f}")
    print(f"  ||V_full - V_trunc5||_∞ = {np.max(np.abs(V - V_t5)):.6f}")


if __name__ == "__main__":
    main()
