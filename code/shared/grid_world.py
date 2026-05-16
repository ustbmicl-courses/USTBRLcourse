"""GridWorld（5×5 标准版）—— 跨讲共享 MDP

对应 Shiyu Zhao 课本图与 USTB PPT 标准网格 (L1-L7)：
    s1  s2  s3  s4  s5
    s6  s7  s8  s9  s10
    s11 s12 s13 s14 s15
    s16 s17 s18 s19 s20
    s21 s22 s23 s24 s25

默认 forbidden = {s7, s8, s13, s17, s19}, target = s18
（与 PPT 中常用图示一致；可通过 GridConfig 改）

L1 用：step / rollout / discounted_return
L2 用：transition_matrix(policy) / reward_vector(policy) → 解 Bellman 方程
L3 用：bellman_optimality_operator / compute_q_table
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ACTIONS = ["up", "down", "left", "right", "stay"]
ACTION_DELTAS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
    "stay": (0, 0),
}


@dataclass
class GridConfig:
    n_rows: int = 5
    n_cols: int = 5
    target: tuple = (3, 2)  # s18 (row=3, col=2) 0-indexed
    forbidden: tuple = ((1, 1), (1, 2), (2, 2), (3, 1), (3, 3))  # s7,s8,s13,s17,s19
    r_target: float = 1.0
    r_forbidden: float = -1.0
    r_step: float = 0.0
    r_boundary: float = -1.0
    gamma: float = 0.9


class GridWorld:
    """确定性 5×5 网格 MDP，显式给出 P(s'|s,a) 与 r(s,a)。"""

    def __init__(self, cfg: GridConfig | None = None) -> None:
        self.cfg = cfg or GridConfig()
        self.actions = ACTIONS
        self.n_states = self.cfg.n_rows * self.cfg.n_cols
        self.n_actions = len(self.actions)
        self._states = [
            (r, c) for r in range(self.cfg.n_rows) for c in range(self.cfg.n_cols)
        ]
        self._state_to_idx = {s: i for i, s in enumerate(self._states)}

    # --- 状态编码 ---
    def state_index(self, s: tuple) -> int:
        """0-indexed 用于矩阵；显示 1-indexed 加 1。"""
        return self._state_to_idx[s]

    def display_index(self, s: tuple) -> int:
        return self.state_index(s) + 1

    def all_states(self) -> list:
        return list(self._states)

    # --- 动力学（确定性 MDP）---
    def step(self, s: tuple, a: str) -> tuple:
        """返回 (s', r, done)。target 是吸收态。"""
        if s == self.cfg.target:
            return s, 0.0, True
        dr, dc = ACTION_DELTAS[a]
        nr, nc = s[0] + dr, s[1] + dc
        if not (0 <= nr < self.cfg.n_rows and 0 <= nc < self.cfg.n_cols):
            return s, self.cfg.r_boundary, False
        s_next = (nr, nc)
        if s_next == self.cfg.target:
            return s_next, self.cfg.r_target, True
        if s_next in self.cfg.forbidden:
            return s_next, self.cfg.r_forbidden, False
        return s_next, self.cfg.r_step, False

    def render(self, s: tuple | None = None) -> str:
        rows = []
        for r in range(self.cfg.n_rows):
            line = []
            for c in range(self.cfg.n_cols):
                if s is not None and (r, c) == s:
                    line.append("A")
                elif (r, c) == self.cfg.target:
                    line.append("T")
                elif (r, c) in self.cfg.forbidden:
                    line.append("X")
                else:
                    line.append(".")
            rows.append(" ".join(line))
        return "\n".join(rows)

    # --- 矩阵接口（L2/L3 用）---
    def transition_matrix(self, policy: dict) -> np.ndarray:
        """返回形状 (|S|, |S|) 的策略转移矩阵 P^π。"""
        n = self.n_states
        P = np.zeros((n, n))
        for s in self._states:
            i = self.state_index(s)
            a = policy[s]
            s_next, _, _ = self.step(s, a)
            j = self.state_index(s_next)
            P[i, j] = 1.0
        return P

    def reward_vector(self, policy: dict) -> np.ndarray:
        """返回形状 (|S|,) 的策略奖励向量 r^π。"""
        r = np.zeros(self.n_states)
        for s in self._states:
            i = self.state_index(s)
            a = policy[s]
            _, rr, _ = self.step(s, a)
            r[i] = rr
        return r

    def solve_bellman(self, policy: dict) -> np.ndarray:
        """直接解 v^π = (I - γ P^π)^{-1} r^π。"""
        n = self.n_states
        P = self.transition_matrix(policy)
        r = self.reward_vector(policy)
        return np.linalg.solve(np.eye(n) - self.cfg.gamma * P, r)

    def compute_q_table(self, V: np.ndarray) -> np.ndarray:
        """给定 V（按 state_index 排），返回 Q (|S|, |A|)。"""
        Q = np.zeros((self.n_states, self.n_actions))
        for s in self._states:
            i = self.state_index(s)
            for ai, a in enumerate(self.actions):
                s_next, rr, _ = self.step(s, a)
                Q[i, ai] = rr + self.cfg.gamma * V[self.state_index(s_next)]
        return Q

    def value_iteration(
        self, tol: float = 1e-9, max_iter: int = 1000
    ) -> tuple:
        """同步 VI，返回 (V_star, policy_star, residuals)。"""
        V = np.zeros(self.n_states)
        residuals = []
        for _ in range(max_iter):
            Q = self.compute_q_table(V)
            V_new = Q.max(axis=1)
            d = float(np.max(np.abs(V_new - V)))
            residuals.append(d)
            V = V_new
            if d < tol:
                break
        Q = self.compute_q_table(V)
        argmax = Q.argmax(axis=1)
        policy = {s: self.actions[argmax[self.state_index(s)]] for s in self._states}
        return V, policy, residuals


# --- 教学辅助 ---


def random_policy(env: GridWorld, rng: np.random.Generator) -> dict:
    return {s: rng.choice(env.actions) for s in env.all_states()}


def rollout(env: GridWorld, policy: dict, s0: tuple, max_steps: int = 100) -> list:
    s, traj = s0, []
    for _ in range(max_steps):
        a = policy[s]
        s_next, r, done = env.step(s, a)
        traj.append((s, a, r, s_next))
        if done:
            break
        s = s_next
    return traj


def discounted_return(traj: list, gamma: float) -> float:
    return sum((gamma**t) * r for t, (_, _, r, _) in enumerate(traj))
