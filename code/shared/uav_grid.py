"""UAVGrid3D —— 3D 网格 UAV MDP（跨讲共享）

教学定位：作为 GridWorld(5x5) 的"维度放大"版本，演示

  |S|: 25  →  8×8×4×16 = 4096   （×164）
  |A|:  5  →  7                  （+ ascend / descend）

故事：无人机在 8×8 街区上空巡航，4 个高度层 (alt=0 地面 / 1 低空 / 2 中空 / 3 高空)，
电量 0..15（16 档）。状态向量：(row, col, alt, battery)。

楼宇高度（即在哪些 alt 层为禁飞）由 `building_height` 给出：alt < height 全部禁飞，
alt >= height 可以越顶飞过。可在配置中自定义。

动作语义（确定性 MDP）：
    north / south / west / east  —— 平面 4 向移动，battery -1
    hover                        —— 原地停留，battery 0
    ascend                       —— alt +1，battery -2（爬升费电）
    descend                      —— alt -1，battery 0（俯冲不耗电）

终止：
    到达 target (row,col,alt=0)   → 吸收态，r=+1
    电量 0 但未到达 target        → "坠机" 吸收态，r=-1（仅在 step 时一次性触发）
    撞建筑/出界                   → 留在原格 + 负奖励，不终止（与 GridWorld 风格一致）

接口（与 shared.grid_world.GridWorld 对齐）：
    step(s, a)            → (s', r, done)
    all_states()          → list[(row,col,alt,battery)]
    state_index(s)        → int
    transition_matrix(π)  → (|S|, |S|)
    reward_vector(π)      → (|S|,)
    solve_bellman(π)      → v^π
    compute_q_table(V)    → (|S|, |A|)
    value_iteration()     → (V*, π*, residuals)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ACTIONS_3D = ["north", "south", "west", "east", "hover", "ascend", "descend"]

# (drow, dcol, dalt)
ACTION_DELTAS_3D = {
    "north":   (-1, 0, 0),
    "south":   (1, 0, 0),
    "west":    (0, -1, 0),
    "east":    (0, 1, 0),
    "hover":   (0, 0, 0),
    "ascend":  (0, 0, 1),
    "descend": (0, 0, -1),
}

BATTERY_COST = {
    "north": 1, "south": 1, "west": 1, "east": 1,
    "hover": 0,
    "ascend": 2,
    "descend": 0,
}


def _default_buildings() -> dict:
    # (row, col) -> 楼宇高度（alt < h 全部禁飞）
    return {
        (1, 5): 2,
        (2, 5): 2,
        (3, 3): 3,
        (4, 3): 2,
        (4, 6): 2,
        (5, 2): 1,
    }


@dataclass
class UAVConfig:
    n_rows: int = 8
    n_cols: int = 8
    n_alts: int = 4
    n_battery: int = 16  # 电量档位 0..15
    target: tuple = (6, 6, 0)  # 着陆点（alt=0）
    building_height: dict = field(default_factory=_default_buildings)
    r_target: float = 1.0
    r_forbidden: float = -1.0   # 撞楼
    r_boundary: float = -1.0    # 出界 / 钻地 / 飞出顶
    r_crash: float = -1.0       # 电量耗尽未到达
    r_step: float = 0.0
    gamma: float = 0.9
    start: tuple = (0, 0, 3, 15)  # 默认起点：左上角，高空，满电


class UAVGrid3D:
    """8×8×4×5 确定性 3D 网格 MDP（带电量约束）。"""

    def __init__(self, cfg: UAVConfig | None = None) -> None:
        self.cfg = cfg or UAVConfig()
        self.actions = ACTIONS_3D
        self.n_actions = len(self.actions)
        self._states = [
            (r, c, a, b)
            for r in range(self.cfg.n_rows)
            for c in range(self.cfg.n_cols)
            for a in range(self.cfg.n_alts)
            for b in range(self.cfg.n_battery)
        ]
        self.n_states = len(self._states)
        self._state_to_idx = {s: i for i, s in enumerate(self._states)}

    def state_index(self, s: tuple) -> int:
        return self._state_to_idx[s]

    def all_states(self) -> list:
        return list(self._states)

    def is_building_at(self, row: int, col: int, alt: int) -> bool:
        h = self.cfg.building_height.get((row, col), 0)
        return alt < h

    def step(self, s: tuple, a: str) -> tuple:
        """返回 (s', r, done)。

        约定：
          - target 与 battery=0 (非 target) 均为吸收态。
          - 撞楼/出界：留在原位 + 负奖励，电量按 BATTERY_COST 扣（仍计代价）。
          - 移动成功但落入 battery<0：截断为 0 并按 r_crash 终止。
        """
        cfg = self.cfg
        r, c, alt, bat = s

        # 1. 吸收态
        if (r, c, alt) == cfg.target:
            return s, 0.0, True
        if bat == 0:
            return s, 0.0, True

        cost = BATTERY_COST[a]
        new_bat = bat - cost
        bat_clip = max(new_bat, 0)

        dr, dc, da = ACTION_DELTAS_3D[a]
        nr, nc, nalt = r + dr, c + dc, alt + da

        out_of_bounds = not (
            0 <= nr < cfg.n_rows and 0 <= nc < cfg.n_cols and 0 <= nalt < cfg.n_alts
        )
        if out_of_bounds:
            return (r, c, alt, bat_clip), cfg.r_boundary, False

        if self.is_building_at(nr, nc, nalt):
            return (r, c, alt, bat_clip), cfg.r_forbidden, False

        s_next = (nr, nc, nalt, bat_clip)

        if (nr, nc, nalt) == cfg.target:
            return s_next, cfg.r_target, True

        if new_bat <= 0:
            return s_next, cfg.r_crash, True

        return s_next, cfg.r_step, False

    def transition_matrix(self, policy: dict) -> np.ndarray:
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
        r = np.zeros(self.n_states)
        for s in self._states:
            i = self.state_index(s)
            a = policy[s]
            _, rr, _ = self.step(s, a)
            r[i] = rr
        return r

    def solve_bellman(self, policy: dict) -> np.ndarray:
        n = self.n_states
        P = self.transition_matrix(policy)
        r = self.reward_vector(policy)
        return np.linalg.solve(np.eye(n) - self.cfg.gamma * P, r)

    def compute_q_table(self, V: np.ndarray) -> np.ndarray:
        Q = np.zeros((self.n_states, self.n_actions))
        for s in self._states:
            i = self.state_index(s)
            for ai, a in enumerate(self.actions):
                s_next, rr, _ = self.step(s, a)
                Q[i, ai] = rr + self.cfg.gamma * V[self.state_index(s_next)]
        return Q

    def value_iteration(self, tol: float = 1e-9, max_iter: int = 2000) -> tuple:
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


def random_policy(env: UAVGrid3D, rng: np.random.Generator) -> dict:
    return {s: rng.choice(env.actions) for s in env.all_states()}


def rollout(env: UAVGrid3D, policy: dict, s0: tuple | None = None, max_steps: int = 100) -> list:
    s = s0 if s0 is not None else env.cfg.start
    traj = []
    for _ in range(max_steps):
        a = policy[s]
        s_next, r, done = env.step(s, a)
        traj.append((s, a, r, s_next))
        if done:
            break
        s = s_next
    return traj


def discounted_return(traj: list, gamma: float) -> float:
    return sum((gamma ** t) * r for t, (_, _, r, _) in enumerate(traj))


def building_mask(env: UAVGrid3D, alt: int) -> np.ndarray:
    """返回 (n_rows, n_cols) bool 矩阵：True 表示该格在该高度被楼挡住。"""
    cfg = env.cfg
    M = np.zeros((cfg.n_rows, cfg.n_cols), dtype=bool)
    for r in range(cfg.n_rows):
        for c in range(cfg.n_cols):
            M[r, c] = env.is_building_at(r, c, alt)
    return M
