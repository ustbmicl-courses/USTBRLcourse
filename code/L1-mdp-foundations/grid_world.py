"""3x3 网格世界 —— L1 教学示例

直接对应 L1-basic.tex 中的 fig_gridDemoState / fig_3x3gridFog 网格图示。
状态编号 s1..s9 与讲义一致：
    s1 s2 s3
    s4 s5 s6
    s7 s8 s9
其中 s7 是禁止区（forbidden），s9 是目标（target）。

本文件只用 numpy，无任何 RL 框架依赖，便于学生逐行阅读。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 五个动作：上、下、左、右、原地停留
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
    """网格世界的奖励参数 —— 直接对应 Shiyu Zhao 讲义"""

    n_rows: int = 3
    n_cols: int = 3
    target: tuple[int, int] = (2, 2)  # s9
    forbidden: tuple[tuple[int, int], ...] = ((2, 0),)  # s7
    r_target: float = 1.0
    r_forbidden: float = -1.0
    r_step: float = 0.0
    r_boundary: float = -1.0  # 撞墙
    gamma: float = 0.9


class GridWorld:
    """3x3 网格世界 MDP —— 显式给出 P(s'|s,a) 与 r(s,a)。

    状态：(row, col) 元组；动作：ACTIONS 中的字符串。
    """

    def __init__(self, cfg: GridConfig | None = None) -> None:
        self.cfg = cfg or GridConfig()
        self.n_states = self.cfg.n_rows * self.cfg.n_cols
        self.actions = ACTIONS

    # --- 状态编码：(row,col) <-> 1..9 编号 ---
    def state_index(self, s: tuple[int, int]) -> int:
        return s[0] * self.cfg.n_cols + s[1] + 1  # 1-indexed: s1..s9

    def all_states(self) -> list[tuple[int, int]]:
        return [(r, c) for r in range(self.cfg.n_rows) for c in range(self.cfg.n_cols)]

    # --- 动力学：本环境是确定性 MDP ---
    def step(self, s: tuple[int, int], a: str) -> tuple[tuple[int, int], float, bool]:
        """返回 (s', r, done)。撞墙 → 留在原处并扣 r_boundary。"""
        if s == self.cfg.target:
            return s, 0.0, True  # 终止状态吸收

        dr, dc = ACTION_DELTAS[a]
        nr, nc = s[0] + dr, s[1] + dc

        # 边界检查
        if not (0 <= nr < self.cfg.n_rows and 0 <= nc < self.cfg.n_cols):
            return s, self.cfg.r_boundary, False

        s_next = (nr, nc)

        if s_next == self.cfg.target:
            return s_next, self.cfg.r_target, True
        if s_next in self.cfg.forbidden:
            # 进入禁区：扣分但仍可继续（讲义版本——并非终止）
            return s_next, self.cfg.r_forbidden, False
        return s_next, self.cfg.r_step, False

    def render(self, s: tuple[int, int]) -> str:
        """返回 ASCII 字符串，方便 print(env.render(s)) 调试。"""
        rows = []
        for r in range(self.cfg.n_rows):
            line = []
            for c in range(self.cfg.n_cols):
                if (r, c) == s:
                    line.append("A")
                elif (r, c) == self.cfg.target:
                    line.append("T")
                elif (r, c) in self.cfg.forbidden:
                    line.append("X")
                else:
                    line.append(".")
            rows.append(" ".join(line))
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# 演示：策略、轨迹、回报
# ---------------------------------------------------------------------------


def random_policy(env: GridWorld, rng: np.random.Generator) -> dict[tuple[int, int], str]:
    """每个状态独立均匀采一个动作（确定性策略，但本身随机生成）。"""
    return {s: rng.choice(env.actions) for s in env.all_states()}


def rollout(
    env: GridWorld,
    policy: dict[tuple[int, int], str],
    s0: tuple[int, int],
    max_steps: int = 50,
) -> list[tuple[tuple[int, int], str, float, tuple[int, int]]]:
    """按策略采一条轨迹，返回 [(s, a, r, s'), ...]"""
    s = s0
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
    """计算从 t=0 起的折扣回报 G_0 = sum gamma^t * r_{t+1}。"""
    return sum((gamma**t) * r for t, (_, _, r, _) in enumerate(traj))


# ---------------------------------------------------------------------------
# 主演示
# ---------------------------------------------------------------------------


def main() -> None:
    rng = np.random.default_rng(seed=42)
    env = GridWorld()
    print("初始网格（A=智能体，T=目标，X=禁区）：")
    print(env.render(s=(0, 0)))
    print()

    # 1) 一个手工设计的"好"策略：尽量朝目标 s9 走
    good_policy = {
        (0, 0): "right", (0, 1): "right", (0, 2): "down",
        (1, 0): "right", (1, 1): "right", (1, 2): "down",
        (2, 0): "right", (2, 1): "right", (2, 2): "stay",
    }

    print("===== 手工策略（沿上行/右列朝 s9）的轨迹 =====")
    traj = rollout(env, good_policy, s0=(0, 0))
    for t, (s, a, r, s_next) in enumerate(traj):
        print(f"  t={t}: s={env.state_index(s)} a={a:>5} r={r:+.1f} s'={env.state_index(s_next)}")
    print(f"  G_0 = {discounted_return(traj, env.cfg.gamma):.4f}")
    print()

    # 2) 随机策略：每个状态采一个固定动作
    print("===== 随机策略 50 episode 的平均回报 =====")
    returns = []
    for _ in range(50):
        rp = random_policy(env, rng)
        tr = rollout(env, rp, s0=(0, 0))
        returns.append(discounted_return(tr, env.cfg.gamma))
    print(f"  mean G_0 = {np.mean(returns):+.4f}  std = {np.std(returns):.4f}")
    print(f"  min = {np.min(returns):+.4f}  max = {np.max(returns):+.4f}")
    print()

    # 3) 形式化检查：列出所有状态、动作、奖励 —— 印证 MDP 五元组
    print("===== MDP 五元组打印（教学诊断） =====")
    print(f"  |S| = {env.n_states},  |A| = {len(env.actions)}")
    print(f"  γ = {env.cfg.gamma}")
    print(f"  r(target) = {env.cfg.r_target},  r(forbidden) = {env.cfg.r_forbidden}")
    print(f"  r(boundary) = {env.cfg.r_boundary},  r(step) = {env.cfg.r_step}")


if __name__ == "__main__":
    main()
