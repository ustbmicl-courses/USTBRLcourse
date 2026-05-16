"""L6 演示：3x3 网格上首次访问蒙特卡洛（FV-MC）的 1/sqrt(N) 收敛速率。

配套讲义：ustbmicl_lectureNotes/L6-monte-carlo-cn/L6-MC.tex
对应帧：例：3×3 网格 MC 预测

运行：
    bash code/run.sh run l6
或：
    python3 code/L6-monte-carlo/mc_convergence_demo.py

输出：
- 终端打印不同 episode 数下的最大估计误差
- 若安装 matplotlib，生成 figures/mc_convergence.png（log-log 曲线对比 1/sqrt(N) 速率）
"""

from __future__ import annotations

import os
import sys

import numpy as np


# 3x3 网格：标号 0..8，按行优先；s8（右下）为终点，奖励 +1 自循环
# 其他位置 r=0
NROWS, NCOLS = 3, 3
NSTATES = NROWS * NCOLS
TERMINAL = NSTATES - 1  # s8
GAMMA = 0.9


def neighbors(s: int):
    """4-邻居（撞墙原地）+ 停留，共 5 个动作。返回 list of next state."""
    r, c = divmod(s, NCOLS)
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]  # up down left right stay
    out = []
    for dr, dc in moves:
        nr, nc = r + dr, c + dc
        if 0 <= nr < NROWS and 0 <= nc < NCOLS:
            out.append(nr * NCOLS + nc)
        else:
            out.append(s)  # 撞墙原地
    return out


def step(s: int, a: int) -> tuple[int, float]:
    """随机策略 + 确定性转移；终点自循环 r=+1。"""
    ns = neighbors(s)[a]
    r = 1.0 if ns == TERMINAL else 0.0
    return ns, r


def fv_mc_one_episode(rng, max_steps: int = 5000) -> list[tuple[int, float]]:
    """跑一条均匀策略的 episode；返回（首次访问的状态, 该次访问后的回报 G_t）列表。

    FV-MC 标准定义：对每个在轨迹中出现的状态 s，使用其\textbf{首次}出现时刻 t
    对应的回报 G_t 作为 V^pi(s) 的一个无偏样本。
    """
    # 随机起点（不在终点）
    s = int(rng.integers(0, NSTATES - 1))
    traj_states = []
    traj_rewards = []
    for _ in range(max_steps):
        a = int(rng.integers(0, 5))
        ns, r = step(s, a)
        traj_states.append(s)
        traj_rewards.append(r)
        s = ns
        if s == TERMINAL:
            break
    # 1) 前向：记录每个状态的首次出现时刻
    first_t = {}
    for t, st in enumerate(traj_states):
        if st not in first_t:
            first_t[st] = t
    # 2) 反向：算出每个时刻的 G_t
    T = len(traj_states)
    G_at_t = np.zeros(T)
    G = 0.0
    for t in range(T - 1, -1, -1):
        G = GAMMA * G + traj_rewards[t]
        G_at_t[t] = G
    # 3) 输出每个首次访问的 (state, G_{first_t[s]})
    return [(st, float(G_at_t[ft])) for st, ft in first_t.items()]


def exact_value() -> np.ndarray:
    """通过线性方程组直接解出均匀策略下的 V^pi（用作参考真值）。
    约定：terminal s8 是吸收态，V(s8)=0；转移到 s8 时给 +1 终止奖励，与 MC episode 行为一致。
    """
    P = np.zeros((NSTATES, NSTATES))
    r = np.zeros(NSTATES)
    for s in range(NSTATES):
        if s == TERMINAL:
            # 吸收态自循环，r=0，V(s8)=0
            P[s, s] = 1.0
            r[s] = 0.0
            continue
        ns_list = neighbors(s)
        for a in range(5):
            ns = ns_list[a]
            P[s, ns] += 0.2  # 均匀 1/5
            r[s] += 0.2 * (1.0 if ns == TERMINAL else 0.0)
    v = np.linalg.solve(np.eye(NSTATES) - GAMMA * P, r)
    return v


def main() -> int:
    rng = np.random.default_rng(0)
    v_true = exact_value()
    print(f"真值 V^pi (3x3 grid, gamma={GAMMA}, 均匀策略):")
    for r in range(NROWS):
        row = ", ".join(f"{v_true[r * NCOLS + c]:6.3f}" for c in range(NCOLS))
        print(f"  [{row}]")
    print()

    sample_points = [100, 300, 1000, 3000, 10000, 30000]
    n_runs = 20  # 多次重跑取平均误差，让 1/sqrt(N) 曲线更平滑

    V_sum = np.zeros(NSTATES)
    N_cnt = np.zeros(NSTATES, dtype=np.int64)

    errors_all = {n: [] for n in sample_points}
    for run in range(n_runs):
        rng = np.random.default_rng(run + 1)
        V_sum.fill(0.0)
        N_cnt.fill(0)
        for k in range(1, max(sample_points) + 1):
            for s, G in fv_mc_one_episode(rng):
                V_sum[s] += G
                N_cnt[s] += 1
            if k in sample_points:
                V_hat = np.where(N_cnt > 0, V_sum / np.maximum(N_cnt, 1), 0.0)
                # 终止状态价值约定为 0；与 v_true 一致
                V_hat[TERMINAL] = 0.0
                err = float(np.max(np.abs(V_hat - v_true)))
                errors_all[k].append(err)

    print(f"{'N (episodes)':>12}  {'平均误差 ‖Vhat-V^pi‖∞':>22}  {'1/sqrt(N) 参考':>14}")
    for n in sample_points:
        avg = float(np.mean(errors_all[n]))
        ref = 1.0 / np.sqrt(n)
        print(f"{n:>12}  {avg:>22.4f}  {ref:>14.4f}")

    # 出图
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib 未安装，跳过出图)")
        return 0

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "mc_convergence.png")

    Ns = np.array(sample_points, dtype=float)
    avg_errs = np.array([np.mean(errors_all[n]) for n in sample_points])
    ref = 5 * Ns ** -0.5  # 截距匹配的参考线

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.loglog(Ns, avg_errs, "o-", color="tab:blue", linewidth=1.4, label="FV-MC 实测误差")
    ax.loglog(Ns, ref, "--", color="gray", linewidth=1.1, label=r"$\propto 1/\sqrt{N}$ 参考")
    ax.set_xlabel("Episode 数 N")
    ax.set_ylabel(r"$\|\hat V_N - V^\pi\|_\infty$")
    ax.set_title("FV-MC 在 3x3 网格上的 $1/\\sqrt{N}$ 收敛速率")
    ax.legend(loc="lower left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"\n图已保存到: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
