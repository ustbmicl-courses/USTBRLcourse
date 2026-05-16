"""L7 演示：5-状态 random walk 上 TD(0) vs FV-MC 的偏差-方差对比。

配套讲义：ustbmicl_lectureNotes/L7-td-learning-cn/L7-TD.tex
对应帧：例：5-状态 random walk

运行：
    bash code/run.sh run l7
或：
    python3 code/L7-td-learning/td_vs_mc_demo.py

输出：
- 终端打印 100 episode 后两种方法的 MSE
- 若装 matplotlib，生成 figures/td_vs_mc.png（多次重跑的均值 + std envelope）
"""

from __future__ import annotations

import os
import sys

import numpy as np


# 5-状态 random walk: A, B, C, D, E
# 从 C 出发，每步 0.5 概率左/右；
# 到 A 终止 r=0；到 E 终止 r=+1；中间步骤 r=0；gamma=1
N_NONTERMINAL = 5
TRUE_V = np.array([1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6])  # 索引 0..4 对应 A..E


def run_episode(rng):
    """随机策略：每步 0.5 左/0.5 右。返回 list of (state_idx, reward)."""
    s = 2  # 从 C 出发
    traj = []
    while True:
        a = int(rng.integers(0, 2))  # 0 = left, 1 = right
        ns = s - 1 if a == 0 else s + 1
        if ns < 0:
            traj.append((s, 0.0))
            return traj
        if ns >= N_NONTERMINAL:
            traj.append((s, 1.0))
            return traj
        traj.append((s, 0.0))
        s = ns


def run_td(rng, n_episodes: int, alpha: float = 0.1) -> np.ndarray:
    """TD(0): V[s] += alpha * (R + gamma V[s'] - V[s]); gamma=1."""
    V = np.full(N_NONTERMINAL, 0.5)
    mse_traj = np.zeros(n_episodes)
    for ep in range(n_episodes):
        traj = run_episode(rng)
        for t, (s, r) in enumerate(traj):
            if t + 1 < len(traj):
                next_s, _ = traj[t + 1]
                v_next = V[next_s]
            else:
                v_next = 0.0  # 终止状态价值定义为 0
            V[s] += alpha * (r + v_next - V[s])
        mse_traj[ep] = float(np.mean((V - TRUE_V) ** 2))
    return mse_traj


def run_fv_mc(rng, n_episodes: int, alpha: float = 0.01) -> np.ndarray:
    """FV-MC: 用回报 G 更新；alpha 为常数步长（最简版本）。"""
    V = np.full(N_NONTERMINAL, 0.5)
    mse_traj = np.zeros(n_episodes)
    for ep in range(n_episodes):
        traj = run_episode(rng)
        # 回报：gamma=1，所以 G = sum of rewards from time t onwards
        G = 0.0
        seen = set()
        for t in range(len(traj) - 1, -1, -1):
            G += traj[t][1]  # gamma=1 简化
        # 重做：FV-MC 需要每个首次访问的 G
        seen = set()
        G_running = 0.0
        Gs = [0.0] * len(traj)
        for t in range(len(traj) - 1, -1, -1):
            G_running = traj[t][1] + G_running  # gamma=1
            Gs[t] = G_running
        for t, (s, _) in enumerate(traj):
            if s not in seen:
                seen.add(s)
                V[s] += alpha * (Gs[t] - V[s])
        mse_traj[ep] = float(np.mean((V - TRUE_V) ** 2))
    return mse_traj


def main() -> int:
    n_runs = 50
    n_episodes = 200
    td_results = np.zeros((n_runs, n_episodes))
    mc_results = np.zeros((n_runs, n_episodes))
    for run in range(n_runs):
        rng = np.random.default_rng(100 + run)
        td_results[run] = run_td(rng, n_episodes, alpha=0.1)
        rng = np.random.default_rng(100 + run)
        mc_results[run] = run_fv_mc(rng, n_episodes, alpha=0.01)

    td_mean = td_results.mean(0)
    mc_mean = mc_results.mean(0)

    print(f"5-状态 random walk: gamma=1, {n_runs} 次重跑取平均")
    print(f"真值 V^pi = {TRUE_V}\n")
    print(f"{'Episode':>8}  {'TD(0) MSE (alpha=0.1)':>22}  {'MC MSE (alpha=0.01)':>22}")
    for ep in [10, 50, 100, 200]:
        idx = min(ep, n_episodes) - 1
        print(f"{ep:>8}  {td_mean[idx]:>22.5f}  {mc_mean[idx]:>22.5f}")
    print(f"\nTD(0) 100-episode 平均 MSE: {td_mean[99]:.5f}")
    print(f"FV-MC 100-episode 平均 MSE: {mc_mean[99]:.5f}")
    print(f"TD/MC 效率比: {mc_mean[99] / max(td_mean[99], 1e-9):.2f}× 倍")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib 未安装，跳过出图)")
        return 0

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "td_vs_mc.png")

    eps = np.arange(1, n_episodes + 1)
    td_std = td_results.std(0)
    mc_std = mc_results.std(0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(eps, td_mean, "-", color="tab:blue", linewidth=1.3, label="TD(0), alpha=0.1")
    ax.fill_between(eps, td_mean - td_std, td_mean + td_std, color="tab:blue", alpha=0.15)
    ax.plot(eps, mc_mean, "-", color="tab:red", linewidth=1.3, label="FV-MC, alpha=0.01")
    ax.fill_between(eps, mc_mean - mc_std, mc_mean + mc_std, color="tab:red", alpha=0.15)
    ax.set_xlabel("Episode")
    ax.set_ylabel("MSE = mean over 5 states of $(\\hat V - V^\\pi)^2$")
    ax.set_title("TD(0) vs FV-MC on 5-state Random Walk (50 runs)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"\n图已保存到: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
