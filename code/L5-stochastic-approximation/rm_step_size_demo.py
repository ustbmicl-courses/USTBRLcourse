"""L5 演示：Robbins-Monro 算法在均值估计上的三种步长行为。

配套讲义：ustbmicl_lectureNotes/L5-stochastic-approximation-cn/L5-SA.tex
对应帧：可视化：三种步长下的 RM 轨迹。

运行：
    bash code/run.sh run l5
或：
    python3 code/L5-stochastic-approximation/rm_step_size_demo.py

输出：终端打印三种步长的最终估计值与累积平方步长；
若安装 matplotlib，则生成 figures/rm_trajectories.png。
"""

from __future__ import annotations

import os
import sys

import numpy as np


def run_rm(alpha_fn, mu_true: float, n_steps: int, seed: int = 42) -> np.ndarray:
    """从 mu_0=0 出发，对 x_k~N(mu_true,1) 做 RM 迭代 mu_{k+1}=mu_k - alpha_k (mu_k - x_{k+1})."""
    rng = np.random.default_rng(seed)
    mu = 0.0
    history = np.empty(n_steps + 1, dtype=np.float64)
    history[0] = mu
    for k in range(n_steps):
        x = mu_true + rng.standard_normal()
        alpha = alpha_fn(k)
        mu = mu + alpha * (x - mu)
        history[k + 1] = mu
    return history


def main() -> int:
    mu_true = 5.0
    n_steps = 200

    # 注：1/k^2 schedule 用 0.05 缩放，避免 alpha_0=1 一步跳到 mu* 掩盖"卡住"现象
    schedules = {
        "1/k 收敛": lambda k: 1.0 / (k + 1),
        "alpha=0.3 抖动": lambda k: 0.3,
        "0.05/k^2 卡住": lambda k: 0.05 / (k + 1) ** 2,
    }

    print(f"RM 均值估计：目标 mu* = {mu_true}, n_steps = {n_steps}\n")
    histories = {}
    for name, fn in schedules.items():
        traj = run_rm(fn, mu_true, n_steps)
        histories[name] = traj
        sum_a = sum(fn(k) for k in range(n_steps))
        sum_a2 = sum(fn(k) ** 2 for k in range(n_steps))
        final = traj[-1]
        residual = abs(final - mu_true)
        print(f"  {name:18s}  最终估计 = {final:6.3f}  残差 |mu_N - mu*| = {residual:6.3f}")
        print(f"  {' ' * 18}  ∑α = {sum_a:8.3f}    ∑α² = {sum_a2:6.3f}")
        print()

    # 可选：matplotlib 出图
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib 未安装，跳过出图)")
        return 0

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "rm_trajectories.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"1/k 收敛": "tab:blue", "alpha=0.3 抖动": "tab:red", "0.05/k^2 卡住": "tab:olive"}
    for name, traj in histories.items():
        ax.plot(traj, label=name, color=colors[name], linewidth=1.3)
    ax.axhline(mu_true, color="gray", linestyle="--", linewidth=1, label="mu* = 5")
    ax.set_xlabel("k")
    ax.set_ylabel("mu_k")
    ax.set_title("RM 均值估计：三种步长下的收敛轨迹")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"图已保存到: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
