"""可视化网格世界中的轨迹 —— matplotlib 简单画图。

依赖：matplotlib（已装则可视化，否则脚本退化为打印 ASCII）。
"""
from __future__ import annotations

import numpy as np

from grid_world import GridConfig, GridWorld, discounted_return, rollout


def plot_trajectory(env: GridWorld, traj: list, title: str = "") -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，回退到 ASCII 输出：")
        for t, (s, a, r, sn) in enumerate(traj):
            print(f"  t={t}: {env.render(s)}  a={a}  r={r}")
            print()
        return

    fig, ax = plt.subplots(figsize=(4, 4))
    cfg = env.cfg

    # 网格底色
    grid_img = np.zeros((cfg.n_rows, cfg.n_cols))
    for fr, fc in cfg.forbidden:
        grid_img[fr, fc] = -1
    grid_img[cfg.target] = 1
    ax.imshow(grid_img, cmap="RdYlGn", vmin=-1.5, vmax=1.5)

    # 标记 target / forbidden
    for r in range(cfg.n_rows):
        for c in range(cfg.n_cols):
            if (r, c) == cfg.target:
                ax.text(c, r, "T", ha="center", va="center", fontsize=14, fontweight="bold")
            elif (r, c) in cfg.forbidden:
                ax.text(c, r, "X", ha="center", va="center", fontsize=14, fontweight="bold")

    # 画轨迹箭头
    for t, (s, a, r, s_next) in enumerate(traj):
        ax.annotate(
            "",
            xy=(s_next[1], s_next[0]),
            xytext=(s[1], s[0]),
            arrowprops=dict(arrowstyle="->", color="blue", lw=2, alpha=0.7),
        )
        ax.text(s[1] - 0.3, s[0] - 0.3, f"t={t}", fontsize=8, color="blue")

    ax.set_xticks(range(cfg.n_cols))
    ax.set_yticks(range(cfg.n_rows))
    ax.set_title(title)
    ax.grid(True, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("trajectory.png", dpi=100)
    print("已保存 trajectory.png")


if __name__ == "__main__":
    env = GridWorld(GridConfig())
    good_policy = {
        (0, 0): "right", (0, 1): "right", (0, 2): "down",
        (1, 0): "right", (1, 1): "right", (1, 2): "down",
        (2, 0): "right", (2, 1): "right", (2, 2): "stay",
    }
    traj = rollout(env, good_policy, s0=(0, 0))
    G = discounted_return(traj, env.cfg.gamma)
    plot_trajectory(env, traj, title=f"Manual policy: $G_0$={G:.3f}")
