"""GridWorld 通用可视化工具（matplotlib）。"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

ARROW = {"up": "↑", "down": "↓", "left": "←", "right": "→", "stay": "·"}


def setup_chinese_font():
    """设置 matplotlib 中文字体（macOS / Windows / Linux 通用回退）。"""
    from matplotlib import font_manager as fm

    candidates = [
        "PingFang SC", "Heiti TC", "Songti SC", "STHeiti",
        "Source Han Sans CN", "Noto Sans CJK SC", "Microsoft YaHei", "SimHei",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name] + plt.rcParams["font.sans-serif"]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    return None


def plot_grid_layout(env, ax=None, title="GridWorld 5×5"):
    """画网格布局：标记 target / forbidden / 普通格。"""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    cfg = env.cfg
    for r in range(cfg.n_rows):
        for c in range(cfg.n_cols):
            color = "white"
            if (r, c) == cfg.target:
                color = "#90EE90"
            elif (r, c) in cfg.forbidden:
                color = "#FFB6B6"
            ax.add_patch(plt.Rectangle((c, cfg.n_rows - 1 - r), 1, 1, facecolor=color, edgecolor="black"))
            label = f"s{env.display_index((r, c))}"
            if (r, c) == cfg.target:
                label += "\nT"
            elif (r, c) in cfg.forbidden:
                label += "\nX"
            ax.text(c + 0.5, cfg.n_rows - 1 - r + 0.5, label, ha="center", va="center", fontsize=9)
    ax.set_xlim(0, cfg.n_cols)
    ax.set_ylim(0, cfg.n_rows)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    return ax


def plot_value_heatmap(env, V, ax=None, title="V", cmap="viridis", vmin=None, vmax=None):
    """画状态价值热图。V 形状可以是 (n_rows, n_cols) 或 (|S|,)。"""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    cfg = env.cfg
    if V.ndim == 1:
        V = V.reshape(cfg.n_rows, cfg.n_cols)
    im = ax.imshow(V, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    for r in range(cfg.n_rows):
        for c in range(cfg.n_cols):
            ax.text(c, r, f"{V[r, c]:+.2f}", ha="center", va="center",
                    color="white" if V[r, c] < (V.max() + V.min()) / 2 else "black", fontsize=9)
    if (cfg.target):
        ax.add_patch(plt.Rectangle((cfg.target[1] - 0.5, cfg.target[0] - 0.5), 1, 1,
                                    fill=False, edgecolor="lime", linewidth=2))
    for f in cfg.forbidden:
        ax.add_patch(plt.Rectangle((f[1] - 0.5, f[0] - 0.5), 1, 1,
                                    fill=False, edgecolor="red", linewidth=1.5, linestyle="--"))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.045)
    return ax


def plot_policy_arrows(env, policy, ax=None, title="π"):
    """画策略箭头图。policy: dict[state -> action]。"""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    cfg = env.cfg
    for r in range(cfg.n_rows):
        for c in range(cfg.n_cols):
            color = "white"
            if (r, c) == cfg.target:
                color = "#90EE90"
            elif (r, c) in cfg.forbidden:
                color = "#FFB6B6"
            ax.add_patch(plt.Rectangle((c, cfg.n_rows - 1 - r), 1, 1, facecolor=color, edgecolor="black"))
            if (r, c) == cfg.target:
                ax.text(c + 0.5, cfg.n_rows - 1 - r + 0.5, "T", ha="center", va="center", fontsize=14, fontweight="bold")
            elif (r, c) in cfg.forbidden:
                ax.text(c + 0.5, cfg.n_rows - 1 - r + 0.5, "X", ha="center", va="center", fontsize=14, fontweight="bold")
            else:
                a = policy.get((r, c), "stay")
                ax.text(c + 0.5, cfg.n_rows - 1 - r + 0.5, ARROW[a], ha="center", va="center", fontsize=18)
    ax.set_xlim(0, cfg.n_cols)
    ax.set_ylim(0, cfg.n_rows)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    return ax


def plot_trajectory(env, traj, ax=None, title="trajectory"):
    """画轨迹路径（叠在网格上）。traj: [(s, a, r, s'), ...]"""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    plot_grid_layout(env, ax=ax, title=title)
    cfg = env.cfg
    if not traj:
        return ax
    xs = [traj[0][0][1] + 0.5]
    ys = [cfg.n_rows - 1 - traj[0][0][0] + 0.5]
    for _, _, _, s_next in traj:
        xs.append(s_next[1] + 0.5)
        ys.append(cfg.n_rows - 1 - s_next[0] + 0.5)
    ax.plot(xs, ys, "b-", alpha=0.6, linewidth=2)
    ax.plot(xs[0], ys[0], "go", markersize=12, label="start")
    ax.plot(xs[-1], ys[-1], "rs", markersize=12, label="end")
    ax.legend(loc="upper right", fontsize=8)
    return ax
