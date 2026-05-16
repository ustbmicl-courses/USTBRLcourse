"""VI 与 PI 收敛速度对比图 —— L4 教学示例

输出: figures/l4_convergence.pdf （若无 matplotlib，则保存 .txt）

教学论断：
  - VI 残差 Δ_k 满足 Δ_k ≤ γ^k Δ_0  ——  半对数曲线呈直线
  - 完整 PI 外层迭代次数 << VI（这里 3x3 网格只需 ~6 轮 vs VI ~60 轮）
  - 截断 PI(1 步评估) ≡ VI
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "L1-mdp-foundations"))
from grid_world import GridConfig, GridWorld  # noqa: E402

from policy_iteration import policy_iteration  # noqa: E402
from value_iteration import value_iteration  # noqa: E402


def main() -> None:
    env = GridWorld(GridConfig(gamma=0.9))

    # 1) VI 残差曲线
    _, _, residuals_vi = value_iteration(env, tol=1e-12, max_iter=200)

    # 2) 不同截断长度的 PI 外层迭代次数
    pi_results = {}
    for eval_iters in [1, 3, 5, 10, None]:
        _, _, ev, _ = policy_iteration(env, eval_iters=eval_iters)
        label = f"eval={eval_iters}" if eval_iters else "eval=full"
        pi_results[label] = ev

    fig_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(fig_dir, exist_ok=True)

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        ax = axes[0]
        ax.semilogy(range(1, len(residuals_vi) + 1), residuals_vi, "o-", label="VI residual")
        # 理论上界 γ^k * Δ_0
        gamma = env.cfg.gamma
        bound = [residuals_vi[0] * gamma**k for k in range(len(residuals_vi))]
        ax.semilogy(range(1, len(bound) + 1), bound, "--", label=r"$\gamma^k \Delta_0$ bound")
        ax.set_xlabel("iteration $k$")
        ax.set_ylabel(r"$\|v_{k+1}-v_k\|_\infty$")
        ax.set_title("Value Iteration: γ-contraction")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        for label, ev in pi_results.items():
            ax.plot(range(1, len(ev) + 1), ev, "o-", label=label)
        ax.set_xlabel("outer iteration")
        ax.set_ylabel("inner eval steps")
        ax.set_title("Truncated PI vs Full PI")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        out_pdf = os.path.join(fig_dir, "l4_convergence.pdf")
        out_png = os.path.join(fig_dir, "l4_convergence.png")
        fig.savefig(out_pdf)
        fig.savefig(out_png, dpi=150)
        print(f"已保存: {out_pdf}")
        print(f"已保存: {out_png}")

    except ImportError:
        out_txt = os.path.join(fig_dir, "l4_convergence.txt")
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("# VI residuals\n")
            for k, d in enumerate(residuals_vi):
                f.write(f"{k}\t{d:.10e}\n")
            f.write("\n# PI outer iterations (eval steps per outer)\n")
            for label, ev in pi_results.items():
                f.write(f"{label}\t{ev}\n")
        print(f"matplotlib 不可用，已保存数值: {out_txt}")

    # 教学诊断
    print("\n===== 教学诊断 =====")
    print(f"VI 收敛步数 (tol=1e-12): {len(residuals_vi)}")
    for label, ev in pi_results.items():
        print(f"PI {label:>10s}: 外层 {len(ev)} 轮，总评估步 {sum(ev)}")
    print(f"\n理论：VI 每轮残差应衰减约 γ={env.cfg.gamma:.1f} 倍")
    n = min(5, len(residuals_vi) - 1)
    ratios = [
        residuals_vi[i] / residuals_vi[i - 1]
        for i in range(1, n + 1)
        if residuals_vi[i - 1] > 0
    ]
    print(f"实测前 {len(ratios)} 轮残差比值: {[f'{r:.4f}' for r in ratios]}")


if __name__ == "__main__":
    main()
