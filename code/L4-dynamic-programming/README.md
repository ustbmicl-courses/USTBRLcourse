# L4 动态规划 — 教学代码

对应 `ustbmicl_lectureNotes/L4-dynamic-programming-cn/L4-DP.tex` 与教材 `miclDRL/chapters/part2-ch03-bellman-optimality.tex`、`part2-ch04-value-policy-iteration.tex`。

## 文件

| 文件 | 教学要点 | 对应 PPT 位置 |
|---|---|---|
| `value_iteration.py` | Bellman 最优算子 T 的不动点迭代 | L4 §VI 算法 |
| `policy_iteration.py` | 完整 PI / 截断 PI / 与 VI 等价性 | L4 §PI + GPI |
| `convergence_plot.py` | γ-收缩可视化 + PI vs VI 对比 | L4 §收敛性 + 总结 |

复用 `code/L1-mdp-foundations/grid_world.py` 中的 `GridWorld`，无新依赖。

## 运行

```bash
cd code/L4-dynamic-programming
python3 value_iteration.py     # VI: 最优 V*, π*, 残差衰减表
python3 policy_iteration.py    # PI 三种变体 + V* 一致性检验
python3 convergence_plot.py    # 生成 figures/l4_convergence.pdf
```

## 课堂诊断（与 PPT 论断对应）

- **VI 收敛**：3×3 网格 γ=0.9，前 10 轮残差比值 ≈ 0.9（验证 γ-收缩）
- **完整 PI**：外层 ~6 轮即收敛 —— 远少于 VI 的 ~60 轮
- **截断 PI (eval=1)** ≡ VI：用同一份代码统一两种算法
- **γ 改变最优策略**：γ=0.5 时部分状态的 π* 与 γ=0.9 不同（教材 §3 例）

## 依赖

仅 `numpy`；`convergence_plot.py` 可选 `matplotlib`（无则降级写 `.txt`）。
