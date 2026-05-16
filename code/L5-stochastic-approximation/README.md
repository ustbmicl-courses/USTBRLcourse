# L5 演示代码：Robbins-Monro 与 SGD

配套讲义：`ustbmicl_lectureNotes/L5-stochastic-approximation-cn/L5-SA.tex`

## 文件

- `rm_step_size_demo.py` —— RM 均值估计在三种步长（$1/k$、常数 $0.3$、$1/k^2$）下的收敛/抖动/卡住对比。

## 运行

```bash
bash code/run.sh run l5
```

或直接：

```bash
python3 code/L5-stochastic-approximation/rm_step_size_demo.py
```

输出：

1. 终端打印三种步长下 $N=200$ 步后的 $\mu_N$ 估计值、残差、$\sum\alpha$、$\sum\alpha^2$
2. 若安装 matplotlib，生成 `figures/rm_trajectories.png`，对应讲义中"可视化：三种步长下的 RM 轨迹"帧

## 课堂讲解顺序

1. $X \sim \mathcal{N}(5, 1)$，目标 $\mu^\star = 5$，从 $\mu_0 = 0$ 出发
2. 三种步长各跑 200 步
3. 看终端数表：$1/k$ 收敛，常数 $0.3$ 残差大但有界，$1/k^2$ 卡在 $\mu_0$ 附近
4. 对照 (C1) 步长两条件：
   - $1/k$：$\sum = \infty$、$\sum^2 < \infty$ ✓
   - $0.3$：$\sum^2 = \infty$ ✗（抖动）
   - $1/k^2$：$\sum < \infty$ ✗（卡住）

## 扩展（留作课后）

- 把 $X$ 换成厚尾分布（Cauchy / Pareto），看 RM 是否仍收敛——挑战 (C2) 噪声方差有限条件
- 加入"漂移目标" $\mu^\star(t) = 5 + 0.01 t$，验证常数步长在非平稳环境的"始终学习"优势
