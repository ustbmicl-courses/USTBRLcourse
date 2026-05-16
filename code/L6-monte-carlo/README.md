# L6 演示代码：蒙特卡洛方法

配套讲义：`ustbmicl_lectureNotes/L6-monte-carlo-cn/L6-MC.tex`

## 文件

- `mc_convergence_demo.py` —— 3x3 网格 + 均匀策略下 FV-MC 的 $1/\sqrt N$ 收敛速率实验

## 运行

```bash
bash code/run.sh run l6
```

或：

```bash
python3 code/L6-monte-carlo/mc_convergence_demo.py
```

## 输出

1. 终端打印解析真值 $V^\pi$（3×3 表格）
2. 不同 episode 数下 FV-MC 估计的最大误差 $\|\hat V_N - V^\pi\|_\infty$
3. 与 $1/\sqrt N$ 参考线对比（验证 MC 收敛速率）
4. 若装有 matplotlib，生成 `figures/mc_convergence.png` 双对数曲线
