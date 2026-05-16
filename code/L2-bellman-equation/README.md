# L2 演示代码：Bellman 期望方程

配套讲义：`ustbmicl_lectureNotes/L2-bellman-equation-cn/L2-Bellman.tex`

运行：

```bash
python3 code/L2-bellman-equation/policy_evaluation.py
```

课堂讲解顺序：

1. 固定一个策略 `pi(s)`。
2. 固定策略后，MDP 变成 Markov reward process。
3. 从代码中打印 `P^pi` 和 `r^pi` 的形状。
4. 对比两种求解方式：
   - 直接解线性方程 `v = (I - gamma P)^(-1) r`
   - 迭代 Bellman 备份 `v_{k+1}=r+gamma P v_k`

这个脚本复用 `code/shared/grid_world.py` 的 5x5 网格世界。
