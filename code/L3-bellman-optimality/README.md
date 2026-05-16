# L3 演示代码：Bellman 最优方程

配套讲义：`ustbmicl_lectureNotes/L3-bellman-optimality-cn/L3-BOE.tex`

运行：

```bash
python3 code/L3-bellman-optimality/bellman_optimality.py
```

课堂讲解顺序：

1. 从任意价值函数 `V_k` 出发。
2. 对每个状态计算所有动作的 `Q_k(s,a)=r+gamma V_k(s')`。
3. 取最大值得到 Bellman 最优备份：
   `V_{k+1}(s)=max_a Q_k(s,a)`。
4. 打印 `V*`、`pi*` 和代表状态的 `Q*(s,a)`。
5. 用残差下降解释：`gamma < 1` 使 Bellman 最优算子成为压缩映射。

这个脚本复用 `code/shared/grid_world.py` 的 5x5 网格世界。
