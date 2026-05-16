# L7 演示代码：时序差分学习

配套讲义：`ustbmicl_lectureNotes/L7-td-learning-cn/L7-TD.tex`

## 文件

- `td_vs_mc_demo.py` —— 5-状态 random walk 上 TD(0) vs FV-MC 的偏差-方差对比（多次重跑取平均）

## 运行

```bash
bash code/run.sh run l7
```

或：

```bash
python3 code/L7-td-learning/td_vs_mc_demo.py
```

## 输出

1. 终端打印 50 次重跑下 TD(0) 与 FV-MC 在 10/50/100/200 episode 时的 MSE
2. TD/MC 效率比（一般在此任务上 TD 快 5-10×）
3. 若装 matplotlib，生成 `figures/td_vs_mc.png`（两条 MSE 曲线 + std envelope）

## 课堂讲解顺序

1. 真值 $V^\pi=(1/6, 2/6, 3/6, 4/6, 5/6)$
2. 起点 C（索引 2），从 0.5 初值出发
3. TD(0) 用 $\alpha=0.1$、MC 用 $\alpha=0.01$
4. 50 episode 后 TD MSE ~ 0.005，MC MSE ~ 0.04
5. 验证讲义第 277 行结论："TD 样本效率高 MC 一个量级"

## 扩展

- 修改 $V$ 初值（如全 0、全随机）观察"初值偏差"对 TD/MC 的影响差异
- 把 gamma 改为 0.95 看 MC 方差爆炸
- 加入 SARSA / Q-learning 控制版本（需扩展为含动作）
