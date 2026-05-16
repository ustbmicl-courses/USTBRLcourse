# L8 · 价值函数逼近（Value Function Approximation）

对应 PPT：`ustbmicl_lectureNotes/L8-value-function-approx-cn/L8-VFA.tex`

## Notebooks

| 文件 | 对应 PPT 节 | 核心内容 |
|---|---|---|
| `01_linear_td_semi_gradient.ipynb` | §2 线性 FA + Tsitsiklis-Van Roy | 5-state random walk 上半梯度 TD(0)；表格 vs 粗特征收敛到投影不动点 |
| `02_baird_counterexample.ipynb`    | §3 Deadly Triad                | Baird 7-state；off-policy 半梯度 TD 指数发散；on-policy 重跑稳定 |
| `03_dqn_cartpole_minimal.ipynb`    | §4 DQN（Mnih 2015）            | 最小化 DQN（QNet + replay + target net + ε-greedy）on CartPole-v1 |

## 一键运行

```bash
bash code/run.sh notebooks l8         # 打开 JupyterLab 到本目录
```

## 依赖

- 01, 02：`numpy`、`matplotlib`（纯 CPU，秒级）
- 03：`torch`、`gymnasium`（CPU 训练 ~3 分钟，300 episode）

均已在 `code/requirements.txt`。
