# L1 教学代码 —— MDP 基础

配套讲义：`ustbmicl_lectureNotes/L1-mdp-foundations-cn/L1-basic.tex`

## 文件清单

| 文件 | 说明 | 依赖 |
|---|---|---|
| `grid_world.py` | 3×3 网格世界 MDP 完整实现：状态/动作/奖励/转移/轨迹/折扣回报 | `numpy` |
| `cartpole_random.py` | CartPole 连续状态 MDP 随机策略基线 | `numpy`, `gymnasium` |

## 快速开始

```bash
# 1) 网格世界 —— 不依赖 gym
python3 grid_world.py

# 2) CartPole（可选）
pip install gymnasium
python3 cartpole_random.py --episodes 500
python3 cartpole_random.py --render        # 可视化一局
```

## 教学要点对照

### `grid_world.py`

直接对应 L1 PPT 中 `fig_gridDemoState` / `fig_3x3gridFog` 三幅图：

| PPT 概念 | 代码位置 |
|---|---|
| 状态 $s_1\ldots s_9$ | `GridWorld.state_index()` |
| 动作集合 $\mathcal A$（5 个） | `ACTIONS = ["up","down","left","right","stay"]` |
| 状态转移 $P(s'\|s,a)$ | `GridWorld.step()` |
| 奖励 $r(s,a)$（target/forbidden/boundary） | `GridConfig` 字段 + `step()` 返回 |
| 策略 $\pi(a\|s)$ | `random_policy()`、`good_policy` 字典 |
| 轨迹 $S_0,A_0,R_1,\ldots$ | `rollout()` 返回 |
| 折扣回报 $G_0=\sum\gamma^t R_{t+1}$ | `discounted_return()` |

运行后会打印：
- 手工策略一条轨迹（从 s1 到 s9）+ $G_0=0.729$
- 随机策略 50 episode 平均回报（用于对比"为什么需要学")
- MDP 五元组诊断信息

### `cartpole_random.py`

对照网格世界的离散状态空间，演示连续状态 MDP：

| 要素 | CartPole-v1 |
|---|---|
| 状态空间 $\mathcal S$ | $\mathbb R^4$（cart 位置/速度、杆角度/角速度）|
| 动作空间 $\mathcal A$ | $\{0,1\}$（左推/右推）|
| 奖励 $r$ | 每步 $+1$（存活） |
| 终止条件 | 杆倒下或步数超 500 |

**核心教学论断**：随机策略平均 $\sim 22$ 步即倒，远低于 500 步上限 → RL 的必要性。

## 课堂演示建议

1. **第一节课**：在终端跑 `grid_world.py`，逐行讲解 `step()` 函数；对照 PPT 的"状态转移：表格表示"slide。
2. **课后作业**：把 `good_policy` 改成"沿对角线走"，看 $G_0$ 如何变化；解释 $\gamma$ 越接近 1，长 horizon 路径优势越明显。
3. **小测**：让学生写一个新的 `forbidden = ((1,1), (2,0))` 配置（两个禁区），观察随机策略的回报分布如何变。

## 与 L4 的衔接

`grid_world.py` 的 `GridWorld` 类\textbf{足以}承载 L4 动态规划（值迭代、策略迭代）的全部演示——L4 教学代码将复用此环境。
