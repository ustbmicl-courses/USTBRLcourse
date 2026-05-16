# USTBRLcourse

本仓库是北京科技大学计算机与通信工程学院段世红老师的“强化学习算法实践”课程的配套代码仓库，面向强化学习基础概念、经典动态规划方法、采样学习方法以及深度强化学习入门实践。

课程代码以“能运行、能改动、能复现实验现象”为目标，帮助同学们在阅读教材和课件之外，通过 Python 脚本与 Jupyter Notebook 观察算法行为，理解强化学习中的状态、动作、奖励、策略、价值函数、Bellman 方程、动态规划、Monte Carlo、TD 学习和函数逼近等核心内容。

## 仓库内容

主要代码位于 [`code/`](code/) 目录：

```text
code/
├── README.md                  代码目录的详细说明
├── requirements.txt           Python 依赖列表
├── run.sh                     统一运行入口
├── setup.sh                   环境安装脚本
├── shared/                    多个章节复用的环境与绘图工具
├── tests/                     pytest 测试用例
├── L1-mdp-foundations/        MDP 基础脚本示例
├── L2-bellman-equation/       Bellman 方程脚本示例
├── L3-bellman-optimality/     Bellman 最优性脚本示例
├── L4-dynamic-programming/    动态规划脚本示例
├── L5-stochastic-approximation/ 随机逼近脚本示例
├── L6-monte-carlo/            Monte Carlo 方法脚本示例
├── L7-td-learning/            TD 学习脚本示例
├── L01_MDP_foundations/       MDP 基础 Notebook
├── L02_Bellman_equation/      Bellman 方程 Notebook
├── L03_Bellman_optimality/    Bellman 最优性 Notebook
├── L04_Dynamic_Programming/   动态规划 Notebook
├── L05_Stochastic_approximation/ 随机逼近 Notebook
└── L08_Value_function_approx/ 函数逼近与 DQN 入门 Notebook
```

其中，`L1-mdp-foundations/`、`L2-bellman-equation/` 这类目录主要放课堂演示脚本，适合快速运行和修改；`L01_MDP_foundations/`、`L02_Bellman_equation/` 这类目录主要放 Notebook，适合按章节阅读、运行单元格、观察图像和实验结果。

## 如何开始

建议在 Git Bash、WSL、Linux 或 macOS 终端中进入 `code/` 目录运行：

```bash
cd code
bash run.sh setup
bash run.sh check
bash run.sh test
```

常用命令如下：

```bash
# 查看所有可用命令
bash run.sh help

# 运行某一讲的脚本示例
bash run.sh run l1
bash run.sh run l4-vi
bash run.sh run l6

# 启动 JupyterLab，浏览和运行 Notebook
bash run.sh jupyter
```

如果你使用 conda，也可以参考：

```bash
cd code
bash setup.sh
```

环境创建完成后，仓库会注册名为 `drlbook` 的 Jupyter 内核，便于在 Notebook 中选择对应环境运行。

## 推荐学习方式

1. 先阅读每一讲目录下的 `README.md`，了解该讲对应的知识点和文件用途。
2. 再运行 `L1-mdp-foundations/`、`L2-bellman-equation/` 等目录中的 Python 脚本，快速观察算法输出。
3. 然后打开对应的 `L01_MDP_foundations/`、`L02_Bellman_equation/` 等 Notebook 目录，逐格运行并修改参数。
4. 对关键算法尝试改动奖励、折扣因子、步长、探索率或环境规模，观察价值函数、策略和收敛曲线如何变化。
5. 最后运行 `bash run.sh test`，确认公共环境和接口仍然可用。

## 章节索引

| 目录 | 主题 |
|---|---|
| `L1-mdp-foundations/` / `L01_MDP_foundations/` | 马尔可夫决策过程、网格世界、CartPole、FrozenLake、UAV 简化建模 |
| `L2-bellman-equation/` / `L02_Bellman_equation/` | Bellman 期望方程、策略评估、状态价值与动作价值 |
| `L3-bellman-optimality/` / `L03_Bellman_optimality/` | Bellman 最优性方程、最优价值函数、最优策略 |
| `L4-dynamic-programming/` / `L04_Dynamic_Programming/` | 价值迭代、策略迭代、广义策略迭代 |
| `L5-stochastic-approximation/` / `L05_Stochastic_approximation/` | Robbins-Monro 随机逼近、步长条件、SGD 与 TD 的联系 |
| `L6-monte-carlo/` | Monte Carlo 预测与采样估计 |
| `L7-td-learning/` | TD(0)、MC 与 TD 的对比 |
| `L08_Value_function_approx/` | 线性价值函数逼近、Baird 反例、DQN CartPole 入门 |

更多运行细节、依赖说明和命令列表请见 [`code/README.md`](code/README.md)。
