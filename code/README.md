# DRLbook 教学代码

深度强化学习课程（北京科技大学，段世红）配套代码。与 `miclDRL/`（教材）、`ustbmicl_lectureNotes/`（讲义 PPT）按章节一一对应。

## 快速开始

```bash
# 1) 一次性环境准备 —— 建立 code/.venv/ + 安装 requirements.txt
bash run.sh setup

# 2) 验证环境
bash run.sh check
bash run.sh test                 # 跑 pytest（30 个测试，覆盖 shared.grid_world / uav_grid）

# 3) 跑某一讲的命令行演示
bash run.sh run l1               # 3×3 网格世界（L1）
bash run.sh run l4-vi            # 价值迭代（L4）

# 4) 启 JupyterLab，浏览 notebook
bash run.sh jupyter              # 自动打开浏览器，内核名 drlbook
```

`bash run.sh help` 列出全部子命令。

## 目录布局

两套并行：`Lx-topic/` 放 `.py` 课堂演示脚本（用 `run.sh run` 派遣），`L0x_Topic/` 放 Jupyter notebook 教学场景（用 `run.sh jupyter` 浏览，或 `run.sh notebooks <id>` 批量重跑）。两者覆盖范围互补——`.py` 是课堂最小子集，notebook 包含完整场景与生成图。

```
code/
├── run.sh              统一入口（setup / check / run / jupyter / compile / notebooks / test / clone / clean）
├── setup.sh            conda 备选入口
├── requirements.txt    依赖清单（按讲分级，深度 RL/UAV 仿真栈在文件内注释）
├── README.md           本文件
│
├── shared/             跨讲复用：grid_world.py / uav_grid.py / plotting.py
├── tests/              pytest 套件
│
│ ── 课堂演示（.py，run.sh run 派遣） ──────────────────────────
├── L1-mdp-foundations/         grid_world.py / cartpole_random.py / visualize_grid.py
├── L2-bellman-equation/        policy_evaluation.py
├── L3-bellman-optimality/      bellman_optimality.py
├── L4-dynamic-programming/     value_iteration.py / policy_iteration.py / q_value_iteration.py / convergence_plot.py
├── L5-stochastic-approximation/  rm_step_size_demo.py
├── L6-monte-carlo/             mc_convergence_demo.py
├── L7-td-learning/             td_vs_mc_demo.py
│
│ ── 教学场景（notebook，run.sh jupyter 浏览） ─────────────────
├── L01_MDP_foundations/        4 notebooks：gridworld / cartpole / frozenlake / uav-3d
├── L02_Bellman_equation/       3 notebooks：solve-linear / iterative / action-value-Q
├── L03_Bellman_optimality/     value iteration / policy iteration / Q*
├── L04_Dynamic_Programming/    GPI / 收敛对比 / UAV
├── L05_Stochastic_approximation/  step-size 失败 / SGD-as-RM / TD0-as-RM
├── L08_Value_function_approx/  linear TD / Baird 反例 / DQN-CartPole
│
└── external/           clone_all.sh：拉外部参考仓库（Gymnasium / VMAS / 可选 CleanRL 等）
                        子目录不入 git，本地用 bash run.sh clone 重建
```

每个 `Lx-topic/` 与 `L0x_Topic/` 目录下另有自己的 `README.md`，写明对应 PPT 章节、文件清单与课堂讲解切入点。

## `run.sh` 命令派遣表

| 子命令 | 用途 |
|---|---|
| `setup` | 建独立 venv（`code/.venv/`）+ 装 `requirements.txt`（清华 PyPI 镜像）|
| `check` | 只读环境审计：Python 版本 / 关键依赖 / 目录结构 |
| `status` | 一览：venv / 依赖 / 外部仓库 / 各讲代码完成情况 |
| `run <id>` | 跑某讲的 `.py` 演示。可用 id：`l1` / `l1-cartpole` / `l1-vis` / `l2` / `l3` / `l4`（=`l4-vi`）/ `l4-pi` / `l4-conv` / `l4-qvi` / `l5` / `l6` / `l7` |
| `jupyter` | 启 JupyterLab。`--notebook` 切回经典 UI；`--port N` 指定端口；`--no-browser` 不自动开 |
| `compile <id>` | 仅编译该讲的 PPT-cn（xelatex 两遍）。id：`l1`-`l8`。`--open` 完成后弹 PDF |
| `notebooks <id>` | 仅重跑该讲所有 `.ipynb`（重生成 `figures/`）。`--check` 只列不跑 |
| `test` | 跑 `code/tests/` pytest 套件。额外参数透传，例如 `bash run.sh test -k bellman` |
| `clone` | 执行 `external/clone_all.sh`，拉教学参考仓库 |
| `clean` | 清 `__pycache__`，可选清 `.venv` |

**重要约定**：PPT 编译（`compile`）与代码执行（`run` / `notebooks` / `jupyter`）**互不触发**。改了 PPT 用 `compile`；重跑代码或重生成图用 `run` / `notebooks` / `jupyter`。没有合并流水。

## 依赖与平台

- Python ≥ 3.9（推荐 3.11）
- 核心：`numpy` / `matplotlib`（L1-L7 表格 RL 纯靠这俩）
- L1 CartPole 与 L7 后续：`gymnasium`
- L8-L10 深度 RL：`torch` / `tensorboard`
- L8+ UAV 仿真栈（按需启用，requirements.txt 内注释了 ArduPilot SITL + Aerial Gym 路径）
- 测试：`pytest`

环境通过 `bash run.sh setup` 一键搭建；conda 用户可改用 `bash setup.sh`。两者都注册名为 `drlbook` 的 Jupyter 内核。

## 不入 git 的内容

- `code/.venv/` — 本机虚拟环境
- `code/external/<repo>/` — 外部参考仓库的 clone 内容（`bash run.sh clone` 重建）
- `**/__pycache__/`、`**/.ipynb_checkpoints/`、`.DS_Store`

## 与教材/讲义的对应关系

| 本目录 | 教材章节（`miclDRL/`）| PPT 讲（`ustbmicl_lectureNotes/`）|
|---|---|---|
| `L1-` / `L01_` | `part2-ch01-mdp` | `L1-mdp-foundations-cn` |
| `L2-` / `L02_` | `part2-ch02-bellman-theory`（前半） | `L2-bellman-equation-cn` |
| `L3-` / `L03_` | `part2-ch02-bellman-theory`（后半） | `L3-bellman-optimality-cn` |
| `L4-` / `L04_` | `part2-ch03-dynamic-programming` | `L4-dynamic-programming-cn` |
| `L5-` / `L05_` | `part2-ch04-stochastic-approximation` | `L5-stochastic-approximation-cn` |
| `L6-` | `part2-ch05-monte-carlo` | `L6-monte-carlo-cn` |
| `L7-` | `part2-ch06-td-learning` | `L7-td-learning-cn` |
| `L08_` | `part2-ch07-value-function-approx` | `L8-value-function-approx-cn` |
