# 外部仓库（教学参考）

本目录用于存放 **clone 自 GitHub** 的外部仓库，作为教学参考、源码阅读、与示例对照用。
**这些目录不进主 git 仓库**（已在 `.gitignore` 中忽略），学生通过 `clone_all.sh` 自取。

## 已规划仓库

| 仓库 | 用途 | 对应讲 | 安装方式 |
|---|---|---|---|
| `Gymnasium` | 通用环境（CartPole / Cliff / Blackjack / Atari） | L1, L6, L7, L8, L9, L10 | clone + `pip install -e .`（或直接 `pip install gymnasium`）|
| `VectorizedMultiAgentSimulator` (VMAS) | GPU 向量化 MARL 仿真 | L12 | clone + `pip install -e .`（或 `pip install vmas`）|
| `reinforcement-learning-an-introduction` (Shangtong) | Sutton-Barto 第二版社区代码 | L2-L7, L14(Dyna) | 仅读源码 / 拷贝单文件 |
| `cleanrl` | 单文件 PyTorch 深度 RL | L8-L10 | clone 读源码 + `pip install` 跑 |
| `spinningup` | 推导级 PG/TRPO 教学代码 | L9, L10（参考）| 读源码（TF1，建议不跑）|
| `GRPO-Zero` | 极简 GRPO（DeepSeek-R1 风格） | L11 | clone 读源码 |
| `SimpleDreamer` | 教学版 Dreamer | L14 | clone 读源码 |
| `muzero-general` | 教学版 MuZero | L14 | clone + 配置 |
| `imitation` | BC/DAgger/GAIL/AIRL 一体库 | L15 | `pip install imitation` |

## 用法

```bash
# 一次性拉所有仓库
bash clone_all.sh

# 或单独拉某个
git clone --depth 1 https://github.com/Farama-Foundation/Gymnasium.git
```

## 已 clone 状态（2026-04-29）

| 仓库 | HEAD | 大小 | 状态 |
|---|---|---|---|
| `Gymnasium` | 8d5043a (2026-04-28) | 242 MB | ✅ 已 clone |
| `VectorizedMultiAgentSimulator` | 7f936ab (2026-02-07) | 6.8 MB | ✅ 已 clone + `pip install -e .` 装入 dc-mcts env |

### 当前 conda env: `dc-mcts`

```
gymnasium 1.1.1   torch 2.8.0   numpy 2.0.2   matplotlib 3.9.4
vmas 1.5.2        gym 0.26.2    pyglet 1.5.27
```

### 安装注意

VMAS 依赖旧 `gym==0.26.2`（非 gymnasium）。本机 macOS 系统代理（127.0.0.1:1082）干扰 pip，须用：
```bash
no_proxy='*' NO_PROXY='*' pip install --proxy "" -e .
```

### VMAS 21 个内置 scenario

`balance / ball_passage / ball_trajectory / buzz_wire / discovery / dispersion / dropout / flocking / football / give_way / joint_passage / joint_passage_size / multi_give_way / navigation / passage / reverse_transport / road_traffic / sampling / transport / wheel / wind_flocking`

L12 教学候选：`navigation`（基础）→ `balance`（协作搬运）→ `discovery`（部分可观测协同）→ `football`（对抗）。
