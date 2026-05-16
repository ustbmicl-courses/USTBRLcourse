#!/usr/bin/env bash
# 拉取教学参考仓库到 code/external/
# 用法: bash clone_all.sh [仓库简称]
# 不带参数则按需注释取舍

set -e
cd "$(dirname "$0")"

clone_shallow() {
    local name="$1"
    local url="$2"
    if [ -d "$name/.git" ]; then
        echo "[skip] $name 已存在"
    else
        echo "[clone] $name <- $url"
        git clone --depth 1 "$url" "$name"
    fi
}

# === 必装：环境 ===
clone_shallow Gymnasium                          https://github.com/Farama-Foundation/Gymnasium.git
clone_shallow VectorizedMultiAgentSimulator     https://github.com/proroklab/VectorizedMultiAgentSimulator.git

# === L8+ 3D UAV 仿真栈（按需启用；课程后置 / 与生产 ArduPilot+MAVLink 对齐）===
#
# 训练层（实验室 NVIDIA GPU 服务器）—— VMAS-style 批量并行
# clone_shallow aerial_gym_simulator              https://github.com/ntnu-arl/aerial_gym_simulator.git
#   v2.0.0 (2025-04-28)；Isaac Gym 后端 (CUDA-only)；bundled PPO + Sim2Real 到 PX4/ArduPilot
#   注：Isaac Gym 已 deprecated，2-3 年内官方会迁 Isaac Lab
#
# 真实层（学生本机；Docker 路径，macOS Apple Silicon 已验证）
# clone_shallow ardupilot                         https://github.com/ArduPilot/ardupilot.git
#   仅看源码；运行推荐用 docker (uxduck/ardupilot-sitl-docker) 避免 arm64 编译坑
# clone_shallow pymavlink                         https://github.com/ArduPilot/pymavlink.git
#   官方 Python MAVLink 绑定；通常 pip 装即可，clone 仅用于看源码示例

# === L2-L7 + L14: Sutton-Barto 单文件代码 ===
# clone_shallow reinforcement-learning-an-introduction https://github.com/ShangtongZhang/reinforcement-learning-an-introduction.git

# === L8-L10: 单文件深度 RL ===
# clone_shallow cleanrl                           https://github.com/vwxyzjn/cleanrl.git

# === L9-L10 参考（TF1 维护停滞）===
# clone_shallow spinningup                        https://github.com/openai/spinningup.git

# === L11 RLHF ===
# clone_shallow GRPO-Zero                         https://github.com/policy-gradient/GRPO-Zero.git
# clone_shallow minRLHF                           https://github.com/ttumiel/minRLHF.git

# === L14 MBRL ===
# clone_shallow SimpleDreamer                     https://github.com/kc-ml2/SimpleDreamer.git
# clone_shallow muzero-general                    https://github.com/werner-duvaud/muzero-general.git

# === L15 IL ===
# clone_shallow imitation                         https://github.com/HumanCompatibleAI/imitation.git

echo
echo "完成。已 clone 的仓库:"
ls -d */ 2>/dev/null
