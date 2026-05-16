"""CartPole 随机策略 —— L1 教学示例

用 gymnasium 的标准 CartPole-v1 演示 MDP 的连续状态空间例子，
对照网格世界的离散状态空间。L1 课程在 fig_cartpole 处提到此环境。

依赖：
    pip install gymnasium

跑法：
    python cartpole_random.py            # 随机策略，500 episode
    python cartpole_random.py --render   # 看一局可视化（需 gym 渲染依赖）

教学要点：
1. 状态 s ∈ R^4：(cart 位置, cart 速度, 杆角度, 杆角速度) —— 连续状态空间
2. 动作 a ∈ {0, 1}：左推 / 右推 —— 离散动作空间
3. 奖励 r = +1（每步存活）；done 时 episode 终止
4. 随机策略期望回报 ~ 22 步 —— 远低于最优（500 步上限）
"""
from __future__ import annotations

import argparse

import numpy as np


def random_policy_episode(env, rng: np.random.Generator) -> tuple[float, int]:
    """跑一个 episode，返回 (累计回报, 步数)。"""
    obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
    total_reward = 0.0
    steps = 0
    while True:
        action = int(rng.integers(0, env.action_space.n))  # 均匀随机
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break
    return total_reward, steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--render", action="store_true", help="render one episode")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import gymnasium as gym
    except ImportError as e:
        raise SystemExit("需要安装 gymnasium：pip install gymnasium") from e

    rng = np.random.default_rng(args.seed)

    if args.render:
        env = gym.make("CartPole-v1", render_mode="human")
        reward, steps = random_policy_episode(env, rng)
        print(f"Render episode: return={reward}, steps={steps}")
        env.close()
        return

    env = gym.make("CartPole-v1")
    print(f"State space:  {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"运行 {args.episodes} 个随机策略 episode...")

    returns = []
    steps_list = []
    for _ in range(args.episodes):
        r, s = random_policy_episode(env, rng)
        returns.append(r)
        steps_list.append(s)
    env.close()

    returns = np.array(returns)
    steps_arr = np.array(steps_list)

    print(f"\n  均值 G_0 = {returns.mean():.2f}  std = {returns.std():.2f}")
    print(f"  min/max = {returns.min():.0f} / {returns.max():.0f}")
    print(f"  存活步数: 平均 {steps_arr.mean():.2f} 步")
    print("\n教学诊断：随机策略远低于 500 步上限 —— 必须用 RL 才能学到稳定平衡。")


if __name__ == "__main__":
    main()
