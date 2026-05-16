"""GridWorld(5x5) 单元测试。

教学目标：让代码改动能立刻发现回归——尤其是 step 边界、Bellman 恒等式与 VI 不动点。
"""
from __future__ import annotations

import numpy as np
import pytest

from shared.grid_world import GridConfig, GridWorld, random_policy


@pytest.fixture
def env() -> GridWorld:
    return GridWorld(GridConfig(gamma=0.9))


# ---------- step 边界 ----------

def test_step_boundary_into_wall(env):
    """(0,0) 向上撞墙：留在原地，奖励 = r_boundary，不终止。"""
    s_next, r, done = env.step((0, 0), "up")
    assert s_next == (0, 0)
    assert r == env.cfg.r_boundary
    assert done is False


def test_step_into_forbidden(env):
    """步入 forbidden 格：移动过去，奖励 = r_forbidden，不终止。"""
    # (3, 1) 是默认 forbidden 之一
    s_next, r, done = env.step((3, 0), "right")
    assert s_next == (3, 1)
    assert r == env.cfg.r_forbidden
    assert done is False


def test_step_target_is_absorbing(env):
    """在 target 上做任意动作：留在 target，r=0，done=True。"""
    target = env.cfg.target
    for a in env.actions:
        s_next, r, done = env.step(target, a)
        assert s_next == target
        assert r == 0.0
        assert done is True


def test_step_into_target_terminates(env):
    """从邻格走到 target：移动过去，r=r_target，done=True。"""
    # target=(3,2)；(4,2) 向上是 target（默认 forbidden 不包含 (4,2)）
    s_next, r, done = env.step((4, 2), "up")
    assert s_next == env.cfg.target
    assert r == env.cfg.r_target
    assert done is True


# ---------- Bellman 恒等式 ----------

def test_bellman_identity_random_policy(env):
    """对随机策略 π：(I - γP^π) v^π == r^π 应成立到机器精度。"""
    rng = np.random.default_rng(0)
    pi = random_policy(env, rng)
    V = env.solve_bellman(pi)
    P = env.transition_matrix(pi)
    r = env.reward_vector(pi)
    residual = (np.eye(env.n_states) - env.cfg.gamma * P) @ V - r
    assert np.max(np.abs(residual)) < 1e-9


# ---------- VI 收敛 ----------

def test_vi_converges_to_fixed_point(env):
    """VI 输出的 V* 应满足 V* = max_a Q*(s, a)。"""
    V, pi, residuals = env.value_iteration(tol=1e-12, max_iter=2000)
    Q = env.compute_q_table(V)
    assert np.allclose(V, Q.max(axis=1), atol=1e-9)


def test_vi_target_value_zero(env):
    """target 作为吸收态 r=0，V*(target) 必须 = 0。"""
    V, _, _ = env.value_iteration(tol=1e-12, max_iter=2000)
    assert abs(V[env.state_index(env.cfg.target)]) < 1e-9


def test_vi_residual_decay_rate(env):
    """残差应近 γ 倍/轮（γ-收缩的实证）。"""
    _, _, residuals = env.value_iteration(tol=1e-12, max_iter=2000)
    # 前 10 轮残差比值
    res = np.array(residuals[:10])
    ratios = res[1:] / res[:-1]
    # 中位数应该 ≤ γ + 一些噪声余量
    assert np.median(ratios) <= env.cfg.gamma + 0.05
