"""UAVGrid3D(8x8x4x16) 单元测试。

覆盖 step 的 4 类边界（出界 / 撞楼 / 越顶飞过 / 电量耗尽）以及 VI 不动点性质。
"""
from __future__ import annotations

import numpy as np
import pytest

from shared.uav_grid import (
    BATTERY_COST,
    UAVConfig,
    UAVGrid3D,
    random_policy,
)


@pytest.fixture
def env() -> UAVGrid3D:
    return UAVGrid3D(UAVConfig(gamma=0.9))


# ---------- step 边界 ----------

def test_step_out_of_bounds(env):
    """(0,0,0,5) 向 west：超出地图边界，留在原地 + r_boundary，电量按动作消耗。"""
    s_next, r, done = env.step((0, 0, 0, 5), "west")
    expected_bat = 5 - BATTERY_COST["west"]
    assert s_next == (0, 0, 0, expected_bat)
    assert r == env.cfg.r_boundary
    assert done is False


def test_step_into_building_blocked(env):
    """(3,2,0,5) 向 east 试图进 (3,3,0)（楼，高度=3）：被挡，留原格 + r_forbidden。"""
    # building_height 默认包含 (3,3)=3，所以 alt=0 处禁飞
    assert env.is_building_at(3, 3, 0)
    s_next, r, done = env.step((3, 2, 0, 5), "east")
    expected_bat = 5 - BATTERY_COST["east"]
    assert s_next == (3, 2, 0, expected_bat)
    assert r == env.cfg.r_forbidden
    assert done is False


def test_step_fly_over_building(env):
    """高度 >= 楼高时可越顶通过：(3,2,3,5) east → (3,3,3,4)。"""
    assert not env.is_building_at(3, 3, 3)  # alt=3 等于楼高，可通过
    s_next, r, done = env.step((3, 2, 3, 5), "east")
    assert s_next == (3, 3, 3, 4)
    assert r == env.cfg.r_step  # 正常移动 r=0
    assert done is False


def test_step_battery_cost(env):
    """每个动作的电量消耗：lateral=1, hover=0, ascend=2, descend=0。"""
    s = (4, 4, 2, 10)
    for action, expected_drain in [
        ("north", 1), ("south", 1), ("east", 1), ("west", 1),
        ("hover", 0), ("ascend", 2), ("descend", 0),
    ]:
        _, _, _ = env.step(s, action)  # ensure no exception
        # 直接验证 BATTERY_COST 表
        assert BATTERY_COST[action] == expected_drain


def test_step_battery_zero_absorbing(env):
    """battery=0 但未到 target：任何动作下都吸收（r=0, done=True）。"""
    s = (1, 1, 1, 0)  # 非 target 且 battery=0
    for a in env.actions:
        s_next, r, done = env.step(s, a)
        assert s_next == s
        assert r == 0.0
        assert done is True


def test_step_target_absorbing(env):
    """到达 target 后任意动作都吸收。"""
    tr, tc, ta = env.cfg.target
    s = (tr, tc, ta, 5)
    for a in env.actions:
        s_next, r, done = env.step(s, a)
        assert s_next == s
        assert r == 0.0
        assert done is True


def test_step_crash_when_battery_drains_to_zero(env):
    """battery=1 做 lateral 动作落空（不撞楼）应导致 battery=0 + r_crash + 终止。"""
    # 从 (0,0,3,1) east → (0,1,3,0)：电量耗尽
    s_next, r, done = env.step((0, 0, 3, 1), "east")
    assert s_next[3] == 0
    assert r == env.cfg.r_crash
    assert done is True


# ---------- Bellman 恒等式 ----------

def test_bellman_identity_random_policy(env):
    """对随机策略：(I - γP^π) v^π == r^π。1280 维线性方程组。"""
    rng = np.random.default_rng(0)
    pi = random_policy(env, rng)
    V = env.solve_bellman(pi)
    P = env.transition_matrix(pi)
    r = env.reward_vector(pi)
    residual = (np.eye(env.n_states) - env.cfg.gamma * P) @ V - r
    assert np.max(np.abs(residual)) < 1e-8  # UAV 矩阵大一点，宽松一档容差


# ---------- VI 收敛 ----------

def test_vi_converges_to_fixed_point(env):
    """V* = max_a Q*(s,a)。"""
    V, pi, residuals = env.value_iteration(tol=1e-9, max_iter=2000)
    Q = env.compute_q_table(V)
    assert np.allclose(V, Q.max(axis=1), atol=1e-7)


def test_vi_optimal_start_value_positive(env):
    """默认配置下从 start=(0,0,3,15) 起 VI 找出的最优值应该 > 0
    （能到达 target 且回报为正）。"""
    V, pi, residuals = env.value_iteration(tol=1e-9, max_iter=2000)
    v_start = V[env.state_index(env.cfg.start)]
    assert v_start > 0.0
    # 理论上 ≥ γ^14 ≈ 0.229（最短 15 步到达）
    assert v_start >= env.cfg.gamma ** 14 - 1e-6
