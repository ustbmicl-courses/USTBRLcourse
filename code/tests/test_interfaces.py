"""GridWorld ↔ UAVGrid3D 接口一致性测试。

L4–L7 的算法（VI / PI / MC / TD）通过这套统一接口在两个环境上无差别复用——
本测试就是把"统一接口"这条承诺钉死，防止某次重构破坏对齐。
"""
from __future__ import annotations

import numpy as np
import pytest

from shared.grid_world import GridWorld
from shared.uav_grid import UAVGrid3D


# 接口契约（核心 API + 数据成员）
REQUIRED_METHODS = (
    "all_states",
    "state_index",
    "step",
    "transition_matrix",
    "reward_vector",
    "solve_bellman",
    "compute_q_table",
    "value_iteration",
)

REQUIRED_ATTRS = ("actions", "n_actions", "n_states", "cfg")


@pytest.fixture(params=[GridWorld, UAVGrid3D], ids=["GridWorld", "UAVGrid3D"])
def env(request):
    return request.param()


def test_has_required_methods(env):
    for name in REQUIRED_METHODS:
        assert callable(getattr(env, name, None)), f"{type(env).__name__} 缺方法 {name}"


def test_has_required_attrs(env):
    for name in REQUIRED_ATTRS:
        assert hasattr(env, name), f"{type(env).__name__} 缺属性 {name}"


def test_n_states_matches_all_states(env):
    states = env.all_states()
    assert len(states) == env.n_states
    assert len(set(states)) == env.n_states, "all_states() 含重复"


def test_state_index_roundtrip(env):
    """state_index(s) 与 all_states()[i] 应一一对应。"""
    states = env.all_states()
    for i, s in enumerate(states):
        assert env.state_index(s) == i


def test_step_returns_3tuple(env):
    """step(s, a) → (s', r, done)。"""
    s = env.all_states()[0]
    a = env.actions[0]
    out = env.step(s, a)
    assert isinstance(out, tuple) and len(out) == 3
    s_next, r, done = out
    assert isinstance(r, float)
    assert isinstance(done, bool)
    # s_next 应该是合法状态
    assert s_next in env.all_states()


def test_transition_matrix_rows_sum_to_one(env):
    """P^π 每行应是概率分布（确定性 MDP 下每行恰好一个 1）。"""
    rng = np.random.default_rng(0)
    pi = {s: rng.choice(env.actions) for s in env.all_states()}
    P = env.transition_matrix(pi)
    assert P.shape == (env.n_states, env.n_states)
    row_sums = P.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-12)
