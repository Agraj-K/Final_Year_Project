"""
test_hetero.py — Pytest unit tests for the HeterogeneousWarehouse wrapper.

Tests cover:
  1. Battery drains at correct type-specific rates
  2. Speed effects fire at roughly expected probabilities
  3. Task info appears correctly in observations
  4. Mismatch penalty applies when expected
  5. Mismatch penalty does NOT apply for compatible pairs
  6. Observation shape is exactly 80
  7. Communication stub returns zero vectors
  8. Robot type one-hot encoding is valid

Run with:  pytest test_hetero.py -v
"""

import numpy as np
import pytest

from hetero_wrapper import (
    COMPATIBILITY_MAP,
    DEFAULT_DRAIN_RATES,
    DEFAULT_MISMATCH_PENALTY,
    CommunicationChannel,
    HeterogeneousWarehouse,
    RobotType,
    TaskType,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

def _make_env(n_agents=4, seed=42):
    """Create a wrapped environment with deterministic seed."""
    from rware.warehouse import Warehouse, RewardType

    base = Warehouse(
        shelf_columns=3,
        column_height=8,
        shelf_rows=1,
        n_agents=n_agents,
        msg_bits=0,
        sensor_range=1,
        request_queue_size=max(n_agents, 2),
        max_inactivity_steps=None,
        max_steps=500,
        reward_type=RewardType.INDIVIDUAL,
    )
    env = HeterogeneousWarehouse(base)
    env.reset(seed=seed)
    return env


@pytest.fixture
def env4():
    """4-agent wrapped environment."""
    return _make_env(n_agents=4, seed=42)


@pytest.fixture
def env2():
    """2-agent wrapped environment (smaller for controlled tests)."""
    return _make_env(n_agents=2, seed=123)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Battery drain rates
# ═══════════════════════════════════════════════════════════════════════════════

class TestBatteryDrain:
    """Verify battery drains at the correct type-specific rate."""

    def test_battery_starts_at_100(self, env4):
        for i in range(env4.n_agents):
            assert env4.battery[i] == pytest.approx(100.0)

    def test_drain_rate_per_type(self):
        """
        Force all agents to a known type, step N times with NOOP,
        and verify exact drain.
        """
        env = _make_env(n_agents=3, seed=99)

        # Force each agent to a different type
        env.robot_types = [RobotType.FAST_LIGHT, RobotType.HEAVY_LOAD, RobotType.BALANCED]
        env.battery[:] = 100.0

        n_steps = 50
        noop_actions = [0] * 3  # NOOP for all

        # Disable speed effects so they don't interfere
        env.fast_light_bonus_prob = 0.0
        env.heavy_load_skip_prob = 0.0

        for _ in range(n_steps):
            env.step(noop_actions)

        expected = {
            RobotType.FAST_LIGHT: 100.0 - n_steps * DEFAULT_DRAIN_RATES[RobotType.FAST_LIGHT],
            RobotType.HEAVY_LOAD: 100.0 - n_steps * DEFAULT_DRAIN_RATES[RobotType.HEAVY_LOAD],
            RobotType.BALANCED:   100.0 - n_steps * DEFAULT_DRAIN_RATES[RobotType.BALANCED],
        }

        for i, rtype in enumerate(env.robot_types):
            assert env.battery[i] == pytest.approx(expected[rtype], abs=0.01), \
                f"Agent {i} ({rtype.name}): expected {expected[rtype]}, got {env.battery[i]}"

    def test_battery_never_negative(self):
        """Battery should clamp to 0, never go negative."""
        env = _make_env(n_agents=2, seed=55)
        env.robot_types = [RobotType.HEAVY_LOAD, RobotType.HEAVY_LOAD]
        env.battery[:] = 1.0  # almost dead

        # Disable speed effects
        env.fast_light_bonus_prob = 0.0
        env.heavy_load_skip_prob = 0.0

        # Step enough to drain past 0
        for _ in range(50):
            env.step([0, 0])

        for i in range(env.n_agents):
            assert env.battery[i] >= 0.0, f"Battery went negative: {env.battery[i]}"

    def test_dead_battery_forces_noop(self):
        """An agent with 0 battery should have its action forced to NOOP."""
        env = _make_env(n_agents=2, seed=66)
        env.robot_types = [RobotType.FAST_LIGHT, RobotType.FAST_LIGHT]
        env.battery[0] = 0.0  # dead
        env.battery[1] = 100.0  # alive

        # Disable speed effects
        env.fast_light_bonus_prob = 0.0

        # Record position before
        pos_before = (env.base_env.agents[0].x, env.base_env.agents[0].y)

        # Try to move agent 0 FORWARD
        env.step([1, 0])  # FORWARD for agent 0, NOOP for agent 1

        pos_after = (env.base_env.agents[0].x, env.base_env.agents[0].y)

        # Dead agent should not have moved (NOOP enforced)
        assert pos_before == pos_after, "Dead-battery agent should not move"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Speed effects probability
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpeedEffects:
    """Verify speed effects fire at roughly expected probabilities."""

    def test_heavy_load_skip_probability(self):
        """
        Over many trials, heavy_load agents should have ~30% of their
        non-NOOP moves skipped.
        """
        env = _make_env(n_agents=1, seed=42)
        env.robot_types = [RobotType.HEAVY_LOAD]
        env.fast_light_bonus_prob = 0.0  # isolate heavy_load behavior
        env.heavy_load_skip_prob = 0.3

        n_trials = 2000
        skip_count = 0

        for trial in range(n_trials):
            env.battery[:] = 100.0  # keep alive
            actions = [1]  # FORWARD

            # Apply speed effects and check if action was changed to NOOP
            modified, _ = env._apply_speed_effects(list(actions))
            if modified[0] == 0:  # was changed to NOOP
                skip_count += 1

        observed_rate = skip_count / n_trials
        expected_rate = 0.3

        # Allow ±5% tolerance (for 2000 trials, this is generous)
        assert abs(observed_rate - expected_rate) < 0.05, \
            f"Heavy load skip rate: expected ~{expected_rate}, got {observed_rate}"

    def test_fast_light_bonus_probability(self):
        """
        Over many trials, fast_light agents should be marked for bonus
        moves ~50% of the time.
        """
        env = _make_env(n_agents=1, seed=42)
        env.robot_types = [RobotType.FAST_LIGHT]
        env.heavy_load_skip_prob = 0.0
        env.fast_light_bonus_prob = 0.5

        n_trials = 2000
        bonus_count = 0

        for trial in range(n_trials):
            env.battery[:] = 100.0
            actions = [1]  # FORWARD

            _, bonus_agents = env._apply_speed_effects(list(actions))
            if 0 in bonus_agents:
                bonus_count += 1

        observed_rate = bonus_count / n_trials
        expected_rate = 0.5

        assert abs(observed_rate - expected_rate) < 0.05, \
            f"Fast light bonus rate: expected ~{expected_rate}, got {observed_rate}"

    def test_balanced_no_speed_effect(self):
        """Balanced agents should never get skips or bonus moves."""
        env = _make_env(n_agents=1, seed=42)
        env.robot_types = [RobotType.BALANCED]

        for _ in range(500):
            env.battery[:] = 100.0
            actions = [1]
            modified, bonus_agents = env._apply_speed_effects(list(actions))
            assert modified[0] == 1, "Balanced agent's action was modified"
            assert len(bonus_agents) == 0, "Balanced agent got bonus move"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Task info in observations
# ═══════════════════════════════════════════════════════════════════════════════

class TestTaskInfo:
    """Verify task information appears correctly in observations."""

    def test_task_info_shape_in_obs(self, env4):
        """Obs indices 75-79 should contain 5 task info values."""
        obs, _ = env4.reset(seed=42)
        for i in range(env4.n_agents):
            task_info = obs[i][75:80]
            assert task_info.shape == (5,), f"Task info shape wrong: {task_info.shape}"

    def test_task_type_onehot_valid(self, env4):
        """Task type one-hot (indices 75-77) should sum to 0 or 1."""
        obs, _ = env4.reset(seed=42)
        for i in range(env4.n_agents):
            type_onehot = obs[i][75:78]
            total = type_onehot.sum()
            assert total == pytest.approx(0.0) or total == pytest.approx(1.0), \
                f"Agent {i}: task type one-hot sum = {total}"

    def test_task_weight_normalized(self, env4):
        """Task weight (index 78) should be in [0, 1]."""
        obs, _ = env4.reset(seed=42)
        for i in range(env4.n_agents):
            weight = obs[i][78]
            assert 0.0 <= weight <= 1.0, f"Agent {i}: weight = {weight}"

    def test_task_distance_normalized(self, env4):
        """Task distance (index 79) should be in [0, 1]."""
        obs, _ = env4.reset(seed=42)
        for i in range(env4.n_agents):
            dist = obs[i][79]
            assert 0.0 <= dist <= 1.0, f"Agent {i}: distance = {dist}"

    def test_task_distance_matches_actual(self):
        """Cross-check the reported distance against actual shelf positions."""
        env = _make_env(n_agents=2, seed=77)

        for i in range(env.n_agents):
            agent = env.base_env.agents[i]
            ax, ay = agent.x, agent.y

            # Find actual nearest requested shelf
            best_dist = float("inf")
            for shelf in env.base_env.request_queue:
                d = abs(ax - shelf.x) + abs(ay - shelf.y)
                if d < best_dist:
                    best_dist = d

            max_dist = env.base_env.grid_size[0] + env.base_env.grid_size[1]
            expected_norm = best_dist / max_dist if max_dist > 0 else 0.0

            # Get from observation
            obs, _ = env.reset(seed=77)
            reported_dist = obs[i][79]

            assert reported_dist == pytest.approx(expected_norm, abs=0.01), \
                f"Agent {i}: expected dist {expected_norm}, got {reported_dist}"


# ═══════════════════════════════════════════════════════════════════════════════
# 4 & 5. Mismatch penalty
# ═══════════════════════════════════════════════════════════════════════════════

class TestMismatchPenalty:
    """Verify mismatch penalty applies (and doesn't apply) correctly."""

    def test_penalty_for_incompatible_delivery(self):
        """
        Simulate a delivery where robot type doesn't match task type.
        Use _apply_mismatch_penalty directly with controlled inputs.
        """
        env = _make_env(n_agents=2, seed=42)

        # Force types
        env.robot_types = [RobotType.FAST_LIGHT, RobotType.BALANCED]

        # Create a fake delivery scenario:
        # Shelf ID 5 was in old queue but not in new queue (= delivered)
        env.task_type_map[5] = TaskType.HEAVY_SHELF  # mismatch with FAST_LIGHT

        old_queue_ids = {5, 10, 15}
        new_queue_ids = {10, 15, 20}  # shelf 5 replaced by 20

        rewards = np.array([1.0, 0.0])

        # We need to mock the delivering agent detection.
        # Directly test the penalty logic by patching _find_delivering_agent
        original_find = env._find_delivering_agent
        env._find_delivering_agent = lambda sid: 0 if sid == 5 else None

        result = env._apply_mismatch_penalty(rewards, old_queue_ids, new_queue_ids)

        # Restore
        env._find_delivering_agent = original_find

        expected_penalty = DEFAULT_MISMATCH_PENALTY
        assert result[0] == pytest.approx(1.0 - expected_penalty), \
            f"Expected {1.0 - expected_penalty}, got {result[0]}"
        assert result[1] == pytest.approx(0.0), \
            f"Agent 1 should be unaffected, got {result[1]}"

    def test_no_penalty_for_compatible_delivery(self):
        """
        Simulate a delivery where robot type matches task type.
        No penalty should be applied.
        """
        env = _make_env(n_agents=2, seed=42)

        # Force compatible types
        env.robot_types = [RobotType.FAST_LIGHT, RobotType.BALANCED]
        env.task_type_map[5] = TaskType.URGENT_LIGHT  # matches FAST_LIGHT

        old_queue_ids = {5, 10, 15}
        new_queue_ids = {10, 15, 20}

        rewards = np.array([1.0, 0.0])

        original_find = env._find_delivering_agent
        env._find_delivering_agent = lambda sid: 0 if sid == 5 else None

        result = env._apply_mismatch_penalty(rewards, old_queue_ids, new_queue_ids)

        env._find_delivering_agent = original_find

        # No penalty — reward stays at 1.0
        assert result[0] == pytest.approx(1.0), \
            f"Compatible delivery should have no penalty, got {result[0]}"

    def test_all_compatibility_pairs(self):
        """Verify the compatibility map covers all expected pairs."""
        assert COMPATIBILITY_MAP[RobotType.FAST_LIGHT] == TaskType.URGENT_LIGHT
        assert COMPATIBILITY_MAP[RobotType.HEAVY_LOAD] == TaskType.HEAVY_SHELF
        assert COMPATIBILITY_MAP[RobotType.BALANCED] == TaskType.STANDARD_DELIVERY

    def test_penalty_magnitude_configurable(self):
        """Custom penalty magnitude should be respected."""
        from rware.warehouse import Warehouse, RewardType

        base = Warehouse(
            shelf_columns=3, column_height=8, shelf_rows=1,
            n_agents=2, msg_bits=0, sensor_range=1,
            request_queue_size=2, max_inactivity_steps=None,
            max_steps=500, reward_type=RewardType.INDIVIDUAL,
        )
        env = HeterogeneousWarehouse(base, mismatch_penalty=0.8)
        env.reset(seed=42)

        env.robot_types = [RobotType.BALANCED, RobotType.BALANCED]
        env.task_type_map[5] = TaskType.HEAVY_SHELF  # mismatch with BALANCED

        old_queue_ids = {5, 10}
        new_queue_ids = {10, 20}
        rewards = np.array([1.0, 0.0])

        original_find = env._find_delivering_agent
        env._find_delivering_agent = lambda sid: 0 if sid == 5 else None

        result = env._apply_mismatch_penalty(rewards, old_queue_ids, new_queue_ids)
        env._find_delivering_agent = original_find

        assert result[0] == pytest.approx(1.0 - 0.8), \
            f"Expected penalty of 0.8, got reward {result[0]}"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Observation shape
# ═══════════════════════════════════════════════════════════════════════════════

class TestObservationShape:
    """Verify observation vector is exactly 80 elements."""

    @pytest.mark.parametrize("n_agents", [2, 4, 8])
    def test_obs_shape_is_80(self, n_agents):
        env = _make_env(n_agents=n_agents, seed=42)
        obs, _ = env.reset(seed=42)

        assert len(obs) == n_agents, f"Expected {n_agents} obs, got {len(obs)}"
        for i in range(n_agents):
            assert obs[i].shape == (80,), \
                f"Agent {i}: expected shape (80,), got {obs[i].shape}"

    def test_obs_shape_after_step(self, env4):
        noop_actions = [0] * env4.n_agents
        obs, _, _, _, _ = env4.step(noop_actions)
        for i in range(env4.n_agents):
            assert obs[i].shape == (80,), \
                f"Agent {i} post-step: expected (80,), got {obs[i].shape}"

    def test_robot_type_onehot_in_obs(self, env4):
        """Indices 71-73 should be a valid one-hot vector."""
        obs, _ = env4.reset(seed=42)
        for i in range(env4.n_agents):
            type_vec = obs[i][71:74]
            assert type_vec.sum() == pytest.approx(1.0), \
                f"Agent {i}: type one-hot sum = {type_vec.sum()}"
            assert all(v in [0.0, 1.0] for v in type_vec), \
                f"Agent {i}: type one-hot has non-binary values"

    def test_battery_in_obs(self, env4):
        """Index 74 should be normalized battery in [0, 1]."""
        obs, _ = env4.reset(seed=42)
        for i in range(env4.n_agents):
            batt = obs[i][74]
            assert 0.0 <= batt <= 1.0, f"Agent {i}: battery = {batt}"
            # After reset, should be 1.0 (100/100)
            assert batt == pytest.approx(1.0), f"Agent {i}: initial battery != 1.0"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Communication stub
# ═══════════════════════════════════════════════════════════════════════════════

class TestCommunicationStub:
    """Verify communication channel returns zero vectors (no-op stub)."""

    def test_messages_are_zeros(self):
        comm = CommunicationChannel(n_agents=4, msg_size=8, comm_range=3)
        for i in range(4):
            msg = comm.encode(i)
            assert msg.shape == (8,)
            assert np.all(msg == 0.0), f"Agent {i}: message should be all zeros"

    def test_gate_always_true(self):
        comm = CommunicationChannel(n_agents=4, msg_size=8, comm_range=3)
        for i in range(4):
            assert comm.gate(i) is True

    def test_receive_from_neighbors(self):
        """Agents within range should receive each other's (zero) messages."""
        comm = CommunicationChannel(n_agents=3, msg_size=8, comm_range=5)
        # Place agents close together
        comm.update_positions([(0, 0), (1, 1), (10, 10)])
        comm.step()

        # Agent 0 and 1 are within range (distance=2), agent 2 is far
        msgs_for_0 = comm.receive(0)
        assert msgs_for_0.shape[0] == 1  # only agent 1
        assert msgs_for_0.shape[1] == 8

        msgs_for_2 = comm.receive(2)
        assert msgs_for_2.shape[0] == 0  # no neighbors in range

    def test_comm_reset_clears_messages(self):
        comm = CommunicationChannel(n_agents=2, msg_size=8)
        comm.messages[0] = np.ones(8)
        comm.reset()
        assert np.all(comm.messages == 0.0)

    def test_wrapper_comm_integration(self, env4):
        """Communication channel should be accessible from wrapper."""
        assert env4.comm is not None
        assert env4.comm.n_agents == env4.n_agents
        assert env4.comm.messages.shape == (env4.n_agents, 8)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge case and integration tests."""

    def test_multiple_resets(self):
        """Environment should handle multiple resets cleanly."""
        env = _make_env(n_agents=4, seed=42)
        for seed in [1, 2, 3, 42, 100]:
            obs, info = env.reset(seed=seed)
            assert len(obs) == 4
            for i in range(4):
                assert obs[i].shape == (80,)
                assert env.battery[i] == 100.0

    def test_step_returns_correct_types(self, env4):
        """Step should return the right types."""
        actions = [0] * env4.n_agents
        result = env4.step(actions)
        obs, rewards, done, truncated, info = result

        assert isinstance(obs, tuple)
        assert isinstance(rewards, list)
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_no_nan_in_observations(self, env4):
        """No NaN values should appear in observations."""
        obs, _ = env4.reset(seed=42)
        for step_num in range(100):
            actions = [env4.action_space[0].sample() for _ in range(env4.n_agents)]
            obs, _, done, _, _ = env4.step(actions)
            for i in range(env4.n_agents):
                assert not np.any(np.isnan(obs[i])), \
                    f"NaN in obs at step {step_num}, agent {i}"
            if done:
                obs, _ = env4.reset(seed=42 + step_num)
