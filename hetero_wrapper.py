"""
hetero_wrapper.py — Heterogeneous Robot Wrapper for RWARE

Extends the RWARE (Robotic Warehouse) environment with:
  1. Robot types (fast_light / heavy_load / balanced)
  2. Battery system with type-dependent drain
  3. Speed effects (bonus moves / move skips)
  4. Task types on requested shelves with priority weights
  5. Capability mismatch penalty on delivery
  6. Communication stub for future learned messaging

Observation vector layout (80 elements total):
  [0:71]  — RWARE default flat observation
  [71:74] — Robot type one-hot (3)
  [74]    — Normalized battery (1)
  [75:80] — Nearest task info: type one-hot (3) + weight (1) + distance (1)

Author: Auto-generated for Final Year Project
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════════

class RobotType(Enum):
    """Three heterogeneous robot archetypes."""
    FAST_LIGHT = 0
    HEAVY_LOAD = 1
    BALANCED = 2


class TaskType(Enum):
    """Three task categories for requested shelves."""
    URGENT_LIGHT = 0
    HEAVY_SHELF = 1
    STANDARD_DELIVERY = 2


# ═══════════════════════════════════════════════════════════════════════════════
# Constants (all configurable via constructor kwargs)
# ═══════════════════════════════════════════════════════════════════════════════

# Battery drain per step for each robot type
DEFAULT_DRAIN_RATES: Dict[RobotType, float] = {
    RobotType.FAST_LIGHT: 0.2,   # slowest drain — efficient motors
    RobotType.HEAVY_LOAD: 0.5,   # fastest drain — powerful but hungry
    RobotType.BALANCED:   0.35,  # middle ground
}

# Speed effect probabilities
DEFAULT_FAST_LIGHT_BONUS_PROB = 0.5   # chance of getting a bonus extra move
DEFAULT_HEAVY_LOAD_SKIP_PROB  = 0.3   # chance of move being silently skipped

# Task type priority weights
DEFAULT_TASK_WEIGHTS: Dict[TaskType, float] = {
    TaskType.URGENT_LIGHT:     2.0,  # highest priority
    TaskType.HEAVY_SHELF:      1.5,  # medium priority
    TaskType.STANDARD_DELIVERY: 1.0, # base priority
}

# Maximum weight (for normalization)
DEFAULT_MAX_TASK_WEIGHT = 2.0

# Mismatch penalty magnitude
DEFAULT_MISMATCH_PENALTY = 0.5

# Robot-type ↔ task-type compatibility mapping
COMPATIBILITY_MAP: Dict[RobotType, TaskType] = {
    RobotType.FAST_LIGHT: TaskType.URGENT_LIGHT,
    RobotType.HEAVY_LOAD: TaskType.HEAVY_SHELF,
    RobotType.BALANCED:   TaskType.STANDARD_DELIVERY,
}

# Number of extra observation elements appended by this wrapper
EXTRA_OBS_SIZE = 9  # 3 (type) + 1 (battery) + 5 (task info)


# ═══════════════════════════════════════════════════════════════════════════════
# Communication Channel Stub
# ═══════════════════════════════════════════════════════════════════════════════

class CommunicationChannel:
    """
    ┌──────────────────────────────────────────────────────────────────────┐
    │  INTENTIONAL NO-OP STUB                                            │
    │                                                                    │
    │  This class provides the *plumbing* for inter-robot communication  │
    │  without any learned logic. Every robot can send/receive a         │
    │  fixed-size message vector to/from robots within grid range.       │
    │                                                                    │
    │  Future replacement plan:                                          │
    │   - Replace gate() with a learned gating network                   │
    │   - Replace encode() with a learned message encoder                │
    │   - Messages will be appended to observations or fed to a          │
    │     separate communication head in the policy network              │
    │                                                                    │
    │  For now: gate() always returns True, encode() returns zeros.      │
    └──────────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        n_agents: int,
        msg_size: int = 8,
        comm_range: int = 3,
    ):
        """
        Args:
            n_agents: Number of robots in the environment.
            msg_size: Size of the placeholder message vector per robot.
            comm_range: Manhattan distance within which robots can communicate.
        """
        self.n_agents = n_agents
        self.msg_size = msg_size
        self.comm_range = comm_range

        # Message buffer: each robot's outgoing message (all zeros = stub)
        self.messages: np.ndarray = np.zeros((n_agents, msg_size), dtype=np.float32)

        # Agent positions cache (updated each step by the wrapper)
        self._positions: List[Tuple[int, int]] = [(0, 0)] * n_agents

    def update_positions(self, positions: List[Tuple[int, int]]) -> None:
        """Update cached agent positions. Called by the wrapper each step."""
        self._positions = list(positions)

    def get_neighbors(self, agent_idx: int) -> List[int]:
        """
        Return indices of agents within comm_range (Manhattan distance)
        of the given agent. Excludes the agent itself.
        """
        ax, ay = self._positions[agent_idx]
        neighbors = []
        for j in range(self.n_agents):
            if j == agent_idx:
                continue
            bx, by = self._positions[j]
            if abs(ax - bx) + abs(ay - by) <= self.comm_range:
                neighbors.append(j)
        return neighbors

    def gate(self, agent_idx: int) -> bool:
        """
        STUB: Decide whether this agent should broadcast a message this step.

        Currently always returns True.
        → Future: replace with a learned gating network that outputs a
          binary decision based on the agent's observation.
        """
        return True  # No-op stub: always communicate

    def encode(self, agent_idx: int) -> np.ndarray:
        """
        STUB: Generate the message content for this agent.

        Currently returns a zero vector.
        → Future: replace with a learned encoder that maps the agent's
          observation/hidden state to a message vector.
        """
        return np.zeros(self.msg_size, dtype=np.float32)  # No-op stub

    def broadcast(self, agent_idx: int, message: Optional[np.ndarray] = None) -> None:
        """
        Set the outgoing message for an agent.
        If message is None, uses encode() to generate it.
        """
        if message is None:
            message = self.encode(agent_idx)
        self.messages[agent_idx] = message

    def receive(self, agent_idx: int) -> np.ndarray:
        """
        Receive messages from all in-range neighbors.

        Returns:
            np.ndarray of shape (len(neighbors), msg_size).
            Empty (0, msg_size) array if no neighbors in range.
        """
        neighbors = self.get_neighbors(agent_idx)
        if not neighbors:
            return np.zeros((0, self.msg_size), dtype=np.float32)
        return self.messages[neighbors].copy()

    def step(self) -> None:
        """
        Run one communication round: every agent broadcasts (using stub encoder).
        Called by the wrapper at the end of each env.step().
        """
        for i in range(self.n_agents):
            if self.gate(i):
                self.broadcast(i)

    def reset(self) -> None:
        """Clear all messages on environment reset."""
        self.messages[:] = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Main Wrapper
# ═══════════════════════════════════════════════════════════════════════════════

class HeterogeneousWarehouse(gym.Wrapper):
    """
    Gymnasium wrapper that adds heterogeneity to a base RWARE Warehouse env.

    Wraps the observation, reward, and step logic to incorporate:
      - Robot type assignment (3 types)
      - Battery with type-specific drain rates
      - Speed effects (bonus moves / skipped moves)
      - Task types on requested shelves
      - Capability mismatch penalty
      - Communication channel stub

    Usage:
        from rware.warehouse import Warehouse, RewardType
        base_env = Warehouse(shelf_columns=3, column_height=8, shelf_rows=1,
                             n_agents=4, msg_bits=0, sensor_range=1,
                             request_queue_size=4, max_inactivity_steps=None,
                             max_steps=500, reward_type=RewardType.INDIVIDUAL)
        env = HeterogeneousWarehouse(base_env)
        obs, info = env.reset()
    """

    def __init__(
        self,
        env: gym.Env,
        drain_rates: Optional[Dict[RobotType, float]] = None,
        fast_light_bonus_prob: float = DEFAULT_FAST_LIGHT_BONUS_PROB,
        heavy_load_skip_prob: float = DEFAULT_HEAVY_LOAD_SKIP_PROB,
        task_weights: Optional[Dict[TaskType, float]] = None,
        max_task_weight: float = DEFAULT_MAX_TASK_WEIGHT,
        mismatch_penalty: float = DEFAULT_MISMATCH_PENALTY,
        comm_msg_size: int = 8,
        comm_range: int = 3,
    ):
        super().__init__(env)

        self.n_agents: int = env.n_agents
        self.base_env = env  # direct reference for clarity

        # ── Configurable parameters ──────────────────────────────────────
        self.drain_rates = drain_rates or dict(DEFAULT_DRAIN_RATES)
        self.fast_light_bonus_prob = fast_light_bonus_prob
        self.heavy_load_skip_prob = heavy_load_skip_prob
        self.task_weights = task_weights or dict(DEFAULT_TASK_WEIGHTS)
        self.max_task_weight = max_task_weight
        self.mismatch_penalty = mismatch_penalty

        # ── State arrays (initialized in reset) ─────────────────────────
        self.robot_types: List[RobotType] = []
        self.battery: np.ndarray = np.zeros(self.n_agents, dtype=np.float32)
        self.task_type_map: Dict[int, TaskType] = {}  # shelf_id → TaskType

        # ── Communication channel ────────────────────────────────────────
        self.comm = CommunicationChannel(
            n_agents=self.n_agents,
            msg_size=comm_msg_size,
            comm_range=comm_range,
        )

        # ── Internal RNG (separate from base env) ───────────────────────
        self._wrapper_rng: np.random.Generator = np.random.default_rng()

        # ── Observation space: extend base by EXTRA_OBS_SIZE ─────────────
        # We assume base env uses FLATTENED observations (Box per agent)
        base_obs_space = env.observation_space[0]  # single agent's space
        base_dim = base_obs_space.shape[0]
        new_dim = base_dim + EXTRA_OBS_SIZE  # 71 + 9 = 80

        sa_obs_space = gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(new_dim,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Tuple(
            tuple([sa_obs_space] * self.n_agents)
        )

        # Store base dim for slicing
        self._base_obs_dim = base_dim

    # ═══════════════════════════════════════════════════════════════════════
    # Reset
    # ═══════════════════════════════════════════════════════════════════════

    def reset(self, seed=None, options=None):
        """Reset base env and initialize all heterogeneous state."""
        base_obs, info = self.env.reset(seed=seed, options=options)

        # Reseed wrapper RNG if a seed is provided
        if seed is not None:
            self._wrapper_rng = np.random.default_rng(seed)

        # ── Assign robot types randomly ──────────────────────────────────
        type_choices = list(RobotType)
        self.robot_types = [
            type_choices[self._wrapper_rng.integers(0, len(type_choices))]
            for _ in range(self.n_agents)
        ]

        # ── Initialize battery to 100 for all agents ────────────────────
        self.battery = np.full(self.n_agents, 100.0, dtype=np.float32)

        # ── Assign task types to initial request queue ───────────────────
        self.task_type_map = {}
        self._assign_task_types_to_queue()

        # ── Reset communication channel ──────────────────────────────────
        self.comm.reset()
        self._update_comm_positions()

        # ── Build extended observations ──────────────────────────────────
        extended_obs = self._extend_observations(base_obs)
        return extended_obs, info

    # ═══════════════════════════════════════════════════════════════════════
    # Step
    # ═══════════════════════════════════════════════════════════════════════

    def step(self, actions):
        """
        Execute one step with heterogeneous modifications.

        Order of operations:
          1. Force NOOP for dead-battery agents
          2. Apply heavy_load skip (pre-step)
          3. Snapshot request queue (for delivery detection)
          4. Call base env.step()
          5. Detect deliveries and apply mismatch penalty
          6. Apply fast_light bonus move (post-step)
          7. Update battery
          8. Assign task types to any new requests
          9. Run communication stub
          10. Build extended observations
        """
        actions = list(actions)  # make mutable copy

        # 1. Force NOOP for dead-battery agents
        actions = self._enforce_battery_constraint(actions)

        # 2. Apply heavy_load move skip (before base env sees the action)
        actions, bonus_agents = self._apply_speed_effects(actions)

        # 3. Snapshot the request queue before step (for delivery detection)
        old_queue_ids = set(s.id for s in self.base_env.request_queue)

        # 4. Base environment step
        base_obs, rewards, done, truncated, info = self.env.step(actions)
        rewards = np.array(rewards, dtype=np.float64)

        # 5. Detect deliveries and apply mismatch penalty
        new_queue_ids = set(s.id for s in self.base_env.request_queue)
        rewards = self._apply_mismatch_penalty(rewards, old_queue_ids, new_queue_ids)

        # 6. Apply fast_light bonus move (extra step for eligible agents)
        if bonus_agents and not done:
            base_obs, rewards, done, truncated, info = self._apply_bonus_moves(
                bonus_agents, base_obs, rewards, done, truncated, info
            )

        # 7. Update battery
        self._update_battery()

        # 8. Assign task types to any newly added requests
        self._assign_task_types_to_queue()

        # 9. Run communication stub
        self._update_comm_positions()
        self.comm.step()

        # 10. Build extended observations
        extended_obs = self._extend_observations(base_obs)

        return extended_obs, list(rewards), done, truncated, info

    # ═══════════════════════════════════════════════════════════════════════
    # Battery Management
    # ═══════════════════════════════════════════════════════════════════════

    def _update_battery(self) -> None:
        """Drain battery for each agent based on its robot type."""
        for i in range(self.n_agents):
            drain = self.drain_rates[self.robot_types[i]]
            self.battery[i] = max(0.0, self.battery[i] - drain)

    def _enforce_battery_constraint(self, actions: list) -> list:
        """Force NOOP for agents whose battery is depleted."""
        for i in range(self.n_agents):
            if self.battery[i] <= 0.0:
                actions[i] = 0  # Action.NOOP = 0
        return actions

    # ═══════════════════════════════════════════════════════════════════════
    # Speed Effects
    # ═══════════════════════════════════════════════════════════════════════

    def _apply_speed_effects(self, actions: list) -> Tuple[list, List[int]]:
        """
        Apply type-dependent speed modifications.

        - heavy_load: 30% chance the move is silently replaced with NOOP
          (done BEFORE base env.step so collision logic is unaffected).
        - fast_light: 50% chance of getting a bonus extra move
          (done AFTER base env.step as a second step call).
        - balanced: no modification.

        Returns:
            (modified_actions, list of agent indices eligible for bonus moves)
        """
        bonus_agents: List[int] = []

        for i in range(self.n_agents):
            if self.battery[i] <= 0.0:
                continue  # dead agents already handled

            rtype = self.robot_types[i]

            if rtype == RobotType.HEAVY_LOAD:
                # 30% chance: silently skip this agent's move
                if self._wrapper_rng.random() < self.heavy_load_skip_prob:
                    actions[i] = 0  # NOOP

            elif rtype == RobotType.FAST_LIGHT:
                # 50% chance: mark for bonus move after base step
                if self._wrapper_rng.random() < self.fast_light_bonus_prob:
                    bonus_agents.append(i)

            # BALANCED: no modification

        return actions, bonus_agents

    def _apply_bonus_moves(
        self,
        bonus_agents: List[int],
        base_obs,
        rewards: np.ndarray,
        done: bool,
        truncated: bool,
        info: dict,
    ):
        """
        Give fast_light agents a bonus extra move by calling env.step()
        again with only their actions; all other agents get NOOP.

        This is safe because RWARE's collision engine handles the full
        set of simultaneous actions — all-NOOP agents simply stay put.
        """
        # We need the original actions for the bonus agents, but we already
        # stepped. We'll repeat the FORWARD action (action=1) for bonus agents
        # as a simple "keep moving" heuristic. This gives them an extra cell
        # of movement in whatever direction they're facing.
        bonus_actions = [0] * self.n_agents  # all NOOP
        for idx in bonus_agents:
            if self.battery[idx] > 0.0:
                bonus_actions[idx] = 1  # Action.FORWARD

        # Snapshot queue again for bonus-step delivery detection
        old_queue_ids = set(s.id for s in self.base_env.request_queue)

        bonus_obs, bonus_rewards, bonus_done, bonus_truncated, bonus_info = \
            self.env.step(bonus_actions)

        bonus_rewards = np.array(bonus_rewards, dtype=np.float64)

        # Detect deliveries in bonus step and apply mismatch penalty
        new_queue_ids = set(s.id for s in self.base_env.request_queue)
        bonus_rewards = self._apply_mismatch_penalty(
            bonus_rewards, old_queue_ids, new_queue_ids
        )

        # Merge results: use bonus step's observations, accumulate rewards
        rewards = rewards + bonus_rewards
        done = done or bonus_done
        truncated = truncated or bonus_truncated

        return bonus_obs, rewards, done, truncated, bonus_info

    # ═══════════════════════════════════════════════════════════════════════
    # Task Type Management
    # ═══════════════════════════════════════════════════════════════════════

    def _assign_task_types_to_queue(self) -> None:
        """
        Assign a random TaskType to any shelf in the request queue
        that doesn't already have one in our task_type_map.
        """
        type_choices = list(TaskType)
        for shelf in self.base_env.request_queue:
            if shelf.id not in self.task_type_map:
                self.task_type_map[shelf.id] = type_choices[
                    self._wrapper_rng.integers(0, len(type_choices))
                ]

    def _compute_nearest_task_info(self, agent_idx: int) -> np.ndarray:
        """
        Compute the 5-element nearest-task info vector for an agent.

        Returns:
            np.ndarray of shape (5,):
              [0:3] — task type one-hot encoding
              [3]   — normalized priority weight (weight / max_weight)
              [4]   — normalized Manhattan distance (dist / max_dist)
        """
        agent = self.base_env.agents[agent_idx]
        ax, ay = agent.x, agent.y

        # Max possible Manhattan distance for normalization
        max_dist = (self.base_env.grid_size[0] + self.base_env.grid_size[1])

        best_dist = float("inf")
        best_shelf = None

        for shelf in self.base_env.request_queue:
            dist = abs(ax - shelf.x) + abs(ay - shelf.y)
            if dist < best_dist:
                best_dist = dist
                best_shelf = shelf

        result = np.zeros(5, dtype=np.float32)

        if best_shelf is not None and best_shelf.id in self.task_type_map:
            task_type = self.task_type_map[best_shelf.id]
            # One-hot encode task type (indices 0-2)
            result[task_type.value] = 1.0
            # Normalized weight (index 3)
            result[3] = self.task_weights[task_type] / self.max_task_weight
            # Normalized distance (index 4)
            result[4] = best_dist / max_dist if max_dist > 0 else 0.0

        return result

    # ═══════════════════════════════════════════════════════════════════════
    # Mismatch Penalty
    # ═══════════════════════════════════════════════════════════════════════

    def _apply_mismatch_penalty(
        self,
        rewards: np.ndarray,
        old_queue_ids: set,
        new_queue_ids: set,
    ) -> np.ndarray:
        """
        Detect which shelves were delivered (left old queue, not in new queue)
        and apply a penalty if the delivering robot's type doesn't match
        the task type.

        Returns:
            Modified rewards array.
        """
        # Shelves that were delivered = in old queue but replaced in new queue
        # RWARE replaces a delivered shelf with a new one at the same index,
        # so delivered IDs = old_ids - new_ids
        delivered_ids = old_queue_ids - new_queue_ids

        if not delivered_ids:
            return rewards

        for shelf_id in delivered_ids:
            # Find which agent is at a goal location carrying this shelf
            delivering_agent_idx = self._find_delivering_agent(shelf_id)
            if delivering_agent_idx is None:
                continue

            # Look up task type
            task_type = self.task_type_map.get(shelf_id)
            if task_type is None:
                continue

            # Check compatibility
            robot_type = self.robot_types[delivering_agent_idx]
            if COMPATIBILITY_MAP[robot_type] != task_type:
                rewards[delivering_agent_idx] -= self.mismatch_penalty

        return rewards

    def _find_delivering_agent(self, shelf_id: int) -> Optional[int]:
        """
        Find the index of the agent that delivered the given shelf.

        Checks each goal location for an agent that was carrying the shelf.
        Since the delivery already happened, the shelf is at a goal and
        the agent is at the same goal cell.
        """
        for goal_x, goal_y in self.base_env.goals:
            # Check if the delivering agent is at this goal
            # RWARE grid stores agent IDs as [_LAYER_AGENTS, y, x]
            # but goals are stored as (x, y) — careful with RWARE's
            # inconsistent (y,x) vs (x,y) conventions
            agent_id_at_goal = self.base_env.grid[0, goal_y, goal_x]  # _LAYER_AGENTS=0
            if agent_id_at_goal > 0:
                agent = self.base_env.agents[agent_id_at_goal - 1]
                # Check if this agent just delivered (has_delivered flag)
                # or is at the goal. We rely on the agent being at the goal
                # cell where the delivery was detected.
                return agent_id_at_goal - 1  # convert to 0-indexed
        return None

    # ═══════════════════════════════════════════════════════════════════════
    # Observation Building
    # ═══════════════════════════════════════════════════════════════════════

    def _extend_observations(self, base_obs) -> tuple:
        """
        Append heterogeneous features to each agent's base observation.

        Appended features (9 elements):
          [0:3] — robot type one-hot
          [3]   — normalized battery (0-1)
          [4:9] — nearest task info (type one-hot + weight + distance)
        """
        extended = []
        for i in range(self.n_agents):
            base = np.array(base_obs[i], dtype=np.float32)

            # Robot type one-hot (3 elements)
            type_onehot = np.zeros(3, dtype=np.float32)
            type_onehot[self.robot_types[i].value] = 1.0

            # Normalized battery (1 element)
            battery_norm = np.array(
                [self.battery[i] / 100.0], dtype=np.float32
            )

            # Nearest task info (5 elements)
            task_info = self._compute_nearest_task_info(i)

            # Concatenate: base(71) + type(3) + battery(1) + task(5) = 80
            obs = np.concatenate([base, type_onehot, battery_norm, task_info])
            extended.append(obs)

        return tuple(extended)

    # ═══════════════════════════════════════════════════════════════════════
    # Communication Helpers
    # ═══════════════════════════════════════════════════════════════════════

    def _update_comm_positions(self) -> None:
        """Sync agent positions to the communication channel."""
        positions = [
            (agent.x, agent.y) for agent in self.base_env.agents
        ]
        self.comm.update_positions(positions)

    # ═══════════════════════════════════════════════════════════════════════
    # Utility / Introspection
    # ═══════════════════════════════════════════════════════════════════════

    def get_robot_type(self, agent_idx: int) -> RobotType:
        """Get the robot type for a specific agent."""
        return self.robot_types[agent_idx]

    def get_battery(self, agent_idx: int) -> float:
        """Get the current battery level (0–100) for a specific agent."""
        return float(self.battery[agent_idx])

    def get_task_type(self, shelf_id: int) -> Optional[TaskType]:
        """Get the task type assigned to a specific shelf, if any."""
        return self.task_type_map.get(shelf_id)

    def is_compatible(self, robot_type: RobotType, task_type: TaskType) -> bool:
        """Check if a robot type is compatible with a task type."""
        return COMPATIBILITY_MAP[robot_type] == task_type
