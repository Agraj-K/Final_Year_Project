"""
simple_env.py — Simplified Environment Wrapper for Mid-Review Demo

This module wraps the RWARE (Robotic Warehouse) Gymnasium environment.
It provides:
  1. 2 Heterogeneous Robot Types:
     - Robot 0: FAST_LIGHT (drains 0.1 battery/step, fast movement)
     - Robot 1: HEAVY_LOAD (drains 0.4 battery/step, heavy carry capacity)
  2. Battery tracking (100.0 down to 0.0)
  3. Simple 80-dimensional observation vector per robot
"""

import gymnasium as gym
import numpy as np
from rware.warehouse import Warehouse, RewardType

class SimpleWarehouseEnv:
    def __init__(self, n_agents=2):
        self.n_agents = n_agents
        
        # Base RWARE warehouse environment (2-robot grid)
        self.base_env = Warehouse(
            shelf_columns=3,
            column_height=6,
            shelf_rows=1,
            n_agents=self.n_agents,
            msg_bits=0,
            sensor_range=1,
            request_queue_size=2,
            max_inactivity_steps=None,
            max_steps=500,
            reward_type=RewardType.INDIVIDUAL,
        )
        
        # Fixed Robot Types for 2-robot Mid-Review demo
        self.robot_type_names = ["FAST_LIGHT", "HEAVY_LOAD"]
        self.drain_rates = [0.1, 0.4]  # battery drain per step
        self.battery = np.array([100.0, 100.0], dtype=np.float32)

    def reset(self, seed=42):
        obs, info = self.base_env.reset(seed=seed)
        self.battery = np.array([100.0, 100.0], dtype=np.float32)
        return self._build_observations(obs), info

    def step(self, actions):
        # Enforce battery depletion constraint (if battery = 0, force NOOP action 0)
        valid_actions = []
        for i, a in enumerate(actions):
            if self.battery[i] <= 0:
                valid_actions.append(0)  # NOOP
            else:
                valid_actions.append(a)
                self.battery[i] = max(0.0, self.battery[i] - self.drain_rates[i])

        obs, rewards, done, truncated, info = self.base_env.step(valid_actions)
        return self._build_observations(obs), rewards, done, truncated, info

    def _build_observations(self, base_obs):
        """Builds 80-dim observation vector (71 base RWARE obs + 9 extra features)."""
        extended_obs = []
        for i in range(self.n_agents):
            b_obs = np.array(base_obs[i], dtype=np.float32)
            
            # Robot type one-hot (3 elements)
            type_onehot = np.zeros(3, dtype=np.float32)
            type_onehot[i % 3] = 1.0
            
            # Normalized battery (1 element)
            batt_norm = np.array([self.battery[i] / 100.0], dtype=np.float32)
            
            # Task vector (5 elements)
            task_vec = np.array([1.0, 0.0, 0.0, 1.0, 0.5], dtype=np.float32)
            
            # Combine features to 80 elements total
            full_obs = np.concatenate([b_obs, type_onehot, batt_norm, task_vec])
            extended_obs.append(full_obs)
            
        return extended_obs

    @property
    def unassigned_requests(self):
        """Returns requested shelf targets."""
        return list(self.base_env.request_queue)
