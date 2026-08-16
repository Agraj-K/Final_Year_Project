"""
network.py — Shared IPPO Network for Heterogeneous Warehouse Robots

This module defines the shared neural network architecture used by all robots.
Since we use Independent PPO (IPPO), every robot uses a copy of the exact same weights.
The robot type (fast_light, heavy_load, balanced) is passed as a one-hot vector within
the 80-dimensional observation, allowing the shared network to learn type-specific behaviors.

Architecture:
  - Input: 80 (Heterogeneous Observation)
  - Shared Base: 2 hidden layers (128 -> 128)
  - Output Heads:
      1. actor_move: logits for Discrete(5) movement actions.
      2. actor_bid: single value (Sigmoid 0-1) indicating task suitability.
      3. actor_comm: logit for Bernoulli(1) communication gate.
      4. critic: single value (Linear) for state value estimation.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli


class AgentNetwork(nn.Module):
    def __init__(self, obs_dim: int = 80, hidden_dim: int = 128):
        super().__init__()
        
        # Shared feature extractor
        self.shared_base = nn.Sequential(
            self._layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            self._layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        
        # ── Output Heads ───────────────────────────────────────────────
        
        # 1. Movement Action Logits (Discrete 5)
        # We initialize the policy head with std=0.01 so actions start roughly uniform
        self.actor_move = self._layer_init(nn.Linear(hidden_dim, 5), std=0.01)
        
        # 2. Bid Value (Sigmoid 0-1)
        # Note: Bids are auxiliary outputs; we treat them as continuous bounded values
        # They don't have an entropy component in the standard PPO action space, but
        # we can supervise them or just let them evolve if they affect the reward.
        # Wait, if they affect downstream mechanisms in Phase 3, they will get gradients.
        self.actor_bid = nn.Sequential(
            self._layer_init(nn.Linear(hidden_dim, 1), std=0.01),
            nn.Sigmoid()
        )
        
        # 3. Communication Gate Logit (Binary)
        # Outputs a single logit which we treat as a Bernoulli distribution.
        self.actor_comm = self._layer_init(nn.Linear(hidden_dim, 1), std=0.01)
        
        # 4. Critic (State Value)
        self.critic = self._layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        """Orthogonal initialization commonly used in PPO."""
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        hidden = self.shared_base(x)
        return self.critic(hidden)

    def get_action_and_value(self, x, move_action=None, comm_action=None):
        hidden = self.shared_base(x)
        
        # Movement
        move_logits = self.actor_move(hidden)
        move_probs = Categorical(logits=move_logits)
        if move_action is None:
            move_action = move_probs.sample()
            
        # Communication
        comm_logits = self.actor_comm(hidden)
        comm_probs = Bernoulli(logits=comm_logits)
        if comm_action is None:
            comm_action = comm_probs.sample()
            
        # Bid Value (Deterministic output, bounded 0-1)
        bid_value = self.actor_bid(hidden).squeeze(-1)
        
        # Critic Value
        value = self.critic(hidden).squeeze(-1)
        
        # Calculate Log Probs and Entropy
        # We treat the joint policy as independent distributions:
        # P(move, comm) = P(move) * P(comm) => log P = log P(move) + log P(comm)
        logprob_move = move_probs.log_prob(move_action)
        entropy_move = move_probs.entropy()
        
        # The Bernoulli log_prob takes the action.
        # Note: Bernoulli sample() returns float tensors (e.g. 0.0 or 1.0)
        # so comm_action needs to be appropriately typed.
        logprob_comm = comm_probs.log_prob(comm_action.float()).squeeze(-1)
        entropy_comm = comm_probs.entropy().squeeze(-1)
        
        total_logprob = logprob_move + logprob_comm
        total_entropy = entropy_move + entropy_comm
        
        return move_action, comm_action, bid_value, total_logprob, total_entropy, value
