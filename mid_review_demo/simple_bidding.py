"""
simple_bidding.py — Simplified Bidding Network for Mid-Review Demo

This module defines a small, clean PyTorch Neural Network for task bidding.
Inputs:
  - 80-dimensional observation vector (includes position, battery, robot type)
Outputs:
  - Bid Value: Continuous scalar between 0.0 and 1.0 representing how suitable
    the robot is to take on the target shelf task.
"""

import torch
import torch.nn as nn

class SimpleBiddingNetwork(nn.Module):
    def __init__(self, obs_dim=80):
        super().__init__()
        # Backbone architecture: Linear(80->64) -> Tanh -> Linear(64->32) -> Tanh
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
        )
        # Bid Head: Linear(32->1) -> Sigmoid (outputs value between 0.0 and 1.0)
        self.bid_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, obs_tensor):
        """Passes observation through backbone and outputs continuous bid value."""
        features = self.backbone(obs_tensor)
        bid = self.bid_head(features)
        return bid

def evaluate_auction(bids, robot_types, batteries):
    """
    Evaluates bids from robots and selects the winning robot.
    In case of close bids, tie-breaks using battery levels.
    
    Returns:
        winner_index (int): Index of the robot that wins the task.
    """
    best_bid = -1.0
    winner = 0
    for i, bid in enumerate(bids):
        # Weighted bid score considering raw bid value + battery availability
        score = bid * 0.7 + (batteries[i] / 100.0) * 0.3
        if score > best_bid:
            best_bid = score
            winner = i
    return winner
