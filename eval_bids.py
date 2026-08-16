"""
eval_bids.py — Verification script for Phase 2

Loads the trained AgentNetwork and evaluates it in deterministic scenarios to prove
that the network has learned to differentiate bid values based on task suitability.

Scenario:
  We spawn a heavy_load robot in an environment. We manually inject two test cases:
  1. The nearest task is a heavy_shelf (Match)
  2. The nearest task is an urgent_light (Mismatch)

We assert that Bid(heavy_shelf) > Bid(urgent_light) on average.
"""

import os
import glob
import numpy as np
import torch
from network import AgentNetwork
from rware.warehouse import Warehouse, RewardType
from hetero_wrapper import HeterogeneousWarehouse, RobotType, TaskType


def main():
    device = torch.device("cpu")
    
    # 1. Find the most recent model
    model_files = glob.glob("models/ppo_hetero_*.pt")
    if not model_files:
        print("No trained models found in models/ directory. Run train_ppo.py first!")
        return
    latest_model = max(model_files, key=os.path.getctime)
    print(f"Loading model: {latest_model}")

    # 2. Setup Environment
    base_env = Warehouse(
        shelf_columns=1,
        column_height=3,
        shelf_rows=1,
        n_agents=1,  # just 1 agent for testing
        msg_bits=0,
        sensor_range=1,
        request_queue_size=1,
        max_inactivity_steps=None,
        max_steps=10,
        reward_type=RewardType.INDIVIDUAL,
    )
    env = HeterogeneousWarehouse(base_env)
    obs_dim = env.observation_space[0].shape[0]

    # 3. Load Network
    agent = AgentNetwork(obs_dim=obs_dim).to(device)
    agent.load_state_dict(torch.load(latest_model, map_location=device, weights_only=True))
    agent.eval()

    # 4. Evaluation Loop
    trials = 50
    bids_heavy_shelf = []
    bids_urgent_light = []

    print("\nRunning deterministic test scenarios...")

    for _ in range(trials):
        obs, _ = env.reset()
        
        # We need to manually manipulate the observation to simulate the scenarios.
        # The agent's type is at indices [71, 72, 73] (fast_light, heavy_load, balanced)
        # The nearest task type is at indices [75, 76, 77] (urgent_light, heavy_shelf, standard)
        
        # Force robot type to heavy_load (index 72)
        obs[0][71] = 0.0
        obs[0][72] = 1.0
        obs[0][73] = 0.0
        
        # We assume distance is valid, battery is full
        obs[0][74] = 1.0  # Battery
        obs[0][78] = 0.5  # Task weight
        obs[0][79] = 0.5  # Task distance

        # --- Scenario A: Task is heavy_shelf ---
        obs_a = np.copy(obs)
        obs_a[0][75] = 0.0  # urgent_light
        obs_a[0][76] = 1.0  # heavy_shelf
        obs_a[0][77] = 0.0  # standard
        
        with torch.no_grad():
            tensor_a = torch.tensor(obs_a, dtype=torch.float32).to(device)
            _, _, bid_a, _, _, _ = agent.get_action_and_value(tensor_a)
            bids_heavy_shelf.append(bid_a.item())

        # --- Scenario B: Task is urgent_light ---
        obs_b = np.copy(obs)
        obs_b[0][75] = 1.0  # urgent_light
        obs_b[0][76] = 0.0  # heavy_shelf
        obs_b[0][77] = 0.0  # standard
        
        with torch.no_grad():
            tensor_b = torch.tensor(obs_b, dtype=torch.float32).to(device)
            _, _, bid_b, _, _, _ = agent.get_action_and_value(tensor_b)
            bids_urgent_light.append(bid_b.item())

    # 5. Results
    avg_heavy = np.mean(bids_heavy_shelf)
    avg_light = np.mean(bids_urgent_light)

    print(f"\n--- RESULTS OVER {trials} TRIALS ---")
    print(f"Robot Type: HEAVY_LOAD")
    print(f"Average Bid for HEAVY_SHELF (Match):     {avg_heavy:.4f}")
    print(f"Average Bid for URGENT_LIGHT (Mismatch): {avg_light:.4f}")

    if avg_heavy > avg_light:
        print("\n✅ SUCCESS: The network successfully learned to bid higher for capability-matched tasks!")
    else:
        print("\n❌ FAILURE: The network did not learn the capability-mismatch distinction.")


if __name__ == "__main__":
    main()
