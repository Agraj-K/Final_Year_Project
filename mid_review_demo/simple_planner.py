"""
simple_planner.py — Simplified A* Pathfinding for Mid-Review Demo

Delegates pathfinding to the robust grid A* implementation in main_simulation.py.
"""

import sys
import os

# Add parent directory to sys.path to ensure main_simulation is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_simulation import astar, path_to_action_queue
import numpy as np

def build_walkable_grid_simple(base_env):
    """Creates boolean walkable grid matrix for RWARE."""
    base = base_env.base_env if hasattr(base_env, "base_env") else base_env
    rows, cols = base.grid_size  # (height, width)
    walkable = np.ones((rows, cols), dtype=bool)

    # Shelf cells are NOT walkable (unless specifically target or starting cell)
    shelf_layer = base.grid[1]  # _LAYER_SHELFS
    walkable[shelf_layer > 0] = False
    return walkable

def plan_path_a_star(start_pos, goal_pos, grid_bounds, base_env=None):
    """
    Computes A* path from start_pos to goal_pos using main_simulation's grid planner.
    """
    if base_env is not None:
        walkable = build_walkable_grid_simple(base_env)
        path = astar(walkable, start_pos, goal_pos)
        if path is not None:
            return path
            
    return [start_pos]

def waypoints_to_actions(path, current_dir):
    """
    Converts path to action queue using main_simulation's path_to_action_queue.
    """
    if not path or len(path) <= 1:
        return []
    return path_to_action_queue(path, current_dir)
