"""
simple_planner.py — Self-Contained A* Pathfinding for Mid-Review Demo

Contains all pathfinding logic locally:
  - A* grid search with Manhattan heuristic
  - Direction helpers for RWARE turn actions
  - Waypoint-to-action-queue conversion
  - Walkable grid construction from RWARE shelf layer

No external imports from main_simulation.py — fully self-contained.
"""

import heapq
import numpy as np
from typing import Dict, List, Optional, Tuple
from rware.warehouse import Direction, Action


# ═══════════════════════════════════════════════════════════════════════════════
# Direction Helpers
# ═══════════════════════════════════════════════════════════════════════════════

_DIR_DELTAS: Dict[Direction, Tuple[int, int]] = {
    Direction.UP:    (0, -1),   # (dx, dy)
    Direction.DOWN:  (0,  1),
    Direction.LEFT:  (-1, 0),
    Direction.RIGHT: (1,  0),
}


def direction_for_delta(dx: int, dy: int) -> Optional[Direction]:
    """Get the RWARE Direction for a given (dx, dy) step."""
    for d, (ddx, ddy) in _DIR_DELTAS.items():
        if ddx == dx and ddy == dy:
            return d
    return None


def turn_actions_to_face(current_dir: Direction, target_dir: Direction) -> List[int]:
    """
    Return a list of turn actions (LEFT=2, RIGHT=3) to rotate from
    current_dir to target_dir. Returns [] if already facing correctly.
    """
    if current_dir == target_dir:
        return []

    # Clockwise order: UP -> RIGHT -> DOWN -> LEFT -> UP
    cw_order = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
    ci = cw_order.index(current_dir)
    ti = cw_order.index(target_dir)

    # Number of clockwise turns
    cw_turns = (ti - ci) % 4
    # Number of counter-clockwise turns
    ccw_turns = (ci - ti) % 4

    if cw_turns <= ccw_turns:
        return [Action.RIGHT.value] * cw_turns  # RIGHT = clockwise turn
    else:
        return [Action.LEFT.value] * ccw_turns  # LEFT = counter-clockwise turn


# ═══════════════════════════════════════════════════════════════════════════════
# Walkable Grid Builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_walkable_grid_simple(base_env):
    """Creates boolean walkable grid matrix for RWARE."""
    base = base_env.base_env if hasattr(base_env, "base_env") else base_env
    rows, cols = base.grid_size  # (height, width)
    walkable = np.ones((rows, cols), dtype=bool)

    # Shelf cells are NOT walkable (unless specifically target or starting cell)
    shelf_layer = base.grid[1]  # _LAYER_SHELFS
    walkable[shelf_layer > 0] = False
    return walkable


# ═══════════════════════════════════════════════════════════════════════════════
# A* Pathfinding
# ═══════════════════════════════════════════════════════════════════════════════

def astar(
    walkable: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    """
    A* pathfinding on a 2D grid.

    Args:
        walkable: 2D boolean array where True = walkable cell.
        start: (x, y) starting coordinate.
        goal: (x, y) target coordinate.

    Returns:
        List of (x, y) waypoints from start to goal, or None if no path exists.
    """
    sx, sy = start
    gx, gy = goal
    rows, cols = walkable.shape

    def heuristic(x, y):
        return abs(x - gx) + abs(y - gy)

    # Priority queue: (f_score, counter, x, y)
    counter = 0
    open_set = [(heuristic(sx, sy), counter, sx, sy)]
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score: Dict[Tuple[int, int], float] = {(sx, sy): 0}

    while open_set:
        _, _, cx, cy = heapq.heappop(open_set)

        if (cx, cy) == (gx, gy):
            path = [(cx, cy)]
            while (cx, cy) in came_from:
                cx, cy = came_from[(cx, cy)]
                path.append((cx, cy))
            path.reverse()
            return path

        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx_, ny_ = cx + dx, cy + dy
            if 0 <= nx_ < cols and 0 <= ny_ < rows:
                # Allow walking into the goal/start cell even if it has a shelf
                if not walkable[ny_, nx_] and (nx_, ny_) != (gx, gy) and (nx_, ny_) != (sx, sy):
                    continue
                new_g = g_score[(cx, cy)] + 1
                if new_g < g_score.get((nx_, ny_), float("inf")):
                    g_score[(nx_, ny_)] = new_g
                    f = new_g + heuristic(nx_, ny_)
                    counter += 1
                    heapq.heappush(open_set, (f, counter, nx_, ny_))
                    came_from[(nx_, ny_)] = (cx, cy)

    return None  # No path found


# ═══════════════════════════════════════════════════════════════════════════════
# Path-to-Action Converter
# ═══════════════════════════════════════════════════════════════════════════════

def path_to_action_queue(
    path: List[Tuple[int, int]],
    start_dir: Direction,
) -> List[int]:
    """
    Convert a list of (x, y) waypoints into a queue of RWARE discrete actions.

    For each consecutive pair of waypoints:
      1. Calculate the direction delta (dx, dy)
      2. Determine which Direction the robot needs to face
      3. Insert turn actions (LEFT/RIGHT) to rotate to that direction
      4. Insert a FORWARD action to move one cell

    Args:
        path: List of (x, y) coordinate waypoints from A*.
        start_dir: The robot's current facing Direction.

    Returns:
        List of integer action codes (FORWARD=1, LEFT=2, RIGHT=3).
    """
    actions = []
    current_dir = start_dir

    for i in range(len(path) - 1):
        cx, cy = path[i]
        nx_, ny_ = path[i + 1]
        dx = nx_ - cx
        dy = ny_ - cy

        target_dir = direction_for_delta(dx, dy)
        if target_dir is None:
            continue

        turns = turn_actions_to_face(current_dir, target_dir)
        actions.extend(turns)
        actions.append(Action.FORWARD.value)
        current_dir = target_dir

    return actions


# ═══════════════════════════════════════════════════════════════════════════════
# Legacy wrapper functions (kept for compatibility)
# ═══════════════════════════════════════════════════════════════════════════════

def plan_path_a_star(start_pos, goal_pos, grid_bounds, base_env=None):
    """
    Computes A* path from start_pos to goal_pos.
    """
    if base_env is not None:
        walkable = build_walkable_grid_simple(base_env)
        path = astar(walkable, start_pos, goal_pos)
        if path is not None:
            return path

    return [start_pos]


def waypoints_to_actions(path, current_dir):
    """
    Converts path to action queue.
    """
    if not path or len(path) <= 1:
        return []
    return path_to_action_queue(path, current_dir)
