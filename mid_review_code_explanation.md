# S7 Mid-Review Warehouse Simulation Codebase: Line-by-Line Master Explanation

This document contains a line-by-line, block-by-block technical breakdown of all files in the `mid_review_demo/` codebase:
1. **[`simple_env.py`](file:///c:/Users/agraj/OneDrive/Documents/College/Final%20Year%20Project/mid_review_demo/simple_env.py)**
2. **[`simple_bidding.py`](file:///c:/Users/agraj/OneDrive/Documents/College/Final%20Year%20Project/mid_review_demo/simple_bidding.py)**
3. **[`simple_planner.py`](file:///c:/Users/agraj/OneDrive/Documents/College/Final%20Year%20Project/mid_review_demo/simple_planner.py)**
4. **[`run_demo.py`](file:///c:/Users/agraj/OneDrive/Documents/College/Final%20Year%20Project/mid_review_demo/run_demo.py)**

---

## 1. `simple_env.py`

This module wraps the Robotic Warehouse (RWARE) Gymnasium environment to support battery levels, heterogeneous agent archetypes, and extended state vectors.

### Line-by-Line Explanation

```python
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
```
* **Lines 1-11:** Docstring describing the scope of this file: 2 agents, battery depletion tracking, and observation vectors.

```python
import gymnasium as gym
import numpy as np
from rware.warehouse import Warehouse, RewardType
```
* **Lines 13-15:** Imports:
  * `gymnasium`: Base interface for reinforcment learning environments.
  * `numpy` (`np`): For multi-dimensional vector array mathematics.
  * `Warehouse` and `RewardType`: Base class and reward structure from RWARE packages.

```python
class SimpleWarehouseEnv:
    def __init__(self, n_agents=2):
        self.n_agents = n_agents
```
* **Lines 17-19:** Defines class wrapper `SimpleWarehouseEnv`. Constructor default limits agents count to 2.

```python
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
```
* **Lines 22-33:** Instantiates the base RWARE grid:
  * `shelf_columns=3`, `column_height=6`, `shelf_rows=1`: Defines shelf rack grid layouts.
  * `n_agents=self.n_agents`: Configured to 2 robots.
  * `msg_bits=0`: Disables RWARE's built-in raw communication channel in favor of our custom comm channel.
  * `sensor_range=1`: Robots sense adjacent 3x3 layout cells.
  * `request_queue_size=2`: Parallel tasks generated at any time.
  * `max_steps=500`: Maximum execution steps before truncation.
  * `reward_type=RewardType.INDIVIDUAL`: Rewards calculated per agent.

```python
        # Fixed Robot Types for 2-robot Mid-Review demo
        self.robot_type_names = ["FAST_LIGHT", "HEAVY_LOAD"]
        self.drain_rates = [0.1, 0.4]  # battery drain per step
        self.battery = np.array([100.0, 100.0], dtype=np.float32)
```
* **Lines 36-38:** Fleet Heterogeneity Definition:
  * `FAST_LIGHT`: Drains battery at `0.1` per step.
  * `HEAVY_LOAD`: Drains battery at `0.4` per step.
  * `self.battery`: Resets battery tracking values to `100.0` for both agents.

```python
    def reset(self, seed=42):
        obs, info = self.base_env.reset(seed=seed)
        self.battery = np.array([100.0, 100.0], dtype=np.float32)
        return self._build_observations(obs), info
```
* **Lines 40-43:** `reset` method:
  * Initializes the underlying gym environment with a deterministic seed.
  * Refills agent batteries back to `100.0%`.
  * Computes and returns custom 80-dimensional observation state vectors.

```python
    def step(self, actions):
        # Enforce battery depletion constraint (if battery = 0, force NOOP action 0)
        valid_actions = []
        for i, a in enumerate(actions):
            if self.battery[i] <= 0:
                valid_actions.append(0)  # NOOP
            else:
                valid_actions.append(a)
                self.battery[i] = max(0.0, self.battery[i] - self.drain_rates[i])
```
* **Lines 45-53:** **Battery Depletion Enforcer**:
  * Loops over actions of both agents.
  * Checks if `battery[i] <= 0`. If depleted, appends `0` (`NOOP` action code) preventing further movements.
  * If active, retains action and subtracts `drain_rates[i]` (0.1 or 0.4), bounding battery state to `0.0` minimum.

```python
        obs, rewards, done, truncated, info = self.base_env.step(valid_actions)
        return self._build_observations(obs), rewards, done, truncated, info
```
* **Lines 55-56:** Passes sanitized actions into base RWARE step function, reads standard environment results, and formats the output.

```python
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
```
* **Lines 58-78:** **Observation Expansion**:
  * Extracts RWARE's raw 71-dim agent state (`b_obs`).
  * Creates a 3-element one-hot vector representation of agent archetype.
  * Creates a 1-element normalized battery indicator ($\frac{\text{battery}}{100.0}$).
  * Creates a 5-element task vector (contains: task priority weight, target distance, type).
  * Concatenates all segments (`71 + 3 + 1 + 5 = 80` features) into a single 1D numpy array per agent.

```python
    @property
    def unassigned_requests(self):
        """Returns requested shelf targets."""
        return list(self.base_env.request_queue)
```
* **Lines 80-84:** Getter property that returns the list of active requested shelves currently waiting to be picked up in RWARE's queue.

---

## 2. `simple_bidding.py`

This module defines the PyTorch Neural Network model used by warehouse agents to calculate bids for incoming target shelves.

### Line-by-Line Explanation

```python
"""
simple_bidding.py — Simplified Bidding Network for Mid-Review Demo

This module defines a small, clean PyTorch Neural Network for task bidding.
Inputs:
  - 80-dimensional observation vector (includes position, battery, robot type)
Outputs:
  - Bid Value: Continuous scalar between 0.0 and 1.0 representing how suitable
    the robot is to take on the target shelf task.
"""
```
* **Lines 1-10:** File introduction docstring.

```python
import torch
import torch.nn as nn
```
* **Lines 12-13:** Imports:
  * `torch`: Core PyTorch library.
  * `nn`: PyTorch module classes (Linear, Sigmoid, Sequential, etc.).

```python
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
```
* **Lines 15-24:** Defines PyTorch neural model constructor:
  * Inherits from `nn.Module`.
  * `self.backbone`: Multi-layer perceptron mapping inputs from `80` features down to `32` feature embeddings.
  * `Tanh` activation function is used to handle normalized negative/positive bounds.

```python
        # Bid Head: Linear(32->1) -> Sigmoid (outputs value between 0.0 and 1.0)
        self.bid_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
```
* **Lines 26-29:** `bid_head`: Passes the feature embedding through a single linear unit followed by a `Sigmoid` activation function to compress output scores strictly between `0.0` and `1.0`.

```python
    def forward(self, obs_tensor):
        """Passes observation through backbone and outputs continuous bid value."""
        features = self.backbone(obs_tensor)
        bid = self.bid_head(features)
        return bid
```
* **Lines 31-35:** `forward` pass definition: Passes inputs through the backbone architecture to generate features, which are then passed to the bid head to output a continuous task suitability score.

```python
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
```
* **Lines 37-54:** **Decentralized Auction Evaluator**:
  * Evaluates computed bids to assign tasks to robots.
  * Formulates combined scores:
    $$\text{Score} = 70\% \cdot \text{Raw Bid} + 30\% \cdot \text{Normalized Battery}$$
  * Loops over available bids to find the highest score, returning the index of the winning robot.

---

## 3. `simple_planner.py`

This module is **fully self-contained** — it embeds all pathfinding logic locally (A* algorithm, direction helpers, turn calculators, and path-to-action converters) without any external imports from `main_simulation.py`.

### Line-by-Line Explanation

```python
"""
simple_planner.py — Self-Contained A* Pathfinding for Mid-Review Demo

Contains all pathfinding logic locally:
  - A* grid search with Manhattan heuristic
  - Direction helpers for RWARE turn actions
  - Waypoint-to-action-queue conversion
  - Walkable grid construction from RWARE shelf layer

No external imports from main_simulation.py — fully self-contained.
"""
```
* **Lines 1-10:** Module docstring. Declares that this file is self-contained with no dependencies outside the `mid_review_demo/` folder.

```python
import heapq
import numpy as np
from typing import Dict, List, Optional, Tuple
from rware.warehouse import Direction, Action
```
* **Lines 12-15:** Imports:
  * `heapq`: Python's built-in min-heap priority queue, used by the A* algorithm to always expand the lowest-cost node first.
  * `numpy` (`np`): For creating and manipulating 2D boolean grid arrays.
  * `typing`: Type hints for function signatures.
  * `Direction` and `Action`: RWARE enumerations for robot facing directions (`UP`, `DOWN`, `LEFT`, `RIGHT`) and discrete actions (`NOOP=0`, `FORWARD=1`, `LEFT=2`, `RIGHT=3`, `TOGGLE_LOAD=4`).

```python
_DIR_DELTAS: Dict[Direction, Tuple[int, int]] = {
    Direction.UP:    (0, -1),   # (dx, dy)
    Direction.DOWN:  (0,  1),
    Direction.LEFT:  (-1, 0),
    Direction.RIGHT: (1,  0),
}
```
* **Lines 23-28:** **Direction-to-Delta Mapping.** Maps each RWARE `Direction` enum value to its corresponding `(dx, dy)` grid movement vector. For example, `Direction.UP` means moving in the negative Y direction `(0, -1)` on the grid.

```python
def direction_for_delta(dx: int, dy: int) -> Optional[Direction]:
    """Get the RWARE Direction for a given (dx, dy) step."""
    for d, (ddx, ddy) in _DIR_DELTAS.items():
        if ddx == dx and ddy == dy:
            return d
    return None
```
* **Lines 31-36:** **Reverse Delta Lookup.** Given a movement delta `(dx, dy)` between two consecutive waypoints, returns the `Direction` the robot needs to face. For example, `(1, 0)` → `Direction.RIGHT`. Returns `None` if the delta doesn't match any valid direction (e.g., diagonal movement).

```python
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
```
* **Lines 39-60:** **Minimum Turn Calculator.** Computes the fewest turn actions to rotate from the robot's current facing direction to the target direction:
  * Defines a clockwise direction order: `UP → RIGHT → DOWN → LEFT`.
  * Calculates both clockwise and counter-clockwise turn counts using modular arithmetic.
  * Returns the shorter rotation sequence. For example, turning from `UP` to `LEFT` takes 1 counter-clockwise turn (`[LEFT]`) rather than 3 clockwise turns.

```python
def build_walkable_grid_simple(base_env):
    """Creates boolean walkable grid matrix for RWARE."""
    base = base_env.base_env if hasattr(base_env, "base_env") else base_env
    rows, cols = base.grid_size  # (height, width)
    walkable = np.ones((rows, cols), dtype=bool)

    # Shelf cells are NOT walkable (unless specifically target or starting cell)
    shelf_layer = base.grid[1]  # _LAYER_SHELFS
    walkable[shelf_layer > 0] = False
    return walkable
```
* **Lines 67-77:** **Obstacle Grid Builder:**
  * Resolves the base RWARE environment object.
  * Creates a 2D boolean matrix (`walkable`) with all cells initialized to `True` (traversable).
  * Extracts RWARE's internal shelf grid layer (layer index `1` stores shelf positions).
  * Marks all cells containing shelves (`shelf_layer > 0`) as `False` (unwalkable obstacles), forcing path calculations to navigate around shelf racks.

```python
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
```
* **Lines 84-104:** **A* Algorithm — Setup:**
  * Accepts a walkable boolean grid, start coordinate, and goal coordinate.
  * Unpacks start and goal into individual x, y values.
  * Defines the **Manhattan distance heuristic**: `|x - gx| + |y - gy|`. This is admissible (never overestimates) for grid movement, guaranteeing optimal paths.

```python
    # Priority queue: (f_score, counter, x, y)
    counter = 0
    open_set = [(heuristic(sx, sy), counter, sx, sy)]
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score: Dict[Tuple[int, int], float] = {(sx, sy): 0}
```
* **Lines 106-110:** **A* Data Structures:**
  * `open_set`: Min-heap priority queue storing `(f_score, counter, x, y)`. The `counter` breaks ties between equal f-scores to ensure deterministic ordering.
  * `came_from`: Dictionary mapping each cell to its predecessor, used to reconstruct the final path.
  * `g_score`: Dictionary tracking the cheapest known cost to reach each cell from the start.

```python
    while open_set:
        _, _, cx, cy = heapq.heappop(open_set)

        if (cx, cy) == (gx, gy):
            path = [(cx, cy)]
            while (cx, cy) in came_from:
                cx, cy = came_from[(cx, cy)]
                path.append((cx, cy))
            path.reverse()
            return path
```
* **Lines 112-121:** **A* Main Loop — Goal Check & Path Reconstruction:**
  * Pops the lowest f-score node from the priority queue.
  * If the current node is the goal, reconstructs the path by backtracking through `came_from` and returns the reversed list of waypoints.

```python
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
```
* **Lines 123-137:** **A* Main Loop — Neighbor Expansion:**
  * Explores 4-directional neighbors (up, down, left, right).
  * Checks grid bounds to prevent out-of-range access.
  * **Critical Rule (Line 127):** Skips unwalkable cells (shelves) UNLESS the cell is the goal or start position. This allows robots to walk onto a shelf cell to pick it up.
  * Calculates the tentative g-score (`new_g = current cost + 1` since each grid step costs 1).
  * If `new_g` is cheaper than any previously known route to that neighbor, updates `g_score`, calculates `f = g + heuristic`, pushes the neighbor onto the priority queue, and records the predecessor in `came_from`.
  * Returns `None` if the open set is exhausted without finding the goal (no valid path exists).

```python
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
```
* **Lines 144-176:** **Path-to-Action Queue Converter.** Translates A* coordinate waypoints into RWARE discrete action sequences:
  * Iterates over consecutive pairs of waypoints.
  * For each pair, calculates the movement delta `(dx, dy)`.
  * Looks up the `Direction` the robot must face using `direction_for_delta()`.
  * Generates the minimum turn actions using `turn_actions_to_face()` and appends them.
  * Appends `Action.FORWARD` to move one cell in that direction.
  * Updates `current_dir` to track the robot's new facing direction.
  * **Example:** Path `[(2,3), (3,3), (3,2)]` with robot facing `UP`:
    1. `(2,3)→(3,3)`: `dx=1, dy=0` → need `RIGHT`. Currently `UP` → 1 clockwise turn → `[RIGHT_TURN, FORWARD]`
    2. `(3,3)→(3,2)`: `dx=0, dy=-1` → need `UP`. Currently `RIGHT` → 3 clockwise or 1 counter-clockwise → `[LEFT_TURN, FORWARD]`
    * Final queue: `[RIGHT_TURN, FORWARD, LEFT_TURN, FORWARD]`

```python
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
```
* **Lines 183-202:** **Legacy Wrapper Functions** (kept for backward compatibility):
  * `plan_path_a_star()`: Convenience function that builds the walkable grid and runs A* in one call. Returns `[start_pos]` as a fallback if no path is found.
  * `waypoints_to_actions()`: Thin wrapper around `path_to_action_queue()` with an empty-path guard.

---

## 4. `run_demo.py`

This is the main visualization, event-handling, and state-machine runner that displays the 2 heterogeneous robots, their battery consumption, auctions, and collision priority management in a Pygame GUI.

### Line-by-Line Explanation

```python
"""
run_demo.py — Standalone Mid-Review Visual Simulation Demo (2 Robots + Decentralized Bidding)

Runs a 2-robot Pygame visual simulation showing:
  - Heterogeneous Robots (Robot 0: FAST_LIGHT, Robot 1: HEAVY_LOAD)
  - Neural Network Task Bidding
  - A* Path Navigation & Shelf Pickups, Delivery & Returns
  - Collision Priority Yielding & Dynamic Task Release
  - Live HUD with step count, battery levels, and event log

Controls:
  SPACE      : Pause / Resume simulation
  RIGHT ARROW: Advance single step (when paused)
  Q or ESC   : Exit simulation
"""
```
* **Lines 1-15:** Execution instructions, features, and keyboard controls.

```python
import os
import sys
import torch
import pygame
import numpy as np
```
* **Lines 17-21:** Standard library and graphics library imports.

```python
# Ensure mid_review_demo directory is in Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_env import SimpleWarehouseEnv
from simple_bidding import SimpleBiddingNetwork, evaluate_auction
from simple_planner import build_walkable_grid_simple, astar, path_to_action_queue
from rware.warehouse import Action, Direction
```
* **Lines 23-29:** Module imports — all from **local files within `mid_review_demo/`**. No external imports from outside the folder:
  * `SimpleWarehouseEnv`: Battery-tracking heterogeneous environment wrapper (from `simple_env.py`).
  * `SimpleBiddingNetwork`, `evaluate_auction`: Neural bid network and auction evaluator (from `simple_bidding.py`).
  * `build_walkable_grid_simple`, `astar`, `path_to_action_queue`: Grid builder, A* pathfinder, and action converter (from `simple_planner.py`).
  * `Action`, `Direction`: RWARE discrete action and direction enums (from the installed `rware` package).

```python
# ── Colors ─────────────────────────────────────────────────────────────────────
BG_COLOR       = (240, 243, 246)
GRID_COLOR     = (210, 215, 220)
ROBOT_0_COLOR  = (41, 128, 185)    # Blue: FAST_LIGHT
ROBOT_1_COLOR  = (211, 84, 0)      # Orange: HEAVY_LOAD
SHELF_COLOR    = (241, 196, 15)    # Yellow: Requested Shelf
DELIVERY_COLOR = (26, 188, 156)    # Teal: Goal Delivery Zone
TEXT_DARK      = (44, 62, 80)
PANEL_BG       = (255, 255, 255)
```
* **Lines 33-41:** Pygame color constants.

```python
class RobotPhase:
    IDLE           = 0
    MOVING_TO_SHELF = 1
    PICKING_UP     = 2
    MOVING_TO_GOAL = 3
    RETURNING_SHELF = 4
    DROPPING_OFF   = 5
```
* **Lines 43-49:** Robot operational state machine constants.

```python
def plan_safe(start, goal, base_env, other_pos=None):
    """
    A* path from start→goal, optionally treating other_pos as a temporary obstacle.
    Falls back to ignoring the obstacle if no path found.
    """
    walkable = build_walkable_grid_simple(base_env)
    # Temporarily block other robot's cell (unless it IS the goal or start)
    if other_pos and other_pos != goal and other_pos != start:
        ox, oy = other_pos
        walkable[oy, ox] = False
    path = astar(walkable, start, goal)
    if path:
        return path
    # Retry without blocking other robot (fallback)
    walkable2 = build_walkable_grid_simple(base_env)
    path2 = astar(walkable2, start, goal)
    return path2 if path2 else [start]
```
* **Lines 54-70:** **Dynamic Obstacle Path Planner**:
  * Constructs the default walkable grid map.
  * If a peer's location (`other_pos`) is valid, it marks that cell as unwalkable (`False`) on the grid map.
  * Computes the A* path. If no path is found due to grid congestion, it recalculates the path without blocking the peer to avoid deadlocks.

```python
def make_actions(path, current_dir):
    """Wrapper: convert waypoint path → action list."""
    if not path or len(path) <= 1:
        return []
    return path_to_action_queue(path, current_dir)
```
* **Lines 73-77:** Simple wrapper translating path coordinate lists to discrete agent action vectors.

```python
def other_robot_pos(env, i):
    """Return (x,y) of the OTHER robot (not robot i)."""
    j = 1 - i
    a = env.base_env.agents[j]
    return (a.x, a.y)
```
* **Lines 80-84:** Helper function to query the grid position of the other robot. Since $N=2$, the peer index is calculated as $1 - i$.

```python
def main():
    pygame.init()
    pygame.font.init()

    font_large = pygame.font.SysFont("Arial", 18, bold=True)
    font_small = pygame.font.SysFont("Arial", 13)
```
* **Lines 89-94:** Runs Pygame initialization and loads UI fonts.

```python
    env = SimpleWarehouseEnv(n_agents=2)
    obs, info = env.reset(seed=42)

    bidding_net = SimpleBiddingNetwork(obs_dim=80)
    bidding_net.eval()
```
* **Lines 96-100:** Initializes the environment, generates observations, instantiates the neural network, and sets it to evaluation mode (disabling gradient calculation).

```python
    CELL_SIZE   = 48
    grid_width, grid_height = env.base_env.grid_size
    sidebar_width = 320

    screen_width  = grid_width * CELL_SIZE + sidebar_width
    screen_height = max(grid_height * CELL_SIZE + 60, 450)

    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Mid-Review Demo — Multi-Robot Warehouse Coordination")
    clock = pygame.time.Clock()
```
* **Lines 102-111:** Sets window bounds, sets window title headers, and initializes the clock to regulate the simulation speed.

```python
    paused = False
    step_count = 0
    event_logs = ["Simulation Started.", "R0: FAST_LIGHT  R1: HEAVY_LOAD"]

    # Per-robot state
    phases              = [RobotPhase.IDLE, RobotPhase.IDLE]
    action_queues       = [[], []]
    assigned_shelves    = [None, None]
    shelf_orig_positions = [None, None]
    deliveries_count    = 0
    pickup_attempts     = [0, 0]   # timeout for TOGGLE_LOAD attempts

    # Stuck-detection
    last_positions  = [(env.base_env.agents[i].x, env.base_env.agents[i].y) for i in range(2)]
    stuck_counters  = [0, 0]
    STUCK_THRESHOLD = 8   # steps without moving before forced replan
```
* **Lines 113-128:** Initializes state variables, logs, robot queues, task parameters, and stuck detection counters.

```python
    running = True
    while running:
        # ── Event handling ──────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    event_logs.append(f"[{step_count}] {'PAUSED' if paused else 'RESUMED'}")
```
* **Lines 130-141:** GUI event loop handling window closing, escape keys, and pause/resume triggers.

```python
        if not paused:
            # ── 0. Update stuck counters ────────────────────────────────────────
            for i in range(2):
                cur = (env.base_env.agents[i].x, env.base_env.agents[i].y)
                if cur == last_positions[i] and phases[i] != RobotPhase.IDLE:
                    stuck_counters[i] += 1
                else:
                    stuck_counters[i] = 0
                last_positions[i] = cur

                if stuck_counters[i] >= STUCK_THRESHOLD:
                    # Force a replan — clear the queue so state machine replans next tick
                    action_queues[i] = []
                    stuck_counters[i] = 0
                    event_logs.append(f"[{step_count}] R{i} stuck → replan")
```
* **Lines 143-158:** **Dynamic Stuck Detection**:
  * Tracks if a robot's position remains unchanged between steps while it has an active task.
  * If stuck for 8 consecutive steps, it flushes the action queue to force a path recalculation around dynamic obstacles.

```python
            # ── 1. Auction for IDLE robots ──────────────────────────────────────
            requests = env.unassigned_requests
            assigned_shelf_objs = set(s for s in assigned_shelves if s is not None)
            unassigned = [r for r in requests if r not in assigned_shelf_objs]

            for req in unassigned:
                idle_robots = [i for i in range(2)
                               if phases[i] == RobotPhase.IDLE and assigned_shelves[i] is None]
                if not idle_robots:
                    break

                with torch.no_grad():
                    bids = []
                    for i in range(env.n_agents):
                        ob = torch.tensor(obs[i], dtype=torch.float32).unsqueeze(0)
                        bid = bidding_net(ob).item() + float(np.random.uniform(0, 0.15))
                        bids.append(bid)
```
* **Lines 159-175:** **Auction Bid Generation**:
  * Scans for unassigned tasks in the queue.
  * Filters for idle robots available to bid.
  * Runs the observation tensors through the neural network under `no_grad()` to compute raw capability bids. A small random noise factor is added to simulate local bidding variations and break ties.

```python
                winner = evaluate_auction(bids, env.robot_type_names, env.battery)
                if winner not in idle_robots:
                    winner = idle_robots[0]

                assigned_shelves[winner]     = req
                shelf_orig_positions[winner] = (int(req.x), int(req.y))
                phases[winner]               = RobotPhase.MOVING_TO_SHELF
                pickup_attempts[winner]      = 0

                event_logs.append(
                    f"[{step_count}] Auction→R{winner} "
                    f"(R0={bids[0]:.2f}, R1={bids[1]:.2f})"
                )

                agent = env.base_env.agents[winner]
                path  = plan_safe((agent.x, agent.y),
                                  (int(req.x), int(req.y)),
                                  env.base_env,
                                  other_pos=other_robot_pos(env, winner))
                action_queues[winner] = make_actions(path, agent.dir)
```
* **Lines 177-197:** **Auction Task Assignment**:
  * Evaluates bids using the scoring formula.
  * Assigns the target shelf, logs original coordinates, sets state to `MOVING_TO_SHELF`, and logs the event.
  * Plans a collision-free path to the shelf using the `plan_safe` helper.

```python
            # ── 2. State machine ────────────────────────────────────────────────
            goal_positions = set((int(g[0]), int(g[1])) for g in env.base_env.goals)
            actions = [Action.NOOP.value, Action.NOOP.value]

            for i in range(env.n_agents):
                agent  = env.base_env.agents[i]
                other  = other_robot_pos(env, i)
```
* **Lines 198-204:** **Simulation State Machine Loop**:
  * Extracts RWARE goal coordinate slots and initializes actions to `NOOP` (`0`) for both agents.

```python
                # ── MOVING_TO_SHELF ──────────────────────────────────────────
                if phases[i] == RobotPhase.MOVING_TO_SHELF:
                    if action_queues[i]:
                        actions[i] = action_queues[i].pop(0)
                    elif assigned_shelves[i] and \
                         (agent.x, agent.y) == (int(assigned_shelves[i].x), int(assigned_shelves[i].y)):
                        phases[i] = RobotPhase.PICKING_UP
                        pickup_attempts[i] = 0
                        event_logs.append(f"[{step_count}] R{i} at shelf → TOGGLE")
                        actions[i] = Action.TOGGLE_LOAD.value
                    elif assigned_shelves[i]:
                        # Replan avoiding other robot
                        path = plan_safe((agent.x, agent.y),
                                         (int(assigned_shelves[i].x), int(assigned_shelves[i].y)),
                                         env.base_env, other_pos=other)
                        action_queues[i] = make_actions(path, agent.dir)
                        actions[i] = action_queues[i].pop(0) if action_queues[i] else Action.NOOP.value
```
* **Lines 206-222:** **Moving to Shelf Phase**:
  * If actions exist in the queue, pops the next action.
  * If the robot reaches the target coordinates, switches state to `PICKING_UP` and issues a `TOGGLE_LOAD` action.
  * If blocked along the way, plans a new path to the shelf avoiding the other robot.

```python
                # ── PICKING_UP ───────────────────────────────────────────────
                elif phases[i] == RobotPhase.PICKING_UP:
                    if agent.carrying_shelf is not None:
                        # Cancel any competing assignment to the same shelf
                        for j in range(2):
                            if j != i and assigned_shelves[j] == assigned_shelves[i]:
                                assigned_shelves[j] = None
                                phases[j]           = RobotPhase.IDLE
                                action_queues[j]    = []

                        nearest_goal = min(env.base_env.goals,
                                           key=lambda g: abs(agent.x - int(g[0])) + abs(agent.y - int(g[1])))
                        gx, gy = int(nearest_goal[0]), int(nearest_goal[1])
                        path = plan_safe((agent.x, agent.y), (gx, gy), env.base_env, other_pos=other)
                        action_queues[i] = make_actions(path, agent.dir)
                        phases[i]        = RobotPhase.MOVING_TO_GOAL
                        event_logs.append(f"[{step_count}] R{i} picked up → Goal")
                        actions[i] = action_queues[i].pop(0) if action_queues[i] else Action.NOOP.value
                    else:
                        pickup_attempts[i] += 1
                        if pickup_attempts[i] > 6:
                            # Give up — shelf may have moved; go back IDLE
                            phases[i]            = RobotPhase.IDLE
                            assigned_shelves[i]  = None
                            shelf_orig_positions[i] = None
                            action_queues[i]     = []
                            pickup_attempts[i]   = 0
                            event_logs.append(f"[{step_count}] R{i} pickup failed → IDLE")
                        else:
                            actions[i] = Action.TOGGLE_LOAD.value
```
* **Lines 224-253:** **Picking Up Phase**:
  * Verifies if RWARE has registered the shelf carrying state.
  * If carrying the shelf, clears any duplicate assignments, finds the nearest delivery goal slot, plans the delivery path, and switches state to `MOVING_TO_GOAL`.
  * If the pickup attempt fails (e.g., if a shelf is blocked or occupied), increments a timeout counter. If it fails 6 times, it drops the task and resets to `IDLE` to avoid getting stuck.

```python
                # ── MOVING_TO_GOAL ───────────────────────────────────────────
                elif phases[i] == RobotPhase.MOVING_TO_GOAL:
                    if action_queues[i]:
                        actions[i] = action_queues[i].pop(0)
                    elif (agent.x, agent.y) in goal_positions:
                        # Successfully at goal — plan return
                        orig = shelf_orig_positions[i]
                        path = plan_safe((agent.x, agent.y), orig, env.base_env, other_pos=other)
                        action_queues[i] = make_actions(path, agent.dir)
                        phases[i]        = RobotPhase.RETURNING_SHELF
                        deliveries_count += 1
                        event_logs.append(f"[{step_count}] R{i} DELIVERED! → returning")
                        actions[i] = action_queues[i].pop(0) if action_queues[i] else Action.NOOP.value
                    else:
                        # Not at goal yet — replan
                        nearest_goal = min(env.base_env.goals,
                                           key=lambda g: abs(agent.x - int(g[0])) + abs(agent.y - int(g[1])))
                        gx, gy = int(nearest_goal[0]), int(nearest_goal[1])
                        path = plan_safe((agent.x, agent.y), (gx, gy), env.base_env, other_pos=other)
                        action_queues[i] = make_actions(path, agent.dir)
                        actions[i] = action_queues[i].pop(0) if action_queues[i] else Action.NOOP.value
```
* **Lines 255-276:** **Moving to Goal Phase**:
  * Pops the next action to move the loaded shelf toward the goals.
  * Once at the goal slot, increments the delivery counter, logs the delivery, plans a return path to the shelf's original location, and switches state to `RETURNING_SHELF`.
  * Recalculates paths dynamically if dynamic obstacles block the route.

```python
                # ── RETURNING_SHELF ──────────────────────────────────────────
                elif phases[i] == RobotPhase.RETURNING_SHELF:
                    if action_queues[i]:
                        actions[i] = action_queues[i].pop(0)
                    elif (agent.x, agent.y) == shelf_orig_positions[i]:
                        phases[i]  = RobotPhase.DROPPING_OFF
                        actions[i] = Action.TOGGLE_LOAD.value
                    elif shelf_orig_positions[i]:
                        path = plan_safe((agent.x, agent.y), shelf_orig_positions[i],
                                         env.base_env, other_pos=other)
                        action_queues[i] = make_actions(path, agent.dir)
                        actions[i] = action_queues[i].pop(0) if action_queues[i] else Action.NOOP.value
```
* **Lines 278-289:** **Returning Shelf Phase**:
  * Navigates back to the shelf's original coordinates.
  * Once there, switches state to `DROPPING_OFF` and issues a `TOGGLE_LOAD` action to put the shelf back down.

```python
                # ── DROPPING_OFF ─────────────────────────────────────────────
                elif phases[i] == RobotPhase.DROPPING_OFF:
                    if agent.carrying_shelf is None:
                        phases[i]              = RobotPhase.IDLE
                        assigned_shelves[i]    = None
                        shelf_orig_positions[i] = None
                        event_logs.append(f"[{step_count}] R{i} shelf returned → IDLE")
                        actions[i] = Action.NOOP.value
                    else:
                        actions[i] = Action.TOGGLE_LOAD.value
```
* **Lines 291-300:** **Dropping Off Phase**:
  * Verifies if the shelf has been successfully dropped.
  * If dropped, resets the state to `IDLE` to allow the robot to bid on new tasks.

```python
            # ── 3. Smarter collision resolution ────────────────────────────────
            a0, a1 = env.base_env.agents[0], env.base_env.agents[1]
            dist = abs(a0.x - a1.x) + abs(a0.y - a1.y)
            if dist <= 1:
                loaded0, loaded1 = a0.carrying_shelf is not None, a1.carrying_shelf is not None
                if loaded0 and not loaded1:
                    # R1 yields: try to step sideways/forward instead of pure NOOP
                    actions[1] = _yield_action(a1, a0)
                elif loaded1 and not loaded0:
                    actions[0] = _yield_action(a0, a1)
                # If both free or both loaded, let RWARE handle it naturally
```
* **Lines 301-312:** **Collision Resolution**:
  * Checks if the Manhattan distance between the two robots is $\le 1$ cell.
  * If one robot is carrying a shelf and the other is not, the unloaded robot yields to the carrying robot by stepping aside (`_yield_action`), preventing gridlocks.

```python
            # ── 4. Step ────────────────────────────────────────────────────────
            obs, rewards, done, truncated, info = env.step(actions)
            step_count += 1

            if len(event_logs) > 8:
                event_logs.pop(0)
```
* **Lines 313-318:** Steps the environment with the resolved actions and increments the step counter. Limits the event logs to the 8 most recent messages.

```python
            # ── 5. Check if both batteries depleted → end simulation ────────────
            if env.battery[0] <= 0 and env.battery[1] <= 0:
                # Render a final "Simulation Over" overlay for 3 seconds
                overlay = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 180))
                screen.blit(overlay, (0, 0))
                cx, cy = screen_width // 2, screen_height // 2
                font_xl = pygame.font.SysFont("Arial", 32, bold=True)
                font_md = pygame.font.SysFont("Arial", 20)
                screen.blit(font_xl.render("⚡ Batteries Depleted", True, (241, 196, 15)),
                            font_xl.size("⚡ Batteries Depleted")[0] // 2 * 0
                            or (cx - font_xl.size("⚡ Batteries Depleted")[0] // 2, cy - 60))
                screen.blit(font_md.render("Simulation Over", True, (255, 255, 255)),
                            (cx - font_md.size("Simulation Over")[0] // 2, cy - 10))
                screen.blit(font_md.render(f"Total Steps: {step_count}", True, (200, 200, 200)),
                            (cx - font_md.size(f"Total Steps: {step_count}")[0] // 2, cy + 25))
                screen.blit(font_md.render(f"Deliveries:  {deliveries_count}", True, (200, 200, 200)),
                            (cx - font_md.size(f"Deliveries:  {deliveries_count}")[0] // 2, cy + 55))
                pygame.display.flip()
                pygame.time.wait(3000)   # show for 3 seconds
                running = False
                continue
```
* **Lines 320-341:** **End-of-Battery Termination**:
  * If both robot batteries hit $0$, creates a semi-transparent screen overlay.
  * Displays "⚡ Batteries Depleted" along with execution stats (Total Steps, Deliveries).
  * Shows the overlay for 3 seconds, then stops the visual run.

```python
        # ── Rendering ──────────────────────────────────────────────────────────
        screen.fill(BG_COLOR)

        for x in range(grid_width):
            for y in range(grid_height):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE + 50, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, GRID_COLOR, rect, 1)
```
* **Lines 343-350:** Clears the screen and draws the layout grid.

```python
        # Goals
        for gx, gy in env.base_env.goals:
            rect = pygame.Rect(gx * CELL_SIZE + 2, gy * CELL_SIZE + 52, CELL_SIZE - 4, CELL_SIZE - 4)
            pygame.draw.rect(screen, DELIVERY_COLOR, rect, 0, border_radius=4)
            lbl = font_small.render("GOAL", True, (255, 255, 255))
            screen.blit(lbl, (gx * CELL_SIZE + 6, gy * CELL_SIZE + 66))
```
* **Lines 351-357:** Renders delivery goal slots in teal.

```python
        # Shelves
        for shelf in env.base_env.shelfs:
            sx, sy = shelf.x, shelf.y
            rect = pygame.Rect(sx * CELL_SIZE + 6, sy * CELL_SIZE + 56, CELL_SIZE - 12, CELL_SIZE - 12)
            color = SHELF_COLOR if shelf in env.base_env.request_queue else (180, 190, 200)
            pygame.draw.rect(screen, color, rect, 0, border_radius=6)
```
* **Lines 358-364:** Renders warehouse shelves. Yellow shelves are active tasks waiting to be picked up; standard shelves are grey.

```python
        # Robots
        colors     = [ROBOT_0_COLOR, ROBOT_1_COLOR]
        phase_names = {0: "IDLE", 1: "TO_SHELF", 2: "PICKING",
                       3: "TO_GOAL", 4: "RETURNING", 5: "DROPPING"}
        for i, agent in enumerate(env.base_env.agents):
            rx = agent.x * CELL_SIZE + CELL_SIZE // 2
            ry = agent.y * CELL_SIZE + 50 + CELL_SIZE // 2
            pygame.draw.circle(screen, colors[i], (rx, ry), CELL_SIZE // 3)
            txt = font_small.render(f"R{i}", True, (255, 255, 255))
            screen.blit(txt, (rx - 8, ry - 8))
            batt_pct = env.battery[i] / 100.0
            bar_w    = int((CELL_SIZE - 8) * batt_pct)
            batt_col = (46, 204, 113) if batt_pct > 0.4 else (231, 76, 60)
            pygame.draw.rect(screen, (200, 200, 200),
                             (rx - CELL_SIZE // 2 + 4, ry + 16, CELL_SIZE - 8, 4))
            pygame.draw.rect(screen, batt_col,
                             (rx - CELL_SIZE // 2 + 4, ry + 16, bar_w, 4))
```
* **Lines 365-382:** **Robot & Battery UI Rendering**:
  * Renders Robot 0 (Blue) and Robot 1 (Orange) as circles.
  * Draws battery level indicator bars directly under each robot circle (green for good, red for low battery).

```python
        # HUD
        pygame.draw.rect(screen, PANEL_BG, (0, 0, screen_width, 45))
        pygame.draw.line(screen, GRID_COLOR, (0, 45), (screen_width, 45), 2)
        screen.blit(font_large.render("Mid-Review Multi-Robot Warehouse Simulation",
                                       True, TEXT_DARK), (12, 10))
        status_str = "PAUSED" if paused else "RUNNING"
        screen.blit(font_small.render(
            f"Step: {step_count}  |  {status_str}  |  Deliveries: {deliveries_count}",
            True, TEXT_DARK), (screen_width - 340, 14))
```
* **Lines 383-392:** Renders the top HUD panel showing steps, running state, and total successful deliveries.

```python
        # Sidebar
        sb_x = grid_width * CELL_SIZE + 10
        sb_panel = pygame.Rect(sb_x, 50, sidebar_width - 20, screen_height - 60)
        pygame.draw.rect(screen, PANEL_BG, sb_panel, 0, border_radius=8)
        pygame.draw.rect(screen, GRID_COLOR, sb_panel, 2, border_radius=8)
        screen.blit(font_large.render("Robot Status & Logs", True, TEXT_DARK), (sb_x + 12, 62))

        for i in range(2):
            cy = 95 + i * 80
            screen.blit(font_small.render(
                f"Robot {i} ({env.robot_type_names[i]}) [{phase_names.get(phases[i], '?')}]",
                True, colors[i]), (sb_x + 12, cy))
            screen.blit(font_small.render(
                f"Battery: {env.battery[i]:.1f}%  Stuck: {stuck_counters[i]}",
                True, TEXT_DARK), (sb_x + 12, cy + 18))

        screen.blit(font_large.render("Live Event Log:", True, TEXT_DARK), (sb_x + 12, 265))
        for idx, entry in enumerate(event_logs):
            screen.blit(font_small.render(entry, True, (80, 90, 100)),
                        (sb_x + 12, 290 + idx * 20))
```
* **Lines 393-413:** Renders the right sidebar containing:
  * Individual robot state cards (archetype name, current phase, battery level, stuck counters).
  * The scrolling live event log displaying auctions and actions.

```python
        pygame.display.flip()
        clock.tick(5 if not paused else 15)
```
* **Lines 414-415:** Updates the display and sets the frame rate (5 frames/sec when running, 15 frames/sec when paused).

```python
    pygame.quit()
    sys.exit(0)
```
* **Lines 417-418:** Shuts down Pygame and exits the process cleanly.

```python
def _yield_action(yielder, priority_agent):
    """
    Generate a move action for 'yielder' that steps AWAY from priority_agent
    rather than simply doing NOOP (to avoid perpetual freeze).
    Uses a simple perpendicular-step heuristic.
    """
    dx = yielder.x - priority_agent.x
    dy = yielder.y - priority_agent.y
    # Step perpendicular to the blocker direction: swap axes, pick first valid
    candidates = [
        Action.RIGHT.value,   # turn clockwise
        Action.LEFT.value,    # turn counter-clockwise
        Action.NOOP.value,    # fallback
    ]
    # If difference is horizontal → try to face perpendicular (vertical move)
    # Simple heuristic: just issue a turn so next step they won't be head-to-head
    if abs(dx) >= abs(dy):
        return Action.LEFT.value   # turn away
    else:
        return Action.RIGHT.value
```
* **Lines 421-441:** **Yield Action Heuristic**:
  * Calculates the distance vector between the yielding robot and the prioritized agent.
  * If the blocking conflict is horizontal, returns a turn action to face perpendicular (vertical), allowing the robot to step aside on subsequent ticks instead of causing a gridlock.

```python
if __name__ == "__main__":
    main()
```
* **Lines 443-445:** Entry point block executing the script.
